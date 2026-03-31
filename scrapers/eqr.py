"""
EQR (Equity Residential) scraper.

Rendering:   Angular SPA — unit data is server-side rendered into
             window.ea5 JavaScript objects on each property page.
             Cloudflare Managed Challenge blocks plain HTTP requests;
             a real Chromium browser (Playwright) is required.

Requires:    playwright + playwright-stealth + Chromium browser:
               pip install playwright playwright-stealth
               playwright install chromium

Discovery:   13 market hub pages → window.ea5.srp.MetroArea.Properties[]
             Typically 300–320 communities.

Unit data:   window.ea5.unitAvailability.BedroomTypes[].AvailableUnits[]
  Each entry is one available unit (not a floor-plan group) with:
  UnitId, Bed, Bath, SqFt, BestTerm {Length, Price}, Floor,
  AvailableDate, FloorplanName, Special {Active, Title}

Property:    window.ea5.property
  Name, Address, City, State, Zip, Coordinates

Concession:  unit.Special.Title (per-unit text, when Special.Active=true)
             → shared parse_concession() engine from maa.py
"""

import logging
import os
import sys
import time
from datetime import date, datetime
from typing import Optional

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.listing import SCHEMA_COLUMNS
from scrapers.maa import parse_concession, make_unit_id
from utils.common import today_str

REIT = "EQR"
BASE_URL = "https://www.equityapartments.com"

# All 13 EQR market hub pages (order matches EQR's own website nav)
_MARKET_SLUGS = [
    "new-york-city-apartments",
    "washington-dc-apartments",
    "san-francisco-bay-apartments",
    "los-angeles-apartments",
    "boston-apartments",
    "seattle-apartments",
    "san-diego-apartments",
    "orange-county-apartments",
    "inland-empire-apartments",
    "denver-apartments",
    "austin-apartments",
    "dallas-apartments",
    "atlanta-apartments",
]

_MARKET_LABEL = {
    "new-york-city-apartments":     "New York",
    "washington-dc-apartments":     "Washington DC",
    "san-francisco-bay-apartments": "San Francisco",
    "los-angeles-apartments":       "Los Angeles",
    "boston-apartments":            "Boston",
    "seattle-apartments":           "Seattle",
    "san-diego-apartments":         "San Diego",
    "orange-county-apartments":     "Orange County",
    "inland-empire-apartments":     "Inland Empire",
    "denver-apartments":            "Denver",
    "austin-apartments":            "Austin",
    "dallas-apartments":            "Dallas",
    "atlanta-apartments":           "Atlanta",
}

logger = logging.getLogger(__name__)


# ── Playwright helpers ────────────────────────────────────────────────────────

# CF challenge page titles (exact match / contains)
_CF_TITLES = ("just a moment", "attention required", "access denied", "403 forbidden")


def _is_cf_blocked(page) -> bool:
    """Return True if the page appears to be a Cloudflare challenge/block."""
    title = (page.title() or "").lower()
    return any(t in title for t in _CF_TITLES)


def _navigate(page, url: str, wait_fn: str, timeout_ms: int = 45_000) -> bool:
    """
    Navigate to url, wait for Angular data to land (via wait_fn JS expression).
    Handles Cloudflare challenge pages with a grace-period retry.
    Returns True on success, False on any timeout or error.
    """
    try:
        page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
    except Exception as e:
        logger.warning(f"  goto failed [{url}]: {type(e).__name__}")
        return False

    # If CF challenge detected, wait up to 20 s for it to auto-clear
    if _is_cf_blocked(page):
        logger.info(f"  CF challenge on {url} — waiting …")
        for _ in range(4):
            page.wait_for_timeout(5_000)
            if not _is_cf_blocked(page):
                break
        if _is_cf_blocked(page):
            logger.warning(f"  CF challenge persisted — skipping {url}")
            return False

    # Wait for Angular data object
    try:
        page.wait_for_function(wait_fn, timeout=20_000)
        return True
    except Exception as e:
        logger.warning(f"  wait_for_function timed out [{url}]: {type(e).__name__}")
        return False


def _nav_market(page, url: str) -> bool:
    return _navigate(
        page, url,
        wait_fn=(
            "window.ea5 && window.ea5.srp && "
            "window.ea5.srp.MetroArea && "
            "window.ea5.srp.MetroArea.Properties"
        ),
    )


def _nav_property(page, url: str) -> bool:
    return _navigate(
        page, url,
        wait_fn="window.ea5 && 'unitAvailability' in window.ea5",
    )


# ── Community discovery ────────────────────────────────────────────────────────

def get_communities(page) -> list[dict]:
    """
    Visit all 13 market pages and collect every unique EQR community.
    Uses window.ea5.srp.MetroArea.Properties (SSR data) for clean extraction.

    Returns list of dicts:
      {id, name, url, address, city, state, zip, latitude, longitude, market}
    """
    seen: dict[int, dict] = {}

    for slug in _MARKET_SLUGS:
        market_url = f"{BASE_URL}/{slug}"
        market_label = _MARKET_LABEL[slug]
        logger.info(f"  Discovering {slug} …")

        if not _nav_market(page, market_url):
            logger.warning(f"  Skipped market: {slug}")
            continue

        try:
            props = page.evaluate("window.ea5.srp.MetroArea.Properties")
        except Exception as e:
            logger.warning(f"  Could not read MetroArea.Properties on {slug}: {e}")
            continue

        if not props:
            logger.warning(f"  Empty property list on {slug}")
            continue

        new_count = 0
        for p in props:
            pid = p.get("Id")
            if pid is None or pid in seen:
                continue

            coords = p.get("Coordinates") or {}
            seen[pid] = {
                "id":        pid,
                "name":      (p.get("Name") or "").strip(),
                "url":       BASE_URL + (p.get("Url") or ""),
                "address":   (p.get("Address") or "").strip(),
                "city":      (p.get("City") or "").strip(),
                "state":     (p.get("State") or "").strip(),
                "zip":       (p.get("Zip") or "").strip(),
                "latitude":  coords.get("Latitude"),
                "longitude": coords.get("Longitude"),
                "market":    market_label,
            }
            new_count += 1

        logger.info(f"    {len(props)} properties ({new_count} new) → {len(seen)} total")
        time.sleep(1.0)

    return list(seen.values())


# ── Unit extraction ────────────────────────────────────────────────────────────

def extract_units(page, community_meta: dict) -> list[dict]:
    """
    Read window.ea5.unitAvailability from an already-loaded property page.
    Expands BedroomTypes → AvailableUnits into one row per unit.
    Returns list of row dicts conforming to SCHEMA_COLUMNS.
    """
    scrape_dt = date.today()
    rows: list[dict] = []

    try:
        ua = page.evaluate("window.ea5.unitAvailability")
    except Exception as e:
        logger.warning(f"  Could not read unitAvailability: {e}")
        return []

    if not ua or not ua.get("BedroomTypes"):
        return []

    # Build full address string
    address_parts = [community_meta["address"]]
    city_state = f"{community_meta['city']}, {community_meta['state']} {community_meta['zip']}".strip(", ")
    if city_state.strip():
        address_parts.append(city_state)
    address_full = ", ".join(p for p in address_parts if p)

    for bt in ua["BedroomTypes"]:
        bed_count = bt.get("BedroomCount", 0)

        for unit in bt.get("AvailableUnits") or []:
            # Rent from BestTerm (cheapest / default lease length)
            best_term = unit.get("BestTerm") or {}
            rent  = best_term.get("Price")
            lease = best_term.get("Length")   # integer months

            if not rent:
                continue   # skip units with no listed price

            # Parse move-in date: "4/17/2026"
            move_in_date = None
            avail_raw = (unit.get("AvailableDate") or "").strip()
            if avail_raw:
                try:
                    move_in_date = datetime.strptime(avail_raw, "%m/%d/%Y").date()
                except ValueError:
                    pass

            # Concession: Special.Title only when Special.Active is True
            # Exclude deposit-reduction offers ("Security Deposit Special") — these
            # are not rent concessions and would inflate concession metrics.
            special = unit.get("Special") or {}
            if special.get("Active"):
                _raw = (special.get("Title") or "").strip()
                _lower = _raw.lower()
                # Skip if the offer is about deposit, admin fee, or application fee
                _is_fee_offer = any(kw in _lower for kw in (
                    "security deposit", "admin fee", "application fee",
                ))
                concession_raw = _raw if (_raw and not _is_fee_offer) else None
            else:
                concession_raw = None

            concession_fields = parse_concession(
                raw=concession_raw,
                rent=float(rent),
                lease_months=lease or 12,
            )

            sqft        = unit.get("SqFt") or None
            unit_number = str(unit.get("UnitId") or "")

            unit_id = make_unit_id(
                reit=REIT,
                community=community_meta["name"],
                unit_number=unit_number,
                beds=bed_count,
                sqft=sqft,
            )

            row = {
                # Core 23 fields
                "scrape_date":   scrape_dt,
                "reit":          REIT,
                "community":     community_meta["name"],
                "address":       address_full,
                "market":        community_meta["market"],
                "unit_id":       unit_id,
                "beds":          bed_count,
                "baths":         unit.get("Bath"),
                "sqft":          int(sqft) if sqft else None,
                "rent":          float(rent),
                "move_in_date":  move_in_date,
                "lease_term":    lease,
                "listing_url":   community_meta["url"],
                "first_seen":    scrape_dt,
                "last_seen":     scrape_dt,
                **concession_fields,
                # Supplemental
                "state":               community_meta["state"],
                "city":                community_meta["city"],
                "latitude":            community_meta.get("latitude"),
                "longitude":           community_meta.get("longitude"),
                "floorplan_name":      unit.get("FloorplanName"),
                "floor_level":         unit.get("Floor"),
                "rentcafe_property_id": str(community_meta["id"]),
            }
            rows.append(row)

    return rows


# ── Main scrape function ───────────────────────────────────────────────────────

def scrape_eqr(
    limit: Optional[int] = None,
    headless: bool = True,
) -> pd.DataFrame:
    """
    Scrape all EQR communities. Returns a DataFrame with SCHEMA_COLUMNS columns.

    Args:
        limit:    If set, restrict to first N communities (for testing).
        headless: Set False if Cloudflare blocks the headless browser.
                  Running headed (visible Chrome) bypasses CF fingerprinting.
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        raise ImportError(
            "playwright is required for EQR scraping.\n"
            "Install:  pip install playwright playwright-stealth && playwright install chromium"
        )

    # Optional stealth — masks automation fingerprints to pass CF managed challenge
    try:
        from playwright_stealth import Stealth as _Stealth
        _stealth_obj = _Stealth()
        _HAS_STEALTH = True
    except ImportError:
        _stealth_obj = None
        _HAS_STEALTH = False
        logger.warning("playwright-stealth not found — CF may block headless browser. "
                       "Install with: pip install playwright-stealth")

    all_rows: list[dict] = []
    errors:   list[str]  = []

    # Prefer system Chrome (harder for CF to fingerprint as automation)
    _CHROME_PATHS = [
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    ]
    _chrome_exe = next((p for p in _CHROME_PATHS if os.path.exists(p)), None)
    _launch_kwargs: dict = dict(
        headless=headless,
        args=[
            "--disable-blink-features=AutomationControlled",
            "--no-sandbox",
            "--disable-dev-shm-usage",
            "--disable-infobars",
            "--window-size=1280,800",
        ],
    )
    if _chrome_exe:
        _launch_kwargs["executable_path"] = _chrome_exe
        logger.info(f"EQR: using system Chrome at {_chrome_exe}")
    else:
        logger.info("EQR: system Chrome not found — using Playwright Chromium")

    with sync_playwright() as pw:
        browser = pw.chromium.launch(**_launch_kwargs)
        ctx = browser.new_context(
            viewport={"width": 1280, "height": 800},
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            locale="en-US",
            java_script_enabled=True,
        )
        page = ctx.new_page()

        # Apply stealth patches (removes navigator.webdriver, etc.)
        if _HAS_STEALTH:
            _stealth_obj.apply_stealth_sync(page)

        # Suppress console noise from the Angular app
        page.on("console", lambda _: None)

        # ── 1. CF warm-up: hit homepage once to establish session ──────────
        logger.info("EQR: warming up Cloudflare session …")
        try:
            page.goto(BASE_URL, wait_until="domcontentloaded", timeout=45_000)
            # If CF challenge appears on homepage, wait for it to clear
            if _is_cf_blocked(page):
                logger.info("  CF challenge on homepage — waiting up to 30 s …")
                page.wait_for_timeout(30_000)
            else:
                page.wait_for_timeout(2_000)
        except Exception as e:
            logger.warning(f"Warm-up issue (may be fine): {e}")

        # ── 2. Discover all communities from market pages ──────────────────
        logger.info("EQR: discovering communities via market pages …")
        communities = get_communities(page)
        if limit:
            communities = communities[:limit]
        logger.info(f"EQR: scraping {len(communities)} communities …")

        # ── 3. Scrape each community page ──────────────────────────────────
        for i, comm in enumerate(communities, 1):
            url = comm["url"]
            logger.info(f"[{i}/{len(communities)}]  {comm['name']}  ({comm['market']})")

            if not _nav_property(page, url):
                errors.append(url)
                logger.warning(f"  Skipped.")
                continue

            rows = extract_units(page, comm)
            logger.info(f"  → {len(rows)} units")
            all_rows.extend(rows)

            time.sleep(1.0)   # polite inter-page delay

        browser.close()

    if errors:
        logger.warning(f"Failed/skipped {len(errors)} URLs: {errors[:5]}")

    if not all_rows:
        logger.error("EQR: 0 rows collected — returning empty DataFrame.")
        df = pd.DataFrame(columns=SCHEMA_COLUMNS)
        return df

    df = pd.DataFrame(all_rows)
    for col in SCHEMA_COLUMNS:
        if col not in df.columns:
            df[col] = None
    df = df[SCHEMA_COLUMNS]

    logger.info(f"EQR: total rows: {len(df):,}")
    return df


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser(description="Scrape EQR rental listings")
    parser.add_argument("--limit",  type=int,  default=None,
                        help="Limit to first N communities (testing)")
    parser.add_argument("--save",   action="store_true",
                        help="Save output CSV to data/raw/")
    parser.add_argument("--headed", action="store_true",
                        help="Run browser visibly (use if CF blocks headless)")
    args = parser.parse_args()

    df = scrape_eqr(limit=args.limit, headless=not args.headed)

    if df.empty:
        print("No data returned.")
    else:
        display_cols = [
            "community", "market", "unit_id", "beds", "sqft", "rent",
            "lease_term", "concession_hardness", "concession_type",
            "concession_value", "effective_monthly_rent",
        ]
        print(df[[c for c in display_cols if c in df.columns]].head(15).to_string())
        print(f"\nShape: {df.shape}")

        if "has_concession" in df.columns and df["has_concession"].any():
            hard = df[df["concession_hardness"] == "hard"]
            soft = df[df["concession_hardness"] == "soft"]
            none_ = df[df["has_concession"] == False]
            print(f"\nConcession breakdown: hard={len(hard)}  soft={len(soft)}  none={len(none_)}")
            if len(hard):
                print("\nConcession types:\n", hard["concession_type"].value_counts().to_string())
                print("\nSample hard concessions:")
                print(hard[[
                    "community", "rent", "concession_type", "concession_value",
                    "concession_pct_lease_value", "effective_monthly_rent", "concession_raw",
                ]].drop_duplicates("community").head(10).to_string())

    if args.save and not df.empty:
        out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw")
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"eqr_raw_{today_str()}.csv")
        df.to_csv(path, index=False)
        print(f"\nSaved -> {path}")
