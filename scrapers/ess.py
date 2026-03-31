"""
ESS (Essex Property Trust) scraper.

Rendering:   Sitecore CMS — floor plan cards are client-side rendered via
             JavaScript. requests returns no unit data; Playwright required.
             ESS has NO Cloudflare protection → default headless Chromium works.

Discovery:   sitemap.xml → ~520 community page URLs
             Pattern: /apartments/{city-slug}/{community-slug}

Floor plans: Navigate to /{community}/floor-plans-and-pricing
             Wait for .floor-plan-card elements (typically 5–40 per community).
             Data is floor-plan-level (not individual unit level) — ESS does not
             expose unit IDs publicly on their website.

Data per floor-plan card:
  .floor-plan-card__content__layout      → floorplan_name ("Plan 1A")
  .floor-plan-card__content__price       → rent ("Starting base rent $2,779")
  .floor-plan-card__content__size        → beds/baths + sqft ("1 Bed / 1 Bath\\n613 sq. ft.")
  .floor-plan-card__content__availability → move_in ("Available as soon as: 04/02/2026")
  .floor-plan-card__special-offer        → per-plan concession text

Community:   window.currentProperty → name
             window.currentPropertyId → numeric property ID (rentcafe_property_id)
             Header banner text → community-level concession fallback

unit_id:     Synthetic composite — ESS-{community}-{beds}br-{sqft}sqft-{plan-slug}
             Tracks floor-plan availability over time (one row per plan per week).

Markets:     ESS operates exclusively in CA (SoCal, Bay Area, OC, SD) + Seattle metro.
"""

import logging
import os
import re
import sys
import time
from datetime import date
from typing import Optional
from xml.etree import ElementTree

import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.listing import SCHEMA_COLUMNS
from scrapers.maa import parse_concession, make_unit_id
from utils.common import today_str

REIT     = "ESS"
BASE_URL = "https://www.essexapartmenthomes.com"
SITEMAP  = "https://www.essexapartmenthomes.com/sitemap.xml"

logger = logging.getLogger(__name__)

# ── Playwright setup ───────────────────────────────────────────────────────────
# ESS has no Cloudflare — standard headless Chromium works fine.

try:
    from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout
    _HAS_PLAYWRIGHT = True
except ImportError:
    _HAS_PLAYWRIGHT = False
    logger.error("playwright not installed — run: pip install playwright && playwright install chromium")


# ── Market mapping ─────────────────────────────────────────────────────────────

_MARKET_MAP: dict[str, str] = {
    # Southern California — LA Basin
    "los-angeles":    "Los Angeles",
    "west-hollywood": "Los Angeles",
    "marina-del-rey": "Los Angeles",
    "culver-city":    "Los Angeles",
    "el-segundo":     "Los Angeles",
    "hawthorne":      "Los Angeles",
    "redondo-beach":  "Los Angeles",
    "torrance":       "Los Angeles",
    "long-beach":     "Los Angeles",
    "burbank":        "Los Angeles",
    "glendale":       "Los Angeles",
    "north-hollywood":"Los Angeles",
    "woodland-hills": "Los Angeles",
    "calabasas":      "Los Angeles",
    "agoura-hills":   "Los Angeles",
    "thousand-oaks":  "Los Angeles",
    "oxnard":         "Los Angeles",
    "camarillo":      "Los Angeles",
    "signal-hill":    "Los Angeles",
    "manhattan-beach":"Los Angeles",
    # Orange County
    "irvine":         "Orange County",
    "costa-mesa":     "Orange County",
    "anaheim":        "Orange County",
    "fullerton":      "Orange County",
    "orange":         "Orange County",
    "aliso-viejo":    "Orange County",
    "laguna-niguel":  "Orange County",
    # San Diego
    "san-diego":      "San Diego",
    "carlsbad":       "San Diego",
    "oceanside":      "San Diego",
    "chula-vista":    "San Diego",
    "del-mar":        "San Diego",
    "encinitas":      "San Diego",
    # Bay Area — South Bay
    "san-jose":       "San Jose",
    "santa-clara":    "San Jose",
    "sunnyvale":      "San Jose",
    "mountain-view":  "San Jose",
    "campbell":       "San Jose",
    "milpitas":       "San Jose",
    "los-gatos":      "San Jose",
    "saratoga":       "San Jose",
    # Bay Area — East / Mid Peninsula
    "fremont":        "Bay Area",
    "newark":         "Bay Area",
    "pleasanton":     "Bay Area",
    "san-mateo":      "Bay Area",
    "redwood-city":   "Bay Area",
    "foster-city":    "Bay Area",
    "san-ramon":      "Bay Area",
    "walnut-creek":   "Bay Area",
    "concord":        "Bay Area",
    # Bay Area — SF
    "san-francisco":  "San Francisco",
    # Seattle metro
    "seattle":        "Seattle",
    "bellevue":       "Seattle",
    "redmond":        "Seattle",
    "kirkland":       "Seattle",
    "bothell":        "Seattle",
    "lynnwood":       "Seattle",
    "renton":         "Seattle",
    "kent":           "Seattle",
    "issaquah":       "Seattle",
    "newcastle":      "Seattle",
}

def _market_from_city_slug(city_slug: str) -> str:
    m = _MARKET_MAP.get(city_slug)
    if m:
        return m
    return city_slug.replace("-", " ").title()


# ── Sitemap discovery ──────────────────────────────────────────────────────────

# Matches /apartments/{city-slug}/{community-slug} (exactly 2 path segments after /apartments/)
_COMM_URL_RE = re.compile(
    r"^https://www\.essexapartmenthomes\.com/apartments/([a-z0-9-]+)/([a-z0-9-]+)$"
)

def get_communities() -> list[dict]:
    """Parse sitemap and return all community page records."""
    logger.info(f"Fetching sitemap: {SITEMAP}")
    try:
        resp = requests.get(SITEMAP, timeout=30,
                            headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
    except requests.RequestException as e:
        logger.error(f"Sitemap fetch failed: {e}")
        return []

    seen: dict[str, dict] = {}
    try:
        root = ElementTree.fromstring(resp.content)
        ns   = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
        for loc in root.findall(".//sm:url/sm:loc", ns):
            url = loc.text.strip().rstrip("/")  # normalize trailing slash
            m   = _COMM_URL_RE.match(url)
            if m and url not in seen:
                city_slug = m.group(1)
                comm_slug = m.group(2)
                seen[url] = {
                    "url":       url,
                    "city_slug": city_slug,
                    "comm_slug": comm_slug,
                    "market":    _market_from_city_slug(city_slug),
                }
    except ElementTree.ParseError as e:
        logger.error(f"Sitemap parse error: {e}")

    logger.info(f"Found {len(seen)} community URLs in sitemap")
    return list(seen.values())


# ── Text parsers for floor-plan card fields ────────────────────────────────────

_RENT_RE      = re.compile(r"\$\s*([\d,]+)")
_BEDS_RE      = re.compile(r"(\d+)\s*Bed|Studio|studio", re.IGNORECASE)
_BATHS_RE     = re.compile(r"([\d.]+)\s*Bath", re.IGNORECASE)
_SQFT_RE      = re.compile(r"([\d,]+)\s*(?:-\s*[\d,]+\s*)?sq\.?\s*ft", re.IGNORECASE)
_AVAIL_RE     = re.compile(r"(\d{1,2}/\d{1,2}/\d{4})", re.IGNORECASE)
_CONCESSION_KW = re.compile(
    r"\b(free|off|save|saving|discount|reduc|waiv|month|week|move.?in|no\s+rent)\b",
    re.IGNORECASE,
)
_DEPOSIT_KW = ("security deposit", "admin fee", "application fee", "holding deposit")


def _parse_rent(price_text: str) -> Optional[float]:
    m = _RENT_RE.search(price_text or "")
    if not m:
        return None
    return float(m.group(1).replace(",", ""))


def _parse_beds(size_text: str) -> Optional[int]:
    t = size_text or ""
    if re.search(r"\bstudio\b", t, re.IGNORECASE):
        return 0
    m = _BEDS_RE.search(t)
    return int(m.group(1)) if m and m.group(1) else None


def _parse_baths(size_text: str) -> Optional[float]:
    m = _BATHS_RE.search(size_text or "")
    return float(m.group(1)) if m else None


def _parse_sqft(size_text: str) -> Optional[int]:
    """Return the minimum sqft value from 'X - Y sq. ft.' or 'X sq. ft.'."""
    for m in _SQFT_RE.finditer(size_text or ""):
        try:
            return int(m.group(1).replace(",", ""))
        except ValueError:
            pass
    return None


def _parse_avail_date(avail_text: str) -> Optional[date]:
    m = _AVAIL_RE.search(avail_text or "")
    if not m:
        return None
    try:
        month, day, year = m.group(1).split("/")
        return date(int(year), int(month), int(day))
    except ValueError:
        return None


def _filter_concession(raw: Optional[str]) -> Optional[str]:
    """Return text only if it's a real concession (not a deposit/fee special)."""
    if not raw or not raw.strip():
        return None
    lower = raw.lower()
    if any(kw in lower for kw in _DEPOSIT_KW):
        return None
    if _CONCESSION_KW.search(raw):
        return raw.strip()
    return None


# ── Per-community Playwright scrape ───────────────────────────────────────────

def _scrape_community(page, comm: dict) -> list[dict]:
    """
    Navigate to the community floor-plans page, extract all floor-plan cards.
    Returns list of row dicts.
    """
    scrape_dt   = date.today()
    fp_url      = comm["url"] + "/floor-plans-and-pricing"
    city_slug   = comm["city_slug"]
    comm_slug   = comm["comm_slug"]

    try:
        page.goto(fp_url, wait_until="domcontentloaded", timeout=45_000)
        # Wait for floor-plan cards to appear
        page.wait_for_selector(".floor-plan-card", timeout=20_000)
    except PWTimeout:
        logger.warning(f"  Timeout waiting for floor-plan-card on {fp_url}")
        return []
    except Exception as e:
        logger.warning(f"  Navigation error on {fp_url}: {e}")
        return []

    # Extract community-level metadata from JS globals
    try:
        comm_name = page.evaluate("window.currentProperty || null")
        prop_id   = page.evaluate("window.currentPropertyId || null")
    except Exception:
        comm_name = None
        prop_id   = None

    # Fallback name from slug
    if not comm_name:
        comm_name = comm_slug.replace("-", " ").title()

    # Community-level concession banner
    comm_concession = None
    try:
        # Common banner selectors in Essex CMS
        for sel in [
            ".floor-plan-card__special-offer",
            ".community-header [class*='special']",
            "[class*='special-offer']",
            "[class*='promo-banner']",
            ".special-offer",
        ]:
            el = page.query_selector(sel)
            if el:
                txt = (el.inner_text() or "").strip()
                comm_concession = _filter_concession(txt)
                if comm_concession:
                    break

        # Also scan the header text for concession keywords
        if not comm_concession:
            header = page.query_selector(".community-header")
            if header:
                header_txt = (header.inner_text() or "").strip()
                for line in header_txt.split("\n"):
                    line = line.strip()
                    candidate = _filter_concession(line)
                    if candidate and len(candidate) > 15:
                        comm_concession = candidate
                        break
    except Exception:
        pass

    # Address / geo from schema.org JSON-LD on page
    address_str = ""
    state_val   = ""
    city_val    = ""
    lat_val     = None
    lon_val     = None
    try:
        ld_data = page.evaluate("""
            () => {
                const scripts = document.querySelectorAll('script[type="application/ld+json"]');
                for (const s of scripts) {
                    try {
                        const d = JSON.parse(s.textContent);
                        if (d && d.address) return d;
                    } catch(e) {}
                }
                return null;
            }
        """)
        if ld_data:
            addr = ld_data.get("address", {})
            street  = addr.get("streetAddress", "")
            city_v  = addr.get("addressLocality", "")
            state_v = addr.get("addressRegion", "")
            zip_v   = addr.get("postalCode", "")
            address_str = ", ".join(p for p in [street, city_v, state_v, zip_v] if p)
            state_val   = state_v
            city_val    = city_v
            geo = ld_data.get("geo", {})
            lat_val = geo.get("latitude")
            lon_val = geo.get("longitude")
    except Exception:
        pass

    # Extract floor-plan cards
    cards = page.query_selector_all(".floor-plan-card")
    rows: list[dict] = []

    for card in cards:
        try:
            get_txt = lambda sel: (card.query_selector(sel) or None) and \
                                  card.query_selector(sel).inner_text().strip()

            plan_name    = get_txt(".floor-plan-card__content__layout")
            price_text   = get_txt(".floor-plan-card__content__price")
            size_text    = get_txt(".floor-plan-card__content__size")
            avail_text   = get_txt(".floor-plan-card__content__availability")
            offer_text   = get_txt(".floor-plan-card__special-offer")

            rent         = _parse_rent(price_text)
            if rent is None:
                continue

            beds         = _parse_beds(size_text)
            baths        = _parse_baths(size_text)
            sqft         = _parse_sqft(size_text)
            move_in_date = _parse_avail_date(avail_text)

            # Concession: card-level first, fall back to community banner
            card_concession = _filter_concession(offer_text)
            concession_raw  = card_concession or comm_concession

            concession_fields = parse_concession(
                raw=concession_raw,
                rent=rent,
                lease_months=12,  # ESS doesn't expose lease term in cards
            )

            plan_slug = re.sub(r"[^a-z0-9]+", "-", (plan_name or "").lower()).strip("-")
            uid = make_unit_id(
                reit=REIT,
                community=comm_name,
                unit_number=plan_slug,
                beds=beds,
                sqft=sqft,
            )

            row = {
                "scrape_date":  scrape_dt,
                "reit":         REIT,
                "community":    comm_name,
                "address":      address_str,
                "market":       comm["market"],
                "unit_id":      uid,
                "beds":         beds,
                "baths":        baths,
                "sqft":         sqft,
                "rent":         rent,
                "move_in_date": move_in_date,
                "lease_term":   None,       # not exposed at floor-plan level
                "listing_url":  fp_url,
                "first_seen":   scrape_dt,
                "last_seen":    scrape_dt,
                **concession_fields,
                "state":               state_val or None,
                "city":                city_val or None,
                "latitude":            lat_val,
                "longitude":           lon_val,
                "floorplan_name":      plan_name,
                "floor_level":         None,
                "rentcafe_property_id": str(prop_id) if prop_id else "",
            }
            rows.append(row)

        except Exception as e:
            logger.debug(f"    Card parse error: {e}")
            continue

    return rows


# ── Main scrape function ───────────────────────────────────────────────────────

def scrape_ess(limit: Optional[int] = None) -> pd.DataFrame:
    """
    Scrape all ESS communities. Returns a DataFrame with SCHEMA_COLUMNS.
    limit: restrict to first N communities (testing).
    """
    if not _HAS_PLAYWRIGHT:
        logger.error("Playwright not available — cannot scrape ESS.")
        return pd.DataFrame(columns=SCHEMA_COLUMNS)

    all_rows: list[dict] = []
    errors:   list[str]  = []

    logger.info("Discovering ESS communities via sitemap ...")
    communities = get_communities()
    if limit:
        communities = communities[:limit]
    logger.info(f"Scraping {len(communities)} communities ...")

    with sync_playwright() as pw:
        browser = pw.chromium.launch(
            headless=True,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--disable-gpu",
            ],
        )
        context = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/123.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1280, "height": 800},
            java_script_enabled=True,
        )
        # Block images, fonts, media to speed up page loads
        context.route(
            "**/*.{png,jpg,jpeg,gif,webp,svg,ico,woff,woff2,ttf,mp4,webm}",
            lambda route: route.abort(),
        )
        page = context.new_page()

        for i, comm in enumerate(communities, 1):
            logger.info(
                f"[{i}/{len(communities)}] {comm['comm_slug']} ({comm['market']})"
            )

            rows = _scrape_community(page, comm)

            if not rows:
                logger.warning(f"  0 rows — skipping")
                errors.append(comm["url"])
            else:
                n_conc = sum(1 for r in rows if r.get("has_concession"))
                logger.info(f"  {len(rows)} plans | concession: {n_conc}/{len(rows)}")
                all_rows.extend(rows)

            # Polite delay between communities
            time.sleep(1.0)

        page.close()
        context.close()
        browser.close()

    if errors:
        logger.warning(f"Failed/skipped {len(errors)} communities: {errors[:5]}")

    df = pd.DataFrame(all_rows)
    for col in SCHEMA_COLUMNS:
        if col not in df.columns:
            df[col] = None
    df = df[SCHEMA_COLUMNS]

    logger.info(f"ESS total rows: {len(df):,}")
    return df


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser(description="Scrape ESS rental listings")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit to first N communities (testing)")
    parser.add_argument("--save",  action="store_true",
                        help="Save output CSV to data/raw/")
    args = parser.parse_args()

    df = scrape_ess(limit=args.limit)

    display_cols = [
        "community", "market", "unit_id", "beds", "sqft", "rent",
        "move_in_date", "concession_hardness", "concession_type",
        "concession_value", "effective_monthly_rent",
    ]
    available = [c for c in display_cols if c in df.columns]
    print(df[available].head(15).to_string())
    print(f"\nShape: {df.shape}")

    if not df.empty and df["has_concession"].any():
        hard = df[df["concession_hardness"] == "hard"]
        soft = df[df["concession_hardness"] == "soft"]
        n_none = len(df) - len(hard) - len(soft)
        print(f"\nConcession breakdown: hard={len(hard)} soft={len(soft)} none={n_none}")
        if len(hard):
            print("\nConcession types:\n", hard["concession_type"].value_counts())
            print("\nSample concessions:")
            print(hard[[
                "community", "rent", "concession_type", "concession_value",
                "concession_pct_lease_value", "effective_monthly_rent", "concession_raw",
            ]].drop_duplicates("community").head(10).to_string())

    if args.save:
        out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw")
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"ess_raw_{today_str()}.csv")
        df.to_csv(path, index=False)
        print(f"\nSaved -> {path}")
