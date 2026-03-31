"""
UDR (UDR, Inc.) scraper.

Rendering:   Custom CMS — server-side rendered. All unit data is embedded in
             window.udr.jsonObjPropertyViewModel on the /apartments-pricing/
             sub-page. Plain requests works; no browser required.

Discovery:   sitemap.xml → ~61 community URLs
             Pattern: /{market-slug}/{neighborhood-slug}/{community-slug}/

Unit data:   window.udr.jsonObjPropertyViewModel.floorPlans[].units[]
  apartmentId         → internal ID (used in unit_id hash)
  marketingName       → unit number (e.g. "1010")
  marketingFullName   → full unit label
  bedrooms / bathrooms / sqFt / floorNumber
  isAvailable         → filter True only
  availableDate       → /Date(ms)/ serialized timestamp
  lowestRent.baseRent → asking rent (before all-in fees)
  lowestRent.leaseTerm → lease months

Concessions: /specials/ sub-page HTML — free-text concession banner scraped and
             passed through parse_concession() engine shared with MAA/CPT/EQR/AVB.

Markets:     Derived from the first segment of the community URL path
             (e.g. "denver-apartments" → "Denver").
"""

import json
import logging
import os
import re
import sys
import time
from datetime import date, datetime, timezone
from typing import Optional
from xml.etree import ElementTree

import pandas as pd
import requests
from bs4 import BeautifulSoup

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.listing import SCHEMA_COLUMNS
from scrapers.maa import parse_concession, make_unit_id
from utils.common import get_page, today_str, parse_float

REIT = "UDR"
BASE_URL = "https://www.udr.com"
SITEMAP_URL = "https://www.udr.com/sitemap.xml"

logger = logging.getLogger(__name__)

# ── Market slug → canonical label ─────────────────────────────────────────────

_MARKET_LABEL: dict[str, str] = {
    "new-york-city-apartments":        "New York",
    "boston-apartments":               "Boston",
    "tampa-apartments":                "Tampa",
    "washington-dc-apartments":        "Washington DC",
    "baltimore-apartments":            "Baltimore",
    "san-diego-apartments":            "San Diego",
    "san-francisco-bay-area-apartments": "San Francisco Bay Area",
    "orange-county-apartments":        "Orange County",
    "los-angeles-apartments":          "Los Angeles",
    "inland-empire-apartments":        "Inland Empire",
    "philadelphia-apartments":         "Philadelphia",
    "dallas-apartments":               "Dallas",
    "orlando-apartments":              "Orlando",
    "denver-apartments":               "Denver",
    "austin-apartments":               "Austin",
    "nashville-apartments":            "Nashville",
    "seattle-apartments":              "Seattle",
}

def _market_from_slug(market_slug: str) -> str:
    label = _MARKET_LABEL.get(market_slug)
    if label:
        return label
    # Fallback: strip '-apartments' suffix and title-case
    label = market_slug.replace("-apartments", "").replace("-", " ").title()
    return label


# ── Window variable extraction ─────────────────────────────────────────────────

# Matches: window.udr.jsonObjPropertyViewModel = {...};
_PROP_VM_RE = re.compile(
    r"window\.udr\.jsonObjPropertyViewModel\s*=\s*(\{.*?\});\s*(?:window|</script>|$)",
    re.DOTALL,
)

def _extract_property_vm(html: str) -> Optional[dict]:
    """Extract and JSON-parse window.udr.jsonObjPropertyViewModel."""
    m = _PROP_VM_RE.search(html)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError as e:
        logger.warning(f"JSON decode error on propertyViewModel: {e}")
        return None


# ── Date parsing ───────────────────────────────────────────────────────────────

_DATE_MS_RE = re.compile(r"/Date\((\d+)\)/")

def _parse_ms_date(raw: Optional[str]) -> Optional[date]:
    """Parse /Date(1775088000000)/ serialized timestamp → date."""
    if not raw:
        return None
    m = _DATE_MS_RE.search(str(raw))
    if not m:
        # Try ISO format fallback
        try:
            return date.fromisoformat(str(raw)[:10])
        except ValueError:
            return None
    ms = int(m.group(1))
    try:
        return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).date()
    except (OSError, ValueError, OverflowError):
        return None


# ── Concession text from /specials/ page ──────────────────────────────────────

_CONCESSION_KW = re.compile(
    r"\b(free|off|save|saving|discount|reduc|waiv|month|week|move.?in|no\s+rent)\b",
    re.IGNORECASE,
)
# Hard concession patterns — must contain a quantifiable concession amount
_HARD_AMOUNT_RE = re.compile(
    r"(\$[\d,]+|\d+\s*months?\s+free|\d+\s*weeks?\s+free|\d+%\s+off)",
    re.IGNORECASE,
)
# Patterns that indicate the text is a PRICE display, not a concession
_PRICE_DISPLAY_RE = re.compile(
    r"\b(starting\s+at|start\s+from|from\s+\$|priced\s+from|as\s+low\s+as"
    r"|beginning\s+at|price\s+from|available\s+from)\b",
    re.IGNORECASE,
)
_DEPOSIT_KW = (
    "security deposit", "admin fee", "administration fee",
    "application fee", "holding deposit",
)

def _scrape_specials(specials_url: str, session: requests.Session) -> Optional[str]:
    """
    Fetch the /specials/ page and extract any concession offer text.
    Returns the best matching text or None.
    """
    html = get_page(specials_url, session)
    if not html:
        return None

    soup = BeautifulSoup(html, "html.parser")
    # Remove nav/footer/script noise
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    hard_candidates: list[str] = []
    soft_candidates: list[str] = []

    for elem in soup.find_all(["h1", "h2", "h3", "h4", "p", "li", "div", "span"]):
        text = elem.get_text(separator=" ", strip=True)
        # Min 20 chars to avoid noise like "Save", "Free Wi-Fi" button text
        if len(text) < 20 or len(text) > 300:
            continue
        lower = text.lower()
        if any(kw in lower for kw in _DEPOSIT_KW):
            continue
        # Skip price displays ("Studio Starting at $3,738") — not concessions
        if _PRICE_DISPLAY_RE.search(text):
            continue
        if _HARD_AMOUNT_RE.search(text):
            hard_candidates.append(text)
        elif _CONCESSION_KW.search(text):
            soft_candidates.append(text)

    # Prefer hard (quantifiable) concession text
    for pool in (hard_candidates, soft_candidates):
        if not pool:
            continue
        # Deduplicate, then take shortest (avoids repeated DOM text)
        seen_texts: set[str] = set()
        unique: list[str] = []
        for t in pool:
            if t not in seen_texts:
                seen_texts.add(t)
                unique.append(t)
        unique.sort(key=len)
        return unique[0]

    return None


# ── Sitemap community discovery ────────────────────────────────────────────────

# Matches community-level URLs (3 path segments, no trailing sub-pages)
# e.g.: /denver-apartments/cherry-creek/steele-creek/
_COMMUNITY_URL_RE = re.compile(
    r"^https://www\.udr\.com/"
    r"([a-z0-9-]+-apartments)/"   # market slug
    r"([a-z0-9-]+)/"              # neighborhood slug
    r"([a-z0-9-]+)/$"             # community slug
)

def get_communities(session: requests.Session) -> list[dict]:
    """
    Parse the UDR sitemap and return all community-level URL records.
    Returns list of dicts: {url, market_slug, neighborhood, community_slug}
    """
    logger.info(f"Fetching sitemap: {SITEMAP_URL}")
    try:
        resp = session.get(SITEMAP_URL, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as e:
        logger.error(f"Failed to fetch sitemap: {e}")
        return []

    seen: dict[str, dict] = {}
    try:
        root = ElementTree.fromstring(resp.content)
        ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
        for url_elem in root.findall(".//sm:url/sm:loc", ns):
            url = url_elem.text.strip()
            m = _COMMUNITY_URL_RE.match(url)
            if m and url not in seen:
                seen[url] = {
                    "url":            url,
                    "market_slug":    m.group(1),
                    "neighborhood":   m.group(2),
                    "community_slug": m.group(3),
                }
    except ElementTree.ParseError as e:
        logger.error(f"Sitemap parse error: {e}")

    logger.info(f"Found {len(seen)} community URLs in sitemap")
    return list(seen.values())


# ── Unit extraction ────────────────────────────────────────────────────────────

def extract_units(
    vm: dict,
    community_meta: dict,
    concession_raw: Optional[str],
) -> list[dict]:
    """
    Parse window.udr.jsonObjPropertyViewModel into unit rows.
    """
    scrape_dt = date.today()
    rows: list[dict] = []

    floor_plans = vm.get("floorPlans") or []

    for fp in floor_plans:
        fp_name  = fp.get("Name") or fp.get("name")
        fp_beds  = fp.get("bedRooms")
        fp_baths = fp.get("bathRooms")
        fp_sqft  = fp.get("sqFtMin")  # floor-plan level sqft

        units = fp.get("units") or []
        for unit in units:
            if not unit.get("isAvailable", False):
                continue

            beds  = unit.get("bedrooms") if unit.get("bedrooms") is not None else fp_beds
            baths = unit.get("bathrooms") if unit.get("bathrooms") is not None else fp_baths
            sqft  = unit.get("sqFt") if unit.get("sqFt") is not None else fp_sqft
            floor = unit.get("floorNumber")

            unit_num = str(unit.get("marketingName") or unit.get("apartmentId") or "")

            # Rent: prefer lowestRent.baseRent (before all-in fees)
            lowest  = unit.get("lowestRent") or {}
            rent    = lowest.get("baseRent") or lowest.get("rent")
            if rent is None:
                rent = unit.get("rent") or unit.get("rentMin")
            if rent is None:
                continue
            rent = float(rent)

            lease_months = lowest.get("leaseTerm") or 12
            try:
                lease_months = int(lease_months)
            except (TypeError, ValueError):
                lease_months = 12

            # Move-in date
            move_in_date = _parse_ms_date(
                unit.get("availableDate") or unit.get("earliestMoveInDate")
            )

            concession_fields = parse_concession(
                raw=concession_raw,
                rent=rent,
                lease_months=lease_months,
            )

            uid = make_unit_id(
                reit=REIT,
                community=community_meta["community"],
                unit_number=unit_num,
                beds=beds,
                sqft=sqft,
            )

            row = {
                # Core 23 fields
                "scrape_date":  scrape_dt,
                "reit":         REIT,
                "community":    community_meta["community"],
                "address":      community_meta["address"],
                "market":       community_meta["market"],
                "unit_id":      uid,
                "beds":         beds,
                "baths":        baths,
                "sqft":         sqft,
                "rent":         rent,
                "move_in_date": move_in_date,
                "lease_term":   lease_months,
                "listing_url":  community_meta["listing_url"],
                "first_seen":   scrape_dt,
                "last_seen":    scrape_dt,
                **concession_fields,
                # Supplemental
                "state":               community_meta.get("state"),
                "city":                community_meta.get("city"),
                "latitude":            community_meta.get("latitude"),
                "longitude":           community_meta.get("longitude"),
                "floorplan_name":      fp_name,
                "floor_level":         str(floor) if floor is not None else None,
                "rentcafe_property_id": str(vm.get("propertyId") or ""),
            }
            rows.append(row)

    return rows


# ── Community metadata from property page ─────────────────────────────────────

_SCHEMA_LD_RE = re.compile(
    r'<script[^>]+type=["\']application/ld\+json["\'][^>]*>(.*?)</script>',
    re.DOTALL | re.IGNORECASE,
)

def _parse_community_meta(html: str, listing_url: str, market_slug: str) -> dict:
    """
    Extract community name, address, coordinates from property page HTML.
    Falls back to slug-derived values if schema.org markup is absent.
    """
    meta = {
        "community": None,
        "address":   None,
        "market":    _market_from_slug(market_slug),
        "state":     None,
        "city":      None,
        "latitude":  None,
        "longitude": None,
        "listing_url": listing_url,
    }

    for m in _SCHEMA_LD_RE.finditer(html):
        try:
            ld = json.loads(m.group(1))
        except json.JSONDecodeError:
            continue

        if not isinstance(ld, dict):
            continue

        schema_type = ld.get("@type", "")
        if "Apartment" in schema_type or "RealEstate" in schema_type or "Place" in schema_type:
            meta["community"]  = meta["community"]  or ld.get("name")
            addr = ld.get("address") or {}
            if isinstance(addr, dict):
                street = addr.get("streetAddress") or ""
                city   = addr.get("addressLocality") or ""
                state  = addr.get("addressRegion") or ""
                zip_c  = addr.get("postalCode") or ""
                meta["address"] = meta["address"] or ", ".join(
                    p for p in [street, city, state, zip_c] if p
                )
                meta["city"]  = meta["city"]  or city
                meta["state"] = meta["state"] or state
            geo = ld.get("geo") or {}
            if isinstance(geo, dict):
                meta["latitude"]  = meta["latitude"]  or geo.get("latitude")
                meta["longitude"] = meta["longitude"] or geo.get("longitude")

    # Fallback name from <title> tag
    if not meta["community"]:
        title_m = re.search(r"<title>([^<]+)</title>", html, re.IGNORECASE)
        if title_m:
            title = title_m.group(1).split("|")[0].strip()
            meta["community"] = title or None

    return meta


# ── Main scrape function ───────────────────────────────────────────────────────

def scrape_udr(limit: Optional[int] = None) -> pd.DataFrame:
    """
    Scrape all UDR communities. Returns a DataFrame with SCHEMA_COLUMNS.
    limit: restrict to first N communities (testing).
    """
    session = requests.Session()
    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/123.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Referer": "https://www.udr.com/",
    })

    all_rows: list[dict] = []
    errors:   list[str]  = []

    logger.info("Discovering UDR communities via sitemap ...")
    communities = get_communities(session)
    if limit:
        communities = communities[:limit]
    logger.info(f"Scraping {len(communities)} communities ...")

    for i, comm in enumerate(communities, 1):
        base_url     = comm["url"]   # e.g. https://www.udr.com/denver-apartments/cherry-creek/steele-creek/
        pricing_url  = base_url + "apartments-pricing/"
        specials_url = base_url + "specials/"
        market_slug  = comm["market_slug"]

        logger.info(f"[{i}/{len(communities)}] {pricing_url}")

        # Fetch pricing page (contains unit JSON)
        html = get_page(pricing_url, session)
        if not html:
            logger.warning(f"  Failed to fetch pricing page — skipping")
            errors.append(pricing_url)
            continue

        vm = _extract_property_vm(html)
        if not vm:
            logger.warning(f"  No jsonObjPropertyViewModel found — skipping")
            errors.append(pricing_url)
            continue

        # Community metadata from pricing page (schema.org)
        community_meta = _parse_community_meta(html, base_url, market_slug)

        # Prefer propertyName from the VM — it's always clean (e.g. "Steele Creek")
        vm_name = vm.get("propertyName")
        if vm_name:
            community_meta["community"] = vm_name
        elif not community_meta["community"]:
            # Final fallback from slug
            community_meta["community"] = (
                comm["community_slug"].replace("-", " ").title()
            )

        # Fetch specials page for concession text (best-effort, don't fail if missing)
        concession_raw = _scrape_specials(specials_url, session)

        rows = extract_units(vm, community_meta, concession_raw)

        logger.info(
            f"  {community_meta['community']} ({community_meta['market']}) — "
            f"{len(rows)} units | "
            f"concession: {'yes' if concession_raw else 'no'}"
        )
        all_rows.extend(rows)

    if errors:
        logger.warning(f"Failed/skipped {len(errors)} communities: {errors[:5]}")

    df = pd.DataFrame(all_rows)
    for col in SCHEMA_COLUMNS:
        if col not in df.columns:
            df[col] = None
    df = df[SCHEMA_COLUMNS]

    logger.info(f"UDR total rows: {len(df):,}")
    return df


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser(description="Scrape UDR rental listings")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit to first N communities (testing)")
    parser.add_argument("--save",  action="store_true",
                        help="Save output CSV to data/raw/")
    args = parser.parse_args()

    df = scrape_udr(limit=args.limit)

    display_cols = [
        "community", "market", "unit_id", "beds", "sqft", "rent",
        "lease_term", "move_in_date", "concession_hardness",
        "concession_type", "concession_value", "effective_monthly_rent",
    ]
    available_cols = [c for c in display_cols if c in df.columns]
    print(df[available_cols].head(15).to_string())
    print(f"\nShape: {df.shape}")

    if not df.empty and df["has_concession"].any():
        hard = df[df["concession_hardness"] == "hard"]
        soft = df[df["concession_hardness"] == "soft"]
        n_none = len(df) - len(hard) - len(soft)
        print(f"\nConcession breakdown: hard={len(hard)} soft={len(soft)} none={n_none}")
        if len(hard):
            print("\nConcession types:\n", hard["concession_type"].value_counts())
            print("\nSample hard concessions:")
            print(hard[[
                "community", "rent", "concession_type", "concession_value",
                "concession_pct_lease_value", "effective_monthly_rent", "concession_raw",
            ]].drop_duplicates("community").head(10).to_string())

    if args.save:
        out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw")
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"udr_raw_{today_str()}.csv")
        df.to_csv(path, index=False)
        print(f"\nSaved -> {path}")
