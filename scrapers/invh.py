"""
INVH (Invitation Homes) scraper.

Rendering:   SvelteKit framework — property data IS server-side rendered into
             the initial HTML.  dom_content_loaded fires with the full page.
             No Cloudflare protection → default headless Chromium works.
             NOTE: wait_until="networkidle" times out (SPA keeps connections
             open); use "domcontentloaded" instead.

Discovery:   https://invitationhomes.com/property/sitemap.xml
             → ~3,600 URLs; individual home listings match the pattern
               https://www.invitationhomes.com/houses-for-rent/{address-slug}
             Sitemap uses a non-standard https:// namespace — namespace
             stripping required before XML parsing.

Data per property (from rendered DOM):
  section.property-metadata  → beds, baths, sqft, address
  [class*=price]             → rent (prefer "base rent" text; fall back to
                               all-in price)
  .static-details-bar        → lease term, availability date
  h1                         → full street address
  schema.org JSON-LD         → address components, geo lat/lon

unit_id:     INVH-{slug}  (slug is unique per listing; stable for active homes)
             Slug-based ID is more reliable than parsing a numeric ID that
             may or may not appear in the URL.

Markets:     INVH's ~16 SFR markets derived from state + city in the address.
             (market_name is not directly exposed in the rendered DOM.)

Note:        INVH is a Single-Family Rental (SFR) REIT.  Each row = one
             individual home, not a unit within an apartment building.
             "community" is set to INVH's market label for grouping/analysis.
"""

import logging
import os
import re
import sys
import time
from datetime import date, datetime
from typing import Optional
from xml.etree import ElementTree

import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.listing import SCHEMA_COLUMNS
from scrapers.maa import parse_concession
from utils.common import today_str

REIT     = "INVH"
BASE_URL = "https://www.invitationhomes.com"
SITEMAP  = "https://invitationhomes.com/property/sitemap.xml"

logger = logging.getLogger(__name__)

# Site-wide generic banner text — NOT a property-specific concession
_GENERIC_BANNERS = frozenset({
    "special offer on select homes",
    "special offer available in select locations",
    "special offer available in select locations. see homes for details.",
    "see homes for details",
})


# ── Playwright setup ───────────────────────────────────────────────────────────

try:
    from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout
    _HAS_PLAYWRIGHT = True
except ImportError:
    _HAS_PLAYWRIGHT = False
    logger.error(
        "playwright not installed — run:\n"
        "  pip install playwright && playwright install chromium"
    )


# ── Sitemap discovery ──────────────────────────────────────────────────────────

# Matches only the direct /houses-for-rent/{slug} pattern (not /markets/ or /search/)
_PROP_URL_RE = re.compile(
    r"^https?://(?:www\.)?invitationhomes\.com/houses-for-rent/([^/]+)$"
)


def _parse_sitemap_locs(xml_bytes: bytes) -> list[str]:
    """
    Parse all <loc> URLs from a sitemap, handling both standard http:// and
    non-standard https:// namespace variants by stripping xmlns declarations.
    """
    # Strip XML namespace declarations before parsing
    clean = re.sub(rb'\s+xmlns(?::\w+)?="[^"]*"', b"", xml_bytes)
    try:
        root = ElementTree.fromstring(clean)
        locs = []
        for el in root.iter():
            tag = el.tag.split("}")[-1] if "}" in el.tag else el.tag
            if tag == "loc" and el.text:
                locs.append(el.text.strip())
        return locs
    except ElementTree.ParseError:
        # Fallback: regex extraction
        return [
            m.decode()
            for m in re.findall(rb"<loc>\s*(https?://[^<]+)\s*</loc>", xml_bytes)
        ]


def get_properties() -> list[dict]:
    """Fetch INVH property sitemap → return list of individual home property dicts."""
    logger.info(f"Fetching sitemap: {SITEMAP}")
    try:
        resp = requests.get(
            SITEMAP,
            timeout=30,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
        )
        resp.raise_for_status()
    except requests.RequestException as e:
        logger.error(f"Sitemap fetch failed: {e}")
        return []

    all_locs      = _parse_sitemap_locs(resp.content)
    # Only individual home pages (pattern /houses-for-rent/{slug}, no sub-paths)
    property_locs = [u for u in all_locs if _PROP_URL_RE.match(u)]
    # Sub-sitemap handling (sitemap index)
    sub_sitemaps  = [u for u in all_locs if u.endswith(".xml") and u not in property_locs]

    if sub_sitemaps and not property_locs:
        logger.info(f"Sitemap index: {len(sub_sitemaps)} sub-sitemaps — fetching")
        for sub_url in sub_sitemaps:
            try:
                sub_resp = requests.get(sub_url, timeout=30,
                                        headers={"User-Agent": "Mozilla/5.0"})
                sub_resp.raise_for_status()
                for u in _parse_sitemap_locs(sub_resp.content):
                    if _PROP_URL_RE.match(u):
                        property_locs.append(u)
            except Exception as e:
                logger.warning(f"Sub-sitemap failed ({sub_url}): {e}")

    props: list[dict] = []
    seen: set[str] = set()
    for url in property_locs:
        if url in seen:
            continue
        seen.add(url)
        m = _PROP_URL_RE.match(url)
        slug = m.group(1) if m else url
        props.append({"url": url, "slug": slug})

    logger.info(f"Found {len(props)} property URLs in sitemap")
    return props


# ── Market inference from address ─────────────────────────────────────────────

# City-level overrides for multi-market states (TX, FL, CA)
_TX_CITY_MARKET: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\b(houston|katy|sugar.land|cypress|pearland|conroe|spring|woodlands|humble|friendswood|league.city|baytown|pasadena|galveston|stafford|rosenberg|richmond|alvin|clear.lake|deer.park|baytown|la.porte|seabrook|kemah|missouri.city|fresno|manvel|angleton|lake.jackson|angleton|clute|freeport|brazoria)\b", re.I), "Houston"),
    (re.compile(r"\b(san.antonio|boerne|new.braunfels|schertz|universal.city|converse|live.oak|helotes|leon.valley|kirby|windcrest|china.grove)\b", re.I), "San Antonio"),
    (re.compile(r"\b(austin|pflugerville|round.rock|cedar.park|leander|liberty.hill|hutto|kyle|buda|bastrop|manor|lakeway|bee.cave|elgin|rollingwood|sunset.valley)\b", re.I), "Austin"),
]
_FL_CITY_MARKET: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\b(jacksonville|fleming.island|orange.park|st..?augustine|clay|duval|nassau)\b", re.I), "Jacksonville"),
    (re.compile(r"\b(miami|fort.laud|miramar|hialeah|hollywood|boca.raton|delray|pompano|coral.spr|dania|davie|hallandale|pembroke|sunrise|plantation|deerfield|margate|coconut.creek|tamarac|weston|lauderhill|lauderdale|north.miami|aventura|sunny.isles|bal.harbour|surfside|opa.locka|cutler|homestead)\b", re.I), "Miami"),
    (re.compile(r"\b(orlando|kissimmee|osceola|daytona|altamonte|winter.park|winter.garden|winter.haven|sanford|clermont|apopka|longwood|lake.mary|deltona|port.orange|new.smyrna|oviedo|ocoee|windermere|dr..?phillips|celebration|buena.vent)\b", re.I), "Orlando"),
]

_STATE_MARKET: dict[str, str] = {
    "GA": "Atlanta",
    "TN": "Nashville",
    "MN": "Minneapolis",
    "NC": "Charlotte",
    "UT": "Salt Lake City",
    "CO": "Denver",
    "NV": "Las Vegas",
    "WA": "Seattle",
    "AZ": "Phoenix",
    "IL": "Chicago",
    "SC": "Charlotte",   # INVH's Charlotte market includes SC suburbs
    "IN": "Indianapolis",
}


def _market_from_address(city: str, state: str) -> str:
    """Derive INVH market name from city and state abbreviation."""
    state_up = (state or "").strip().upper()
    city_str = (city or "").strip()

    if state_up == "TX":
        for pattern, market in _TX_CITY_MARKET:
            if pattern.search(city_str):
                return market
        return "Dallas"  # default TX

    if state_up == "FL":
        for pattern, market in _FL_CITY_MARKET:
            if pattern.search(city_str):
                return market
        return "Tampa"   # default FL

    if state_up == "CA":
        return "Southern California"

    market = _STATE_MARKET.get(state_up)
    if market:
        return market

    # Fallback: City, ST
    if city_str and state_up:
        return f"{city_str}, {state_up}"
    return state_up or "Unknown"


# ── Text parsers ───────────────────────────────────────────────────────────────

_DOLLAR_RE       = re.compile(r"\$([\d,]+)")
_BEDS_RE         = re.compile(r"(\d+)\s*bed", re.IGNORECASE)
_BATHS_RE        = re.compile(r"([\d.]+)\s*bath", re.IGNORECASE)
_SQFT_RE         = re.compile(r"([\d,]+)\s*sqft", re.IGNORECASE)
_LEASE_TERM_RE   = re.compile(r"(\d+)\s*month\s*lease", re.IGNORECASE)
_AVAIL_DATE_RE   = re.compile(
    r"Available\s+(?:Now|(\w+\s+\d{1,2}(?:,\s*\d{4})?|\d{1,2}/\d{1,2}(?:/\d{2,4})?))",
    re.IGNORECASE,
)
_DATE_RE         = re.compile(r"(\d{4}-\d{2}-\d{2})")


def _parse_date_str(raw: str) -> Optional[date]:
    """Parse 'Available Now', 'Apr 15', '4/15/2026', '2026-04-15' → date."""
    if not raw:
        return None
    raw = raw.strip()
    if re.search(r"\bnow\b", raw, re.IGNORECASE):
        return date.today()
    # ISO format
    m = _DATE_RE.search(raw)
    if m:
        try:
            y, mo, d = m.group(1).split("-")
            return date(int(y), int(mo), int(d))
        except ValueError:
            pass
    # M/D/YYYY or M/D/YY
    m2 = re.search(r"(\d{1,2})/(\d{1,2})/(\d{2,4})", raw)
    if m2:
        try:
            yr = int(m2.group(3))
            if yr < 100:
                yr += 2000
            return date(yr, int(m2.group(1)), int(m2.group(2)))
        except ValueError:
            pass
    # "Apr 15" or "April 15, 2026"
    try:
        from dateutil.parser import parse as dateparse
        d = dateparse(raw, default=datetime(date.today().year, 1, 1))
        return d.date()
    except Exception:
        pass
    return None


# ── Per-property Playwright scrape ─────────────────────────────────────────────

def _get_text(page, *selectors: str) -> Optional[str]:
    """Try selectors in order; return inner_text() of first matching element."""
    for sel in selectors:
        try:
            el = page.query_selector(sel)
            if el:
                txt = (el.inner_text() or "").strip()
                if txt:
                    return txt
        except Exception:
            pass
    return None


def _scrape_property(page, prop: dict) -> Optional[dict]:
    """
    Navigate to one INVH property page and extract all data from the rendered DOM.
    Returns a raw dict of extracted values, or None if rent is unavailable.
    """
    url  = prop["url"]
    slug = prop["slug"]

    try:
        page.goto(url, wait_until="domcontentloaded", timeout=25_000)
    except PWTimeout:
        logger.warning(f"  Timeout: {url}")
        return None
    except Exception as e:
        logger.warning(f"  Navigation error: {e}")
        return None

    result: dict = {
        "_url":  url,
        "_slug": slug,
    }

    # ── Property metadata: beds / baths / sqft ──────────────────────────
    meta_text = _get_text(page,
                          "section.property-metadata",
                          "[class*=property-metadata]",
                          "[class*=propertymetadata]")
    if meta_text:
        m = _BEDS_RE.search(meta_text)
        if m: result["beds"] = m.group(1)
        m = _BATHS_RE.search(meta_text)
        if m: result["baths"] = m.group(1)
        m = _SQFT_RE.search(meta_text)
        if m: result["sqft"] = m.group(1).replace(",", "")

    # ── Rent: prefer "base rent", fall back to all-in / first price ─────
    rent = None
    rent_type = "unknown"

    # Scan [class*=price] elements for "base rent" label
    try:
        price_els = page.query_selector_all("[class*=price]")
        for el in price_els:
            try:
                txt = (el.inner_text() or "").strip()
                if "base rent" in txt.lower() or "base price" in txt.lower():
                    m = _DOLLAR_RE.search(txt)
                    if m:
                        rent = m.group(1).replace(",", "")
                        rent_type = "base"
                        break
            except Exception:
                continue
    except Exception:
        pass

    # ── Always fetch details bar for lease term + availability ────────────
    details_text = _get_text(page,
                             ".static-details-bar",
                             "[class*=static-details-bar]",
                             "[class*=details-container]",
                             "[class*=detailsbar]") or ""
    if details_text:
        lt_m = _LEASE_TERM_RE.search(details_text)
        if lt_m:
            result["lease_term"] = f"{lt_m.group(1)} months"
        # Normalise newlines before matching (inner_text may split on block elements)
        details_flat = " ".join(details_text.split())
        avail_m = _AVAIL_DATE_RE.search(details_flat)
        if avail_m:
            avail_raw = avail_m.group(0).replace("Available", "").strip()
            avail_date = _parse_date_str(avail_raw)
            if avail_date:
                result["available_on"] = str(avail_date)

    if not rent:
        # Fallback: first dollar amount in details bar
        if details_text:
            m = _DOLLAR_RE.search(details_text)
            if m:
                rent = m.group(1).replace(",", "")
                rent_type = "allin" if "all-in" in details_text.lower() else "listed"

    if not rent:
        logger.debug(f"  No rent found for {slug}")
        return None

    result["rent"] = rent
    result["_rent_type"] = rent_type

    # ── Address from H1 ────────────────────────────────────────────────
    h1_text = _get_text(page, "h1") or ""
    result["address_h1"] = h1_text

    # ── Address from H1 (always parse — reliable and consistent) ──────
    # INVH H1 format: "{street}, {city}, {state_abbr}, {zip}"
    # e.g. "336 Brady St, Elko New Market, MN, 55054"
    if h1_text:
        h1_parts = [p.strip() for p in h1_text.split(",")]
        if len(h1_parts) >= 4:
            result["address_street"] = h1_parts[0]
            result["city"]           = h1_parts[1]
            st_raw = h1_parts[2]   # "MN" or "CA" etc.
            st_m = re.search(r"\b([A-Z]{2})\b", st_raw)
            result["state"]          = st_m.group(1) if st_m else st_raw.strip()
            result["zip_code"]       = h1_parts[3]
        elif len(h1_parts) == 3:
            # "{street}, {city state}, {zip}"
            result["address_street"] = h1_parts[0]
            cs = h1_parts[1]  # "Corona CA" or "Corona, CA"
            st_m = re.search(r"\b([A-Z]{2})\b", cs)
            result["state"] = st_m.group(1) if st_m else ""
            result["city"]  = re.sub(r"\s*\b[A-Z]{2}\b\s*$", "", cs).strip()
            result["zip_code"] = h1_parts[2]

    # ── Geo from Schema.org JSON-LD (SingleFamilyResidence only) ──────────
    try:
        import json as _json
        html = page.content()
        # Find ALL JSON-LD script blocks, pick SingleFamilyResidence
        for ld_raw in re.findall(
            r'<script[^>]+type=["\']application/ld\+json["\'][^>]*>(.*?)</script>',
            html, re.DOTALL | re.IGNORECASE
        ):
            try:
                ld = _json.loads(ld_raw)
                if not isinstance(ld, dict):
                    continue
                if ld.get("@type") != "SingleFamilyResidence":
                    continue
                # We found the property-specific block
                addr = ld.get("address", {})
                if isinstance(addr, dict):
                    # Only override H1-parsed address_street if it adds value
                    if addr.get("streetAddress"):
                        result.setdefault("address_street", addr["streetAddress"])
                    # addressLocality may be "Elko New Market" or "Elko New Market, MN"
                    raw_city = addr.get("addressLocality", "")
                    if "," in raw_city:
                        result.setdefault("city", raw_city.split(",")[0].strip())
                        st_m = re.search(r"\b([A-Z]{2})\b", raw_city.split(",")[-1])
                        if st_m:
                            result.setdefault("state", st_m.group(1))
                    else:
                        result.setdefault("city", raw_city)
                    result.setdefault("zip_code", addr.get("postalCode", ""))
                geo = ld.get("geo", {})
                if isinstance(geo, dict):
                    result["latitude"]  = geo.get("latitude")
                    result["longitude"] = geo.get("longitude")
                break
            except Exception:
                continue
    except Exception:
        pass

    # ── Property-specific concession ───────────────────────────────────
    # The site-wide "Special offer on select homes!" banner is generic.
    # Look for per-listing special text that's more specific.
    concession_raw = None
    try:
        special_selectors = [
            "[class*=listing-special]",
            "[class*=property-special]",
            "[class*=home-special]",
            "[class*=promo]",
            "[class*=concession]",
            "[class*=discount]",
            "[class*=incentive]",
        ]
        for sel in special_selectors:
            els = page.query_selector_all(sel)
            for el in els:
                txt = (el.inner_text() or "").strip()
                if (txt
                        and txt.lower() not in _GENERIC_BANNERS
                        and re.search(
                            r"\b(free|off|save|month|week|discount|waiv|reduc)\b",
                            txt, re.IGNORECASE
                        )):
                    concession_raw = txt
                    break
            if concession_raw:
                break
    except Exception:
        pass

    result["leasing_special"] = concession_raw

    return result


# ── Build row dict from extracted data ────────────────────────────────────────

def _build_row(raw: dict) -> Optional[dict]:
    """Convert the per-property extraction dict into a SCHEMA_COLUMNS row."""
    scrape_dt = date.today()

    # Rent
    try:
        rent = float(str(raw["rent"]).replace(",", "").replace("$", "").strip())
    except (KeyError, ValueError, TypeError):
        return None
    if rent < 100:
        return None

    def _int(key: str) -> Optional[int]:
        v = raw.get(key)
        try:
            return int(float(str(v))) if v is not None else None
        except (ValueError, TypeError):
            return None

    def _flt(key: str) -> Optional[float]:
        v = raw.get(key)
        try:
            return float(str(v)) if v is not None else None
        except (ValueError, TypeError):
            return None

    beds   = _int("beds")
    baths  = _flt("baths")
    sqft   = _int("sqft")

    # Availability
    avail_raw   = raw.get("available_on")
    move_in_date = _parse_date_str(avail_raw) if avail_raw else None

    # Lease term
    lease_term = raw.get("lease_term")

    # Address
    addr_street = raw.get("address_street") or raw.get("address_h1", "").split(",")[0].strip()
    city        = raw.get("city", "")
    state       = raw.get("state", "")
    zip_code    = raw.get("zip_code", "")
    address_str = ", ".join(p for p in [addr_street, city, state, zip_code] if p)

    # Market derived from address
    market = _market_from_address(city, state)

    # Geo
    lat = _flt("latitude")
    lon = _flt("longitude")

    # Unit ID — use slug (unique, stable per active listing)
    slug   = raw.get("_slug", "")
    unit_id = f"INVH-{slug}"

    # Concession
    leasing_special = raw.get("leasing_special") or None
    concession_fields = parse_concession(
        raw=leasing_special,
        rent=rent,
        lease_months=12,
    )

    row = {
        "scrape_date":  scrape_dt,
        "reit":         REIT,
        "community":    market,       # SFR REIT: group by market for analysis
        "address":      address_str,
        "market":       market,
        "unit_id":      unit_id,
        "beds":         beds,
        "baths":        baths,
        "sqft":         sqft,
        "rent":         rent,
        "move_in_date": move_in_date,
        "lease_term":   lease_term,
        "listing_url":  raw.get("_url", ""),
        "first_seen":   scrape_dt,
        "last_seen":    scrape_dt,
        **concession_fields,
        "state":               state or None,
        "city":                city  or None,
        "latitude":            lat,
        "longitude":           lon,
        "floorplan_name":      None,    # N/A for SFR
        "floor_level":         None,    # N/A for SFR
        "rentcafe_property_id": slug,
    }
    return row


# ── Main scrape function ───────────────────────────────────────────────────────

def scrape_invh(limit: Optional[int] = None) -> pd.DataFrame:
    """
    Scrape all INVH SFR listings. Returns a DataFrame with SCHEMA_COLUMNS.
    limit: restrict to first N listings (testing).
    """
    if not _HAS_PLAYWRIGHT:
        logger.error("Playwright not available — cannot scrape INVH.")
        return pd.DataFrame(columns=SCHEMA_COLUMNS)

    all_rows: list[dict] = []
    errors:   list[str]  = []

    logger.info("Discovering INVH properties via sitemap ...")
    properties = get_properties()
    if limit:
        properties = properties[:limit]
    logger.info(f"Scraping {len(properties)} properties ...")

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
        # Block heavy resources that don't affect data rendering
        context.route(
            "**/*.{png,jpg,jpeg,gif,webp,svg,ico,woff,woff2,ttf,mp4,webm}",
            lambda route: route.abort(),
        )
        page = context.new_page()

        for i, prop in enumerate(properties, 1):
            logger.info(f"[{i}/{len(properties)}]  {prop['slug']}")

            raw = _scrape_property(page, prop)
            if raw is None:
                errors.append(prop["url"])
                continue

            row = _build_row(raw)
            if row is None:
                logger.warning(f"  Could not build row — skipping")
                errors.append(prop["url"])
                continue

            logger.info(
                f"  ${row['rent']:,.0f}  "
                f"{row['beds']}bd/{row['baths']}ba  "
                f"{row['sqft'] or '?'}sqft  "
                f"market={row['market']}  "
                f"avail={row['move_in_date']}  "
                f"type={raw.get('_rent_type','?')}"
            )
            all_rows.append(row)

            time.sleep(0.5)

        page.close()
        context.close()
        browser.close()

    if errors:
        logger.warning(
            f"Failed/skipped {len(errors)} properties: "
            f"{[e.split('/')[-1] for e in errors[:5]]}"
        )

    df = pd.DataFrame(all_rows)
    for col in SCHEMA_COLUMNS:
        if col not in df.columns:
            df[col] = None
    df = df[SCHEMA_COLUMNS]

    logger.info(f"INVH total rows: {len(df):,}")
    return df


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser(description="Scrape INVH SFR rental listings")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit to first N properties (testing)")
    parser.add_argument("--save",  action="store_true",
                        help="Save output CSV to data/raw/")
    args = parser.parse_args()

    df = scrape_invh(limit=args.limit)

    display_cols = [
        "community", "market", "unit_id", "beds", "baths", "sqft", "rent",
        "move_in_date", "lease_term", "address", "state",
        "concession_hardness", "concession_type", "concession_value",
        "effective_monthly_rent",
    ]
    available = [c for c in display_cols if c in df.columns]
    print(df[available].head(20).to_string())
    print(f"\nShape: {df.shape}")

    if not df.empty:
        print("\nListings by market:")
        print(df["market"].value_counts().to_string())
        if df["has_concession"].any():
            hard = df[df["concession_hardness"] == "hard"]
            soft = df[df["concession_hardness"] == "soft"]
            print(f"\nConcession breakdown: hard={len(hard)} soft={len(soft)} none={len(df)-len(hard)-len(soft)}")

    if args.save:
        out_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "data", "raw"
        )
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"invh_raw_{today_str()}.csv")
        df.to_csv(path, index=False)
        print(f"\nSaved → {path}")
