"""
AMH (American Homes 4 Rent) scraper.

Rendering:   Next.js SSR — all property data is embedded in a
             <script id="__NEXT_DATA__"> JSON blob on the query page.
             No JavaScript execution required; plain HTTP GET + JSON parse.

Discovery:   AMH sitemap (https://www.amh.com/sitemap.xml) lists ~47
             /query?criteria={Market} URLs.  Each query page returns ALL
             available listings for that market in a single page load
             (no pagination).

Data per property (from __NEXT_DATA__ → props.pageProps.results[]):
  propertyNo      → unique property identifier (e.g. "TX13894")
  addressLine1    → street address
  city, state, zipCode → address components
  latitude, longitude  → geolocation
  rent            → base monthly rent
  totalRent       → rent + fees (~$10 more)
  bedrooms, bathrooms, unitSqFt → unit specs
  yearBuilt       → year built
  availableDate   → ISO date string
  propertyDesc    → description (contains concession text if any)
  images[]        → photo URLs
  pills[]         → tags (e.g. "Solar")

unit_id:     AMH-{propertyNo}  (stable, unique per listing)

Markets:     47 markets from sitemap query URLs.  The "criteria" param
             is used as the market label (e.g. "Dallas, TX").

Note:        AMH is a Single-Family Rental (SFR) REIT.  Each row = one
             individual home, not a unit within an apartment building.
             "community" is set to the market label for grouping.
"""

import json
import logging
import os
import re
import sys
import time
from datetime import date
from typing import Optional
from urllib.parse import unquote
from xml.etree import ElementTree

import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.listing import SCHEMA_COLUMNS
from scrapers.maa import parse_concession
from utils.common import today_str

REIT     = "AMH"
BASE_URL = "https://www.amh.com"
SITEMAP  = "https://www.amh.com/sitemap.xml"

logger = logging.getLogger(__name__)

# ── Request config ────────────────────────────────────────────────────────────

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

_DELAY_BETWEEN_MARKETS = 2.0   # seconds between market requests (polite)
_MAX_RETRIES = 3
_REQUEST_TIMEOUT = 30

# ── Market discovery via sitemap ──────────────────────────────────────────────

_QUERY_URL_RE = re.compile(r"https?://(?:www\.)?amh\.com/query\?criteria=(.+)")


def _parse_sitemap_locs(xml_bytes: bytes) -> list[str]:
    """Parse all <loc> URLs from a sitemap XML, handling namespace variants."""
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
        return [
            m.decode()
            for m in re.findall(rb"<loc>\s*(https?://[^<]+)\s*</loc>", xml_bytes)
        ]


def get_markets() -> list[dict]:
    """
    Fetch AMH sitemap and extract all /query?criteria= URLs.
    Returns list of dicts: {"url": ..., "market": "Dallas, TX"}.
    """
    logger.info(f"Fetching sitemap: {SITEMAP}")
    try:
        resp = requests.get(SITEMAP, timeout=_REQUEST_TIMEOUT, headers=_HEADERS)
        resp.raise_for_status()
    except requests.RequestException as e:
        logger.error(f"Sitemap fetch failed: {e}")
        return []

    all_locs = _parse_sitemap_locs(resp.content)
    markets: list[dict] = []
    seen: set[str] = set()

    for url in all_locs:
        m = _QUERY_URL_RE.match(url)
        if not m:
            continue
        criteria_raw = m.group(1)
        # Decode URL encoding: "Dallas%2C%20TX" → "Dallas, TX"
        market_name = unquote(criteria_raw).replace("+", " ").strip()
        if market_name in seen:
            continue
        seen.add(market_name)
        markets.append({"url": url, "market": market_name})

    logger.info(f"Found {len(markets)} markets in sitemap")
    return markets


# ── __NEXT_DATA__ extraction ─────────────────────────────────────────────────

_NEXT_DATA_RE = re.compile(
    r'<script\s+id="__NEXT_DATA__"[^>]*>\s*(.*?)\s*</script>',
    re.DOTALL,
)


def _fetch_market_listings(url: str, session: requests.Session) -> list[dict]:
    """
    Fetch a single AMH query page and extract property dicts from __NEXT_DATA__.
    Returns a list of raw property dicts, or empty list on failure.
    """
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            resp = session.get(url, headers=_HEADERS, timeout=_REQUEST_TIMEOUT)
            resp.raise_for_status()
            break
        except requests.RequestException as e:
            logger.warning(f"  Attempt {attempt}/{_MAX_RETRIES} failed for {url}: {e}")
            if attempt < _MAX_RETRIES:
                time.sleep(_DELAY_BETWEEN_MARKETS * attempt)
            else:
                logger.error(f"  All retries exhausted for {url}")
                return []

    html = resp.text
    m = _NEXT_DATA_RE.search(html)
    if not m:
        logger.warning(f"  No __NEXT_DATA__ found in {url}")
        return []

    try:
        data = json.loads(m.group(1))
    except json.JSONDecodeError as e:
        logger.warning(f"  JSON parse error in __NEXT_DATA__: {e}")
        return []

    # Navigate to results array
    try:
        results = data["props"]["pageProps"]["results"]
    except (KeyError, TypeError):
        logger.warning(f"  No results in __NEXT_DATA__ for {url}")
        return []

    if not isinstance(results, list):
        logger.warning(f"  results is not a list for {url}")
        return []

    return results


# ── Concession extraction from description ───────────────────────────────────

# Sentence-boundary split for context-aware matching
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

# Deposit / non-rent contexts — if a numeric pattern appears within a sentence
# containing any of these phrases, it is NOT a rent concession.
_DEPOSIT_CONTEXT_RE = re.compile(
    r"security\s+deposit|deposit\s+offer|application\s+fee|"
    r"admin(?:istrative)?\s+fee|processing\s+fee|pet\s+fee|"
    r"holding\s+deposit|move[- ]in\s+fee",
    re.I,
)

_CONCESSION_PATTERNS = [
    # "X month(s) free" or "X weeks free"
    re.compile(r"\b(\d+)\s*months?\s*free\b", re.I),
    # "$X off" patterns
    re.compile(r"\$[\d,]+\s*off\b", re.I),
    # "X% off" patterns
    re.compile(r"\d+%\s*off\b", re.I),
    # Generic "special offer" or "limited time" (non-deposit)
    re.compile(r"(?:special|limited[- ]time)\s+(?:offer|promotion|deal)[^.]*", re.I),
]


def _extract_concession_text(description: str) -> Optional[str]:
    """
    Extract concession/promo text from property description.

    Returns the sentence containing the match (so downstream parse_concession
    can apply its own deposit-exclusion logic on the full context).
    Skips matches whose sentence mentions security deposit / application fee /
    admin fee / etc., since those are not rent concessions.

    Patterns are tried in priority order — specific (months/dollars/percent off)
    before generic ("special offer") — so a description containing both a
    generic intro and a specific concession returns the specific one.
    """
    if not description:
        return None

    # Split description into sentences for context-aware matching.
    sentences = _SENTENCE_SPLIT_RE.split(description)

    # Filter out sentences that are deposit/fee offers up front.
    rent_sentences = [s for s in sentences if not _DEPOSIT_CONTEXT_RE.search(s)]
    if not rent_sentences:
        return None

    # Iterate patterns in priority order so specific matches win.
    for pat in _CONCESSION_PATTERNS:
        for sent in rent_sentences:
            if pat.search(sent):
                return sent.strip()
    return None


# ── Build row from raw property dict ─────────────────────────────────────────

def _build_row(prop: dict, market: str) -> Optional[dict]:
    """Convert a raw AMH property dict into a SCHEMA_COLUMNS row."""
    scrape_dt = date.today()

    # Rent
    rent_raw = prop.get("rent")
    if rent_raw is None:
        return None
    try:
        rent = float(rent_raw)
    except (ValueError, TypeError):
        return None
    if rent < 100:
        return None

    # Property ID
    prop_no = prop.get("propertyNo", "")
    if not prop_no:
        return None
    unit_id = f"AMH-{prop_no}"

    # Address
    street   = (prop.get("addressLine1") or "").strip()
    city     = (prop.get("city") or "").strip()
    state    = (prop.get("state") or "").strip()
    zip_code = (prop.get("zipCode") or "").strip()
    address  = ", ".join(p for p in [street, city, state, zip_code] if p)

    # Specs
    beds  = prop.get("bedrooms")
    baths = prop.get("bathrooms")
    sqft  = prop.get("unitSqFt")

    try:
        beds = int(beds) if beds is not None else None
    except (ValueError, TypeError):
        beds = None
    try:
        baths = float(baths) if baths is not None else None
    except (ValueError, TypeError):
        baths = None
    try:
        sqft = int(sqft) if sqft is not None else None
    except (ValueError, TypeError):
        sqft = None

    # Availability date
    avail_raw = prop.get("availableDate")
    move_in_date = None
    if avail_raw:
        try:
            # ISO format: "2026-03-05T10:59:00+00:00"
            move_in_date = date.fromisoformat(avail_raw[:10])
        except (ValueError, TypeError):
            pass

    # Geo
    lat = prop.get("latitude")
    lon = prop.get("longitude")

    # Listing URL
    listing_url = f"{BASE_URL}/query?criteria={market.replace(' ', '+').replace(',', '%2C')}"

    # Concession from description
    description = prop.get("propertyDesc") or ""
    concession_raw = _extract_concession_text(description)
    concession_fields = parse_concession(
        raw=concession_raw,
        rent=rent,
        lease_months=12,
    )

    row = {
        "scrape_date":  scrape_dt,
        "reit":         REIT,
        "community":    market,
        "address":      address,
        "market":       market,
        "unit_id":      unit_id,
        "beds":         beds,
        "baths":        baths,
        "sqft":         sqft,
        "rent":         rent,
        "move_in_date": move_in_date,
        "lease_term":   "12 months",       # AMH standard
        "listing_url":  listing_url,
        "first_seen":   scrape_dt,
        "last_seen":    scrape_dt,
        **concession_fields,
        "state":               state or None,
        "city":                city or None,
        "latitude":            lat,
        "longitude":           lon,
        "floorplan_name":      None,    # N/A for SFR
        "floor_level":         None,    # N/A for SFR
        "rentcafe_property_id": prop_no,
    }
    return row


# ── Main scrape function ─────────────────────────────────────────────────────

def scrape_amh(limit: Optional[int] = None) -> pd.DataFrame:
    """
    Scrape all AMH SFR listings across all markets.

    Returns a DataFrame with SCHEMA_COLUMNS.

    Parameters
    ----------
    limit : int, optional
        Restrict to the first N *markets* (for testing).
    """
    all_rows: list[dict] = []
    errors:   list[str]  = []

    logger.info("Discovering AMH markets via sitemap ...")
    markets = get_markets()
    if not markets:
        logger.error("No markets found — returning empty DataFrame.")
        return pd.DataFrame(columns=SCHEMA_COLUMNS)

    if limit:
        markets = markets[:limit]
    logger.info(f"Scraping {len(markets)} markets ...")

    session = requests.Session()

    for i, mkt in enumerate(markets, 1):
        url    = mkt["url"]
        market = mkt["market"]
        logger.info(f"[{i}/{len(markets)}]  {market}")

        props = _fetch_market_listings(url, session)
        logger.info(f"  → {len(props)} listings")

        market_rows = 0
        for prop in props:
            row = _build_row(prop, market)
            if row is None:
                continue
            all_rows.append(row)
            market_rows += 1

        if market_rows == 0 and len(props) > 0:
            errors.append(market)
            logger.warning(f"  All {len(props)} listings failed to parse for {market}")

        if i < len(markets):
            time.sleep(_DELAY_BETWEEN_MARKETS)

    session.close()

    if errors:
        logger.warning(f"Markets with parse errors: {errors}")

    df = pd.DataFrame(all_rows)
    for col in SCHEMA_COLUMNS:
        if col not in df.columns:
            df[col] = None
    df = df[SCHEMA_COLUMNS]

    logger.info(f"AMH total rows: {len(df):,}")
    return df


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser(description="Scrape AMH SFR rental listings")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit to first N markets (testing)")
    parser.add_argument("--save", action="store_true",
                        help="Save output CSV to data/raw/")
    args = parser.parse_args()

    df = scrape_amh(limit=args.limit)

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
        if "has_concession" in df.columns:
            n_conc = df["has_concession"].sum()
            print(f"\nConcession rate: {n_conc}/{len(df)} = {df['has_concession'].mean():.1%}")

    if args.save:
        out_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "data", "raw"
        )
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"amh_raw_{today_str()}.csv")
        df.to_csv(path, index=False)
        print(f"\nSaved → {path}")
