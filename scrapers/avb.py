"""
AVB (AvalonBay Communities) scraper.

Rendering:   Arc XP / Fusion CMS (Washington Post headless CMS) — all unit data
             is server-side rendered into a <script id="fusion-metadata"> tag as
             a 400KB+ JSON blob assigned to window.Fusion.globalContent.
             No JavaScript execution or browser required — plain requests works.

Discovery:   13 state-slug pages → Fusion.globalContent.communityResults.communities.items[]
             ~300+ communities across high-cost coastal markets.

URL pattern: https://www.avaloncommunities.com/{state}/{city-slug}-apartments/{community-slug}
Unit data:   Embedded in the same page's Fusion.globalContent.units[] array.
             All available units are pre-loaded — no API pagination needed.

Data model:  units[] entry fields used:
  unitId                                    → unit number
  bedroomNumber                             → beds
  bathroomNumber                            → baths
  squareFeet                                → sqft
  floorNumber                               → floor_level
  availableDateUnfurnished                  → move_in_date (ISO string)
  startingAtPricesUnfurnished.prices.price  → rent
  startingAtPricesUnfurnished.leaseTerm     → lease_term (months, int)
  promotions[]                              → unit-level concession signals

Community:   communityId (e.g. "AVB-NY037"), name, address, coordinates,
             url (relative), hasSpecials, unitsSummary.promotions[] (community-level)

Concession:  Unit promotions → promotionTitle (filtered for deposit/fee specials)
             → parse_concession() engine shared with MAA/CPT/EQR.
             Falls back to community-level unitsSummary.promotions[] if no unit promo.
"""

import json
import logging
import os
import re
import sys
import time
from datetime import date
from typing import Optional

import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.listing import SCHEMA_COLUMNS
from scrapers.maa import parse_concession, make_unit_id
from utils.common import get_page, today_str, parse_float

REIT = "AVB"
BASE_URL = "https://www.avaloncommunities.com"

# All state-level discovery pages
STATE_SLUGS = [
    "southern-california",
    "northern-california",
    "colorado",
    "district-of-columbia",
    "florida",
    "maryland",
    "massachusetts",
    "new-jersey",
    "new-york",
    "north-carolina",
    "texas",
    "virginia",
    "washington",
]

# Keywords that disqualify a promotion from being a concession
# (deposit/fee specials are NOT rent concessions)
_FEE_KEYWORDS = (
    "security deposit",
    "admin fee",
    "administration fee",
    "application fee",
    "holding deposit",
)

# Keywords required for a promotion to be treated as a concession
_CONCESSION_KW = re.compile(
    r"\b(free|off|save|saving|discount|reduc|waiv|month|week|move.?in|no\s+rent)\b",
    re.IGNORECASE,
)

logger = logging.getLogger(__name__)


# ── Fusion.globalContent extraction ───────────────────────────────────────────

# Matches:  Fusion.globalContent={...};Fusion.globalContentConfig=
# The blob can be 400KB+ so we use re.DOTALL to span newlines.
_FUSION_RE = re.compile(
    r"Fusion\.globalContent=(\{.*?\});Fusion\.globalContentConfig=",
    re.DOTALL,
)


def _extract_fusion(html: str) -> Optional[dict]:
    """Extract and JSON-parse Fusion.globalContent from page HTML."""
    m = _FUSION_RE.search(html)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError as e:
        logger.warning(f"JSON decode error on Fusion.globalContent: {e}")
        return None


# ── Market mapping ─────────────────────────────────────────────────────────────
# Maps (state_slug, city) → canonical market label.
# City names are lower-cased for matching.

_MARKET_MAP: dict[tuple[str, str], str] = {
    # New York
    ("new-york", "new york"):        "New York",
    ("new-york", "brooklyn"):        "New York",
    ("new-york", "queens"):          "New York",
    ("new-york", "bronx"):           "New York",
    ("new-york", "jersey city"):     "New York",
    ("new-york", "weehawken"):       "New York",
    ("new-york", "edgewater"):       "New York",
    ("new-york", "hoboken"):         "New York",
    ("new-york", "stamford"):        "New York",
    ("new-york", "mount kisco"):     "New York",
    ("new-york", "yonkers"):         "New York",
    # New Jersey (NYC metro)
    ("new-jersey", "jersey city"):   "New York",
    ("new-jersey", "weehawken"):     "New York",
    ("new-jersey", "edgewater"):     "New York",
    ("new-jersey", "hoboken"):       "New York",
    ("new-jersey", "fort lee"):      "New York",
    ("new-jersey", "princeton"):     "New York",
    ("new-jersey", "parsippany"):    "New York",
    ("new-jersey", "wayne"):         "New York",
    ("new-jersey", "woodbridge"):    "New York",
    ("new-jersey", "metuchen"):      "New York",
    # Massachusetts (Boston)
    ("massachusetts", "boston"):          "Boston",
    ("massachusetts", "cambridge"):       "Boston",
    ("massachusetts", "waltham"):         "Boston",
    ("massachusetts", "lexington"):       "Boston",
    ("massachusetts", "newton"):          "Boston",
    ("massachusetts", "marlborough"):     "Boston",
    ("massachusetts", "chelmsford"):      "Boston",
    ("massachusetts", "woburn"):          "Boston",
    ("massachusetts", "burlington"):      "Boston",
    ("massachusetts", "billerica"):       "Boston",
    ("massachusetts", "wilmington"):      "Boston",
    ("massachusetts", "andover"):         "Boston",
    ("massachusetts", "canton"):          "Boston",
    ("massachusetts", "quincy"):          "Boston",
    ("massachusetts", "south weymouth"):  "Boston",
    ("massachusetts", "peabody"):         "Boston",
    ("massachusetts", "beverly"):         "Boston",
    # DC / MD / VA
    ("district-of-columbia", "washington"):  "Washington DC",
    ("maryland", "rockville"):        "Washington DC",
    ("maryland", "gaithersburg"):     "Washington DC",
    ("maryland", "germantown"):       "Washington DC",
    ("maryland", "bethesda"):         "Washington DC",
    ("maryland", "silver spring"):    "Washington DC",
    ("maryland", "college park"):     "Washington DC",
    ("maryland", "bowie"):            "Washington DC",
    ("maryland", "largo"):            "Washington DC",
    ("maryland", "baltimore"):        "Baltimore",
    ("maryland", "owings mills"):     "Baltimore",
    ("virginia", "arlington"):        "Washington DC",
    ("virginia", "alexandria"):       "Washington DC",
    ("virginia", "reston"):           "Washington DC",
    ("virginia", "herndon"):          "Washington DC",
    ("virginia", "mclean"):           "Washington DC",
    ("virginia", "fairfax"):          "Washington DC",
    ("virginia", "falls church"):     "Washington DC",
    ("virginia", "ashburn"):          "Washington DC",
    ("virginia", "centreville"):      "Washington DC",
    ("virginia", "chantilly"):        "Washington DC",
    ("virginia", "leesburg"):         "Washington DC",
    ("virginia", "tysons"):           "Washington DC",
    ("virginia", "woodbridge"):       "Washington DC",
    ("virginia", "richmond"):         "Richmond",
    # Northern California (Bay Area)
    ("northern-california", "san jose"):       "San Jose",
    ("northern-california", "santa clara"):    "San Jose",
    ("northern-california", "sunnyvale"):      "San Jose",
    ("northern-california", "mountain view"):  "San Jose",
    ("northern-california", "campbell"):       "San Jose",
    ("northern-california", "milpitas"):       "San Jose",
    ("northern-california", "los gatos"):      "San Jose",
    ("northern-california", "fremont"):        "Bay Area",
    ("northern-california", "pleasanton"):     "Bay Area",
    ("northern-california", "walnut creek"):   "Bay Area",
    ("northern-california", "concord"):        "Bay Area",
    ("northern-california", "san francisco"):  "San Francisco",
    ("northern-california", "san mateo"):      "San Francisco",
    ("northern-california", "redwood city"):   "San Francisco",
    ("northern-california", "foster city"):    "San Francisco",
    # Southern California
    ("southern-california", "los angeles"):    "Los Angeles",
    ("southern-california", "west hollywood"): "Los Angeles",
    ("southern-california", "burbank"):        "Los Angeles",
    ("southern-california", "glendale"):       "Los Angeles",
    ("southern-california", "redondo beach"):  "Los Angeles",
    ("southern-california", "hawthorne"):      "Los Angeles",
    ("southern-california", "long beach"):     "Los Angeles",
    ("southern-california", "torrance"):       "Los Angeles",
    ("southern-california", "irvine"):         "Orange County",
    ("southern-california", "costa mesa"):     "Orange County",
    ("southern-california", "anaheim"):        "Orange County",
    ("southern-california", "orange"):         "Orange County",
    ("southern-california", "fullerton"):      "Orange County",
    ("southern-california", "san diego"):      "San Diego",
    ("southern-california", "carlsbad"):       "San Diego",
    ("southern-california", "oceanside"):      "San Diego",
    # Colorado
    ("colorado", "denver"):         "Denver",
    ("colorado", "aurora"):         "Denver",
    ("colorado", "lakewood"):       "Denver",
    ("colorado", "broomfield"):     "Denver",
    ("colorado", "westminster"):    "Denver",
    ("colorado", "boulder"):        "Denver",
    ("colorado", "fort collins"):   "Denver",
    ("colorado", "englewood"):      "Denver",
    ("colorado", "highlands ranch"):"Denver",
    # Washington (Seattle)
    ("washington", "seattle"):        "Seattle",
    ("washington", "bellevue"):       "Seattle",
    ("washington", "redmond"):        "Seattle",
    ("washington", "kirkland"):       "Seattle",
    ("washington", "issaquah"):       "Seattle",
    ("washington", "bothell"):        "Seattle",
    ("washington", "kent"):           "Seattle",
    ("washington", "renton"):         "Seattle",
    ("washington", "federal way"):    "Seattle",
    # Florida
    ("florida", "miami"):             "South Florida",
    ("florida", "fort lauderdale"):   "South Florida",
    ("florida", "plantation"):        "South Florida",
    ("florida", "pembroke pines"):    "South Florida",
    ("florida", "boca raton"):        "South Florida",
    ("florida", "delray beach"):      "South Florida",
    ("florida", "palm beach gardens"):"South Florida",
    ("florida", "orlando"):           "Orlando",
    ("florida", "lake nona"):         "Orlando",
    ("florida", "tampa"):             "Tampa",
    ("florida", "st. petersburg"):    "Tampa",
    ("florida", "jacksonville"):      "Jacksonville",
    # North Carolina
    ("north-carolina", "raleigh"):      "Raleigh",
    ("north-carolina", "cary"):         "Raleigh",
    ("north-carolina", "durham"):       "Raleigh",
    ("north-carolina", "apex"):         "Raleigh",
    ("north-carolina", "morrisville"):  "Raleigh",
    ("north-carolina", "charlotte"):    "Charlotte",
    ("north-carolina", "mooresville"):  "Charlotte",
    ("north-carolina", "huntersville"): "Charlotte",
    ("north-carolina", "matthews"):     "Charlotte",
    # Texas
    ("texas", "austin"):              "Austin",
    ("texas", "houston"):             "Houston",
    ("texas", "dallas"):              "Dallas",
    ("texas", "frisco"):              "Dallas",
    ("texas", "plano"):               "Dallas",
    # New Jersey (additional NYC metro)
    ("new-jersey", "roseland"):       "New York",
    ("new-jersey", "bloomfield"):     "New York",
    ("new-jersey", "bloomingdale"):   "New York",
    ("new-jersey", "boonton"):        "New York",
    ("new-jersey", "maplewood"):      "New York",
    ("new-jersey", "montville"):      "New York",
    ("new-jersey", "north bergen"):   "New York",
    ("new-jersey", "old bridge"):     "New York",
    ("new-jersey", "piscataway"):     "New York",
    ("new-jersey", "metuchen"):       "New York",
    ("new-jersey", "princeton"):      "New York",
    ("new-jersey", "pine brook"):     "New York",
    ("new-jersey", "parsippany"):     "New York",
    # Southern California (additional)
    ("southern-california", "woodland hills"): "Los Angeles",
    ("southern-california", "canoga park"):    "Los Angeles",
    ("southern-california", "chatsworth"):     "Los Angeles",
    ("southern-california", "sherman oaks"):   "Los Angeles",
    ("southern-california", "studio city"):    "Los Angeles",
    ("southern-california", "encino"):         "Los Angeles",
    ("southern-california", "tarzana"):        "Los Angeles",
    ("southern-california", "reseda"):         "Los Angeles",
    ("southern-california", "thousand oaks"):  "Los Angeles",
    ("southern-california", "camarillo"):      "Los Angeles",
    ("southern-california", "ventura"):        "Los Angeles",
    ("southern-california", "oxnard"):         "Los Angeles",
    ("southern-california", "signal hill"):    "Los Angeles",
    ("southern-california", "culver city"):    "Los Angeles",
    ("southern-california", "el segundo"):     "Los Angeles",
    ("southern-california", "manhattan beach"):"Los Angeles",
    ("southern-california", "calabasas"):      "Los Angeles",
    ("southern-california", "agoura hills"):   "Los Angeles",
    # Maryland (additional)
    ("maryland", "annapolis"):        "Baltimore",
    ("maryland", "hanover"):          "Baltimore",
    ("maryland", "columbia"):         "Baltimore",
    ("maryland", "ellicott city"):    "Baltimore",
    ("maryland", "towson"):           "Baltimore",
    ("maryland", "pikesville"):       "Baltimore",
    # Washington state (additional)
    ("washington", "lynnwood"):       "Seattle",
    ("washington", "bothell"):        "Seattle",
    ("washington", "newcastle"):      "Seattle",
    ("washington", "shoreline"):      "Seattle",
    ("washington", "kirkland"):       "Seattle",
    ("washington", "tukwila"):        "Seattle",
    ("washington", "kent"):           "Seattle",
}


def _market_label(state_slug: str, city: str) -> str:
    """Resolve city+state_slug to canonical market label."""
    key = (state_slug, city.lower())
    if key in _MARKET_MAP:
        return _MARKET_MAP[key]
    # Fallback: title-case city + state abbreviation derived from slug
    _STATE_ABBREV = {
        "new-york": "NY", "new-jersey": "NJ", "massachusetts": "MA",
        "district-of-columbia": "DC", "maryland": "MD", "virginia": "VA",
        "northern-california": "CA", "southern-california": "CA",
        "colorado": "CO", "washington": "WA", "florida": "FL",
        "north-carolina": "NC", "texas": "TX",
    }
    abbrev = _STATE_ABBREV.get(state_slug, state_slug.upper()[:2])
    return f"{city.title()}, {abbrev}"


# ── Community discovery ────────────────────────────────────────────────────────

def get_communities(session: requests.Session) -> list[dict]:
    """
    Iterate state-slug pages and collect community metadata.
    Returns list of dicts: {communityId, name, url, address, coordinates,
                             hasSpecials, state_slug, area}
    """
    seen: dict[str, dict] = {}   # communityId → dict

    for slug in STATE_SLUGS:
        url = f"{BASE_URL}/{slug}"
        logger.info(f"  Discovery: {url}")
        html = get_page(url, session)
        if not html:
            logger.warning(f"  Could not fetch state page: {slug}")
            continue

        gc = _extract_fusion(html)
        if not gc:
            logger.warning(f"  No Fusion.globalContent on {slug} state page")
            continue

        try:
            comms_obj = gc.get("communityResults", {}).get("communities", {})
            items = comms_obj.get("items", [])
        except (AttributeError, KeyError):
            logger.warning(f"  Unexpected communityResults structure on {slug}")
            continue

        new_this_page = 0
        for item in items:
            cid = item.get("communityId")
            if not cid or cid in seen:
                continue
            seen[cid] = {**item, "state_slug": slug}
            new_this_page += 1

        logger.info(f"  {slug}: {len(items)} communities "
                    f"({new_this_page} new, {len(seen)} total)")

    return list(seen.values())


# ── Promotion helpers ──────────────────────────────────────────────────────────

def _best_promo_text(promotions: list) -> Optional[str]:
    """
    Pick the best promotion title from a promotions array.
    Returns None if every promo is a deposit/fee special or lacks concession keywords.
    """
    if not promotions:
        return None
    for promo in promotions:
        title = (promo.get("promotionTitle") or promo.get("title") or "").strip()
        desc  = (promo.get("promotionDescription") or promo.get("description") or "").strip()
        text  = title or desc
        if not text:
            continue
        lower = text.lower()
        # Skip deposit/fee specials
        if any(kw in lower for kw in _FEE_KEYWORDS):
            continue
        # Require at least one concession keyword
        if _CONCESSION_KW.search(text):
            return text
    return None


# ── Unit extraction ────────────────────────────────────────────────────────────

def extract_units(
    gc: dict,
    community_meta: dict,
) -> list[dict]:
    """
    Parse the Fusion.globalContent dict into unit rows.
    Returns a list of row dicts conforming to SCHEMA_COLUMNS.
    """
    scrape_dt = date.today()
    rows: list[dict] = []

    # Community-level concession (from unitsSummary.promotions[])
    unit_summary = gc.get("unitsSummary") or {}
    comm_promos = unit_summary.get("promotions") or []
    comm_concession_raw = _best_promo_text(comm_promos)

    units = gc.get("units") or []
    if not units:
        return rows

    for unit in units:
        # Skip unavailable units
        if unit.get("availabilityStatus", "").lower() not in ("", "available"):
            if not unit.get("availableDateUnfurnished"):
                continue

        beds  = unit.get("bedroomNumber")
        baths = unit.get("bathroomNumber")
        sqft  = unit.get("squareFeet")
        floor_num = unit.get("floorNumber")
        unit_id_raw = str(unit.get("unitId") or "").strip()

        # Rent & lease term
        pricing = unit.get("startingAtPricesUnfurnished") or {}
        prices_obj = pricing.get("prices") or {}
        rent = None
        # prices may be a dict with numeric keys → grab first value
        if isinstance(prices_obj, dict):
            for v in prices_obj.values():
                r = parse_float(str(v))
                if r and r > 0:
                    rent = r
                    break
        elif isinstance(prices_obj, list) and prices_obj:
            rent = parse_float(str(prices_obj[0].get("price", "")))

        if rent is None:
            # Try top-level price field
            rent = parse_float(str(unit.get("price") or unit.get("rent") or ""))
        if rent is None:
            continue

        lease_term_months = pricing.get("leaseTerm")
        if lease_term_months is None:
            lease_term_months = 12
        try:
            lease_term_months = int(lease_term_months)
        except (TypeError, ValueError):
            lease_term_months = 12

        # Move-in date
        move_in_raw = unit.get("availableDateUnfurnished")
        move_in_date = None
        if move_in_raw:
            try:
                move_in_date = date.fromisoformat(str(move_in_raw)[:10])
            except ValueError:
                pass

        # Concession: unit-level promotions first, fall back to community-level
        unit_promos = unit.get("promotions") or []
        unit_concession_raw = _best_promo_text(unit_promos)
        concession_raw = unit_concession_raw or comm_concession_raw

        concession_fields = parse_concession(
            raw=concession_raw,
            rent=rent,
            lease_months=lease_term_months,
        )

        uid = make_unit_id(
            reit=REIT,
            community=community_meta["community"] or "",
            unit_number=unit_id_raw,
            beds=beds,
            sqft=sqft,
        )

        floor_level_str = str(floor_num) if floor_num is not None else None

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
            "lease_term":   lease_term_months,
            "listing_url":  community_meta["listing_url"],
            "first_seen":   scrape_dt,
            "last_seen":    scrape_dt,
            **concession_fields,
            # Supplemental
            "state":               community_meta["state"],
            "city":                community_meta["city"],
            "latitude":            community_meta.get("latitude"),
            "longitude":           community_meta.get("longitude"),
            "floorplan_name":      unit.get("floorplanName") or unit.get("floorPlanName"),
            "floor_level":         floor_level_str,
            "rentcafe_property_id": community_meta.get("community_id"),
        }
        rows.append(row)

    return rows


# ── Main scrape function ───────────────────────────────────────────────────────

def scrape_avb(limit: Optional[int] = None) -> pd.DataFrame:
    """
    Scrape all AVB communities. Returns a DataFrame with SCHEMA_COLUMNS.
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
    })

    all_rows: list[dict] = []
    errors: list[str] = []

    logger.info("Discovering AVB communities via state pages ...")
    communities = get_communities(session)
    if limit:
        communities = communities[:limit]
    logger.info(f"Scraping {len(communities)} communities ...")

    for i, comm in enumerate(communities, 1):
        raw_url = comm.get("url") or ""
        if raw_url.startswith("http"):
            # Some entries carry a full absolute URL — use directly
            page_url = raw_url
        else:
            if not raw_url.startswith("/"):
                raw_url = "/" + raw_url
            page_url = BASE_URL + raw_url

        # Extract city and state from address
        addr_obj = comm.get("address") or {}
        city     = addr_obj.get("city") or ""
        state    = addr_obj.get("state") or ""
        zip_code = addr_obj.get("zip") or addr_obj.get("postalCode") or ""
        addr_line = addr_obj.get("addressLine1") or ""
        full_address = ", ".join(
            p for p in [addr_line, city, state, zip_code] if p
        )

        coords    = comm.get("coordinates") or {}
        lat       = coords.get("latitude")
        lon       = coords.get("longitude")
        state_slug = comm.get("state_slug", "")

        community_meta = {
            "community":    comm.get("name") or "",
            "address":      full_address,
            "market":       _market_label(state_slug, city),
            "state":        state,
            "city":         city,
            "latitude":     lat,
            "longitude":    lon,
            "listing_url":  page_url,
            "community_id": comm.get("communityId") or "",
        }

        logger.info(
            f"[{i}/{len(communities)}] {community_meta['community']} "
            f"({community_meta['market']}) — {page_url}"
        )

        html = get_page(page_url, session)
        if not html:
            logger.warning(f"  Failed to fetch page — skipping")
            errors.append(page_url)
            continue

        gc = _extract_fusion(html)
        if not gc:
            logger.warning(f"  No Fusion.globalContent — skipping")
            errors.append(page_url)
            continue

        rows = extract_units(gc, community_meta)

        has_specials = comm.get("hasSpecials", False)
        logger.info(
            f"  {len(rows)} units | hasSpecials={has_specials} | "
            f"concession_rate={sum(1 for r in rows if r.get('has_concession'))}/{len(rows)}"
        )
        all_rows.extend(rows)

    if errors:
        logger.warning(f"Failed/skipped {len(errors)} communities: {errors[:5]}")

    df = pd.DataFrame(all_rows)
    # Ensure all schema columns are present
    for col in SCHEMA_COLUMNS:
        if col not in df.columns:
            df[col] = None
    df = df[SCHEMA_COLUMNS]

    logger.info(f"AVB total rows: {len(df):,}")
    return df


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser(description="Scrape AVB rental listings")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit to first N communities (testing)")
    parser.add_argument("--save",  action="store_true",
                        help="Save output CSV to data/raw/")
    args = parser.parse_args()

    df = scrape_avb(limit=args.limit)

    display_cols = [
        "community", "market", "unit_id", "beds", "sqft", "rent",
        "lease_term", "concession_hardness", "concession_type",
        "concession_value", "effective_monthly_rent",
    ]
    print(df[display_cols].head(15).to_string())
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
        path = os.path.join(out_dir, f"avb_raw_{today_str()}.csv")
        df.to_csv(path, index=False)
        print(f"\nSaved -> {path}")
