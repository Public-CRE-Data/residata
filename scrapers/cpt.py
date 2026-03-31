"""
CPT (Camden Property Trust) scraper.

Rendering:   Next.js (ISR) — unit data fully server-side rendered in
             __NEXT_DATA__ JSON blob; no Selenium needed.

Discovery:   Fetch 15 metro-hub pages → deduplicate community {cityState, slug} pairs.
             ~160 communities.
URL pattern: https://www.camdenliving.com/apartments/{cityState}/{communitySlug}/
Unit data:   https://www.camdenliving.com/apartments/{cityState}/{communitySlug}/available-apartments

Data model:  __NEXT_DATA__ → pageProps.data.availableApartments[]
  Each entry is a floor-plan group (one rent, one lease term, one move-in date)
  but may cover multiple available unit IDs (availableUnitIds list).
  → Expand each floor-plan entry into one row per unit ID.

Concession:  pageProps.specialCtaTitle  (community-level HTML banner)
             + pageProps.data.availableApartments[].floorPlanBanner (per floor-plan)
  Both are parsed through the same concession engine as MAA.
"""

import hashlib
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
from bs4 import BeautifulSoup

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.listing import SCHEMA_COLUMNS
from scrapers.maa import parse_concession, make_unit_id   # reuse shared engines
from utils.common import get_page, today_str, parse_int, parse_float

REIT = "CPT"
BASE_URL = "https://www.camdenliving.com"

# Metro hub pages that aggregate ALL Camden communities
# Keywords that indicate a text string is an actual concession offer
_CONCESSION_KW = re.compile(
    r"\b(free|off|save|saving|discount|reduc|waiv|month|week|move.?in)\b",
    re.IGNORECASE,
)

METRO_PAGES = [
    "houston-metro",
    "dallas-fort-worth-metro",
    "charlotte-metro",
    "atlanta-metro",
    "dc-maryland-metro",
    "nashville-metro",
    "phoenix-metro",
    "austin-metro",
    "tampa-metro",
    "southeast-florida-metro",
    "orlando-metro",
    "la-orange-county-metro",
    "san-diego-metro",
    "denver-metro",
    "raleigh-metro",
]

logger = logging.getLogger(__name__)


# ── HTML strip helper ──────────────────────────────────────────────────────────

def _strip_html(text: Optional[str]) -> Optional[str]:
    """Remove HTML tags from a string (e.g. '<p>Save $500!</p>' → 'Save $500!')"""
    if not text:
        return None
    cleaned = BeautifulSoup(text, "html.parser").get_text(separator=" ", strip=True)
    return cleaned if cleaned else None


# ── Market label from cityState slug ──────────────────────────────────────────

_CAMDEN_MARKET_MAP = {
    # TX
    "houston-tx":     "Houston",
    "richmond-tx":    "Houston",
    "cypress-tx":     "Houston",
    "tomball-tx":     "Houston",
    "spring-tx":      "Houston",
    "katy-tx":        "Houston",
    "stafford-tx":    "Houston",
    "dallas-tx":      "Dallas",
    "addison-tx":     "Dallas",
    "richardson-tx":  "Dallas",
    "plano-tx":       "Dallas",
    "frisco-tx":      "Dallas",
    "grapevine-tx":   "Dallas",
    "fort-worth-tx":  "Dallas",
    "austin-tx":      "Austin",
    "leander-tx":     "Austin",
    "cedar-park-tx":  "Austin",
    "round-rock-tx":  "Austin",
    # FL
    "tampa-fl":           "Tampa",
    "brandon-fl":         "Tampa",
    "clearwater-fl":      "Tampa",
    "st-petersburg-fl":   "Tampa",
    "orlando-fl":         "Orlando",
    "kissimmee-fl":       "Orlando",
    "plantation-fl":      "South Florida",
    "fort-lauderdale-fl": "South Florida",
    "aventura-fl":        "South Florida",
    "pembroke-pines-fl":  "South Florida",
    "doral-fl":           "South Florida",
    "miami-fl":           "South Florida",
    "boca-raton-fl":      "South Florida",
    # NC
    "charlotte-nc":    "Charlotte",
    "huntersville-nc": "Charlotte",
    "raleigh-nc":      "Raleigh",
    "apex-nc":         "Raleigh",
    "morrisville-nc":  "Raleigh",
    "chapel-hill-nc":  "Raleigh",
    "durham-nc":       "Raleigh",
    # GA
    "atlanta-ga":       "Atlanta",
    "alpharetta-ga":    "Atlanta",
    "dunwoody-ga":      "Atlanta",
    "kennesaw-ga":      "Atlanta",
    "stockbridge-ga":   "Atlanta",
    "peachtree-city-ga":"Atlanta",
    # CO
    "denver-co":          "Denver",
    "golden-co":          "Denver",
    "broomfield-co":      "Denver",
    "lakewood-co":        "Denver",
    "lone-tree-co":       "Denver",
    "englewood-co":       "Denver",
    "highlands-ranch-co": "Denver",
    # DC/MD/VA
    "washington-dc":  "Washington DC",
    "rockville-md":   "Washington DC",
    "gaithersburg-md":"Washington DC",
    "college-park-md":"Washington DC",
    "arlington-va":   "Washington DC",
    # AZ
    "phoenix-az":    "Phoenix",
    "scottsdale-az": "Phoenix",
    "chandler-az":   "Phoenix",
    "tempe-az":      "Phoenix",
    # TN
    "nashville-tn": "Nashville",
    "franklin-tn":  "Nashville",
    # CA
    "hollywood-ca":     "Los Angeles",
    "glendale-ca":      "Los Angeles",
    "irvine-ca":        "Orange County",
    "mission-viejo-ca": "Orange County",
    "chula-vista-ca":   "San Diego",
    "san-diego-ca":     "San Diego",
    "san-marcos-ca":    "San Diego",
    "murrieta-ca":      "Inland Empire",
    "ontario-ca":       "Inland Empire",
    "long-beach-ca":    "Los Angeles",
}

def market_from_city_state(city_state: str) -> str:
    """Map cityState slug to canonical market label."""
    mapped = _CAMDEN_MARKET_MAP.get(city_state)
    if mapped:
        return mapped
    # Fallback: title-case the city portion
    state = city_state.split("-")[-1].upper()
    city = " ".join(p.capitalize() for p in city_state.split("-")[:-1])
    return f"{city}, {state}"


def state_city_from_city_state(city_state: str) -> tuple[str, str]:
    """'houston-tx' → (state='TX', city='Houston')"""
    parts = city_state.split("-")
    state = parts[-1].upper()
    city  = " ".join(p.capitalize() for p in parts[:-1])
    return state, city


# ── Community discovery ────────────────────────────────────────────────────────

def get_community_slugs(session: requests.Session) -> list[dict]:
    """
    Fetch all Camden communities by scraping metro hub pages.
    Returns list of dicts: {cityState, slug, name, address}
    """
    seen: dict[tuple, dict] = {}

    for metro in METRO_PAGES:
        url = f"{BASE_URL}/apartments/{metro}"
        html = get_page(url, session)
        if not html:
            logger.warning(f"Could not fetch metro page: {metro}")
            continue

        m = re.search(
            r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>',
            html,
        )
        if not m:
            logger.warning(f"No __NEXT_DATA__ on {metro}")
            continue

        try:
            data = json.loads(m.group(1))
            comms = data["props"]["pageProps"].get("communities", [])
            for c in comms:
                key = (c["cityState"], c["slug"])
                if key not in seen:
                    seen[key] = c
            logger.info(f"  {metro}: {len(comms)} communities ({len(seen)} unique total)")
        except Exception as e:
            logger.warning(f"Error parsing {metro}: {e}")

    return list(seen.values())


# ── Unit extraction from __NEXT_DATA__ ────────────────────────────────────────

def extract_units(
    data: dict,
    community_meta: dict,
) -> list[dict]:
    """
    Parse the available-apartments __NEXT_DATA__ into unit rows.
    Expands each floor-plan group into one row per available unit ID.
    """
    scrape_dt   = date.today()
    rows: list[dict] = []

    pp = data["props"]["pageProps"]

    # Community-level concession (HTML banner).
    # Filter out pure marketing CTAs ("Looking for a 2BR?") that lack concession keywords.
    community_concession_raw = _strip_html(pp.get("specialCtaTitle"))
    has_community_concession = bool(
        community_concession_raw and _CONCESSION_KW.search(community_concession_raw)
    )

    apts = pp.get("data", {}).get("availableApartments", [])

    for apt in apts:
        if not apt.get("available", True):
            continue

        beds        = apt.get("bedrooms")
        baths       = apt.get("bathrooms")
        sqft        = apt.get("squareFeet")
        rent        = parse_float(str(apt.get("monthlyRent") or ""))
        lease_term  = apt.get("leaseTerm")        # actual lease term in months!
        fp_name     = apt.get("name")             # floor plan name
        move_in_raw = apt.get("moveInDate")       # ISO string
        fp_banner   = apt.get("floorPlanBanner")  # per-fp promotional text

        if rent is None:
            continue

        move_in_date = None
        if move_in_raw:
            try:
                move_in_date = date.fromisoformat(move_in_raw[:10])
            except ValueError:
                pass

        # Available unit IDs for this floor plan
        unit_ids = apt.get("availableUnitIds") or []
        if not unit_ids:
            # Fallback: use unitNumber if available
            unit_num = str(apt.get("unitNumber") or "")
            unit_ids = [unit_num] if unit_num else [""]

        # Concession: use floor-plan banner only if it looks like a concession
        # (contains money/time keywords). Plain feature banners ("Vaulted Ceilings")
        # must NOT override the community-level concession banner.
        fp_banner_text = _strip_html(fp_banner) if fp_banner else None
        fp_is_concession = bool(fp_banner_text and _CONCESSION_KW.search(fp_banner_text))
        if fp_is_concession:
            concession_raw = fp_banner_text
        elif has_community_concession:
            concession_raw = community_concession_raw
        else:
            concession_raw = None

        effective_lease_months = lease_term or 12
        concession_fields = parse_concession(
            raw=concession_raw,
            rent=rent,
            lease_months=effective_lease_months,
        )

        # Expand into one row per unit
        for unit_number in unit_ids:
            unit_id = make_unit_id(
                reit=REIT,
                community=community_meta["community"] or "",
                unit_number=unit_number,
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
                "unit_id":      unit_id,
                "beds":         beds,
                "baths":        baths,
                "sqft":         sqft,
                "rent":         rent,
                "move_in_date": move_in_date,
                "lease_term":   lease_term,          # CPT provides this!
                "listing_url":  community_meta["listing_url"],
                "first_seen":   scrape_dt,
                "last_seen":    scrape_dt,
                **concession_fields,
                # Supplemental
                "state":               community_meta["state"],
                "city":                community_meta["city"],
                "latitude":            community_meta.get("latitude"),
                "longitude":           community_meta.get("longitude"),
                "floorplan_name":      fp_name,
                "floor_level":         None,   # CPT does not expose floor number
                "rentcafe_property_id": community_meta.get("realpage_id"),
            }
            rows.append(row)

    return rows


# ── Main scrape function ───────────────────────────────────────────────────────

def scrape_cpt(limit: Optional[int] = None) -> pd.DataFrame:
    """
    Scrape all CPT communities. Returns a DataFrame with 30 schema columns.
    limit: restrict to first N communities (testing).
    """
    session = requests.Session()
    all_rows: list[dict] = []
    errors: list[str] = []

    logger.info("Discovering CPT communities via metro pages …")
    communities = get_community_slugs(session)
    if limit:
        communities = communities[:limit]
    logger.info(f"Scraping {len(communities)} communities …")

    for i, comm in enumerate(communities, 1):
        city_state   = comm["cityState"]
        slug         = comm["slug"]
        avail_url    = f"{BASE_URL}/apartments/{city_state}/{slug}/available-apartments"

        logger.info(f"[{i}/{len(communities)}] {avail_url}")
        html = get_page(avail_url, session)
        if not html:
            errors.append(avail_url)
            continue

        m = re.search(
            r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>',
            html,
        )
        if not m:
            logger.warning(f"  No __NEXT_DATA__ — skipping")
            errors.append(avail_url)
            continue

        try:
            data = json.loads(m.group(1))
        except json.JSONDecodeError as e:
            logger.warning(f"  JSON parse error: {e}")
            errors.append(avail_url)
            continue

        pp  = data["props"]["pageProps"]
        cd  = pp.get("data", {}).get("community", {})
        state, city = state_city_from_city_state(city_state)

        community_meta = {
            "community":    comm.get("name") or cd.get("name"),
            "address":      cd.get("address") or comm.get("address"),
            "market":       market_from_city_state(city_state),
            "state":        state,
            "city":         city,
            "latitude":     None,   # not in __NEXT_DATA__; geocode separately if needed
            "longitude":    None,
            "listing_url":  avail_url,
            "realpage_id":  str(cd.get("realPageCommunityId") or ""),
        }

        rows = extract_units(data, community_meta)

        logger.info(
            f"  {community_meta['community']} — "
            f"{len(rows)} units | "
            f"concession: {bool(pp.get('specialCtaTitle'))} | "
            f"raw: {str(pp.get('specialCtaTitle') or '')[:60]}"
        )
        all_rows.extend(rows)

    if errors:
        logger.warning(f"Failed/skipped {len(errors)} URLs: {errors[:5]}")

    df = pd.DataFrame(all_rows)
    for col in SCHEMA_COLUMNS:
        if col not in df.columns:
            df[col] = None
    df = df[SCHEMA_COLUMNS]

    logger.info(f"Total rows: {len(df)}")
    return df


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser(description="Scrape CPT rental listings")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--save",  action="store_true")
    args = parser.parse_args()

    df = scrape_cpt(limit=args.limit)

    print(df[[
        "community", "market", "unit_id", "beds", "sqft", "rent",
        "lease_term", "concession_hardness", "concession_type",
        "concession_value", "effective_monthly_rent",
    ]].head(15).to_string())
    print(f"\nShape: {df.shape}")

    if df["has_concession"].any():
        hard = df[df["concession_hardness"] == "hard"]
        soft = df[df["concession_hardness"] == "soft"]
        print(f"\nConcession breakdown: hard={len(hard)} soft={len(soft)} none={len(df)-len(hard)-len(soft)}")
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
        path = os.path.join(out_dir, f"cpt_raw_{today_str()}.csv")
        df.to_csv(path, index=False)
        print(f"\nSaved → {path}")
