"""
MAA (Mid-America Apartment Communities) scraper.

Rendering:   Static HTML — server-rendered via Sitecore / RentCafe backend.
             No Selenium or JS execution required.

Discovery:   https://www.maac.com/sitemap.xml  →  298 community pages
URL pattern: https://www.maac.com/{state}/{city}/{slug}/

Key selectors per community page:
  Community metadata:
    - Name       : h1 inside .property-information
    - Address    : #property-address  (href also contains lat/lon)
    - Lat/Lon    : center= param in Google Maps href inside #property-address
    - Zip        : parsed from address string (last 5 digits of state abbreviation block)
    - RentCafe ID: first occurrence of propertyId=\d+ in page source

  Unit listings (.available-apartments__body--apt):
    - Unit #     : .unit  →  "Unit #181807"
    - Price      : span[class="price"][style]  →  "$1138"
    - Beds/baths : .apt-details ul li[0]       →  "1 Bed, 1 Bath"
    - Sqft       : .apt-details ul li[1]       →  "800 Sq. Ft."
    - Floor      : .apt-details ul li[2]       →  "First Floor" or blank
    - Move-in    : .apt-details ul li[3]       →  "Move-in: 03/30 - 04/02"
    - FP name    : .apt-amenities text         →  contains "11C-FP" token
    - Concession : presence of .special-offer-btn inside unit block

  Community-level concession text:
    - .move-in-special-wrapper .move-in-special p
    - Classified as "hard" if text mentions "free", "off", or a dollar amount;
      "soft" otherwise (e.g., waived fees, reduced deposit).
"""

import re
import sys
import os
import logging
import requests
import pandas as pd
from bs4 import BeautifulSoup
from datetime import date
from typing import Optional

# Allow running directly from scrapers/ or from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SCHEMA_COLUMNS, RAW_DIR
from utils.common import get_page, today_str, iso_week_str, parse_int, parse_float

REIT = "MAA"
SITEMAP_URL = "https://www.maac.com/sitemap.xml"
BASE_URL = "https://www.maac.com"

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)


# ---------------------------------------------------------------------------
# Sitemap discovery
# ---------------------------------------------------------------------------

def get_community_urls(session: requests.Session) -> list[str]:
    """
    Parse the sitemap to return all active community page URLs.
    Pattern: 3-segment paths like /state/city/maa-community-name/
    Excludes about-us, new-development, careers pages.
    """
    html = get_page(SITEMAP_URL, session)
    if not html:
        logger.error("Could not fetch sitemap")
        return []

    soup = BeautifulSoup(html, "xml")
    urls = []
    for loc in soup.find_all("loc"):
        url = loc.text.strip()
        path = url.replace(BASE_URL, "").strip("/")
        segments = [s for s in path.split("/") if s]
        if len(segments) == 3 and not segments[0].startswith(("about", "new-dev", "career")):
            urls.append(url)

    logger.info(f"Discovered {len(urls)} community URLs from sitemap")
    return urls


# ---------------------------------------------------------------------------
# Community metadata extraction
# ---------------------------------------------------------------------------

def extract_community_meta(soup: BeautifulSoup, url: str) -> dict:
    """Extract property-level fields from a community page."""
    meta = {
        "community_name": None,
        "address": None,
        "zip_code": None,
        "latitude": None,
        "longitude": None,
        "rentcafe_property_id": None,
    }

    # Community name — <h1> inside .property-information
    prop_info = soup.find(class_="property-information")
    if prop_info:
        h1 = prop_info.find("h1")
        if h1:
            meta["community_name"] = h1.get_text(strip=True)

    # Address + lat/lon — #property-address anchor
    addr_tag = soup.find(id="property-address")
    if addr_tag:
        meta["address"] = addr_tag.get_text(strip=True)

        # Lat/lon from Google Maps href: center=33.354, -86.7901
        href = addr_tag.get("href", "")
        coord_match = re.search(r"center=([-\d.]+),\s*([-\d.]+)", href)
        if coord_match:
            meta["latitude"] = parse_float(coord_match.group(1))
            meta["longitude"] = parse_float(coord_match.group(2))

        # Zip — only extract from the property's own address string.
        # MAA omits zip from many property-address anchors; lat/lon is always
        # present for downstream geocoding if needed.
        zip_match = re.search(r"\b(\d{5})\b", meta["address"] or "")
        if zip_match:
            meta["zip_code"] = zip_match.group(1)

    # RentCafe property ID — first propertyId= in source
    page_text = str(soup)
    pid_match = re.search(r"propertyId=(\d+)", page_text)
    if pid_match:
        meta["rentcafe_property_id"] = pid_match.group(1)

    return meta


# ---------------------------------------------------------------------------
# Concession extraction
# ---------------------------------------------------------------------------

# Patterns that indicate a "hard" (dollar/time-quantified) concession
_HARD_CONCESSION_RE = re.compile(
    r"\b(free|off|\$\d|\d+\s*month|\d+\s*week|reduced rent|look-and-lease)\b",
    re.IGNORECASE,
)

def extract_concession(soup: BeautifulSoup) -> tuple[bool, Optional[str]]:
    """
    Returns (community_has_any_concession: bool, concession_description: str|None).
    Looks for the community-level move-in-special banner.
    """
    wrapper = soup.find(class_="move-in-special-wrapper")
    if not wrapper:
        return False, None

    p_tag = wrapper.find(class_="move-in-special")
    if not p_tag:
        return False, None

    p_text = p_tag.find("p")
    description = p_text.get_text(strip=True) if p_text else None
    return True, description


def classify_concession(description: Optional[str]) -> Optional[str]:
    """
    Classify concession as 'hard' or 'soft'.
    Hard: quantified — free rent weeks/months, specific dollar amount.
    Soft: unquantified — waived fees, deposit reductions, looser terms.
    Returns None if no description.
    """
    if not description:
        return None
    return "hard" if _HARD_CONCESSION_RE.search(description) else "soft"


# ---------------------------------------------------------------------------
# Unit listing extraction
# ---------------------------------------------------------------------------

def parse_move_in_dates(text: str) -> Optional[str]:
    """
    Parse 'Move-in: 03/30 - 04/02' → ISO date of start: '2026-03-30'.
    Assumes current or next calendar year.
    """
    text = text.strip()
    m = re.search(r"Move-in:\s*(\d{1,2}/\d{1,2})", text, re.IGNORECASE)
    if not m:
        return None
    raw = m.group(1)  # e.g. "03/30"
    month, day = raw.split("/")
    year = date.today().year
    try:
        d = date(year, int(month), int(day))
        # If inferred date is in the past by >60 days, bump to next year
        if (d - date.today()).days < -60:
            d = date(year + 1, int(month), int(day))
        return d.isoformat()
    except ValueError:
        return None


def parse_floorplan_name(amenities_text: str) -> Optional[str]:
    """
    Extract floor plan code from amenities string.
    MAA embeds the RentCafe floor plan code in amenities, e.g. '11C-FP'.
    Pattern: alphanumeric token ending in '-FP' or '-fp'.
    """
    m = re.search(r"\b([A-Z0-9]+-FP)\b", amenities_text, re.IGNORECASE)
    return m.group(1).upper() if m else None


def extract_units(
    soup: BeautifulSoup,
    community_meta: dict,
    community_url: str,
    has_concession: bool,
    concession_desc: Optional[str],
    state: str,
    city: str,
) -> list[dict]:
    """
    Extract all unit rows from a community page.
    Returns list of dicts conforming to SCHEMA_COLUMNS.
    """
    scrape_date = today_str()
    scrape_week = iso_week_str()
    rows = []

    unit_blocks = soup.find_all("div", class_="available-apartments__body--apt")
    if not unit_blocks:
        logger.debug(f"No units found on {community_url}")
        return rows

    for block in unit_blocks:
        row = {col: None for col in SCHEMA_COLUMNS}
        row.update({
            "reit": REIT,
            "scrape_date": scrape_date,
            "scrape_week": scrape_week,
            "state": state,
            "city": city,
            "community_url": community_url,
            **community_meta,
        })

        # Unit number
        unit_span = block.find(class_="unit")
        if unit_span:
            raw = unit_span.get_text(strip=True)
            m = re.search(r"#(\S+)", raw)
            row["unit_number"] = m.group(1) if m else raw

        # Price — span with class="price" and inline style
        price_span = block.find("span", attrs={"class": "price", "style": True})
        if price_span:
            row["asking_rent"] = parse_int(price_span.get_text(strip=True))

        # Details list: [beds/baths, sqft, floor, move-in]
        detail_lis = block.select("div.apt-details ul li")
        if len(detail_lis) >= 1:
            bed_bath = detail_lis[0].get_text(strip=True)
            bed_m = re.search(r"(\d+)\s*Bed", bed_bath, re.IGNORECASE)
            bath_m = re.search(r"(\d+)\s*Bath", bed_bath, re.IGNORECASE)
            if bed_m:
                row["bedrooms"] = int(bed_m.group(1))
            if bath_m:
                row["bathrooms"] = int(bath_m.group(1))
        if len(detail_lis) >= 2:
            row["sqft"] = parse_int(detail_lis[1].get_text(strip=True))
        if len(detail_lis) >= 3:
            floor_text = detail_lis[2].get_text(strip=True)
            # Accept only recognised floor-level values; reject amenity tags
            # e.g. "First Floor", "Second Floor" → OK; "Premium Flooring" → None
            if re.match(r"^\w[\w\s]*\bFloor\b$", floor_text, re.IGNORECASE):
                row["floor_level"] = floor_text
            else:
                row["floor_level"] = None
        if len(detail_lis) >= 4:
            row["available_from"] = parse_move_in_dates(detail_lis[3].get_text())

        # Amenities / floor plan name
        amenities_div = block.find(class_="apt-amenities")
        if amenities_div:
            amenities_text = amenities_div.get_text(strip=True)
            row["floorplan_name"] = parse_floorplan_name(amenities_text)

        # Unit-level concession flag (Special Offer badge on image)
        unit_has_special = bool(block.find(class_="special-offer-btn"))
        # A unit is marked as having a concession if:
        #   (a) it has the special-offer-btn, OR
        #   (b) the community has a move-in-special banner
        row["has_concession"] = unit_has_special or has_concession
        row["concession_description"] = concession_desc if row["has_concession"] else None

        rows.append(row)

    return rows


# ---------------------------------------------------------------------------
# State/city derivation from URL
# ---------------------------------------------------------------------------

def state_city_from_url(url: str) -> tuple[str, str]:
    """
    Extract human-readable state and city from URL segments.
    '/georgia/atlanta/maa-buckhead/' -> ('Georgia', 'Atlanta')
    '/north-carolina/raleigh-cary/'  -> ('North Carolina', 'Raleigh-Cary')
    """
    path = url.replace(BASE_URL, "").strip("/")
    segments = path.split("/")
    state = segments[0].replace("-", " ").title() if len(segments) > 0 else ""
    city  = segments[1].replace("-", " ").title() if len(segments) > 1 else ""
    return state, city


# ---------------------------------------------------------------------------
# Main scrape entry point
# ---------------------------------------------------------------------------

def scrape_maa(limit: Optional[int] = None) -> pd.DataFrame:
    """
    Scrape all MAA communities.
    limit: if set, only scrape the first N communities (for testing).
    Returns DataFrame with all scraped rows.
    """
    session = requests.Session()
    all_rows = []
    errors = []

    urls = get_community_urls(session)
    if limit:
        urls = urls[:limit]

    total = len(urls)
    for i, url in enumerate(urls, 1):
        logger.info(f"[{i}/{total}] {url}")

        html = get_page(url, session)
        if not html:
            errors.append(url)
            continue

        soup = BeautifulSoup(html, "lxml")

        # Skip 404 pages
        if soup.title and "Page Not Found" in soup.title.string:
            logger.warning(f"  404 — skipping {url}")
            errors.append(url)
            continue

        state, city = state_city_from_url(url)
        community_meta = extract_community_meta(soup, url)
        has_concession, concession_desc = extract_concession(soup)

        rows = extract_units(
            soup=soup,
            community_meta=community_meta,
            community_url=url,
            has_concession=has_concession,
            concession_desc=concession_desc,
            state=state,
            city=city,
        )

        logger.info(
            f"  {community_meta.get('community_name', '?')} — "
            f"{len(rows)} units | concession: {has_concession}"
        )
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows, columns=SCHEMA_COLUMNS)

    if errors:
        logger.warning(f"Failed/skipped {len(errors)} communities: {errors[:10]}")

    logger.info(f"Total rows scraped: {len(df)}")
    return df


def save_raw(df: pd.DataFrame) -> str:
    """Save raw CSV with date-stamped filename in data/raw/."""
    os.makedirs(RAW_DIR, exist_ok=True)
    filename = os.path.join(RAW_DIR, f"maa_raw_{today_str()}.csv")
    df.to_csv(filename, index=False)
    logger.info(f"Saved raw data → {filename}")
    return filename


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Scrape MAA rental listings")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit to N communities (for testing)")
    parser.add_argument("--save", action="store_true",
                        help="Save output to data/raw/")
    args = parser.parse_args()

    df = scrape_maa(limit=args.limit)
    print(df.head(10).to_string())
    print(f"\nShape: {df.shape}")
    print(f"\nConcession summary:\n{df['has_concession'].value_counts()}")
    print(f"\nBed type distribution:\n{df['bedrooms'].value_counts().sort_index()}")

    if args.save:
        save_raw(df)
