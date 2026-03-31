"""
MAA (Mid-America Apartment Communities) scraper.

Rendering:   Static HTML — server-rendered via Sitecore / RentCafe backend.
             No Selenium or JS execution required.

Discovery:   https://www.maac.com/sitemap.xml  →  ~298 community pages
URL pattern: https://www.maac.com/{state}/{city}/{slug}/

Concession classification
─────────────────────────
Hard (quantifiable):
  • months_free  – "X months/weeks free", "no rent until <date>"
  • dollar_off   – "$X off", "save $X"
  • percent_off  – "X% off"

Soft (vague):
  • "look & lease", "call for specials", "limited time offer" without a number

Calculated fields (require lease_term; default 12 months when unknown):
  • concession_value         — months or dollars
  • concession_pct_lease_value  — concession $ ÷ (rent × lease_months) × 100
  • concession_pct_lease_term   — free_months ÷ lease_months × 100
  • effective_monthly_rent   — rent adjusted for concession spread over lease
"""

import hashlib
import logging
import os
import re
import sys
from datetime import date, timedelta
from typing import Optional

import pandas as pd
import requests
import yaml
from bs4 import BeautifulSoup

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.listing import SCHEMA_COLUMNS, ConcessionHardness, ConcessionType
from utils.common import get_page, today_str, parse_int, parse_float

REIT = "MAA"
BASE_URL = "https://www.maac.com"
SITEMAP_URL = "https://www.maac.com/sitemap.xml"
DEFAULT_LEASE_MONTHS = 12  # used when lease_term not scraped

logger = logging.getLogger(__name__)

# ── Load market map ────────────────────────────────────────────────────────────

def _load_market_map() -> dict[str, str]:
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "config", "markets.yaml"
    )
    with open(config_path, "r") as f:
        data = yaml.safe_load(f)
    return data.get("slug_to_market", {})

_MARKET_MAP = _load_market_map()


# ── Concession parsing engine ──────────────────────────────────────────────────

# Months/weeks free: "1 month free", "6 weeks free", "up to one month free"
_WORD_TO_NUM = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}
_MONTHS_FREE_RE = re.compile(
    r"(?:up\s+to\s+)?(\d+(?:\.\d+)?|one|two|three|four|five|six|seven|eight|nine|ten)"
    r"\s*(month|week)['\u2019]?s?\s+free",
    re.IGNORECASE,
)
# Dollar off: "$1,000 off", "save $750", "save—$1000", "save $ 600", "reduced $150",
#             "reduced -$100", "$1500 in savings", "1500 in savings"
_DOLLAR_OFF_RE = re.compile(
    # Require captured number is NOT followed by % (that's percent_off territory)
    r"(?:save\s*[\u2013\u2014\-]?\s*|reduced\s*[\u2013\u2014\-]?\s*|\$)\s*-?\$?\s*([\d,]+(?:\.\d+)?)(?!\s*%)\s*(?:off|in savings|!)?"
    r"|(?<!\d)([\d,]{3,}(?:\.\d+)?)(?!\s*%)\s+in\s+savings",
    re.IGNORECASE,
)
# Percent off: "10% off", "1/2 off", "half off", "50% discount"
_PERCENT_OFF_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*%\s*(?:off|discount)"
    r"|(?:^|\s)(1/2|half)\s+off",
    re.IGNORECASE,
)
# Date-based free rent with day: "no rent until May 15", "free until April 30"
_DATE_FREE_RE = re.compile(
    r"(?:no\s+rent|free)\s+until\s+(\w+\s+\d+)",
    re.IGNORECASE,
)
# Month-name free variants:
#   "for April free", "for May free"         (whole calendar month)
#   "no rent until April", "free until April" (month-only, no day number)
_MONTH_FREE_RE = re.compile(
    r"(?:\bfor\s+|\buntil\s+)(January|February|March|April|May|June|July|August"
    r"|September|October|November|December)(?:\s+\w+)?\s*(?:free\b|[!.,]|$)",
    re.IGNORECASE,
)
# Soft-only markers (no quantity)
_SOFT_ONLY_RE = re.compile(
    r"\b(call\s+for\s+specials|look\s+(?:and|&)\s+lease|limited\s+time"
    r"|ask\s+about|contact\s+us|special\s+offer)\b",
    re.IGNORECASE,
)


def parse_concession(
    raw: Optional[str],
    rent: float,
    lease_months: int = DEFAULT_LEASE_MONTHS,
) -> dict:
    """
    Parse a concession description string into all plan-specified fields.
    Returns a dict of concession-related fields.
    """
    nulls = {
        "has_concession": False,
        "concession_hardness": None,
        "concession_raw": None,
        "concession_type": None,
        "concession_value": None,
        "concession_pct_lease_value": None,
        "concession_pct_lease_term": None,
        "effective_monthly_rent": None,
    }

    if not raw or not raw.strip():
        return nulls

    text = raw.strip()
    result = {**nulls, "has_concession": True, "concession_raw": text}

    # ── months/weeks free ─────────────────────────────────────────────────
    m = _MONTHS_FREE_RE.search(text)
    if m:
        qty_raw = m.group(1).lower()
        qty = float(_WORD_TO_NUM.get(qty_raw, qty_raw))
        unit = m.group(2).lower()
        if unit.startswith("week"):
            qty = round(qty / 4.333, 4)   # convert weeks → months
        result["concession_hardness"] = ConcessionHardness.hard.value
        result["concession_type"] = ConcessionType.months_free.value
        result["concession_value"] = qty
        concession_dollars = rent * qty
        result["concession_pct_lease_value"] = round(
            concession_dollars / (rent * lease_months) * 100, 2
        )
        result["concession_pct_lease_term"] = round(qty / lease_months * 100, 2)
        result["effective_monthly_rent"] = round(
            rent * (lease_months - qty) / lease_months, 2
        )
        return result

    # ── dollar off ────────────────────────────────────────────────────────
    m = _DOLLAR_OFF_RE.search(text)
    if m:
        raw_amt = m.group(1) or m.group(2)   # group 2 = "N in savings" branch
        dollars = parse_float(raw_amt.replace(",", "")) if raw_amt else None
        if dollars and dollars >= 10:   # sanity filter: ignore noise values
            result["concession_hardness"] = ConcessionHardness.hard.value
            result["concession_type"] = ConcessionType.dollar_off.value
            result["concession_value"] = dollars
            total_lease = rent * lease_months
            result["concession_pct_lease_value"] = round(dollars / total_lease * 100, 2)
            result["concession_pct_lease_term"] = None  # not term-based
            result["effective_monthly_rent"] = round(
                rent - dollars / lease_months, 2
            )
            return result

    # ── percent off ───────────────────────────────────────────────────────
    m = _PERCENT_OFF_RE.search(text)
    if m:
        raw_pct = m.group(1) or m.group(2)  # group(2) = "1/2" or "half"
        if raw_pct and raw_pct.lower() in ("1/2", "half"):
            pct = 50.0
        else:
            pct = float(raw_pct) if raw_pct else 0.0
        dollars = rent * lease_months * pct / 100
        result["concession_hardness"] = ConcessionHardness.hard.value
        result["concession_type"] = ConcessionType.percent_off.value
        result["concession_value"] = pct
        result["concession_pct_lease_value"] = round(pct, 2)
        result["concession_pct_lease_term"] = None
        result["effective_monthly_rent"] = round(rent * (1 - pct / 100), 2)
        return result

    # ── date-free with specific day (e.g. "no rent until May 15") ───────────
    m = _DATE_FREE_RE.search(text)
    if m:
        result["concession_hardness"] = ConcessionHardness.hard.value
        result["concession_type"] = ConcessionType.date_free.value
        try:
            from dateutil.parser import parse as dateparse
            free_until = dateparse(m.group(1) + f" {date.today().year}")
            days_free = max(0, (free_until.date() - date.today()).days)
            months_free = round(days_free / 30.4, 2)
        except Exception:
            months_free = None
        result["concession_value"] = months_free
        if months_free:
            result["concession_pct_lease_value"] = round(
                months_free / lease_months * 100, 2
            )
            result["concession_pct_lease_term"] = round(
                months_free / lease_months * 100, 2
            )
            result["effective_monthly_rent"] = round(
                rent * (lease_months - months_free) / lease_months, 2
            )
        return result

    # ── month-name free (e.g. "for April free", "until April", "for May free") ─
    m = _MONTH_FREE_RE.search(text)
    if m:
        month_name = m.group(1)
        result["concession_hardness"] = ConcessionHardness.hard.value
        result["concession_type"] = ConcessionType.date_free.value
        try:
            import calendar
            yr = date.today().year
            month_num = list(calendar.month_name).index(month_name.capitalize())
            days_in_month = calendar.monthrange(yr, month_num)[1]
            months_free = round(days_in_month / 30.4, 2)
        except Exception:
            months_free = 1.0  # fallback: one month
        result["concession_value"] = months_free
        result["concession_pct_lease_value"] = round(
            months_free / lease_months * 100, 2
        )
        result["concession_pct_lease_term"] = round(
            months_free / lease_months * 100, 2
        )
        result["effective_monthly_rent"] = round(
            rent * (lease_months - months_free) / lease_months, 2
        )
        return result

    # ── soft fallback ─────────────────────────────────────────────────────
    result["concession_hardness"] = ConcessionHardness.soft.value
    return result


# ── Synthetic unit ID ──────────────────────────────────────────────────────────

def make_unit_id(
    reit: str,
    community: str,
    unit_number: str,
    beds: Optional[int],
    sqft: Optional[int],
) -> str:
    """
    Construct a stable synthetic unit identifier.
    Format: REIT-CommSlug-BEDSbr-SQFTsqft-UNITNUM
    e.g.   MAA-Riverchase-1br-800sqft-181807
    Falls back to a short hash if unit_number is unavailable.
    """
    comm_slug = re.sub(r"[^a-z0-9]+", "-", community.lower()).strip("-")
    # Strip leading reit ticker prefix if the community name starts with it
    # e.g. "maa-eagle-ridge" → "eagle-ridge"
    ticker_prefix = reit.lower() + "-"
    if comm_slug.startswith(ticker_prefix):
        comm_slug = comm_slug[len(ticker_prefix):]
    bed_part = f"{beds}br" if beds is not None else "Xbr"
    sqft_part = f"{sqft}sqft" if sqft is not None else "Xsqft"

    if unit_number:
        return f"{reit}-{comm_slug}-{bed_part}-{sqft_part}-{unit_number}"

    # No unit number: hash the community+bed+sqft combo
    raw = f"{reit}-{comm_slug}-{bed_part}-{sqft_part}"
    short_hash = hashlib.md5(raw.encode()).hexdigest()[:6].upper()
    return f"{reit}-{comm_slug}-{bed_part}-{sqft_part}-{short_hash}"


# ── Sitemap discovery ──────────────────────────────────────────────────────────

def get_community_urls(session: requests.Session) -> list[str]:
    html = get_page(SITEMAP_URL, session)
    if not html:
        logger.error("Could not fetch sitemap")
        return []
    soup = BeautifulSoup(html, "xml")
    urls = []
    for loc in soup.find_all("loc"):
        url = loc.text.strip()
        path = url.replace(BASE_URL, "").strip("/")
        segs = [s for s in path.split("/") if s]
        if len(segs) == 3 and not segs[0].startswith(("about", "new-dev", "career")):
            urls.append(url)
    logger.info(f"Discovered {len(urls)} community URLs")
    return urls


# ── State / city / market from URL ────────────────────────────────────────────

def meta_from_url(url: str) -> dict:
    path = url.replace(BASE_URL, "").strip("/")
    segs = path.split("/")
    state_slug = segs[0] if len(segs) > 0 else ""
    city_slug  = segs[1] if len(segs) > 1 else ""
    state = state_slug.replace("-", " ").title()
    city  = city_slug.replace("-", " ").title()
    market = _MARKET_MAP.get(city_slug, city)   # fallback to city name
    return {"state": state, "city": city, "market": market}


# ── Community metadata extraction ─────────────────────────────────────────────

def extract_community_meta(soup: BeautifulSoup, url: str) -> dict:
    meta = {
        "community": None,
        "address": None,
        "latitude": None,
        "longitude": None,
        "rentcafe_property_id": None,
    }

    prop_info = soup.find(class_="property-information")
    if prop_info:
        h1 = prop_info.find("h1")
        if h1:
            meta["community"] = h1.get_text(strip=True)

    addr_tag = soup.find(id="property-address")
    if addr_tag:
        meta["address"] = addr_tag.get_text(strip=True)
        href = addr_tag.get("href", "")
        m = re.search(r"center=([-\d.]+),\s*([-\d.]+)", href)
        if m:
            meta["latitude"]  = parse_float(m.group(1))
            meta["longitude"] = parse_float(m.group(2))

    pid = re.search(r"propertyId=(\d+)", str(soup))
    if pid:
        meta["rentcafe_property_id"] = pid.group(1)

    return meta


# ── Floor level validation ─────────────────────────────────────────────────────

_FLOOR_RE = re.compile(r"^\w[\w\s]*\bFloor\b$", re.IGNORECASE)

def clean_floor(text: str) -> Optional[str]:
    t = text.strip()
    return t if t and _FLOOR_RE.match(t) else None


# ── Move-in date parsing ───────────────────────────────────────────────────────

def parse_move_in(text: str) -> Optional[date]:
    """'Move-in: 03/30 - 04/02' → date(2026, 3, 30)"""
    m = re.search(r"Move-in:\s*(\d{1,2}/\d{1,2})", text, re.IGNORECASE)
    if not m:
        return None
    month, day = m.group(1).split("/")
    y = date.today().year
    try:
        d = date(y, int(month), int(day))
        if (d - date.today()).days < -60:
            d = date(y + 1, int(month), int(day))
        return d
    except ValueError:
        return None


# ── Floor plan name ────────────────────────────────────────────────────────────

def parse_floorplan(amenities: str) -> Optional[str]:
    m = re.search(r"\b([A-Z0-9]+-FP)\b", amenities, re.IGNORECASE)
    return m.group(1).upper() if m else None


# ── Unit extraction ────────────────────────────────────────────────────────────

def extract_units(
    soup: BeautifulSoup,
    community_meta: dict,
    url_meta: dict,
    listing_url: str,
    has_community_concession: bool,
    concession_raw: Optional[str],
) -> list[dict]:
    scrape_dt = date.today()
    rows = []

    blocks = soup.find_all("div", class_="available-apartments__body--apt")
    for block in blocks:
        # ── Unit number ───────────────────────────────────────────────
        unit_number = None
        unit_span = block.find(class_="unit")
        if unit_span:
            raw = unit_span.get_text(strip=True)
            m = re.search(r"#(\S+)", raw)
            unit_number = m.group(1) if m else raw

        # ── Price ─────────────────────────────────────────────────────
        rent = None
        price_span = block.find("span", attrs={"class": "price", "style": True})
        if price_span:
            rent = parse_float(re.sub(r"[^\d.]", "", price_span.get_text(strip=True)))

        if rent is None:
            continue   # rent is mandatory — skip row

        # ── Detail list ───────────────────────────────────────────────
        lis = block.select("div.apt-details ul li")
        beds = baths = sqft = None
        floor_level = move_in_date = None

        if len(lis) >= 1:
            bed_bath = lis[0].get_text(strip=True)
            bm = re.search(r"(\d+)\s*Bed", bed_bath, re.IGNORECASE)
            bam = re.search(r"(\d+(?:\.\d+)?)\s*Bath", bed_bath, re.IGNORECASE)
            if bm:  beds  = int(bm.group(1))
            if bam: baths = float(bam.group(1))
        if len(lis) >= 2:
            sqft = parse_int(lis[1].get_text(strip=True))
        if len(lis) >= 3:
            floor_level = clean_floor(lis[2].get_text(strip=True))
        if len(lis) >= 4:
            move_in_date = parse_move_in(lis[3].get_text())

        # ── Amenities / floor plan ────────────────────────────────────
        fp_name = None
        amen_div = block.find(class_="apt-amenities")
        if amen_div:
            fp_name = parse_floorplan(amen_div.get_text(strip=True))

        # ── Concession ────────────────────────────────────────────────
        # Unit has a concession if it has the badge OR community has a banner
        unit_has_badge = bool(block.find(class_="special-offer-btn"))
        effective_concession = (unit_has_badge or has_community_concession)
        raw_text = concession_raw if effective_concession else None

        concession_fields = parse_concession(
            raw=raw_text,
            rent=rent,
            lease_months=DEFAULT_LEASE_MONTHS,
        )

        # ── Synthetic unit ID ─────────────────────────────────────────
        unit_id = make_unit_id(
            reit=REIT,
            community=community_meta.get("community") or "",
            unit_number=unit_number or "",
            beds=beds,
            sqft=sqft,
        )

        row = {
            # Core 23 fields
            "scrape_date":   scrape_dt,
            "reit":          REIT,
            "community":     community_meta.get("community"),
            "address":       community_meta.get("address"),
            "market":        url_meta["market"],
            "unit_id":       unit_id,
            "beds":          beds,
            "baths":         baths,
            "sqft":          sqft,
            "rent":          rent,
            "move_in_date":  move_in_date,
            "lease_term":    None,   # MAA does not show lease term on listing card
            "listing_url":   listing_url,
            "first_seen":    scrape_dt,   # set to scrape_date; tracker will update
            "last_seen":     scrape_dt,
            **concession_fields,
            # Supplemental
            "state":               url_meta["state"],
            "city":                url_meta["city"],
            "latitude":            community_meta.get("latitude"),
            "longitude":           community_meta.get("longitude"),
            "floorplan_name":      fp_name,
            "floor_level":         floor_level,
            "rentcafe_property_id": community_meta.get("rentcafe_property_id"),
        }
        rows.append(row)

    return rows


# ── Main scrape function ───────────────────────────────────────────────────────

def scrape_maa(limit: Optional[int] = None) -> pd.DataFrame:
    """
    Scrape all MAA communities. Returns a DataFrame.
    limit: restrict to first N communities (for testing).
    """
    session = requests.Session()
    all_rows: list[dict] = []
    errors: list[str] = []

    urls = get_community_urls(session)
    if limit:
        urls = urls[:limit]

    for i, url in enumerate(urls, 1):
        logger.info(f"[{i}/{len(urls)}] {url}")
        html = get_page(url, session)
        if not html:
            errors.append(url)
            continue

        soup = BeautifulSoup(html, "lxml")
        if soup.title and "Page Not Found" in (soup.title.string or ""):
            logger.warning(f"  404 — skipping")
            errors.append(url)
            continue

        url_meta       = meta_from_url(url)
        community_meta = extract_community_meta(soup, url)

        # Community-level concession
        has_community_concession = False
        concession_raw = None
        wrapper = soup.find(class_="move-in-special-wrapper")
        if wrapper:
            special = wrapper.find(class_="move-in-special")
            if special:
                p = special.find("p")
                if p:
                    concession_raw = p.get_text(strip=True)
                    has_community_concession = bool(concession_raw)

        rows = extract_units(
            soup=soup,
            community_meta=community_meta,
            url_meta=url_meta,
            listing_url=url,
            has_community_concession=has_community_concession,
            concession_raw=concession_raw,
        )
        logger.info(
            f"  {community_meta.get('community','?')} — "
            f"{len(rows)} units | "
            f"concession: {has_community_concession} | "
            f"raw: {concession_raw[:60] if concession_raw else None}"
        )
        all_rows.extend(rows)

    if errors:
        logger.warning(f"Failed/skipped {len(errors)} URLs: {errors[:5]}")

    df = pd.DataFrame(all_rows)
    # Enforce column order, adding any missing columns as NaN
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

    parser = argparse.ArgumentParser(description="Scrape MAA rental listings")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--save",  action="store_true")
    args = parser.parse_args()

    df = scrape_maa(limit=args.limit)

    print(df[["community", "market", "unit_id", "beds", "sqft", "rent",
              "concession_hardness", "concession_type", "concession_value",
              "effective_monthly_rent"]].head(15).to_string())
    print(f"\nShape: {df.shape}")

    if df["has_concession"].any():
        hard = df[df["concession_hardness"] == "hard"]
        soft = df[df["concession_hardness"] == "soft"]
        print(f"\nConcession breakdown:  hard={len(hard)}  soft={len(soft)}  none={len(df)-len(hard)-len(soft)}")
        print("\nConcession types:\n", hard["concession_type"].value_counts())
        print("\nSample concessions:")
        print(hard[["community", "rent", "concession_type", "concession_value",
                     "concession_pct_lease_value", "effective_monthly_rent",
                     "concession_raw"]].drop_duplicates("community").to_string())

    if args.save:
        out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw")
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"maa_raw_{today_str()}.csv")
        df.to_csv(path, index=False)
        print(f"\nSaved → {path}")
