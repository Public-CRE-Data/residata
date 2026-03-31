"""
Pydantic data model for a single rental listing row.
All scrapers must map their raw output to this model.
"""
from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class ConcessionHardness(str, Enum):
    hard = "hard"   # quantifiable: months free, $ off, % off
    soft = "soft"   # vague: "call for specials", "look & lease"


class ConcessionType(str, Enum):
    months_free  = "months_free"   # e.g. "1 month free", "6 weeks free"
    dollar_off   = "dollar_off"    # e.g. "$1,000 off first month"
    percent_off  = "percent_off"   # e.g. "10% off"
    date_free    = "date_free"     # e.g. "no rent until May 15"
    # null → soft concession (no enum value needed, just None)


class Listing(BaseModel):
    # ── Identity ──────────────────────────────────────────────────────────
    scrape_date:  date
    reit:         str                    # ticker: MAA, CPT, EQR …
    community:    str                    # property name
    address:      str                    # full street address
    market:       str                    # MSA/market label (normalized)
    unit_id:      str                    # synthetic composite key

    # ── Unit specs ────────────────────────────────────────────────────────
    beds:         int
    baths:        float
    sqft:         int
    rent:         float                  # listed asking rent (before concessions)
    move_in_date: Optional[date] = None  # earliest available move-in
    lease_term:   Optional[str]  = None  # e.g. "12 months"; None if not shown

    # ── Concessions ───────────────────────────────────────────────────────
    has_concession:            bool
    concession_hardness:       Optional[ConcessionHardness]     = None
    concession_raw:            Optional[str]                    = None
    concession_type:           Optional[ConcessionType]         = None
    concession_value:          Optional[float]                  = None  # months or $
    concession_pct_lease_value: Optional[float]                 = None  # % of lease $
    concession_pct_lease_term:  Optional[float]                 = None  # % of lease term
    effective_monthly_rent:    Optional[float]                  = None  # rent adj. for concession

    # ── Provenance ────────────────────────────────────────────────────────
    listing_url:  str
    first_seen:   date                   # first scrape this unit appeared
    last_seen:    date                   # last scrape this unit still listed

    # ── Extra (not in base 23-field spec, kept for research utility) ──────
    state:              Optional[str]   = None
    city:               Optional[str]   = None
    latitude:           Optional[float] = None
    longitude:          Optional[float] = None
    floorplan_name:     Optional[str]   = None  # e.g. "11C-FP"
    floor_level:        Optional[str]   = None  # "First Floor", "Second Floor"
    rentcafe_property_id: Optional[str] = None

    @field_validator("rent", "baths", mode="before")
    @classmethod
    def coerce_numeric(cls, v):
        if v is None:
            return v
        if isinstance(v, str):
            import re
            digits = re.sub(r"[^\d.]", "", v)
            return float(digits) if digits else None
        return v

    @field_validator("beds", "sqft", mode="before")
    @classmethod
    def coerce_int(cls, v):
        if v is None:
            return v
        if isinstance(v, float):
            return int(v)
        if isinstance(v, str):
            import re
            digits = re.sub(r"[^\d]", "", v)
            return int(digits) if digits else None
        return v

    class Config:
        use_enum_values = True


# ── Column order for DataFrame / Excel output ──────────────────────────────
SCHEMA_COLUMNS = [
    # Core 23 fields (plan spec)
    "scrape_date",
    "reit",
    "community",
    "address",
    "market",
    "unit_id",
    "beds",
    "baths",
    "sqft",
    "rent",
    "move_in_date",
    "lease_term",
    "has_concession",
    "concession_hardness",
    "concession_raw",
    "concession_type",
    "concession_value",
    "concession_pct_lease_value",
    "concession_pct_lease_term",
    "effective_monthly_rent",
    "listing_url",
    "first_seen",
    "last_seen",
    # Supplemental fields (equity research extras)
    "state",
    "city",
    "latitude",
    "longitude",
    "floorplan_name",
    "floor_level",
    "rentcafe_property_id",
]
