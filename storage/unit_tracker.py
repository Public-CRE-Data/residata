"""
Unit tracker — maintains longitudinal first_seen / last_seen for each unit.

The tracker holds a CSV registry at data/registry/unit_registry.csv.
On each scrape run:
  1. Load existing registry (or start fresh).
  2. For every unit in today's scrape:
       - If unit_id already in registry → update last_seen.
       - If unit_id is new → add row with first_seen = last_seen = today.
  3. Units that were in the registry but NOT in today's scrape are left
     unchanged (last_seen stays at the previous scrape date — signals de-listing).
  4. Merge registry first_seen / last_seen back into the scraped DataFrame
     so the output CSV reflects accurate longitudinal dates.
  5. Save updated registry.

Unit identity key: unit_id (synthetic composite key set by scraper).
"""

import logging
import os
from datetime import date

import pandas as pd

logger = logging.getLogger(__name__)

_REGISTRY_COLS = ["unit_id", "first_seen", "last_seen", "reit", "community"]


def _registry_path(data_dir: str) -> str:
    return os.path.join(data_dir, "registry", "unit_registry.csv")


def load_registry(data_dir: str) -> pd.DataFrame:
    """Load the unit registry CSV, or return an empty DataFrame."""
    path = _registry_path(data_dir)
    if os.path.exists(path):
        df = pd.read_csv(path, parse_dates=["first_seen", "last_seen"])
        # Ensure date columns are date (not datetime)
        for col in ("first_seen", "last_seen"):
            df[col] = pd.to_datetime(df[col]).dt.date
        logger.info(f"Loaded registry: {len(df):,} units from {path}")
        return df
    logger.info("No existing registry — starting fresh.")
    return pd.DataFrame(columns=_REGISTRY_COLS)


def save_registry(registry: pd.DataFrame, data_dir: str) -> str:
    """Save the unit registry CSV."""
    path = _registry_path(data_dir)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    registry.to_csv(path, index=False)
    logger.info(f"Saved registry: {len(registry):,} units → {path}")
    return path


def update_registry(
    registry: pd.DataFrame,
    scraped: pd.DataFrame,
    scrape_date: date,
) -> pd.DataFrame:
    """
    Merge today's scrape into the registry.
    Returns the updated registry DataFrame.
    """
    today = scrape_date

    if registry.empty:
        # First ever run — seed registry from scraped data
        new_reg = scraped[["unit_id", "reit", "community"]].drop_duplicates("unit_id").copy()
        new_reg["first_seen"] = today
        new_reg["last_seen"] = today
        logger.info(f"First run: registered {len(new_reg):,} new units.")
        return new_reg[_REGISTRY_COLS]

    # Scope "gone" detection to only the REIT(s) present in this scrape.
    # Units belonging to other REITs are untouched — they weren't scraped today.
    scraped_reits = set(scraped["reit"].dropna().unique())
    reg_in_scope  = registry[registry["reit"].isin(scraped_reits)]
    existing_ids  = set(reg_in_scope["unit_id"])
    scraped_ids   = set(scraped["unit_id"].dropna())

    # Units seen today that are already in registry → bump last_seen
    returning = scraped_ids & existing_ids
    registry.loc[registry["unit_id"].isin(returning), "last_seen"] = today

    # New units not yet in registry
    new_ids = scraped_ids - existing_ids
    if new_ids:
        new_rows = (
            scraped[scraped["unit_id"].isin(new_ids)][["unit_id", "reit", "community"]]
            .drop_duplicates("unit_id")
            .copy()
        )
        new_rows["first_seen"] = today
        new_rows["last_seen"] = today
        registry = pd.concat([registry, new_rows[_REGISTRY_COLS]], ignore_index=True)

    # Units in-scope (same REIT) but not seen today → likely de-listed
    gone = existing_ids - scraped_ids
    if gone:
        logger.info(f"  {len(gone):,} units not seen today (de-listed or no availability).")

    logger.info(
        f"Registry update ({', '.join(sorted(scraped_reits))}): "
        f"{len(returning):,} returning | "
        f"{len(new_ids):,} new | "
        f"{len(gone):,} gone | "
        f"total registry {len(registry):,}"
    )
    return registry


def apply_registry_dates(
    scraped: pd.DataFrame,
    registry: pd.DataFrame,
) -> pd.DataFrame:
    """
    Overwrite first_seen / last_seen in the scraped DataFrame with values
    from the authoritative registry. Units with no registry entry keep the
    scrape-date values set by the scraper.
    """
    if registry.empty:
        return scraped

    reg_index = registry.set_index("unit_id")[["first_seen", "last_seen"]]
    scraped = scraped.copy()
    scraped["first_seen"] = scraped["unit_id"].map(reg_index["first_seen"]).fillna(scraped["first_seen"])
    scraped["last_seen"]  = scraped["unit_id"].map(reg_index["last_seen"]).fillna(scraped["last_seen"])
    # Convert back to date objects (map may return mixed types)
    for col in ("first_seen", "last_seen"):
        scraped[col] = pd.to_datetime(scraped[col]).dt.date
    return scraped


def run_tracker(
    scraped: pd.DataFrame,
    data_dir: str,
    scrape_date: date | None = None,
) -> pd.DataFrame:
    """
    Full tracker pipeline:
      1. Load registry
      2. Update with today's scrape
      3. Save updated registry
      4. Apply accurate first_seen / last_seen to scraped DataFrame
      5. Return the updated scraped DataFrame

    This is the single entry point called from main.py.
    """
    if scrape_date is None:
        scrape_date = date.today()

    registry = load_registry(data_dir)
    registry = update_registry(registry, scraped, scrape_date)
    save_registry(registry, data_dir)
    scraped  = apply_registry_dates(scraped, registry)
    return scraped
