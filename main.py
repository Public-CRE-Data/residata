"""
REIT Rental Scraper — main entry point.

Usage:
  py main.py                      # scrape all configured REITs
  py main.py --reit MAA           # scrape one REIT
  py main.py --reit MAA --limit 5 # test: first 5 communities only
  py main.py --no-save            # run without writing files

Output:
  data/raw/maa_raw_YYYY-MM-DD.csv   — per-REIT raw CSVs
  data/registry/unit_registry.csv   — longitudinal unit tracker
"""

import argparse
import logging
import os
import sys
from datetime import date

import pandas as pd

from config import BASE_DIR, DATA_DIR, RAW_DIR, REITS, CUMULATIVE_EXCEL
from storage.unit_tracker import load_registry, run_tracker
from storage.excel_writer import write_excel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Scraper registry ───────────────────────────────────────────────────────────
# Maps REIT ticker → scrape function.
# Phase 2 scrapers will be added here as they are built.

def _load_scrapers() -> dict:
    scrapers = {}
    try:
        from scrapers.maa import scrape_maa
        scrapers["MAA"] = scrape_maa
    except ImportError as e:
        logger.warning(f"Could not load MAA scraper: {e}")
    try:
        from scrapers.cpt import scrape_cpt
        scrapers["CPT"] = scrape_cpt
    except ImportError as e:
        logger.warning(f"Could not load CPT scraper: {e}")
    try:
        from scrapers.eqr import scrape_eqr
        scrapers["EQR"] = scrape_eqr
    except ImportError as e:
        logger.warning(f"Could not load EQR scraper: {e}")
    try:
        from scrapers.avb import scrape_avb
        scrapers["AVB"] = scrape_avb
    except ImportError as e:
        logger.warning(f"Could not load AVB scraper: {e}")
    try:
        from scrapers.udr import scrape_udr
        scrapers["UDR"] = scrape_udr
    except ImportError as e:
        logger.warning(f"Could not load UDR scraper: {e}")
    try:
        from scrapers.ess import scrape_ess
        scrapers["ESS"] = scrape_ess
    except ImportError as e:
        logger.warning(f"Could not load ESS scraper: {e}")
    try:
        from scrapers.invh import scrape_invh
        scrapers["INVH"] = scrape_invh
    except ImportError as e:
        logger.warning(f"Could not load INVH scraper: {e}")
    # AMH (American Homes 4 Rent): BTR community REIT — does NOT publish
    # individual home rents publicly.  api.ah4rc.com/mapjson/ returns only
    # visual floor-plan map data (no prices).  AMH is not scrapeable.
    return scrapers


# ── Main ───────────────────────────────────────────────────────────────────────

def run(reit: str | None, limit: int | None, save: bool) -> pd.DataFrame:
    scrapers = _load_scrapers()
    targets = [reit.upper()] if reit else list(scrapers.keys())

    all_frames = []
    for ticker in targets:
        if ticker not in scrapers:
            logger.warning(f"No scraper available for {ticker} — skipping.")
            continue

        logger.info(f"{'='*60}")
        logger.info(f"  Scraping {ticker}")
        logger.info(f"{'='*60}")

        scrape_fn = scrapers[ticker]
        df = scrape_fn(limit=limit)

        if df.empty:
            logger.warning(f"{ticker}: 0 rows returned — skipping tracker/save.")
            continue

        # Run longitudinal tracker — updates first_seen / last_seen
        df = run_tracker(df, data_dir=DATA_DIR, scrape_date=date.today())

        logger.info(f"{ticker}: {len(df):,} rows after tracker merge.")

        if save:
            os.makedirs(RAW_DIR, exist_ok=True)
            out_path = os.path.join(RAW_DIR, f"{ticker.lower()}_raw_{date.today().isoformat()}.csv")
            df.to_csv(out_path, index=False)
            logger.info(f"  Saved → {out_path}")

        all_frames.append(df)

    if not all_frames:
        logger.error("No data collected.")
        return pd.DataFrame()

    combined = pd.concat(all_frames, ignore_index=True)
    logger.info(f"\nTotal rows across all REITs: {len(combined):,}")

    if save:
        registry_df = load_registry(DATA_DIR)
        write_excel(combined, registry_df, CUMULATIVE_EXCEL)

    return combined


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="REIT Rental Scraper")
    parser.add_argument("--reit",    type=str,  default=None,
                        help="Single REIT ticker to scrape (e.g. MAA)")
    parser.add_argument("--limit",   type=int,  default=None,
                        help="Limit to N communities per REIT (testing)")
    parser.add_argument("--no-save", action="store_true",
                        help="Skip writing output files")
    args = parser.parse_args()

    df = run(
        reit=args.reit,
        limit=args.limit,
        save=not args.no_save,
    )

    if not df.empty:
        print("\n--- Sample output (first 5 rows) ---")
        cols = ["scrape_date", "reit", "community", "market", "unit_id",
                "beds", "sqft", "rent", "concession_hardness",
                "concession_type", "concession_value", "effective_monthly_rent",
                "first_seen", "last_seen"]
        print(df[cols].head(5).to_string(index=False))
        print(f"\nShape: {df.shape}")
        if "has_concession" in df.columns:
            print(f"\nConcession rate: {df['has_concession'].mean():.1%}")
