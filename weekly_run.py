# weekly_run.py — End-to-end weekly REIT scrape + split + push + rebuild Excel
#
# Runs automatically via Windows Task Scheduler every Saturday at 11 PM.
# Also safe to run manually anytime: py weekly_run.py
#
# Pipeline:
#   0. Pre-flight: ensure Playwright browsers are installed
#   1. Run all scrapers (MAA, CPT, EQR, AVB, UDR, ESS, INVH, AMH)
#   2. Split any CSV > 4,000 rows or > 1.5 MB into _part1 / _part2
#   3. Git add + commit + push to GitHub
#   4. Rebuild Excel workbook via build_excel.py (pulls from GitHub)
#   5. Run week-over-week data quality checks (wow_qa.py)
#   6. Build equity research charts workbook (build_charts.py)
#
# Logs to: logs/weekly_YYYY-MM-DD.log

import os
import sys
import csv
import logging
import subprocess
from datetime import date
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "raw"
LOG_DIR = BASE_DIR / "logs"
OUTPUT_DIR = BASE_DIR / "output"

MAX_ROWS = 4000
MAX_BYTES = 1_500_000  # 1.5 MB

today = date.today().isoformat()

LOG_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_DIR / f"weekly_{today}.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


def preflight_checks():
    """Step 0: Ensure dependencies are ready (Playwright browsers, etc.)."""
    logger.info("=" * 60)
    logger.info("  STEP 0: Pre-flight checks")
    logger.info("=" * 60)

    # Ensure Playwright chromium is installed — this is a no-op if already
    # up to date, and auto-fixes after pip upgrades playwright.
    try:
        result = subprocess.run(
            [sys.executable, "-m", "playwright", "install", "chromium"],
            cwd=str(BASE_DIR),
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode == 0:
            logger.info("  Playwright chromium: OK")
        else:
            logger.warning(f"  Playwright install returned {result.returncode}: "
                           f"{result.stderr.strip()[:200]}")
    except Exception as e:
        logger.warning(f"  Playwright pre-flight failed: {e}")

    # Verify data directories exist
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)
    logger.info("  Directories: OK")


def run_scrapers():
    """Step 1: Run main.py to scrape all REITs."""
    logger.info("=" * 60)
    logger.info("  STEP 1: Running scrapers")
    logger.info("=" * 60)
    result = subprocess.run(
        [sys.executable, str(BASE_DIR / "main.py")],
        cwd=str(BASE_DIR),
        capture_output=False,
        timeout=43200,  # 12-hour timeout
    )
    if result.returncode != 0:
        logger.error(f"main.py exited with code {result.returncode}")
    else:
        logger.info("Scrapers completed successfully.")
    return result.returncode


def split_large_csvs():
    """Step 2: Split any CSV over MAX_ROWS or MAX_BYTES into _part1/_part2."""
    logger.info("=" * 60)
    logger.info("  STEP 2: Splitting large CSVs")
    logger.info("=" * 60)

    # Use glob for today AND tomorrow (scrape starts ~11PM, may cross midnight)
    from datetime import timedelta
    tomorrow = (date.today() + timedelta(days=1)).isoformat()
    candidates = sorted(set(
        list(RAW_DIR.glob(f"*_raw_{today}.csv")) +
        list(RAW_DIR.glob(f"*_raw_{tomorrow}.csv"))
    ))
    for csv_path in candidates:
        # Skip already-split files
        if "_part1" in csv_path.name or "_part2" in csv_path.name:
            continue

        file_size = csv_path.stat().st_size
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = list(reader)

        row_count = len(rows) - 1  # exclude header
        needs_split = row_count > MAX_ROWS or file_size > MAX_BYTES

        if not needs_split:
            logger.info(f"  {csv_path.name}: {row_count:,} rows, {file_size/1e6:.1f} MB — OK")
            continue

        logger.info(f"  {csv_path.name}: {row_count:,} rows, {file_size/1e6:.1f} MB — SPLITTING")

        header = rows[0]
        data_rows = rows[1:]
        mid = len(data_rows) // 2

        base = csv_path.stem  # e.g. maa_raw_2026-04-04
        for suffix, chunk in [("_part1", data_rows[:mid]), ("_part2", data_rows[mid:])]:
            out_path = csv_path.parent / f"{base}{suffix}.csv"
            with open(out_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(header)
                writer.writerows(chunk)
            logger.info(f"    -> {out_path.name}: {len(chunk):,} rows")

        # Remove the original unsplit file
        csv_path.unlink()
        logger.info(f"    Removed original {csv_path.name}")


def git_push():
    """Step 3: Git add, commit, push new CSVs to GitHub."""
    logger.info("=" * 60)
    logger.info("  STEP 3: Pushing to GitHub")
    logger.info("=" * 60)

    def git(*args):
        result = subprocess.run(
            ["git"] + list(args),
            cwd=str(BASE_DIR),
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.stdout.strip():
            logger.info(f"  git {' '.join(args)}: {result.stdout.strip()[:200]}")
        if result.stderr.strip():
            logger.info(f"  stderr: {result.stderr.strip()[:200]}")
        return result

    git("add", "data/raw/", "data/registry/", "data/summary/")

    status = git("status", "--porcelain")
    if not status.stdout.strip():
        logger.info("  No changes to commit.")
        return

    git("commit", "-m", f"Weekly scrape {today}")
    result = git("push")
    if "-> main" in (result.stderr + result.stdout):
        logger.info("  Push successful.")
    else:
        logger.warning("  Push may have failed — check stderr above.")


def rebuild_excel():
    """Step 4: Rebuild the Excel workbook from GitHub data."""
    logger.info("=" * 60)
    logger.info("  STEP 4: Rebuilding Excel workbook")
    logger.info("=" * 60)
    result = subprocess.run(
        [sys.executable, str(BASE_DIR / "build_excel.py")],
        cwd=str(BASE_DIR),
        capture_output=False,
        timeout=600,
    )
    if result.returncode != 0:
        logger.error(f"build_excel.py exited with code {result.returncode}")
    else:
        logger.info("Excel workbook rebuilt successfully.")


def run_wow_qa():
    """Step 5: Run WoW data quality checks and log any flags."""
    logger.info("=" * 60)
    logger.info("  STEP 5: Week-over-Week Data Quality Check")
    logger.info("=" * 60)
    try:
        result = subprocess.run(
            [sys.executable, str(BASE_DIR / "wow_qa.py")],
            cwd=str(BASE_DIR),
            capture_output=True,
            text=True,
            timeout=300,
        )
        # Mirror QA stdout to pipeline log
        for line in result.stdout.splitlines():
            logger.info(f"  {line}")
        if result.stderr.strip():
            logger.warning(f"  QA stderr: {result.stderr.strip()[:500]}")

        # Parse flags from output for a one-line summary
        out = result.stdout
        fails = out.count("[FAIL]") - (1 if "[FAIL] (" in out else 0)
        warns = out.count("[WARN]") - (1 if "[WARN] (" in out else 0)
        import re as _re
        fm = _re.search(r"\[FAIL\] \((\d+)\)", out)
        wm = _re.search(r"\[WARN\] \((\d+)\)", out)
        n_fail = int(fm.group(1)) if fm else 0
        n_warn = int(wm.group(1)) if wm else 0
        if n_fail:
            logger.warning(f"  QA: {n_fail} FAIL, {n_warn} WARN — review logs/wow_qa_*.md")
        elif n_warn:
            logger.info(f"  QA: 0 FAIL, {n_warn} WARN — review logs/wow_qa_*.md")
        else:
            logger.info(f"  QA: all checks passed.")
    except Exception as e:
        logger.error(f"  WoW QA failed to run: {type(e).__name__}: {e}")


def build_research_charts():
    """Step 6: Rebuild the research charts workbook for publication."""
    logger.info("=" * 60)
    logger.info("  STEP 6: Building research charts workbook")
    logger.info("=" * 60)
    try:
        result = subprocess.run(
            [sys.executable, str(BASE_DIR / "build_charts.py")],
            cwd=str(BASE_DIR),
            capture_output=True,
            text=True,
            timeout=600,
        )
        for line in result.stdout.splitlines():
            logger.info(f"  {line}")
        if result.stderr.strip():
            logger.warning(f"  build_charts stderr: {result.stderr.strip()[:500]}")
        if result.returncode != 0:
            logger.error(f"  build_charts.py exited with code {result.returncode}")
        else:
            logger.info("  Research charts built successfully.")
    except Exception as e:
        logger.error(f"  build_charts.py failed to run: {type(e).__name__}: {e}")


def main():
    logger.info(f"Weekly REIT pipeline started — {today}")
    logger.info(f"Base directory: {BASE_DIR}")
    logger.info("")

    # Step 0: Pre-flight
    preflight_checks()

    # Step 1: Scrape
    rc = run_scrapers()
    if rc != 0:
        logger.warning("Scrapers had errors but continuing with available data...")

    # Step 2: Split large CSVs
    split_large_csvs()

    # Step 3: Push to GitHub
    git_push()

    # Step 4: Rebuild Excel
    rebuild_excel()

    # Step 5: Week-over-week data quality checks
    run_wow_qa()

    # Step 6: Research charts workbook
    build_research_charts()

    logger.info("")
    logger.info("=" * 60)
    logger.info(f"  PIPELINE COMPLETE — {today}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
