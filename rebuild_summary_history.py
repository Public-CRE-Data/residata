"""
Rebuild data/summary/summary_history.csv from scratch using ALL raw CSVs.

Why this exists:
  The weekly pipeline APPENDS to summary_history each run, but only for the
  current week. If a REIT or a week was missed at first-run time (AMH/INVH
  joined late, March 28 week was never captured), that data is permanently
  absent from the index charts.

This script scans every CSV in data/raw/, applies the same deduplication
and fixes as build_excel.py (AMH deposit-offer false positive, ESS/UDR
week-1 scraper bugs), then produces a complete week-by-week same-property
history across every REIT.

Usage:
  py rebuild_summary_history.py          # writes to data/summary/
  py rebuild_summary_history.py --dry    # just prints coverage, no write
"""

import argparse
import re
import pathlib
import warnings
from datetime import timedelta

import pandas as pd

warnings.filterwarnings("ignore")

BASE_DIR = pathlib.Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "raw"
SUMMARY_PATH = BASE_DIR / "data" / "summary" / "summary_history.csv"

# Reuse build_excel's macro_market resolver by importing it
import sys
sys.path.insert(0, str(BASE_DIR))
from build_excel import _resolve_macro_market


def saturday_anchor(d):
    """Map a scrape date to the Saturday on or before (week anchor)."""
    return d - pd.Timedelta(days=(d.weekday() - 5) % 7)


def load_all_raw() -> pd.DataFrame:
    """Load every CSV in data/raw/, dedupe on (reit, unit_id, week_anchor)."""
    files = sorted(RAW_DIR.glob("*_raw_*.csv"))
    parts = []
    for f in files:
        try:
            parts.append(pd.read_csv(f, low_memory=False))
        except Exception as e:
            print(f"  [skip] {f.name}: {e}")
    df = pd.concat(parts, ignore_index=True)
    df["scrape_date"] = pd.to_datetime(df["scrape_date"])
    df["week"] = df["scrape_date"].apply(saturday_anchor)
    # Dedupe: keep last observation per (reit, unit_id, week)
    df = (df.sort_values("scrape_date")
            .groupby(["reit", "unit_id", "week"], as_index=False)
            .last())
    df["scrape_date"] = df["week"]
    return df


def apply_fixes(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the same known-issue mitigations as build_excel.py."""
    df = df.copy()

    # Re-parse concession_raw with the current parse_concession (ensures
    # consistent classification — same logic build_excel uses at load time).
    try:
        from scrapers.maa import parse_concession
        conc_cols = ["has_concession", "concession_hardness", "concession_type",
                     "concession_value", "concession_pct_lease_value",
                     "concession_pct_lease_term", "effective_monthly_rent"]
        mask = df["concession_raw"].notna() & (df["concession_raw"] != "")
        reparsed = 0
        for idx in df.loc[mask].index:
            raw = df.at[idx, "concession_raw"]
            rent = df.at[idx, "rent"]
            if pd.isna(rent) or rent <= 0:
                continue
            parsed = parse_concession(raw, float(rent))
            for c in conc_cols:
                df.at[idx, c] = parsed[c]
            reparsed += 1
        # No-concession-raw rows → ensure defaults
        no_mask = df["concession_raw"].isna() | (df["concession_raw"] == "")
        df.loc[no_mask, "has_concession"] = False
        for c in conc_cols[1:]:
            df.loc[no_mask, c] = None
        print(f"  [RE-PARSE] Re-parsed {reparsed:,} concession_raw values.")
    except ImportError:
        print("  [WARN] parse_concession unavailable — using CSV values as-is.")

    # ESS week 1: scraper bug nulls on earliest period
    ess_first = df[df["reit"] == "ESS"]["scrape_date"].min() if (df["reit"] == "ESS").any() else None
    if ess_first is not None:
        mask = (df["reit"] == "ESS") & (df["scrape_date"] == ess_first)
        if mask.any():
            df["has_concession"] = df["has_concession"].astype("object")
            for c in ["has_concession", "concession_hardness", "concession_raw",
                      "concession_type", "concession_value",
                      "concession_pct_lease_value", "concession_pct_lease_term",
                      "effective_monthly_rent"]:
                if c in df.columns:
                    df.loc[mask, c] = None
            print(f"  [FIX] Nulled ESS first-period concessions ({ess_first.date()}, {int(mask.sum()):,} rows).")

    # UDR week 1: deposit-text overwrite on earliest period
    udr_first = df[df["reit"] == "UDR"]["scrape_date"].min() if (df["reit"] == "UDR").any() else None
    if udr_first is not None:
        mask = (df["reit"] == "UDR") & (df["scrape_date"] == udr_first)
        if mask.any():
            df["has_concession"] = df["has_concession"].astype("object")
            for c in ["has_concession", "concession_hardness", "concession_raw",
                      "concession_type", "concession_value",
                      "concession_pct_lease_value", "concession_pct_lease_term",
                      "effective_monthly_rent"]:
                if c in df.columns:
                    df.loc[mask, c] = None
            print(f"  [FIX] Nulled UDR first-period concessions ({udr_first.date()}, {int(mask.sum()):,} rows).")

    # AMH bare-percent deposit-offer false positive (all periods)
    bare_re = re.compile(r"^\s*\d+\s*%\s*off\s*$", re.I)
    amh_mask = ((df["reit"] == "AMH")
                & df["concession_raw"].fillna("").astype(str).str.match(bare_re))
    if amh_mask.any():
        df["has_concession"] = df["has_concession"].astype("object")
        for c in ["has_concession", "concession_hardness", "concession_raw",
                  "concession_type", "concession_value",
                  "concession_pct_lease_value", "concession_pct_lease_term",
                  "effective_monthly_rent"]:
            if c in df.columns:
                df.loc[amh_mask, c] = None
        print(f"  [FIX] Nulled {int(amh_mask.sum()):,} AMH bare-percent deposit FPs.")

    # Attach macro_market
    df["macro_market"] = df["market"].apply(_resolve_macro_market)

    # has_concession → bool (post-nulling it was object)
    df["has_concession"] = df["has_concession"].fillna(False).astype(bool)

    # NER coalesce — see build_excel.build_panel for full rationale.
    # Cases: (a) no concession, NER missing; (b) soft concession (has_concession=True
    # but concession_value=None, e.g. "Check out current specials" banner).
    # Both → NER = gross rent.
    if "effective_monthly_rent" in df.columns and "rent" in df.columns:
        has_conc = df["has_concession"].fillna(False).astype(bool)
        missing_ner = df["effective_monthly_rent"].isna()
        has_rent = df["rent"].notna() & (df["rent"] > 0)
        no_value = df["concession_value"].isna() if "concession_value" in df.columns else True
        fill = ((~has_conc) | (has_conc & no_value)) & missing_ner & has_rent
        n = int(fill.sum())
        if n:
            df.loc[fill, "effective_monthly_rent"] = df.loc[fill, "rent"]
            print(f"  [NER] Coalesced {n:,} no-NER rows to NER=gross_rent "
                  f"(no concession or soft/unparseable concession).")

    # ── Scraper coverage gap detection ────────────────────────────────
    # If a REIT's FIRST-week scrape missed a macro_market that appears
    # with >10% of portfolio weight in week 2, the same-property
    # intersection for that pair is composition-biased (the matched
    # pool lacks the under-covered market). Flag those REIT-date pairs
    # so downstream can null their sp_* values.
    print("  [COVERAGE] Checking first-week market coverage per REIT...")
    coverage_gaps = {}
    weeks = sorted(df["scrape_date"].unique())
    if len(weeks) >= 2:
        w0, w1 = weeks[0], weeks[1]
        for reit in df["reit"].unique():
            w0_mkts = df[(df["reit"] == reit) & (df["scrape_date"] == w0)]
            w1_mkts = df[(df["reit"] == reit) & (df["scrape_date"] == w1)]
            if len(w0_mkts) == 0 or len(w1_mkts) == 0:
                continue
            w0_mkt_counts = w0_mkts["macro_market"].value_counts()
            w1_mkt_counts = w1_mkts["macro_market"].value_counts()
            w1_total = w1_mkt_counts.sum()
            # Find markets in week 2 that are missing or <10% of week 2 presence in week 1
            for mkt, w1_n in w1_mkt_counts.items():
                w0_n = w0_mkt_counts.get(mkt, 0)
                if w1_n >= 50 and (w0_n / max(w1_n, 1)) < 0.10:
                    coverage_gaps.setdefault(reit, []).append(
                        (mkt, int(w0_n), int(w1_n))
                    )
    for reit, gaps in coverage_gaps.items():
        gap_desc = ", ".join(f"{m} ({a}->{b})" for m, a, b in gaps)
        print(f"    [FLAG] {reit} first-week ({w0.date()}) coverage gaps: {gap_desc}")

    # Stash for use in compute_history
    df.attrs["_coverage_gaps"] = coverage_gaps
    return df


def _safe_div(a, b):
    if b is None or b == 0 or pd.isna(a) or pd.isna(b):
        return None
    return a / b


def compute_history(panel: pd.DataFrame) -> pd.DataFrame:
    """Build a complete week-by-week same-property history."""
    dates = sorted(panel["scrape_date"].dropna().unique())
    print(f"  Distinct weeks: {len(dates)}  ({[d.date().isoformat() for d in dates]})")

    all_rows = []

    for i, curr_date in enumerate(dates):
        curr = panel[panel["scrape_date"] == curr_date].copy()

        # unit-level derived columns
        curr["_rent_psf"] = curr.apply(
            lambda r: r["rent"] / r["sqft"] if pd.notna(r["sqft"]) and r["sqft"] > 0 and pd.notna(r["rent"]) else None,
            axis=1)
        curr["_eff_rent_psf"] = curr.apply(
            lambda r: r["effective_monthly_rent"] / r["sqft"]
            if pd.notna(r.get("effective_monthly_rent")) and pd.notna(r["sqft"]) and r["sqft"] > 0 else None,
            axis=1)

        # ── Non-SP aggregates (always computable) ─────────────────────
        nonsp = curr.groupby(["reit", "macro_market", "beds"], dropna=False).agg(
            listing_count=("unit_id", "count"),
            avg_rent=("rent", "mean"),
            median_rent=("rent", "median"),
            avg_sqft=("sqft", "mean"),
            rent_per_sqft=("_rent_psf", "mean"),
            concession_rate=("has_concession", "mean"),
            avg_concession_value=("concession_value", "mean"),
            avg_rent_psf=("_rent_psf", "mean"),
            median_rent_psf=("_rent_psf", "median"),
            avg_eff_rent=("effective_monthly_rent", "mean"),
            avg_eff_rent_psf=("_eff_rent_psf", "mean"),
        ).reset_index()
        nonsp["scrape_date"] = curr_date

        # ── SP aggregates (need prior period) ────────────────────────
        if i > 0:
            prev_date = dates[i - 1]
            prev = panel[panel["scrape_date"] == prev_date].copy()
            prev["_rent_psf"] = prev.apply(
                lambda r: r["rent"] / r["sqft"] if pd.notna(r["sqft"]) and r["sqft"] > 0 and pd.notna(r["rent"]) else None,
                axis=1)
            prev["_eff_rent_psf"] = prev.apply(
                lambda r: r["effective_monthly_rent"] / r["sqft"]
                if pd.notna(r.get("effective_monthly_rent")) and pd.notna(r["sqft"]) and r["sqft"] > 0 else None,
                axis=1)

            sp_ids = set(prev["unit_id"].dropna()) & set(curr["unit_id"].dropna())
            prev_sp = prev[prev["unit_id"].isin(sp_ids)].copy()
            curr_sp = curr[curr["unit_id"].isin(sp_ids)].copy()

            # NER treatment: NER is already coalesced with gross rent for
            # no-concession units (in apply_fixes), so every matched unit
            # contributes to the NER aggregate. Concession flips register
            # as real NER moves.
            prev_sp["_eff_matched"] = prev_sp["effective_monthly_rent"]
            prev_sp["_eff_psf_matched"] = prev_sp["_eff_rent_psf"]
            curr_sp["_eff_matched"] = curr_sp["effective_monthly_rent"]
            curr_sp["_eff_psf_matched"] = curr_sp["_eff_rent_psf"]

            keys = ["reit", "macro_market", "beds"]
            prev_grp = prev_sp.groupby(keys, dropna=False).agg(
                sp_avg_rent_prev=("rent", "mean"),
                sp_concession_rate_prev=("has_concession", "mean"),
                sp_count_prev=("unit_id", "count"),
                sp_avg_rent_psf_prev=("_rent_psf", "mean"),
                sp_avg_eff_rent_prev=("_eff_matched", "mean"),
                sp_avg_eff_rent_psf_prev=("_eff_psf_matched", "mean"),
            ).reset_index()
            curr_grp = curr_sp.groupby(keys, dropna=False).agg(
                sp_avg_rent_curr=("rent", "mean"),
                sp_concession_rate_curr=("has_concession", "mean"),
                sp_count_curr=("unit_id", "count"),
                sp_avg_rent_psf_curr=("_rent_psf", "mean"),
                sp_avg_eff_rent_curr=("_eff_matched", "mean"),
                sp_avg_eff_rent_psf_curr=("_eff_psf_matched", "mean"),
            ).reset_index()
            sp = pd.merge(prev_grp, curr_grp, on=keys, how="inner")
            sp["sp_count"] = sp["sp_count_curr"]
            sp["sp_wow_pct"] = (sp["sp_avg_rent_curr"] - sp["sp_avg_rent_prev"]) / sp["sp_avg_rent_prev"]
            sp["sp_wow_pct_psf"] = sp.apply(
                lambda r: _safe_div(r["sp_avg_rent_psf_curr"] - r["sp_avg_rent_psf_prev"], r["sp_avg_rent_psf_prev"]), axis=1)
            sp["sp_wow_pct_eff"] = sp.apply(
                lambda r: _safe_div(r["sp_avg_eff_rent_curr"] - r["sp_avg_eff_rent_prev"], r["sp_avg_eff_rent_prev"]), axis=1)
            sp["sp_wow_pct_eff_psf"] = sp.apply(
                lambda r: _safe_div(r["sp_avg_eff_rent_psf_curr"] - r["sp_avg_eff_rent_psf_prev"], r["sp_avg_eff_rent_psf_prev"]), axis=1)

            merged = nonsp.merge(sp, on=keys, how="left")
        else:
            merged = nonsp.copy()
            # Fill SP columns with NaN for first period
            for c in ["sp_count", "sp_avg_rent_curr", "sp_avg_rent_prev", "sp_wow_pct",
                      "sp_concession_rate_curr", "sp_concession_rate_prev",
                      "sp_avg_rent_psf_curr", "sp_avg_rent_psf_prev", "sp_wow_pct_psf",
                      "sp_avg_eff_rent_curr", "sp_avg_eff_rent_prev", "sp_wow_pct_eff",
                      "sp_avg_eff_rent_psf_curr", "sp_avg_eff_rent_psf_prev", "sp_wow_pct_eff_psf"]:
                merged[c] = None

        all_rows.append(merged)

    hist = pd.concat(all_rows, ignore_index=True)

    # ── Null SP values for REIT-week pairs flagged for coverage gap ─
    # If week N had incomplete coverage for REIT X, then the SP pair
    # (N, N+1) is composition-biased. Null X's sp_* in week N+1.
    coverage_gaps = panel.attrs.get("_coverage_gaps", {})
    if coverage_gaps and len(dates) >= 2:
        second_week = dates[1]
        sp_null_cols = ["sp_count", "sp_avg_rent_curr", "sp_avg_rent_prev", "sp_wow_pct",
                        "sp_concession_rate_curr", "sp_concession_rate_prev",
                        "sp_avg_rent_psf_curr", "sp_avg_rent_psf_prev", "sp_wow_pct_psf",
                        "sp_avg_eff_rent_curr", "sp_avg_eff_rent_prev", "sp_wow_pct_eff",
                        "sp_avg_eff_rent_psf_curr", "sp_avg_eff_rent_psf_prev",
                        "sp_wow_pct_eff_psf"]
        for reit in coverage_gaps:
            mask = (hist["reit"] == reit) & (hist["scrape_date"] == second_week)
            n = int(mask.sum())
            if n:
                for c in sp_null_cols:
                    if c in hist.columns:
                        hist.loc[mask, c] = None
                print(f"  [FIX] Nulled {n} rows of {reit} SP metrics on {second_week.date()} "
                      f"(first-week coverage gap means N-1->N pair is biased).")

    # Column order matches existing summary_history.csv
    col_order = ["scrape_date", "reit", "macro_market", "beds", "listing_count",
                 "avg_rent", "median_rent", "avg_sqft", "rent_per_sqft",
                 "concession_rate", "avg_concession_value",
                 "sp_count", "sp_avg_rent_curr", "sp_avg_rent_prev", "sp_wow_pct",
                 "sp_concession_rate_curr", "sp_concession_rate_prev",
                 "avg_rent_psf", "median_rent_psf", "avg_eff_rent", "avg_eff_rent_psf",
                 "sp_avg_rent_psf_curr", "sp_avg_rent_psf_prev", "sp_wow_pct_psf",
                 "sp_avg_eff_rent_curr", "sp_avg_eff_rent_prev", "sp_wow_pct_eff",
                 "sp_avg_eff_rent_psf_curr", "sp_avg_eff_rent_psf_prev", "sp_wow_pct_eff_psf"]
    col_order = [c for c in col_order if c in hist.columns] + [c for c in hist.columns if c not in col_order]
    hist = hist[col_order]
    return hist


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry", action="store_true", help="Print coverage only, don't write")
    args = parser.parse_args()

    print("[Rebuild] Loading all raw CSVs...")
    panel = load_all_raw()
    print(f"[Rebuild] Loaded {len(panel):,} deduped rows across "
          f"{panel['scrape_date'].nunique()} weeks and {panel['reit'].nunique()} REITs.")

    print("[Rebuild] Applying data-quality fixes...")
    panel = apply_fixes(panel)

    print("[Rebuild] Computing week-by-week same-property history...")
    hist = compute_history(panel)

    print(f"[Rebuild] Output: {len(hist):,} rows across {hist['scrape_date'].nunique()} weeks.")
    print()
    # Coverage table
    cov = hist.groupby(["reit", "scrape_date"]).size().unstack(fill_value=0)
    print("Coverage (rows per REIT × week):")
    print(cov.to_string())
    print()

    # sp_count coverage
    sp_cov = hist.dropna(subset=["sp_avg_rent_curr"]).groupby(["reit", "scrape_date"]).size().unstack(fill_value=0)
    print("SP coverage (rows with sp_avg_rent_curr):")
    print(sp_cov.to_string())

    if args.dry:
        print("\n[Rebuild] DRY RUN — not writing.")
        return

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Serialize scrape_date as plain ISO date string
    hist_out = hist.copy()
    hist_out["scrape_date"] = pd.to_datetime(hist_out["scrape_date"]).dt.strftime("%Y-%m-%d")
    hist_out.to_csv(SUMMARY_PATH, index=False)
    print(f"\n[Rebuild] Wrote: {SUMMARY_PATH}  ({len(hist_out):,} rows)")


if __name__ == "__main__":
    main()
