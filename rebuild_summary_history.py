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

    # ── ESS / UDR week-1 nulling (AFTER NER coalesce) ────────────────
    # Must run after coalesce; otherwise these rows would be coalesced
    # back to NER=gross_rent. See build_excel.build_panel for rationale.
    earliest = df["scrape_date"].min() if not df.empty else None
    if earliest is not None:
        for reit_name in ("ESS", "UDR"):
            mask = (df["reit"] == reit_name) & (df["scrape_date"] == earliest)
            if not mask.any():
                continue
            df["has_concession"] = df["has_concession"].astype("object")
            for c in ["has_concession", "concession_hardness", "concession_raw",
                      "concession_type", "concession_value",
                      "concession_pct_lease_value", "concession_pct_lease_term",
                      "effective_monthly_rent"]:
                if c in df.columns:
                    df.loc[mask, c] = None
            print(f"  [FIX] Nulled {reit_name} first-period concession+NER "
                  f"({earliest.date()}, {int(mask.sum()):,} rows).")

    # ── Scraper coverage gap detection ────────────────────────────────
    # If a REIT's FIRST-week scrape missed a macro_market that appears
    # with >10% of portfolio weight in week 2, the same-property
    # intersection for that pair is composition-biased (the matched
    # pool lacks the under-covered market). Flag those REIT-date pairs
    # so downstream can null their sp_* values.
    print("  [COVERAGE] Checking first-week market coverage per REIT...")
    # Coverage-gap detection generalized across ALL pair-week combinations.
    # For each (REIT, pair = N-1 -> N): if either side has communities missing
    # from the other side that account for >10% of either week's communities,
    # the matched-pool composition is biased and that pair's sp_* values
    # should be nulled for that REIT.
    #
    # Two flavors of gap:
    #   missing_in_curr = communities in PREV but not in CURR  (under-rep)
    #   new_in_curr     = communities in CURR but not in PREV  (over-rep)
    # Either ≥10% triggers exclusion.
    coverage_gaps = {}    # (reit, curr_date) -> list of gap descriptions
    weeks = sorted(df["scrape_date"].unique())
    THRESHOLD = 0.10
    for i in range(1, len(weeks)):
        prev_wk, curr_wk = weeks[i - 1], weeks[i]
        for reit in df["reit"].unique():
            p_comms = set(df[(df["reit"] == reit) & (df["scrape_date"] == prev_wk)]
                          ["community"].dropna())
            c_comms = set(df[(df["reit"] == reit) & (df["scrape_date"] == curr_wk)]
                          ["community"].dropna())
            if len(p_comms) < 30 or len(c_comms) < 30:
                continue
            missing = p_comms - c_comms
            new_in = c_comms - p_comms
            miss_pct = len(missing) / max(len(p_comms), 1)
            new_pct = len(new_in) / max(len(c_comms), 1)
            if miss_pct >= THRESHOLD or new_pct >= THRESHOLD:
                key = (reit, curr_wk)
                gap_msg = (f"missing {len(missing)}/{len(p_comms)} ({miss_pct:.0%}), "
                           f"new {len(new_in)}/{len(c_comms)} ({new_pct:.0%})")
                coverage_gaps[key] = gap_msg

    if coverage_gaps:
        # Group flags by REIT for readable output
        from collections import defaultdict
        by_reit = defaultdict(list)
        for (reit, wk), msg in coverage_gaps.items():
            by_reit[reit].append((wk, msg))
        for reit in sorted(by_reit):
            for wk, msg in sorted(by_reit[reit]):
                print(f"    [GAP] {reit} pair ending {wk.date()}: {msg}")

    df.attrs["_coverage_gaps"] = coverage_gaps
    return df


def _safe_div(a, b):
    if b is None or b == 0 or pd.isna(a) or pd.isna(b):
        return None
    return a / b


def _compute_reit_sp(panel, reit, prev_date, curr_date):
    """Compute SP aggregates for one (REIT, prev_date, curr_date) triple.
    Returns a DataFrame with one row per (reit, macro_market, beds) bucket,
    containing all sp_* columns. Cohort = unit_ids in BOTH dates for this
    REIT. Used to support cross-week bridging when prev_date is not the
    immediate predecessor of curr_date."""
    prev = panel[(panel["reit"] == reit) & (panel["scrape_date"] == prev_date)].copy()
    curr = panel[(panel["reit"] == reit) & (panel["scrape_date"] == curr_date)].copy()
    if prev.empty or curr.empty:
        return None

    for d in (prev, curr):
        d["_rent_psf"] = d.apply(
            lambda r: r["rent"] / r["sqft"]
            if pd.notna(r["sqft"]) and r["sqft"] > 0 and pd.notna(r["rent"]) else None,
            axis=1)
        d["_eff_rent_psf"] = d.apply(
            lambda r: r["effective_monthly_rent"] / r["sqft"]
            if pd.notna(r.get("effective_monthly_rent"))
               and pd.notna(r["sqft"]) and r["sqft"] > 0 else None,
            axis=1)

    sp_ids = set(prev["unit_id"].dropna()) & set(curr["unit_id"].dropna())
    prev_sp = prev[prev["unit_id"].isin(sp_ids)].copy()
    curr_sp = curr[curr["unit_id"].isin(sp_ids)].copy()
    if prev_sp.empty or curr_sp.empty:
        return None

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
    if sp.empty:
        return None
    sp["sp_count"] = sp["sp_count_curr"]
    sp["sp_wow_pct"] = (sp["sp_avg_rent_curr"] - sp["sp_avg_rent_prev"]) / sp["sp_avg_rent_prev"]
    sp["sp_wow_pct_psf"] = sp.apply(
        lambda r: _safe_div(r["sp_avg_rent_psf_curr"] - r["sp_avg_rent_psf_prev"],
                            r["sp_avg_rent_psf_prev"]), axis=1)
    sp["sp_wow_pct_eff"] = sp.apply(
        lambda r: _safe_div(r["sp_avg_eff_rent_curr"] - r["sp_avg_eff_rent_prev"],
                            r["sp_avg_eff_rent_prev"]), axis=1)
    sp["sp_wow_pct_eff_psf"] = sp.apply(
        lambda r: _safe_div(r["sp_avg_eff_rent_psf_curr"] - r["sp_avg_eff_rent_psf_prev"],
                            r["sp_avg_eff_rent_psf_prev"]), axis=1)
    return sp


def compute_history(panel: pd.DataFrame) -> pd.DataFrame:
    """Build a complete week-by-week same-property history.

    Cross-week bridging: when a REIT has a community-coverage gap with
    its immediate prior week, we look further back for the most recent
    "clean" prior week (one without a gap to the current week) and use
    THAT as the SP comparison anchor. The chain-link factor still works
    because the chain just multiplies wow_factors; spacing between
    factors can be irregular.

    Example: MAA had 19% gap from 3-28 to 4-4 and 14% gap from 4-4 to
    4-11. Instead of nulling MAA at both 4-4 and 4-11, we:
      - Null MAA at 4-4 (no clean prior to compare with)
      - Compute MAA at 4-11 by matching units against 3-28 directly
        (3-28 to 4-11 had only ~6% community gap — passes threshold)
    """
    dates = sorted(panel["scrape_date"].dropna().unique())
    print(f"  Distinct weeks: {len(dates)}  ({[d.date().isoformat() for d in dates]})")

    coverage_gaps = panel.attrs.get("_coverage_gaps", {})
    GAP_THRESHOLD = 0.10

    def _has_gap(reit, prev_d, curr_d, panel_df):
        """Check if (reit, prev_d -> curr_d) has community coverage gap."""
        p_comms = set(panel_df[(panel_df["reit"] == reit)
                                & (panel_df["scrape_date"] == prev_d)]["community"].dropna())
        c_comms = set(panel_df[(panel_df["reit"] == reit)
                                & (panel_df["scrape_date"] == curr_d)]["community"].dropna())
        if len(p_comms) < 30 or len(c_comms) < 30:
            return True   # not enough data to be confident
        miss_pct = len(p_comms - c_comms) / max(len(p_comms), 1)
        new_pct = len(c_comms - p_comms) / max(len(c_comms), 1)
        return miss_pct >= GAP_THRESHOLD or new_pct >= GAP_THRESHOLD

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

        # ── SP aggregates (per-REIT prev-period selection w/ bridging) ──
        if i > 0:
            # For each REIT, find the appropriate prev_date — by default
            # the immediate prior, but bridge over gapped weeks.
            sp_per_reit_chunks = []
            reits_in_curr = curr["reit"].unique()
            for reit in reits_in_curr:
                # Walk backwards to find most recent non-gap prior week
                prev_date = None
                for j in range(i - 1, -1, -1):
                    candidate = dates[j]
                    if not _has_gap(reit, candidate, curr_date, panel):
                        prev_date = candidate
                        break
                if prev_date is None:
                    continue   # no clean prior — skip SP for this REIT
                if prev_date != dates[i - 1]:
                    print(f"    [BRIDGE] {reit} {curr_date.date()}: bridging to {prev_date.date()} "
                          f"(skipped {(curr_date - prev_date).days // 7 - 1} gap week(s))")
                # Compute SP for this REIT alone using prev_date
                reit_sp = _compute_reit_sp(panel, reit, prev_date, curr_date)
                if reit_sp is not None and not reit_sp.empty:
                    sp_per_reit_chunks.append(reit_sp)

            sp = pd.concat(sp_per_reit_chunks, ignore_index=True) if sp_per_reit_chunks else pd.DataFrame()
            keys = ["reit", "macro_market", "beds"]
            if not sp.empty:
                merged = nonsp.merge(sp, on=keys, how="left")
            else:
                merged = nonsp.copy()
                # Fill SP columns with NaN
                for c in ["sp_count", "sp_avg_rent_curr", "sp_avg_rent_prev", "sp_wow_pct",
                          "sp_concession_rate_curr", "sp_concession_rate_prev",
                          "sp_avg_rent_psf_curr", "sp_avg_rent_psf_prev", "sp_wow_pct_psf",
                          "sp_avg_eff_rent_curr", "sp_avg_eff_rent_prev", "sp_wow_pct_eff",
                          "sp_avg_eff_rent_psf_curr", "sp_avg_eff_rent_psf_prev",
                          "sp_wow_pct_eff_psf"]:
                    merged[c] = None

            all_rows.append(merged)
            continue   # skip the legacy single-prev path below

        # Legacy single-prev path (i == 0 only — first week, no SP)
        if False:   # never executes; kept to preserve indentation below
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

    # Note: coverage-gap handling is now done INLINE in compute_history
    # via cross-week bridging — for each (reit, curr_week) we walk back
    # to the most recent clean prior week. If no clean prior exists, the
    # SP merge yields no rows for that (reit, week) so the merged row
    # naturally has NaN SP columns. No post-hoc nulling needed.

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
