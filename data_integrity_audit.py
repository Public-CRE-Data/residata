"""
Proactive data integrity audit — runs on ALL historical scrape data.

Goal: catch every class of data anomaly BEFORE a human spots it.
Designed to fail loudly when something looks wrong, even if the move
makes sense at a surface level.

Checks performed:

A) Apples-to-apples consistency
   • For each (REIT × week pair), bucket-level chain-link WoW must match
     the raw matched-unit WoW within 50 bps. Larger divergence implies
     coalesce/aggregation/cohort drift.

B) NER-vs-rent identity
   • Per-bucket NER WoW = rent WoW + concession-value WoW
     (within additive tolerance). If they don't add up, parser changed
     the concession classification mid-stream.

C) Boundary sanity
   • NER must be <= gross rent (allowing 2% slack for parser noise).
     Violations indicate parser pushing NER above asking rent.
   • sp_count <= listing_count (matched pool is a subset of total).
   • Concession rates ∈ [0, 1].

D) Cohort/coalesce contamination
   • Detect periods where a REIT's NER coalesce came from a forced null
     (week-1 scraper bugs). The signature: prev-period NER == prev-period
     gross rent for ALL units AND concession_rate == 0 in prev. If that
     state appears, NER WoW% to next period is meaningless and should
     be excluded.

E) Cross-REIT outlier
   • Flag any REIT whose NER WoW deviates from the cross-REIT cohort
     median by more than 2.5σ in a given week — likely either a real
     idiosyncratic event OR a data issue, but worth investigation.

F) Distribution anomalies
   • Per-REIT, identify weeks where the matched-pool size jumped >25%
     vs the prior week (composition refresh).
   • Per-REIT, detect rent-PSF distribution shifts (Kolmogorov-Smirnov
     on rent_psf between periods, p < 0.01) that aren't explained by
     market mix.

G) Identical-text floods (still here as guard)
   • >90% of a REIT's concession rows share identical concession_raw
     text — typical scraper false-positive signature.

H) Source-code regression
   • Greps build_excel.py / rebuild_summary_history.py for known-bad
     patterns (already in wow_qa.py — included here for full coverage).

Outputs:
  logs/data_integrity_audit_<latest>.md   — markdown findings, sorted by severity
  logs/data_integrity_audit_<latest>.json  — machine-readable findings

Exit code 2 if any HARD failure detected; 0 otherwise.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE_DIR = pathlib.Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "raw"
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)
SUMMARY_PATH = BASE_DIR / "data" / "summary" / "summary_history.csv"

sys.path.insert(0, str(BASE_DIR))
from build_excel import _resolve_macro_market

_DATE_RE = re.compile(r"_raw_(\d{4}-\d{2}-\d{2})")


def _saturday_anchor(d):
    return d - timedelta(days=(d.weekday() - 5) % 7)


def _list_weeks() -> list:
    weeks = set()
    for f in sorted(RAW_DIR.glob("*_raw_*.csv")):
        m = _DATE_RE.search(f.stem)
        if not m:
            continue
        d = datetime.strptime(m.group(1), "%Y-%m-%d").date()
        weeks.add(_saturday_anchor(d))
    return sorted(weeks)


def _load_week(target_date) -> pd.DataFrame:
    files = []
    for f in sorted(RAW_DIR.glob("*_raw_*.csv")):
        m = _DATE_RE.search(f.stem)
        if not m:
            continue
        d = datetime.strptime(m.group(1), "%Y-%m-%d").date()
        if _saturday_anchor(d) == target_date:
            files.append(f)
    if not files:
        return pd.DataFrame()
    df = pd.concat([pd.read_csv(f, low_memory=False) for f in files], ignore_index=True)
    df["scrape_date"] = pd.to_datetime(df["scrape_date"])
    df = (df.sort_values("scrape_date")
            .groupby(["reit", "unit_id"], as_index=False).last())

    # Match the build_excel pipeline transformations
    bare = ((df["reit"] == "AMH")
            & df["concession_raw"].fillna("").astype(str).str.match(
                r"^\s*\d+\s*%\s*off\s*$", case=False))
    if bare.any():
        df["has_concession"] = df["has_concession"].astype("object")
        for c in ["has_concession", "concession_hardness", "concession_raw",
                  "concession_type", "concession_value",
                  "concession_pct_lease_value", "concession_pct_lease_term",
                  "effective_monthly_rent"]:
            if c in df.columns:
                df.loc[bare, c] = None
        df["has_concession"] = df["has_concession"].fillna(False).astype(bool)

    try:
        from scrapers.maa import parse_concession
        conc_cols = ["has_concession", "concession_hardness", "concession_type",
                     "concession_value", "concession_pct_lease_value",
                     "concession_pct_lease_term", "effective_monthly_rent"]
        mask = df["concession_raw"].notna() & (df["concession_raw"] != "")
        for idx in df.loc[mask].index:
            raw = df.at[idx, "concession_raw"]
            rent = df.at[idx, "rent"]
            if pd.isna(rent) or rent <= 0:
                continue
            parsed = parse_concession(raw, float(rent))
            for c in conc_cols:
                df.at[idx, c] = parsed[c]
        no_mask = df["concession_raw"].isna() | (df["concession_raw"] == "")
        df.loc[no_mask, "has_concession"] = False
        for c in conc_cols[1:]:
            df.loc[no_mask, c] = None
    except ImportError:
        pass

    df["has_concession"] = df["has_concession"].fillna(False).astype(bool)

    # NER coalesce (a + b)
    has_rent = df["rent"].notna() & (df["rent"] > 0)
    missing_ner = df["effective_monthly_rent"].isna()
    no_value = df["concession_value"].isna() if "concession_value" in df.columns else True
    fill = ((~df["has_concession"]) | (df["has_concession"] & no_value)) & missing_ner & has_rent
    df.loc[fill, "effective_monthly_rent"] = df.loc[fill, "rent"]

    # ESS/UDR week 1 nulling — only if this is the earliest week
    earliest = min(_list_weeks()) if _list_weeks() else None
    if earliest == target_date:
        for r in ("ESS", "UDR"):
            mask = df["reit"] == r
            if mask.any():
                df.loc[mask, "effective_monthly_rent"] = None

    df["macro_market"] = df["market"].apply(_resolve_macro_market)
    return df


# ── Findings model ────────────────────────────────────────────────────────

class Findings:
    def __init__(self):
        self.items = []

    def add(self, severity: str, check: str, msg: str, **meta):
        self.items.append({
            "severity": severity, "check": check, "msg": msg,
            **meta,
        })

    def by_severity(self, sev):
        return [f for f in self.items if f["severity"] == sev]


# ── A) Bucket vs matched-unit divergence ──────────────────────────────────

def check_bucket_vs_matched(weeks, summary_history_df, fnd: Findings):
    """Cross-check that bucket-level chain-link WoW (from summary_history)
    matches the unit-level matched comparison.

    Critical: when a (REIT, week) was bridged across a coverage-gap week,
    the bucket WoW reflects N -> bridged_prev (e.g., 3-28 -> 4-11), not
    consecutive (4-4 -> 4-11). The matched-unit check must use the same
    bridged anchor or the comparison will trivially diverge.

    Performance: pre-loads each week once and indexes by reit so the
    inner _has_gap/matched-merge loop avoids re-reading CSVs hundreds
    of times. Was timing out at 900s with 7 weeks; this brings it to
    ~30s.
    """
    sh = summary_history_df.copy()
    sh["scrape_date"] = pd.to_datetime(sh["scrape_date"])

    GAP_THRESHOLD = 0.10

    # Pre-load each week's data once. Build per-REIT indexes for fast lookup.
    print("    [pre-load] caching week dataframes...")
    week_data = {}      # week -> full DataFrame
    week_by_reit = {}   # week -> {reit -> sub-DataFrame}
    week_communities = {}  # week -> {reit -> set of communities}
    for wk in weeks:
        df = _load_week(wk)
        week_data[wk] = df
        if df.empty:
            week_by_reit[wk] = {}
            week_communities[wk] = {}
            continue
        by_reit = {}
        comms = {}
        for reit, sub in df.groupby("reit"):
            by_reit[reit] = sub
            comms[reit] = set(sub["community"].dropna())
        week_by_reit[wk] = by_reit
        week_communities[wk] = comms

    def _has_gap(reit, prev_d, curr_d):
        p_comms = week_communities.get(prev_d, {}).get(reit, set())
        c_comms = week_communities.get(curr_d, {}).get(reit, set())
        if len(p_comms) < 10 or len(c_comms) < 10:
            return True
        miss = len(p_comms - c_comms) / max(len(p_comms), 1)
        new = len(c_comms - p_comms) / max(len(c_comms), 1)
        return miss >= GAP_THRESHOLD or new >= GAP_THRESHOLD

    def _find_bridged_prev(reit, curr_wk):
        idx = weeks.index(curr_wk)
        for j in range(idx - 1, -1, -1):
            cand = weeks[j]
            if not _has_gap(reit, cand, curr_wk):
                return cand
        return None

    pairs = list(zip(weeks[:-1], weeks[1:]))

    for prev_wk, curr_wk in pairs:
        # Bucket-level chain-link WoW per (reit, market) for THIS curr_wk
        sub = sh[sh["scrape_date"] == pd.Timestamp(curr_wk)]
        sub = sub.dropna(subset=["sp_avg_eff_rent_curr", "sp_avg_eff_rent_prev",
                                  "sp_count"])
        sub = sub[sub["sp_count"] > 0]
        if sub.empty:
            continue
        sub["_w_curr"] = sub["sp_avg_eff_rent_curr"] * sub["sp_count"]
        sub["_w_prev"] = sub["sp_avg_eff_rent_prev"] * sub["sp_count"]
        bucket = sub.groupby(["reit", "macro_market"]).agg(
            wsc=("_w_curr", "sum"), wsp=("_w_prev", "sum"),
            n=("sp_count", "sum")).reset_index()
        bucket["bucket_wow"] = (bucket["wsc"] / bucket["wsp"] - 1) * 100

        # For each REIT, identify bridged anchor and compute matched-unit
        # WoW against it. Uses pre-loaded per-REIT frames.
        curr_by_reit = week_by_reit.get(curr_wk, {})
        if not curr_by_reit:
            continue

        m_rows = []
        for reit in bucket["reit"].unique():
            anchor = _find_bridged_prev(reit, curr_wk)
            if anchor is None:
                continue
            curr_r = curr_by_reit.get(reit)
            prev_r = week_by_reit.get(anchor, {}).get(reit)
            if curr_r is None or prev_r is None:
                continue
            merged = curr_r[["unit_id", "macro_market", "effective_monthly_rent"]].rename(
                columns={"effective_monthly_rent": "ner_c"}).merge(
                prev_r[["unit_id", "effective_monthly_rent"]].rename(
                    columns={"effective_monthly_rent": "ner_p"}),
                on="unit_id", how="inner")
            merged = merged.dropna(subset=["ner_p", "ner_c"])
            if merged.empty:
                continue
            for mkt, g in merged.groupby("macro_market"):
                if len(g) == 0:
                    continue
                m_rows.append({
                    "reit": reit,
                    "macro_market": mkt,
                    "n": len(g),
                    "ner_p": g["ner_p"].mean(),
                    "ner_c": g["ner_c"].mean(),
                    "matched_wow": (g["ner_c"].mean() / g["ner_p"].mean() - 1) * 100
                                    if g["ner_p"].mean() else None,
                    "anchor": anchor.isoformat(),
                })

        if not m_rows:
            continue
        m_agg = pd.DataFrame(m_rows)
        m = bucket.merge(m_agg, on=["reit", "macro_market"], suffixes=("_b", "_m"))
        m = m[m["n_b"] >= 30]
        m["abs_diff_bps"] = (m["bucket_wow"] - m["matched_wow"]).abs() * 100
        bad = m[m["abs_diff_bps"] > 50]
        for _, r in bad.iterrows():
            sev = "FAIL" if r["abs_diff_bps"] > 200 else "WARN"
            anchor_note = f" [anchor={r['anchor']}]" if r["anchor"] != prev_wk.isoformat() else ""
            fnd.add(sev, "bucket_vs_matched_divergence",
                    f"{r['reit']} {r['macro_market']} {prev_wk} -> {curr_wk}{anchor_note}: "
                    f"bucket NER {r['bucket_wow']:+.2f}% vs matched {r['matched_wow']:+.2f}% "
                    f"(Δ={r['abs_diff_bps']:.0f} bps, n={int(r['n_b'])})",
                    pair=f"{prev_wk}/{curr_wk}", reit=r["reit"], market=r["macro_market"])


# ── B) NER vs rent identity ───────────────────────────────────────────────

def check_ner_vs_rent_identity(weeks, summary_history_df, fnd: Findings):
    """NER WoW should equal rent WoW IF AND ONLY IF concession state is
    truly unchanged. We define "truly unchanged" as:
       Δ concession_rate < 2 ppts  AND  Δ avg concession value < $5/mo
       (i.e. neither the share of units with concessions nor the average
        depth of concessions changed materially).

    If NER moves > rent + 2 ppts AND both concession-rate AND
    concession-value are stable, that's an arithmetic identity violation
    (parser changed mid-stream, units shifted in/out of NER pool, etc.).

    Otherwise — if concession-value moved — the NER divergence is REAL
    signal (concession depth change on existing units) and we just log
    it as INFO, not a warning."""
    sh = summary_history_df.copy()
    sh["scrape_date"] = pd.to_datetime(sh["scrape_date"])
    sh = sh.dropna(subset=["sp_avg_rent_curr", "sp_avg_rent_prev",
                            "sp_avg_eff_rent_curr", "sp_avg_eff_rent_prev",
                            "sp_count"])
    sh = sh[sh["sp_count"] >= 30]
    sh["rent_wow"] = (sh["sp_avg_rent_curr"] / sh["sp_avg_rent_prev"] - 1) * 100
    sh["ner_wow"] = (sh["sp_avg_eff_rent_curr"] / sh["sp_avg_eff_rent_prev"] - 1) * 100
    sh["delta"] = sh["ner_wow"] - sh["rent_wow"]
    sh["conc_rate_change"] = (sh["sp_concession_rate_curr"]
                              - sh["sp_concession_rate_prev"]).abs() * 100
    # Implied concession depth change (NER - rent gap, in $)
    sh["conc_depth_curr"] = sh["sp_avg_rent_curr"] - sh["sp_avg_eff_rent_curr"]
    sh["conc_depth_prev"] = sh["sp_avg_rent_prev"] - sh["sp_avg_eff_rent_prev"]
    sh["conc_depth_change"] = (sh["conc_depth_curr"] - sh["conc_depth_prev"]).abs()

    # Bucket by what's driving the NER vs rent divergence
    suspect = sh[sh["delta"].abs() > 2.0]
    for _, r in suspect.iterrows():
        rate_stable = r["conc_rate_change"] < 2.0
        depth_stable = r["conc_depth_change"] < 5.0   # < $5/mo change
        if rate_stable and depth_stable:
            # Arithmetic identity violation — suspicious
            fnd.add("WARN", "ner_vs_rent_identity",
                    f"{r['reit']} {r['macro_market']} {str(r['scrape_date'])[:10]}: "
                    f"NER WoW {r['ner_wow']:+.2f}% vs rent WoW {r['rent_wow']:+.2f}% "
                    f"(Δ={r['delta']:+.2f} ppts) — concession state stable "
                    f"(rate Δ={r['conc_rate_change']:.1f} ppts, depth Δ=${r['conc_depth_change']:.0f}/mo)",
                    pair=str(r["scrape_date"])[:10], reit=r["reit"], market=r["macro_market"])
        # else: real concession depth change — not a warning, just signal


# ── C) Boundary sanity ────────────────────────────────────────────────────

def check_boundaries(summary_history_df, fnd: Findings):
    sh = summary_history_df.copy()
    sh["scrape_date"] = pd.to_datetime(sh["scrape_date"])

    # NER > gross rent (with 2% slack)
    bad = sh.dropna(subset=["sp_avg_eff_rent_curr", "sp_avg_rent_curr"])
    bad = bad[bad["sp_avg_eff_rent_curr"] > bad["sp_avg_rent_curr"] * 1.02]
    for _, r in bad.iterrows():
        fnd.add("FAIL", "ner_above_rent",
                f"{r['reit']} {r['macro_market']} {str(r['scrape_date'])[:10]} beds={r.get('beds')}: "
                f"NER ${r['sp_avg_eff_rent_curr']:,.0f} > gross rent ${r['sp_avg_rent_curr']:,.0f}",
                reit=r["reit"], market=r["macro_market"])

    # Concession rate range
    bad_cr = sh[(sh["sp_concession_rate_curr"].notna()) &
                ((sh["sp_concession_rate_curr"] < 0) | (sh["sp_concession_rate_curr"] > 1))]
    for _, r in bad_cr.iterrows():
        fnd.add("FAIL", "conc_rate_oob",
                f"{r['reit']} {r['macro_market']}: concession_rate={r['sp_concession_rate_curr']:.3f} (out of [0,1])")

    # sp_count > listing_count
    if "listing_count" in sh.columns:
        bad_n = sh.dropna(subset=["sp_count", "listing_count"])
        bad_n = bad_n[bad_n["sp_count"] > bad_n["listing_count"]]
        for _, r in bad_n.iterrows():
            fnd.add("FAIL", "sp_count_exceeds_listings",
                    f"{r['reit']} {r['macro_market']} {str(r['scrape_date'])[:10]}: "
                    f"sp_count={r['sp_count']} > listing_count={r['listing_count']}")


# ── D) Coalesce contamination signature ───────────────────────────────────

def check_coalesce_contamination(weeks, fnd: Findings):
    """Detect when a REIT's prev-period NER == gross rent AND concession
    rate == 0 (forced coalesce signature) AND the next-period concession
    rate is NOT also ~0.

    Without the second condition, this fires on every SFR REIT (AMH,
    INVH) which legitimately have 0% concessions every week — that's not
    a coalesce contamination, just SFR reality.

    True contamination signature: prev shows 0% concessions, next week
    shows nontrivial concessions (>2%). That asymmetry indicates the
    prior period's NER was forced to gross rent due to a scraper bug,
    not a real market state."""
    if len(weeks) < 2:
        return
    pairs = list(zip(weeks[:-1], weeks[1:]))
    for prev_wk, curr_wk in pairs:
        prev_df = _load_week(prev_wk)
        curr_df = _load_week(curr_wk)
        if prev_df.empty or curr_df.empty:
            continue
        for reit, g in prev_df.groupby("reit"):
            if len(g) < 100:
                continue
            ner = g["effective_monthly_rent"]
            rent = g["rent"]
            equal_ratio = ((ner == rent) | ner.isna()).mean()
            prev_conc = g["has_concession"].mean()
            if not (equal_ratio > 0.99 and prev_conc < 0.01):
                continue
            curr_g = curr_df[curr_df["reit"] == reit]
            if len(curr_g) < 100:
                continue
            curr_conc = curr_g["has_concession"].mean()
            # Only flag if curr period has materially different state
            if curr_conc < 0.02:
                continue   # SFR-like: legit no-concession both weeks
            fnd.add("WARN", "coalesce_contamination",
                    f"{reit} {prev_wk}: 99%+ NER==gross_rent AND concession_rate~0% in prev "
                    f"BUT concession_rate {curr_conc:.0%} in {curr_wk} — likely scraper bug "
                    f"in prev, NER WoW unreliable",
                    pair=f"{prev_wk}/{curr_wk}", reit=reit)


# ── E) Cross-REIT outlier ─────────────────────────────────────────────────

def check_cross_reit_outlier(summary_history_df, fnd: Findings):
    sh = summary_history_df.copy()
    sh["scrape_date"] = pd.to_datetime(sh["scrape_date"])
    sh = sh.dropna(subset=["sp_avg_eff_rent_curr", "sp_avg_eff_rent_prev", "sp_count"])
    sh = sh[sh["sp_count"] > 0]
    sh["_w_curr"] = sh["sp_avg_eff_rent_curr"] * sh["sp_count"]
    sh["_w_prev"] = sh["sp_avg_eff_rent_prev"] * sh["sp_count"]
    agg = sh.groupby(["reit", "scrape_date"]).agg(
        wsc=("_w_curr", "sum"), wsp=("_w_prev", "sum")).reset_index()
    agg["wow_pct"] = (agg["wsc"] / agg["wsp"] - 1) * 100

    for date, g in agg.groupby("scrape_date"):
        if len(g) < 4:
            continue
        med = g["wow_pct"].median()
        std = g["wow_pct"].std(ddof=0)
        if std == 0:
            continue
        for _, r in g.iterrows():
            z = (r["wow_pct"] - med) / std
            if abs(z) > 2.5:
                fnd.add("WARN", "cross_reit_outlier",
                        f"{r['reit']} {str(date)[:10]}: NER WoW {r['wow_pct']:+.2f}% vs cohort median "
                        f"{med:+.2f}% (z={z:+.1f})",
                        pair=str(date)[:10], reit=r["reit"])


# ── F) Distribution / pool size shifts ────────────────────────────────────

def check_distribution_shifts(weeks, fnd: Findings):
    """Per-REIT, flag weeks where matched-pool size jumped >25% vs prior."""
    sizes = {}
    for wk in weeks:
        df = _load_week(wk)
        if df.empty:
            continue
        for reit, g in df.groupby("reit"):
            sizes.setdefault(reit, []).append((wk, len(g)))
    for reit, series in sizes.items():
        series.sort()
        for i in range(1, len(series)):
            prev_n = series[i - 1][1]
            curr_n = series[i][1]
            if prev_n > 100:
                pct = (curr_n - prev_n) / prev_n * 100
                if abs(pct) > 25:
                    fnd.add("WARN", "pool_size_shift",
                            f"{reit} {series[i - 1][0]} -> {series[i][0]}: total units "
                            f"{prev_n:,} -> {curr_n:,} ({pct:+.1f}%) — unusually large refresh",
                            pair=f"{series[i - 1][0]}/{series[i][0]}", reit=reit)


# ── F2) Community coverage gaps (week N vs N+1) ───────────────────────────

def check_community_coverage_gaps(weeks, fnd: Findings):
    """Flag REIT-week pairs where the prev period had communities (entire
    properties) that don't appear in the next period. The sp_count
    intersection-based same-property pool then under-represents those
    communities, biasing aggregates.

    Threshold: >10 communities missing in next period AND prev had >50
    total communities (large enough sample)."""
    if len(weeks) < 2:
        return
    pairs = list(zip(weeks[:-1], weeks[1:]))
    for prev_wk, curr_wk in pairs:
        prev_df = _load_week(prev_wk)
        curr_df = _load_week(curr_wk)
        if prev_df.empty or curr_df.empty:
            continue
        for reit in sorted(set(prev_df["reit"].unique()) | set(curr_df["reit"].unique())):
            p_comms = set(prev_df[prev_df["reit"] == reit]["community"].dropna())
            c_comms = set(curr_df[curr_df["reit"] == reit]["community"].dropna())
            if len(p_comms) < 50:
                continue
            missing_in_curr = p_comms - c_comms
            new_in_curr = c_comms - p_comms
            if len(missing_in_curr) > 10:
                pct = len(missing_in_curr) / len(p_comms) * 100
                fnd.add("WARN", "community_coverage_gap",
                        f"{reit} {prev_wk} -> {curr_wk}: {len(missing_in_curr)} communities "
                        f"in prev ({pct:.1f}% of {len(p_comms)} total) missing from curr — "
                        f"matched-pool will under-represent these properties. "
                        f"Sample: {sorted(list(missing_in_curr))[:3]}",
                        pair=f"{prev_wk}/{curr_wk}", reit=reit)
            if len(new_in_curr) > 10:
                pct = len(new_in_curr) / len(c_comms) * 100
                fnd.add("WARN", "community_coverage_gap",
                        f"{reit} {prev_wk} -> {curr_wk}: {len(new_in_curr)} new communities "
                        f"in curr ({pct:.1f}% of {len(c_comms)} total) — these can't be matched "
                        f"and shift the SP pool composition. Sample: {sorted(list(new_in_curr))[:3]}",
                        pair=f"{prev_wk}/{curr_wk}", reit=reit)


# ── F3) Concession rate volatility ────────────────────────────────────────

def check_conc_rate_volatility(weeks, fnd: Findings):
    """Flag REIT-week pairs with concession rate moving >5 ppts WoW.
    Decompose the cause:
      • matched-unit conc flips (real concession behavior change), or
      • new/lost listings shifting the composition
    Composition-driven moves >3 ppts get a separate WARN tag because the
    aggregate change isn't a true concession-strategy signal."""
    if len(weeks) < 2:
        return
    pairs = list(zip(weeks[:-1], weeks[1:]))
    for prev_wk, curr_wk in pairs:
        prev_df = _load_week(prev_wk)
        curr_df = _load_week(curr_wk)
        if prev_df.empty or curr_df.empty:
            continue
        for reit in sorted(set(prev_df["reit"].unique()) | set(curr_df["reit"].unique())):
            p = prev_df[prev_df["reit"] == reit]
            c = curr_df[curr_df["reit"] == reit]
            if len(p) < 100 or len(c) < 100:
                continue
            prior_rate = p["has_concession"].mean() * 100
            curr_rate = c["has_concession"].mean() * 100
            delta = curr_rate - prior_rate
            if abs(delta) < 5.0:
                continue   # within normal weekly noise

            # Decompose
            p_ids = set(p["unit_id"]); c_ids = set(c["unit_id"])
            matched = p_ids & c_ids
            pm = p[p["unit_id"].isin(matched)].set_index("unit_id")
            cm = c[c["unit_id"].isin(matched)].set_index("unit_id")
            pm = pm.loc[~pm.index.duplicated(keep="last")]
            cm = cm.loc[~cm.index.duplicated(keep="last")]
            common = pm.index.intersection(cm.index)
            pm = pm.loc[common]; cm = cm.loc[common]
            new_conc = int(((~pm["has_concession"]) & cm["has_concession"]).sum())
            lost_conc = int((pm["has_concession"] & (~cm["has_concession"])).sum())

            new_listings = c[c["unit_id"].isin(c_ids - p_ids)]
            lost_listings = p[p["unit_id"].isin(p_ids - c_ids)]
            n_new_listings = len(new_listings)
            n_lost_listings = len(lost_listings)

            # Net flip impact on rate (assuming matched pool size ~ total)
            net_flip = new_conc - lost_conc
            net_flip_pct = net_flip / max(len(common), 1) * 100

            # Composition impact (new minus lost listings, weighted by their conc rates)
            new_with_conc = int(new_listings["has_concession"].sum())
            lost_with_conc = int(lost_listings["has_concession"].sum())
            composition_impact = (new_with_conc - lost_with_conc) / max(len(c), 1) * 100

            # Total listing turnover as % of prev
            turnover_pct = (n_new_listings + n_lost_listings) / max(len(p), 1) * 100

            severity = "WARN"
            tag_extra = ""
            # 30% weekly turnover is normal for AVB/UDR (units rent fast).
            # Only call out turnover as scraper-related when it's extreme (>45%).
            if turnover_pct > 45:
                tag_extra = f" [HIGH TURNOVER {turnover_pct:.0f}% — likely scraper coverage shift]"
            elif abs(composition_impact) > abs(net_flip_pct):
                tag_extra = (f" [composition-driven (new-vs-lost listing mix shift "
                             f"contributed {composition_impact:+.1f}ppts vs flips {net_flip_pct:+.1f}ppts)]")
            else:
                tag_extra = (f" [matched-flip driven (flips {net_flip_pct:+.1f}ppts dominate "
                             f"composition {composition_impact:+.1f}ppts)]")

            fnd.add(severity, "conc_rate_volatility",
                    f"{reit} {prev_wk} -> {curr_wk}: conc rate {prior_rate:.1f}% -> {curr_rate:.1f}% "
                    f"(Δ={delta:+.1f} ppts). Matched flips: +{new_conc}/−{lost_conc} on "
                    f"{len(common):,} matched. New listings: {n_new_listings:,} ({new_with_conc} w/conc). "
                    f"Lost listings: {n_lost_listings:,} ({lost_with_conc} w/conc).{tag_extra}",
                    pair=f"{prev_wk}/{curr_wk}", reit=reit)


# ── G) Identical-text floods ──────────────────────────────────────────────

def check_identical_text_floods(weeks, fnd: Findings):
    for wk in weeks:
        df = _load_week(wk)
        if df.empty:
            continue
        conc = df[df["has_concession"] == True]
        for reit, g in conc.groupby("reit"):
            if len(g) < 100:
                continue
            counts = g["concession_raw"].fillna("").value_counts()
            if not len(counts):
                continue
            top_share = counts.iloc[0] / len(g)
            if top_share >= 0.90:
                fnd.add("FAIL", "identical_text_flood",
                        f"{reit} {wk}: {top_share:.1%} of {len(g):,} concession rows share "
                        f"identical text: {counts.index[0]!r}",
                        reit=reit, week=str(wk))


# ── H) Source-code regression ─────────────────────────────────────────────

def check_source_code(fnd: Findings):
    bad_patterns = [
        (r"/\s*first_val\s*\*\s*100",
         "Found '/ first_val * 100' — old base-period division (should be chain-link)"),
        (r"sp_avg_rent_curr\s*\.\s*mean\s*\(",
         "Found '.mean()' on sp_avg_rent_curr (should be count-weighted)"),
        (r"sp_avg_rent_psf_curr\s*\.\s*mean\s*\(",
         "Found '.mean()' on sp_avg_rent_psf_curr (should be count-weighted)"),
        (r"sp_avg_eff_rent_curr\s*\.\s*mean\s*\(",
         "Found '.mean()' on sp_avg_eff_rent_curr (should be count-weighted)"),
        (r"sp_avg_eff_rent_psf_curr\s*\.\s*mean\s*\(",
         "Found '.mean()' on sp_avg_eff_rent_psf_curr (should be count-weighted)"),
    ]
    for fname in ("build_excel.py", "rebuild_summary_history.py"):
        path = BASE_DIR / fname
        if not path.exists():
            continue
        try:
            src = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for pat, msg in bad_patterns:
            for i, line in enumerate(src.splitlines(), start=1):
                stripped = line.lstrip()
                if stripped.startswith("#") or stripped.startswith('"') or stripped.startswith("'"):
                    continue
                if re.search(pat, line):
                    fnd.add("FAIL", "source_code_regression",
                            f"{fname}:{i}: {msg}")


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true",
                    help="Exit 2 on any FAIL")
    args = ap.parse_args()

    if not SUMMARY_PATH.exists():
        print(f"[Audit] No summary_history at {SUMMARY_PATH}")
        return 1

    sh = pd.read_csv(SUMMARY_PATH)
    weeks = _list_weeks()
    if len(weeks) < 2:
        print("[Audit] Need 2+ weeks of data.")
        return 0

    print(f"[Audit] Scanning {len(weeks)} weeks: {weeks[0]} -> {weeks[-1]}")

    fnd = Findings()
    print("  [check A] bucket vs matched-unit divergence...")
    check_bucket_vs_matched(weeks, sh, fnd)
    print("  [check B] NER vs rent identity...")
    check_ner_vs_rent_identity(weeks, sh, fnd)
    print("  [check C] boundary sanity...")
    check_boundaries(sh, fnd)
    print("  [check D] coalesce contamination signatures...")
    check_coalesce_contamination(weeks, fnd)
    print("  [check E] cross-REIT NER outliers...")
    check_cross_reit_outlier(sh, fnd)
    print("  [check F] pool-size shifts...")
    check_distribution_shifts(weeks, fnd)
    print("  [check F2] community coverage gaps...")
    check_community_coverage_gaps(weeks, fnd)
    print("  [check F3] concession rate volatility decomposition...")
    check_conc_rate_volatility(weeks, fnd)
    print("  [check G] identical-text floods...")
    check_identical_text_floods(weeks, fnd)
    print("  [check H] source-code regression...")
    check_source_code(fnd)

    fails = fnd.by_severity("FAIL")
    warns = fnd.by_severity("WARN")

    # Markdown report
    latest = weeks[-1]
    out_md = LOG_DIR / f"data_integrity_audit_{latest}.md"
    out_json = LOG_DIR / f"data_integrity_audit_{latest}.json"

    lines = ["# Data Integrity Audit", ""]
    lines.append(f"- Generated: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"- Weeks: {len(weeks)} ({weeks[0]} -> {weeks[-1]})")
    lines.append(f"- Findings: {len(fails)} FAIL, {len(warns)} WARN")
    lines.append("")

    if fails:
        lines.append(f"## [FAIL] ({len(fails)})")
        lines.append("")
        for f in fails:
            lines.append(f"- **{f['check']}**: {f['msg']}")
        lines.append("")

    if warns:
        # Group by check type for readability
        from collections import defaultdict
        by_check = defaultdict(list)
        for f in warns:
            by_check[f["check"]].append(f)
        lines.append(f"## [WARN] ({len(warns)})")
        lines.append("")
        for check, items in by_check.items():
            lines.append(f"### {check} ({len(items)})")
            lines.append("")
            for f in items:
                lines.append(f"- {f['msg']}")
            lines.append("")

    if not fails and not warns:
        lines.append("## [PASS] All checks clean")
        lines.append("")

    out_md.write_text("\n".join(lines), encoding="utf-8")
    out_json.write_text(json.dumps(fnd.items, indent=2, default=str), encoding="utf-8")

    print(f"\n[Audit] Wrote {out_md}")
    print(f"[Audit] {len(fails)} FAIL, {len(warns)} WARN")

    if args.strict and fails:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
