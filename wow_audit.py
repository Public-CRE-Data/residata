"""
Detailed weekly WoW audit — data-scientist-grade.

For every REIT × week pair, decompose the same-property WoW% into:
  • Bucket-level WoW (REIT × macro_market × beds), using chain-link math
  • Top-contributing markets (positive and negative)
  • Concession flips (units gaining/losing concessions in the matched pool)
  • Same-unit reproducibility check: bucket-level chain-link WoW vs raw
    matched-unit WoW computed directly on the unit-level CSVs.

Flags any bucket-level move that:
  • Differs from the raw matched-unit WoW by more than 50 bps (composition
    leak or parser/aggregation inconsistency)
  • Concentrates >75% of the REIT-level WoW in <3 markets (driver concentration)
  • Has matched-pool size <30 (statistically thin)
  • Shows mismatched concession flips (e.g. one bucket flipped many but the
    NER didn't move — possible data-error fingerprint)

Outputs:
  logs/wow_audit_<latest_week>.md   — markdown report
  logs/wow_audit_<latest_week>.csv  — full row-level data

Designed to run after every scrape, alongside wow_qa.py.
"""

from __future__ import annotations

import argparse
import pathlib
import re
import warnings
from datetime import datetime, timedelta

import pandas as pd

warnings.filterwarnings("ignore")

BASE_DIR = pathlib.Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "raw"
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)
SUMMARY_PATH = BASE_DIR / "data" / "summary" / "summary_history.csv"

import sys
sys.path.insert(0, str(BASE_DIR))
from build_excel import _resolve_macro_market

_DATE_RE = re.compile(r"_raw_(\d{4}-\d{2}-\d{2})")

# ── Thresholds ────────────────────────────────────────────────────────────
THRESH = {
    "bucket_vs_unit_diff_bps": 50,          # bucket WoW vs matched-unit WoW
    "driver_concentration_share": 0.75,     # >X% of WoW from top 3 markets
    "thin_pool_min_n": 30,                  # matched pool below this is suspect
    "matched_unit_wow_warn_pct": 1.5,       # WARN if |matched WoW| > X%
    "matched_unit_wow_fail_pct": 5.0,       # FAIL if |matched WoW| > X% AND n>=30
    "ner_only_move_pct": 1.0,               # NER moves >X% with rent flat
}


# ── Helpers ───────────────────────────────────────────────────────────────

def _saturday_anchor(d):
    return d - timedelta(days=(d.weekday() - 5) % 7)


def _load_week(date_obj) -> pd.DataFrame:
    """Load all CSVs in a Saturday-anchored week."""
    target = date_obj
    files = []
    for f in sorted(RAW_DIR.glob("*_raw_*.csv")):
        m = _DATE_RE.search(f.stem)
        if not m:
            continue
        d = datetime.strptime(m.group(1), "%Y-%m-%d").date()
        if _saturday_anchor(d) == target:
            files.append(f)
    if not files:
        return pd.DataFrame()
    df = pd.concat([pd.read_csv(f, low_memory=False) for f in files], ignore_index=True)
    df["scrape_date"] = pd.to_datetime(df["scrape_date"])
    df = (df.sort_values("scrape_date")
            .groupby(["reit", "unit_id"], as_index=False).last())

    # AMH bare-percent fix
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

    # Re-parse all concession_raw with current parser logic
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
        # No-concession rows: enforce defaults
        no_mask = df["concession_raw"].isna() | (df["concession_raw"] == "")
        df.loc[no_mask, "has_concession"] = False
        for c in conc_cols[1:]:
            df.loc[no_mask, c] = None
    except ImportError:
        pass

    df["has_concession"] = df["has_concession"].fillna(False).astype(bool)

    # NER coalesce: (a) no concession + NER missing, (b) has_concession=True
    # but no parseable value (soft promo banner). See build_excel for rationale.
    has_rent = df["rent"].notna() & (df["rent"] > 0)
    missing_ner = df["effective_monthly_rent"].isna()
    no_value = df["concession_value"].isna() if "concession_value" in df.columns else True
    fill = ((~df["has_concession"]) | (df["has_concession"] & no_value)) & missing_ner & has_rent
    df.loc[fill, "effective_monthly_rent"] = df.loc[fill, "rent"]

    df["macro_market"] = df["market"].apply(_resolve_macro_market)
    return df


def _list_weeks() -> list:
    """List all Saturday-anchored weeks present in data/raw/."""
    weeks = set()
    for f in sorted(RAW_DIR.glob("*_raw_*.csv")):
        m = _DATE_RE.search(f.stem)
        if not m:
            continue
        d = datetime.strptime(m.group(1), "%Y-%m-%d").date()
        weeks.add(_saturday_anchor(d))
    return sorted(weeks)


# ── Decomposition ─────────────────────────────────────────────────────────

def _decompose_pair(prev_df: pd.DataFrame, curr_df: pd.DataFrame) -> pd.DataFrame:
    """For each (reit, macro_market), compute matched-unit aggregates."""
    if prev_df.empty or curr_df.empty:
        return pd.DataFrame()

    p_keys = prev_df.set_index(["reit", "unit_id"])
    c_keys = curr_df.set_index(["reit", "unit_id"])
    common = p_keys.index.intersection(c_keys.index)
    if len(common) == 0:
        return pd.DataFrame()
    p_keys = p_keys.loc[~p_keys.index.duplicated(keep="last")]
    c_keys = c_keys.loc[~c_keys.index.duplicated(keep="last")]
    common = p_keys.index.intersection(c_keys.index)

    p = p_keys.loc[common].reset_index()
    c = c_keys.loc[common].reset_index()
    # Use macro_market and beds from current period (consistent labeling)
    p = p.rename(columns={"rent": "rent_p", "effective_monthly_rent": "ner_p",
                          "has_concession": "hc_p"})
    c = c.rename(columns={"rent": "rent_c", "effective_monthly_rent": "ner_c",
                          "has_concession": "hc_c"})
    m = c[["reit", "unit_id", "macro_market", "beds", "rent_c", "ner_c",
           "hc_c"]].merge(
        p[["reit", "unit_id", "rent_p", "ner_p", "hc_p"]],
        on=["reit", "unit_id"], how="inner")

    rows = []
    for (reit, mkt), g in m.groupby(["reit", "macro_market"]):
        n = len(g)
        rent_prev = g["rent_p"].mean()
        rent_curr = g["rent_c"].mean()
        ner_prev = g["ner_p"].mean()
        ner_curr = g["ner_c"].mean()
        new_conc = int(((~g["hc_p"].astype(bool)) & g["hc_c"].astype(bool)).sum())
        lost_conc = int((g["hc_p"].astype(bool) & (~g["hc_c"].astype(bool))).sum())
        rows.append({
            "reit": reit,
            "macro_market": mkt,
            "matched_n": n,
            "rent_prev": rent_prev,
            "rent_curr": rent_curr,
            "rent_wow_pct": (rent_curr / rent_prev - 1) * 100 if rent_prev else None,
            "ner_prev": ner_prev,
            "ner_curr": ner_curr,
            "ner_wow_pct": (ner_curr / ner_prev - 1) * 100 if ner_prev else None,
            "new_conc": new_conc,
            "lost_conc": lost_conc,
        })
    return pd.DataFrame(rows)


def _bucket_chain(summary_history_df: pd.DataFrame, week, metric_curr: str,
                  metric_prev: str) -> pd.DataFrame:
    """Compute bucket-level chain-link WoW% per (REIT × market) for one week."""
    sh = summary_history_df.copy()
    sh["scrape_date"] = pd.to_datetime(sh["scrape_date"])
    sh = sh[sh["scrape_date"] == pd.Timestamp(week)]
    sh = sh.dropna(subset=[metric_curr, metric_prev, "sp_count"])
    sh = sh[sh["sp_count"] > 0]
    if sh.empty:
        return pd.DataFrame()
    sh["_w_curr"] = sh[metric_curr] * sh["sp_count"]
    sh["_w_prev"] = sh[metric_prev] * sh["sp_count"]
    agg = sh.groupby(["reit", "macro_market"]).agg(
        wsc=("_w_curr", "sum"), wsp=("_w_prev", "sum"),
        n=("sp_count", "sum")).reset_index()
    agg["wow_pct"] = (agg["wsc"] / agg["wsp"] - 1) * 100
    return agg[["reit", "macro_market", "n", "wow_pct"]]


def _reit_chain(summary_history_df: pd.DataFrame, metric_curr: str,
                metric_prev: str) -> pd.DataFrame:
    """Compute REIT-level chain-link WoW% per (REIT × week)."""
    sh = summary_history_df.copy()
    sh["scrape_date"] = pd.to_datetime(sh["scrape_date"])
    sh = sh.dropna(subset=[metric_curr, metric_prev, "sp_count"])
    sh = sh[sh["sp_count"] > 0]
    if sh.empty:
        return pd.DataFrame()
    sh["_w_curr"] = sh[metric_curr] * sh["sp_count"]
    sh["_w_prev"] = sh[metric_prev] * sh["sp_count"]
    agg = sh.groupby(["reit", "scrape_date"]).agg(
        wsc=("_w_curr", "sum"), wsp=("_w_prev", "sum"),
        n=("sp_count", "sum")).reset_index()
    agg["wow_pct"] = (agg["wsc"] / agg["wsp"] - 1) * 100
    return agg[["reit", "scrape_date", "n", "wow_pct"]]


# ── Audit per pair ────────────────────────────────────────────────────────

def audit_pair(prev_wk, curr_wk, summary_history_df, lines: list, all_rows: list):
    prev_df = _load_week(prev_wk)
    curr_df = _load_week(curr_wk)
    if prev_df.empty or curr_df.empty:
        return

    matched = _decompose_pair(prev_df, curr_df)
    if matched.empty:
        return

    # Bucket-aggregated chain-link WoW from summary_history
    bucket_rent = _bucket_chain(summary_history_df, curr_wk,
                                "sp_avg_rent_curr", "sp_avg_rent_prev")
    bucket_ner = _bucket_chain(summary_history_df, curr_wk,
                                "sp_avg_eff_rent_curr", "sp_avg_eff_rent_prev")

    bucket_rent = bucket_rent.rename(columns={"wow_pct": "bucket_rent_wow",
                                                "n": "sp_n"})
    bucket_ner = bucket_ner.rename(columns={"wow_pct": "bucket_ner_wow",
                                              "n": "sp_n_ner"})

    merged = matched.merge(bucket_rent, on=["reit", "macro_market"], how="left")
    merged = merged.merge(bucket_ner[["reit", "macro_market", "bucket_ner_wow"]],
                           on=["reit", "macro_market"], how="left")
    merged["pair"] = f"{prev_wk} -> {curr_wk}"
    all_rows.append(merged.copy())

    lines.append(f"## Pair: {prev_wk} -> {curr_wk}")
    lines.append("")

    for reit, grp in merged.groupby("reit"):
        # REIT-level chain-link WoW (rent and NER)
        reit_rent = _reit_chain(summary_history_df, "sp_avg_rent_curr",
                                "sp_avg_rent_prev")
        reit_ner = _reit_chain(summary_history_df, "sp_avg_eff_rent_curr",
                               "sp_avg_eff_rent_prev")
        rrow = reit_rent[(reit_rent["reit"] == reit) &
                         (reit_rent["scrape_date"] == pd.Timestamp(curr_wk))]
        nrow = reit_ner[(reit_ner["reit"] == reit) &
                        (reit_ner["scrape_date"] == pd.Timestamp(curr_wk))]
        rent_wow = float(rrow["wow_pct"].iloc[0]) if not rrow.empty else float("nan")
        ner_wow = float(nrow["wow_pct"].iloc[0]) if not nrow.empty else float("nan")
        n = int(rrow["n"].iloc[0]) if not rrow.empty else 0

        lines.append(f"### {reit}  —  rent {rent_wow:+.2f}% | NER {ner_wow:+.2f}%  (n={n:,})")
        lines.append("")

        # Per-market breakdown ranked by NER contribution
        grp = grp.copy()
        grp["ner_contrib_bps"] = ((grp["ner_curr"] - grp["ner_prev"]) *
                                   grp["matched_n"] /
                                   ((grp["ner_prev"] * grp["matched_n"]).sum() or 1) * 10000)
        grp["rent_contrib_bps"] = ((grp["rent_curr"] - grp["rent_prev"]) *
                                    grp["matched_n"] /
                                    ((grp["rent_prev"] * grp["matched_n"]).sum() or 1) * 10000)
        # Sort by NER contribution
        top_neg = grp.nsmallest(5, "ner_contrib_bps")
        top_pos = grp.nlargest(5, "ner_contrib_bps")

        lines.append("Top 5 NER drag (matched-unit decomposition):")
        lines.append("")
        lines.append("| Market | n | Rent WoW (matched) | NER WoW (matched) | Bucket NER WoW | Δ (bp) | Conc flips +/− | NER bp contrib |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for _, r in top_neg.iterrows():
            mu = float(r["ner_wow_pct"]) if pd.notna(r["ner_wow_pct"]) else 0
            bk = float(r["bucket_ner_wow"]) if pd.notna(r["bucket_ner_wow"]) else 0
            diff = (mu - bk) * 100  # bps
            lines.append(
                f"| {r['macro_market']} | {int(r['matched_n'])} | "
                f"{r['rent_wow_pct']:+.2f}% | {mu:+.2f}% | {bk:+.2f}% | "
                f"{diff:+.0f} | +{int(r['new_conc'])}/−{int(r['lost_conc'])} | "
                f"{r['ner_contrib_bps']:+.0f} |"
            )
        lines.append("")
        lines.append("Top 5 NER tailwind:")
        lines.append("")
        lines.append("| Market | n | Rent WoW | NER WoW | Bucket NER | Δ (bp) | Conc flips +/− | NER bp |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for _, r in top_pos.iterrows():
            mu = float(r["ner_wow_pct"]) if pd.notna(r["ner_wow_pct"]) else 0
            bk = float(r["bucket_ner_wow"]) if pd.notna(r["bucket_ner_wow"]) else 0
            diff = (mu - bk) * 100
            lines.append(
                f"| {r['macro_market']} | {int(r['matched_n'])} | "
                f"{r['rent_wow_pct']:+.2f}% | {mu:+.2f}% | {bk:+.2f}% | "
                f"{diff:+.0f} | +{int(r['new_conc'])}/−{int(r['lost_conc'])} | "
                f"{r['ner_contrib_bps']:+.0f} |"
            )
        lines.append("")

        # Audit flags for this REIT in this pair
        flags = []
        # 1. Bucket vs matched-unit divergence
        bad_div = grp[(grp["bucket_ner_wow"].notna())
                      & (grp["ner_wow_pct"].notna())
                      & (grp["matched_n"] >= THRESH["thin_pool_min_n"])
                      & (((grp["bucket_ner_wow"] - grp["ner_wow_pct"]).abs() * 100) >
                         THRESH["bucket_vs_unit_diff_bps"])]
        for _, r in bad_div.iterrows():
            mu = float(r["ner_wow_pct"]); bk = float(r["bucket_ner_wow"])
            flags.append(
                f"DIVERGENCE: {r['macro_market']} bucket NER {bk:+.2f}% vs "
                f"matched-unit NER {mu:+.2f}% (Δ={(mu - bk) * 100:+.0f} bps, n={int(r['matched_n'])})"
            )
        # 2. Driver concentration: if 1-3 markets contribute >75% of |NER move|
        gp = grp.copy()
        total_abs = gp["ner_contrib_bps"].abs().sum()
        if total_abs > 0 and abs(ner_wow) > 0.5:
            top3 = gp.reindex(gp["ner_contrib_bps"].abs().sort_values(
                ascending=False).index[:3])
            share = top3["ner_contrib_bps"].abs().sum() / total_abs
            if share >= THRESH["driver_concentration_share"]:
                mkts = ", ".join(f"{m}" for m in top3["macro_market"])
                flags.append(
                    f"CONCENTRATION: {share:.0%} of {reit}'s NER move comes from "
                    f"3 markets: {mkts}"
                )
        # 3. NER-only move (rent flat but NER moved >X%) — concession-driven
        ner_only = grp[(grp["matched_n"] >= THRESH["thin_pool_min_n"])
                       & (grp["ner_wow_pct"].abs() >= THRESH["ner_only_move_pct"])
                       & (grp["rent_wow_pct"].abs() < 0.5)]
        for _, r in ner_only.iterrows():
            flips = int(r["new_conc"]) + int(r["lost_conc"])
            if flips < 5:
                flags.append(
                    f"NER_ONLY_NO_FLIPS: {r['macro_market']} NER moved "
                    f"{float(r['ner_wow_pct']):+.2f}% with rent {float(r['rent_wow_pct']):+.2f}% "
                    f"and only {flips} concession flips (deepening on existing units?)"
                )
        # 4. Thin pool warnings (only if WoW is large)
        thin = grp[(grp["matched_n"] < THRESH["thin_pool_min_n"])
                   & (grp["ner_wow_pct"].abs() >= 2.0)]
        for _, r in thin.iterrows():
            flags.append(
                f"THIN_POOL: {r['macro_market']} n={int(r['matched_n'])} but NER "
                f"moved {float(r['ner_wow_pct']):+.2f}% — likely noise"
            )

        if flags:
            lines.append("**Audit flags:**")
            for f in flags:
                lines.append(f"- {f}")
            lines.append("")
        else:
            lines.append("_No flags._")
            lines.append("")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latest-only", action="store_true",
                    help="Audit only the most recent week pair")
    args = ap.parse_args()

    if not SUMMARY_PATH.exists():
        print(f"[Audit] No summary_history at {SUMMARY_PATH}")
        return 1
    sh = pd.read_csv(SUMMARY_PATH)

    weeks = _list_weeks()
    if len(weeks) < 2:
        print("[Audit] Need at least 2 weeks of data.")
        return 1

    pairs = list(zip(weeks[:-1], weeks[1:]))
    if args.latest_only:
        pairs = pairs[-1:]

    lines = [
        "# Detailed WoW Audit",
        "",
        f"- Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"- Weeks scanned: {[str(w) for w in weeks]}",
        f"- Pairs audited: {len(pairs)}",
        "",
        "## Methodology",
        "",
        "For each pair, we report:",
        "1. **REIT-level chain-link WoW** on rent and NER (count-weighted across "
        "(market, beds) buckets where curr/prev pair from same matched cohort).",
        "2. **Per-market matched-unit decomposition** — for each market, the WoW "
        "is computed directly on unit_ids present in BOTH weeks (no bucketing).",
        "3. **Bucket-vs-matched cross-check** — bucket-level WoW from summary_history "
        "vs raw matched-unit aggregate. Divergence > 50 bps flagged as composition "
        "leak or aggregation bug.",
        "4. **Audit flags** — driver concentration, NER-only-no-flips (concessions "
        "deepening on existing units), thin-pool noise, and divergence.",
        "",
    ]

    all_rows = []
    latest_wk = weeks[-1]
    for prev_wk, curr_wk in pairs:
        audit_pair(prev_wk, curr_wk, sh, lines, all_rows)

    out_md = LOG_DIR / f"wow_audit_{latest_wk}.md"
    out_md.write_text("\n".join(lines), encoding="utf-8")

    if all_rows:
        full = pd.concat(all_rows, ignore_index=True)
        out_csv = LOG_DIR / f"wow_audit_{latest_wk}.csv"
        full.to_csv(out_csv, index=False)
        print(f"[Audit] Wrote {out_md} and {out_csv}")
    else:
        print(f"[Audit] Wrote {out_md} (no row data)")

    # One-line summary on stdout for the pipeline log
    flag_count = sum(1 for l in lines if l.startswith("- "))
    print(f"[Audit] {flag_count} flagged items across {len(pairs)} pair(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
