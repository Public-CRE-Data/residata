"""
Weekly Week-over-Week Data Quality Checker.

Compares the latest week of scrape data vs the prior week and flags:
  • Unit count swings beyond normal listing churn
  • Concession rate moves that are suspiciously large
  • Rent level shifts that look non-organic
  • Identical-text concession floods (scraper false-positive signature)
  • Matched-unit rent outliers

Writes a markdown report to logs/wow_qa_<date>.md and prints a terminal summary.
Exit code 0 = all checks pass or only soft warnings.
Exit code 2 = hard data-quality failure (e.g., identical-text flood).
"""

from __future__ import annotations

import sys
import argparse
import pathlib
import warnings
from datetime import date, datetime

import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

BASE_DIR = pathlib.Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "raw"
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

# ── Thresholds ────────────────────────────────────────────────────────────
# Tuned against observed REIT data: weekly listing turnover is 15-20%, but
# aggregate metrics (unit count, rent level) should be much more stable.
THRESH = {
    "unit_pct_change": 10.0,          # |% chg in total units| above 10% is suspicious
    "community_change": 15,           # |# chg in communities| above 15 is suspicious
    "conc_rate_ppts_soft": 10.0,      # |WoW ppts| above 10 is warn
    "conc_rate_ppts_hard": 25.0,      # |WoW ppts| above 25 is hard fail (unless known migration)
    "avg_rent_pct": 3.0,              # |avg rent WoW %| above 3 is suspicious
    "matched_median_rent_pct": 1.0,   # |matched median rent WoW %| above 1 is suspicious
    "identical_text_share": 0.90,     # if >90% of conc rows have identical text, hard fail
    "identical_text_min_n": 100,      # only flag if there are at least this many conc rows
    "conc_value_single_value_share": 0.90,  # all conc values identical = suspicious
}


# ── Helpers ───────────────────────────────────────────────────────────────

def _load_week(date_patterns: list[str]) -> pd.DataFrame:
    """Load all CSVs matching any of the given date patterns and dedupe."""
    files = []
    for pat in date_patterns:
        files.extend(sorted(RAW_DIR.glob(pat)))
    if not files:
        return pd.DataFrame()
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df["scrape_date"] = pd.to_datetime(df["scrape_date"])
    df = df.sort_values("scrape_date").groupby(["reit", "unit_id"]).last().reset_index()
    return df


def _find_latest_two_weeks() -> tuple[list[str], list[str], date, date]:
    """
    Inspect data/raw/ and return the most recent week + prior week patterns.
    A "week" is grouped by Saturday — files dated Sat–Fri (inclusive) belong
    to the same week.
    """
    all_files = sorted(RAW_DIR.glob("*_raw_*.csv"))
    dates = set()
    for f in all_files:
        # Filename pattern:  reit_raw_YYYY-MM-DD[_partN].csv
        parts = f.stem.split("_")
        for p in parts:
            try:
                d = datetime.strptime(p, "%Y-%m-%d").date()
                dates.add(d)
            except ValueError:
                continue
    if not dates:
        return [], [], None, None

    # Normalize each date to its Saturday (weekday 5) — the scrape week anchor
    def to_saturday(d: date) -> date:
        # Python weekday: Mon=0 ... Sat=5, Sun=6
        delta = (d.weekday() - 5) % 7
        return d - pd.Timedelta(days=delta).to_pytimedelta()

    grouped = {}
    for d in dates:
        # Accept dates within 3 days after Saturday as part of that week
        wk = None
        for days_back in range(7):
            candidate = d - pd.Timedelta(days=days_back).to_pytimedelta()
            if candidate.weekday() == 5:  # Saturday
                wk = candidate
                break
        if wk is None:
            continue
        grouped.setdefault(wk, []).append(d)

    if len(grouped) < 2:
        # Just return latest week only
        latest_wk = max(grouped) if grouped else None
        pats = [f"*_raw_{d.isoformat()}*.csv" for d in grouped.get(latest_wk, [])]
        return pats, [], latest_wk, None

    sorted_wks = sorted(grouped.keys(), reverse=True)
    latest_wk, prior_wk = sorted_wks[0], sorted_wks[1]
    latest_pats = [f"*_raw_{d.isoformat()}*.csv" for d in grouped[latest_wk]]
    prior_pats = [f"*_raw_{d.isoformat()}*.csv" for d in grouped[prior_wk]]
    return latest_pats, prior_pats, latest_wk, prior_wk


# ── QA Checks ─────────────────────────────────────────────────────────────

def check_unit_counts(cur: pd.DataFrame, prior: pd.DataFrame) -> list[dict]:
    """Flag REITs with unusual unit-count WoW changes."""
    flags = []
    cs = cur.groupby("reit").size()
    ps = prior.groupby("reit").size()
    for reit in sorted(set(cs.index) | set(ps.index)):
        c = int(cs.get(reit, 0))
        p = int(ps.get(reit, 0))
        if p == 0:
            flags.append({"reit": reit, "severity": "WARN",
                          "check": "unit_count",
                          "msg": f"{reit}: new REIT this week ({c} units)"})
            continue
        pct = (c - p) / p * 100
        if abs(pct) > THRESH["unit_pct_change"]:
            flags.append({"reit": reit, "severity": "WARN",
                          "check": "unit_count",
                          "msg": f"{reit}: {p:,} -> {c:,} units ({pct:+.1f}%)"})
    return flags


def check_community_counts(cur: pd.DataFrame, prior: pd.DataFrame) -> list[dict]:
    flags = []
    for reit in sorted(cur["reit"].unique()):
        c = cur[cur["reit"] == reit]["community"].nunique()
        p = prior[prior["reit"] == reit]["community"].nunique() if reit in prior["reit"].values else 0
        if p == 0:
            continue
        chg = c - p
        if abs(chg) > THRESH["community_change"]:
            flags.append({"reit": reit, "severity": "WARN",
                          "check": "community_count",
                          "msg": f"{reit}: {p} -> {c} communities ({chg:+d})"})
    return flags


def check_concession_rate(cur: pd.DataFrame, prior: pd.DataFrame) -> list[dict]:
    flags = []
    for reit in sorted(cur["reit"].unique()):
        c_rate = cur[cur["reit"] == reit]["has_concession"].mean() * 100
        p_sub = prior[prior["reit"] == reit]
        if len(p_sub) == 0:
            continue
        p_rate = p_sub["has_concession"].mean() * 100
        delta = c_rate - p_rate
        if abs(delta) > THRESH["conc_rate_ppts_hard"]:
            flags.append({"reit": reit, "severity": "FAIL",
                          "check": "concession_rate",
                          "msg": f"{reit}: concession rate {p_rate:.1f}% -> {c_rate:.1f}% ({delta:+.1f} ppts)"})
        elif abs(delta) > THRESH["conc_rate_ppts_soft"]:
            flags.append({"reit": reit, "severity": "WARN",
                          "check": "concession_rate",
                          "msg": f"{reit}: concession rate {p_rate:.1f}% -> {c_rate:.1f}% ({delta:+.1f} ppts)"})
    return flags


def check_avg_rent(cur: pd.DataFrame, prior: pd.DataFrame) -> list[dict]:
    flags = []
    for reit in sorted(cur["reit"].unique()):
        c_rent = cur[cur["reit"] == reit]["rent"].mean()
        p_sub = prior[prior["reit"] == reit]
        if len(p_sub) == 0 or pd.isna(c_rent):
            continue
        p_rent = p_sub["rent"].mean()
        if pd.isna(p_rent) or p_rent == 0:
            continue
        pct = (c_rent / p_rent - 1) * 100
        if abs(pct) > THRESH["avg_rent_pct"]:
            flags.append({"reit": reit, "severity": "WARN",
                          "check": "avg_rent",
                          "msg": f"{reit}: avg rent ${p_rent:,.0f} -> ${c_rent:,.0f} ({pct:+.2f}%)"})
    return flags


def check_matched_rent(cur: pd.DataFrame, prior: pd.DataFrame) -> list[dict]:
    """Flag REITs whose matched-unit median rent moved beyond expectation."""
    flags = []
    cur = cur.copy()
    prior = prior.copy()
    cur["key"] = cur["reit"] + "|" + cur["unit_id"].astype(str)
    prior["key"] = prior["reit"] + "|" + prior["unit_id"].astype(str)
    matched = set(cur["key"]) & set(prior["key"])
    if not matched:
        return flags
    cm = cur[cur["key"].isin(matched)][["key", "reit", "rent"]]
    pm = prior[prior["key"].isin(matched)][["key", "rent"]]
    merged = cm.merge(pm, on="key", suffixes=("_cur", "_prior"))
    merged["pct"] = (merged["rent_cur"] / merged["rent_prior"] - 1) * 100
    for reit, grp in merged.groupby("reit"):
        med = grp["pct"].median()
        if abs(med) > THRESH["matched_median_rent_pct"]:
            flags.append({"reit": reit, "severity": "WARN",
                          "check": "matched_rent",
                          "msg": f"{reit}: matched-unit median rent WoW {med:+.2f}% (sample n={len(grp):,})"})
    return flags


def check_identical_text_flood(cur: pd.DataFrame) -> list[dict]:
    """
    Detect the 'scraper false-positive' signature where a massive share of
    concession rows for a REIT share identical concession_raw text.
    Classic case: AMH '25% off' deposit-offer false positive (1,946 rows).
    """
    flags = []
    conc = cur[cur["has_concession"] == True].copy()
    for reit, grp in conc.groupby("reit"):
        if len(grp) < THRESH["identical_text_min_n"]:
            continue
        raw_counts = grp["concession_raw"].fillna("").value_counts()
        if len(raw_counts) == 0:
            continue
        top_share = raw_counts.iloc[0] / len(grp)
        if top_share >= THRESH["identical_text_share"]:
            top_text = raw_counts.index[0]
            flags.append({"reit": reit, "severity": "FAIL",
                          "check": "identical_text_flood",
                          "msg": f"{reit}: {top_share:.1%} of {len(grp):,} concession rows share identical text: "
                                 f"{top_text!r}"})
    return flags


def check_identical_value_flood(cur: pd.DataFrame) -> list[dict]:
    """Detect REITs where almost all concession values are a single number."""
    flags = []
    conc = cur[cur["has_concession"] == True].copy()
    for reit, grp in conc.groupby("reit"):
        vals = grp["concession_value"].dropna()
        if len(vals) < THRESH["identical_text_min_n"]:
            continue
        top = vals.value_counts()
        share = top.iloc[0] / len(vals)
        if share >= THRESH["conc_value_single_value_share"]:
            flags.append({"reit": reit, "severity": "WARN",
                          "check": "identical_value_flood",
                          "msg": f"{reit}: {share:.1%} of {len(vals):,} concession values "
                                 f"are identical ({top.index[0]})"})
    return flags


def check_ner_rate_explosion(cur: pd.DataFrame, prior: pd.DataFrame) -> list[dict]:
    """REITs where NER coverage explodes WoW often indicates fake concessions."""
    flags = []
    for reit in sorted(cur["reit"].unique()):
        c = cur[cur["reit"] == reit]["effective_monthly_rent"].notna().mean() * 100
        p_sub = prior[prior["reit"] == reit]
        if len(p_sub) == 0:
            continue
        p = p_sub["effective_monthly_rent"].notna().mean() * 100
        delta = c - p
        if abs(delta) > 20.0:
            flags.append({"reit": reit, "severity": "WARN",
                          "check": "ner_rate_explosion",
                          "msg": f"{reit}: NER coverage {p:.1f}% -> {c:.1f}% ({delta:+.1f} ppts)"})
    return flags


# ── Main runner ───────────────────────────────────────────────────────────

def run_qa(cur_patterns: list[str], prior_patterns: list[str],
           latest_wk: date, prior_wk: date) -> tuple[list[dict], str]:
    cur = _load_week(cur_patterns)
    prior = _load_week(prior_patterns)

    if cur.empty:
        return [{"severity": "FAIL", "check": "load",
                 "msg": "No current-week data found"}], ""
    if prior.empty:
        return [{"severity": "WARN", "check": "load",
                 "msg": "No prior-week data found — WoW checks skipped"}], ""

    all_flags: list[dict] = []
    all_flags.extend(check_unit_counts(cur, prior))
    all_flags.extend(check_community_counts(cur, prior))
    all_flags.extend(check_concession_rate(cur, prior))
    all_flags.extend(check_avg_rent(cur, prior))
    all_flags.extend(check_matched_rent(cur, prior))
    all_flags.extend(check_ner_rate_explosion(cur, prior))
    all_flags.extend(check_identical_text_flood(cur))
    all_flags.extend(check_identical_value_flood(cur))

    # ── Build markdown report ─────────────────────────────────────────────
    lines = []
    lines.append(f"# Week-over-Week Data Quality Report")
    lines.append(f"")
    lines.append(f"- Current week: **{latest_wk}**  ({len(cur):,} rows)")
    lines.append(f"- Prior week:   **{prior_wk}**  ({len(prior):,} rows)")
    lines.append(f"- Generated:    {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"")

    fails = [f for f in all_flags if f["severity"] == "FAIL"]
    warns = [f for f in all_flags if f["severity"] == "WARN"]

    if not all_flags:
        lines.append(f"## [PASS] All checks passed")
        lines.append(f"")
    else:
        if fails:
            lines.append(f"## [FAIL] ({len(fails)})")
            for f in fails:
                lines.append(f"- **{f['check']}**: {f['msg']}")
            lines.append(f"")
        if warns:
            lines.append(f"## [WARN] ({len(warns)})")
            for f in warns:
                lines.append(f"- **{f['check']}**: {f['msg']}")
            lines.append(f"")

    # ── Summary table ─────────────────────────────────────────────────────
    lines.append(f"## WoW Summary by REIT")
    lines.append(f"")
    lines.append(f"| REIT | Units WoW | Conc Rate WoW | Avg Rent WoW | Matched Median WoW |")
    lines.append(f"|------|----------:|--------------:|-------------:|-------------------:|")

    cur["key"] = cur["reit"] + "|" + cur["unit_id"].astype(str)
    prior["key"] = prior["reit"] + "|" + prior["unit_id"].astype(str)
    matched = set(cur["key"]) & set(prior["key"])
    cm = cur[cur["key"].isin(matched)][["key", "reit", "rent"]]
    pm = prior[prior["key"].isin(matched)][["key", "rent"]]
    m = cm.merge(pm, on="key", suffixes=("_cur", "_prior"))
    m["pct"] = (m["rent_cur"] / m["rent_prior"] - 1) * 100

    for reit in sorted(cur["reit"].unique()):
        cu = len(cur[cur["reit"] == reit])
        pu = len(prior[prior["reit"] == reit])
        cr = cur[cur["reit"] == reit]["has_concession"].mean() * 100
        pr = prior[prior["reit"] == reit]["has_concession"].mean() * 100 if pu else 0
        cw_rent = cur[cur["reit"] == reit]["rent"].mean()
        pw_rent = prior[prior["reit"] == reit]["rent"].mean() if pu else 0
        rent_pct = (cw_rent / pw_rent - 1) * 100 if pw_rent else 0
        med = m[m["reit"] == reit]["pct"].median() if (m["reit"] == reit).any() else float("nan")

        u_pct = (cu - pu) / pu * 100 if pu else 0
        lines.append(
            f"| {reit} | {cu:,} ({u_pct:+.1f}%) | {pr:.1f}% → {cr:.1f}% ({cr - pr:+.1f}) | "
            f"${pw_rent:,.0f} → ${cw_rent:,.0f} ({rent_pct:+.2f}%) | "
            f"{med:+.2f}% |"
        )

    lines.append(f"")
    return all_flags, "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Weekly WoW Data Quality Check")
    parser.add_argument("--strict", action="store_true",
                        help="Exit non-zero on any FAIL (otherwise warnings only)")
    args = parser.parse_args()

    cur_pats, prior_pats, latest_wk, prior_wk = _find_latest_two_weeks()
    if not cur_pats:
        print("[QA] No scrape data found.")
        return 1

    flags, report = run_qa(cur_pats, prior_pats, latest_wk, prior_wk)

    out_path = LOG_DIR / f"wow_qa_{latest_wk.isoformat() if latest_wk else 'unknown'}.md"
    out_path.write_text(report, encoding="utf-8")
    # Print to stdout — handle non-UTF-8 terminals safely
    try:
        print(report)
    except UnicodeEncodeError:
        print(report.encode("ascii", errors="replace").decode("ascii"))
    print(f"\n[QA] Report written: {out_path}")

    fail_count = sum(1 for f in flags if f["severity"] == "FAIL")
    if args.strict and fail_count:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
