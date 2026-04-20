"""
Historical outlier scan across all scrape weeks.

Goal: identify suspicious outliers and data-error signatures across the full
CSV archive, not just the latest WoW pair.

Outputs a markdown report to logs/outlier_scan.md.
"""

import pathlib
import pandas as pd
import numpy as np
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

BASE_DIR = pathlib.Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "raw"
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)


def load_all() -> pd.DataFrame:
    files = sorted(RAW_DIR.glob("*_raw_*.csv"))
    dfs = []
    for f in files:
        try:
            d = pd.read_csv(f, low_memory=False)
        except Exception as e:
            print(f"  [skip] {f.name}: {e}")
            continue
        dfs.append(d)
    df = pd.concat(dfs, ignore_index=True)
    df["scrape_date"] = pd.to_datetime(df["scrape_date"])
    # Bucket to week-ending-Saturday
    df["scrape_wk"] = df["scrape_date"].apply(
        lambda d: (d + pd.Timedelta(days=(5 - d.weekday()) % 7)).date()
    )
    return df


def build_report(df: pd.DataFrame) -> str:
    lines = []
    lines.append("# Historical Outlier Scan")
    lines.append("")
    lines.append(f"- Rows scanned: {len(df):,}")
    lines.append(f"- Date range:   {df['scrape_date'].min().date()} to {df['scrape_date'].max().date()}")
    lines.append(f"- Weeks:        {df['scrape_wk'].nunique()}")
    lines.append(f"- REITs:        {df['reit'].nunique()}")
    lines.append(f"- Generated:    {datetime.now().isoformat(timespec='seconds')}")
    lines.append("")

    # ── 1. Bare-percent concession_raw (the AMH-style false positive) ─────
    lines.append("## 1. Bare-percent concession_raw (AMH-style false-positive signature)")
    lines.append("")
    lines.append("Pattern: concession_raw is exactly `N% off` with no rent/month/week context.")
    lines.append("These are almost always security-deposit / app-fee / admin-fee waivers, not rent concessions.")
    lines.append("")
    bare_mask = df["concession_raw"].fillna("").str.match(r"^\s*\d+\s*%\s*off\s*$", case=False)
    bare = df[bare_mask]
    if len(bare):
        summary = bare.groupby(["reit", "scrape_wk"]).size().unstack(fill_value=0)
        lines.append("| REIT | " + " | ".join(str(c) for c in summary.columns) + " |")
        lines.append("|------|" + "|".join(["------:"] * len(summary.columns)) + "|")
        for reit, row in summary.iterrows():
            lines.append(f"| {reit} | " + " | ".join(f"{v:,}" if v else "-" for v in row) + " |")
        lines.append("")
        lines.append(f"**Total bare-percent rows: {len(bare):,}**")
    else:
        lines.append("_None found._")
    lines.append("")

    # ── 2. Extreme individual rents (per-REIT) ────────────────────────────
    lines.append("## 2. Extreme individual rents (|z-score| > 5 within REIT × scrape week)")
    lines.append("")
    lines.append("Per-REIT, per-week z-scores on `rent`. Rents this far from the mean are")
    lines.append("usually penthouse/ultra-luxury units or scraper parse errors.")
    lines.append("")
    df["rent_z"] = df.groupby(["reit", "scrape_wk"])["rent"].transform(
        lambda s: (s - s.mean()) / s.std(ddof=0) if s.std(ddof=0) > 0 else 0
    )
    extreme = df[df["rent_z"].abs() > 5].sort_values("rent_z", ascending=False)
    lines.append(f"- Rows flagged: **{len(extreme):,}** of {len(df):,}  "
                 f"({len(extreme) / max(len(df), 1) * 100:.2f}%)")
    lines.append("")
    if len(extreme):
        lines.append("**Top 20 highest z-scores:**")
        lines.append("")
        lines.append("| REIT | Week | Market | Community | Rent | Sqft | Z-score |")
        lines.append("|------|------|--------|-----------|-----:|-----:|--------:|")
        for _, r in extreme.head(20).iterrows():
            lines.append(
                f"| {r['reit']} | {r['scrape_wk']} | {str(r.get('market',''))[:20]} | "
                f"{str(r.get('community',''))[:30]} | ${r['rent']:,.0f} | {r.get('sqft',0):.0f} | "
                f"{r['rent_z']:.1f} |"
            )
        lines.append("")
        lines.append("**Bottom 10 lowest z-scores:**")
        lines.append("")
        lines.append("| REIT | Week | Market | Community | Rent | Sqft | Z-score |")
        lines.append("|------|------|--------|-----------|-----:|-----:|--------:|")
        for _, r in extreme.sort_values("rent_z").head(10).iterrows():
            lines.append(
                f"| {r['reit']} | {r['scrape_wk']} | {str(r.get('market',''))[:20]} | "
                f"{str(r.get('community',''))[:30]} | ${r['rent']:,.0f} | {r.get('sqft',0):.0f} | "
                f"{r['rent_z']:.1f} |"
            )
        lines.append("")

    # ── 3. Impossible rent (<$500 or >$20K/mo) ────────────────────────────
    lines.append("## 3. Implausible rent levels")
    lines.append("")
    impossible = df[(df["rent"] < 500) | (df["rent"] > 20000)]
    lines.append(f"- Rows with rent < $500 or > $20,000: **{len(impossible):,}**")
    if len(impossible):
        by_reit = impossible.groupby("reit").agg(
            n=("unit_id", "count"),
            min_rent=("rent", "min"),
            max_rent=("rent", "max"),
        )
        lines.append("")
        lines.append("| REIT | Count | Min Rent | Max Rent |")
        lines.append("|------|------:|---------:|---------:|")
        for reit, row in by_reit.iterrows():
            lines.append(f"| {reit} | {int(row['n'])} | ${row['min_rent']:,.0f} | ${row['max_rent']:,.0f} |")
        lines.append("")

    # ── 4. Implausible sqft ───────────────────────────────────────────────
    lines.append("## 4. Implausible sqft values")
    lines.append("")
    bad_sqft = df[(df["sqft"] < 200) | (df["sqft"] > 10000)]
    lines.append(f"- Rows with sqft < 200 or > 10,000: **{len(bad_sqft):,}**")
    if len(bad_sqft):
        by_reit = bad_sqft.groupby("reit").agg(
            n=("unit_id", "count"),
            min_sqft=("sqft", "min"),
            max_sqft=("sqft", "max"),
        )
        lines.append("")
        lines.append("| REIT | Count | Min Sqft | Max Sqft |")
        lines.append("|------|------:|---------:|---------:|")
        for reit, row in by_reit.iterrows():
            lines.append(f"| {reit} | {int(row['n'])} | {row['min_sqft']:,.0f} | {row['max_sqft']:,.0f} |")
        lines.append("")

    # ── 5. Rent jumps week-over-week on matched units ─────────────────────
    lines.append("## 5. Matched-unit rent jumps (|WoW %| > 15%)")
    lines.append("")
    lines.append("Same unit_id seen in consecutive weeks with >15% rent move.")
    lines.append("Real weekly rent movement is rarely more than ±5%.")
    lines.append("")
    df = df.sort_values(["reit", "unit_id", "scrape_wk"])
    df["prev_rent"] = df.groupby(["reit", "unit_id"])["rent"].shift(1)
    df["prev_wk"] = df.groupby(["reit", "unit_id"])["scrape_wk"].shift(1)
    df["wow_pct"] = (df["rent"] / df["prev_rent"] - 1) * 100
    jumps = df[df["wow_pct"].abs() > 15].dropna(subset=["prev_rent"])

    lines.append(f"- Total matched-unit rent jumps > 15%: **{len(jumps):,}**")
    if len(jumps):
        by_reit = jumps.groupby("reit").size().to_dict()
        lines.append("- By REIT: " + ", ".join(f"{k}: {v}" for k, v in sorted(by_reit.items())))
        lines.append("")
        lines.append("**Top 20 largest absolute jumps:**")
        lines.append("")
        lines.append("| REIT | Unit | Market | Week | Prior Rent | New Rent | WoW % |")
        lines.append("|------|------|--------|------|-----------:|---------:|------:|")
        for _, r in jumps.assign(abs_pct=jumps["wow_pct"].abs()).sort_values("abs_pct", ascending=False).head(20).iterrows():
            lines.append(
                f"| {r['reit']} | {str(r['unit_id'])[:22]} | {str(r.get('market',''))[:18]} | "
                f"{r['scrape_wk']} | ${r['prev_rent']:,.0f} | ${r['rent']:,.0f} | {r['wow_pct']:+.1f}% |"
            )
        lines.append("")

    # ── 6. REIT-week snapshots: concession-rate step changes ──────────────
    lines.append("## 6. Concession-rate step changes (|WoW ppts| > 15)")
    lines.append("")
    lines.append("Historical REIT-level concession rates across all weeks, flagging step changes.")
    lines.append("")
    weekly = df.groupby(["reit", "scrape_wk"])["has_concession"].mean().unstack().round(3)
    lines.append("| REIT | " + " | ".join(str(c) for c in weekly.columns) + " |")
    lines.append("|------|" + "|".join(["------:"] * len(weekly.columns)) + "|")
    for reit, row in weekly.iterrows():
        lines.append(f"| {reit} | " + " | ".join(
            f"{v*100:.1f}%" if not pd.isna(v) else "-" for v in row
        ) + " |")
    lines.append("")

    step_flags = []
    for reit, row in weekly.iterrows():
        prev_val = None
        for wk, val in row.items():
            if pd.isna(val) or prev_val is None:
                prev_val = val
                continue
            ppts = (val - prev_val) * 100
            if abs(ppts) > 15:
                step_flags.append((reit, wk, prev_val * 100, val * 100, ppts))
            prev_val = val

    if step_flags:
        lines.append("**Step changes > 15 ppts:**")
        lines.append("")
        lines.append("| REIT | Week | Prior | Current | WoW ppts |")
        lines.append("|------|------|------:|--------:|---------:|")
        for reit, wk, p, c, d in step_flags:
            lines.append(f"| {reit} | {wk} | {p:.1f}% | {c:.1f}% | {d:+.1f} |")
        lines.append("")
    else:
        lines.append("_No step changes > 15 ppts detected._")
        lines.append("")

    # ── 7. NER discount extremes ──────────────────────────────────────────
    lines.append("## 7. Net effective rent extremes")
    lines.append("")
    m = df["effective_monthly_rent"].notna() & df["rent"].notna() & (df["rent"] > 0)
    d2 = df[m].copy()
    d2["discount_pct"] = (1 - d2["effective_monthly_rent"] / d2["rent"]) * 100
    neg = d2[d2["discount_pct"] < -1]  # NER > gross — clearly wrong
    huge = d2[d2["discount_pct"] > 50]  # implausible discount
    lines.append(f"- NER > asking rent (negative discount): **{len(neg):,}**")
    lines.append(f"- Discount > 50% (suspicious): **{len(huge):,}**")
    if len(huge):
        lines.append("")
        lines.append("**Top 15 largest NER discounts:**")
        lines.append("")
        lines.append("| REIT | Week | Market | Rent | NER | Discount | Raw text |")
        lines.append("|------|------|--------|-----:|----:|---------:|----------|")
        for _, r in huge.sort_values("discount_pct", ascending=False).head(15).iterrows():
            lines.append(
                f"| {r['reit']} | {r['scrape_wk']} | {str(r.get('market',''))[:18]} | "
                f"${r['rent']:,.0f} | ${r['effective_monthly_rent']:,.0f} | "
                f"{r['discount_pct']:.1f}% | {str(r.get('concession_raw',''))[:40]} |"
            )
        lines.append("")

    return "\n".join(lines)


def main():
    print("[Scan] Loading all scrape CSVs...")
    df = load_all()
    print(f"[Scan] Loaded {len(df):,} rows across {df['scrape_wk'].nunique()} weeks.")
    report = build_report(df)
    out = LOG_DIR / "outlier_scan.md"
    out.write_text(report, encoding="utf-8")
    print(f"[Scan] Report written: {out}")
    print(f"[Scan] {report.count('**') // 2} key sections.")


if __name__ == "__main__":
    main()
