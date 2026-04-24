"""
Scan every REIT x macro_market for suspicious WoW moves on gross rent and
net effective rent between the two most recent scrape weeks.

Flags when the matched-unit change diverges materially from the aggregate
change (indicating composition / mix shift), and when either magnitude is
above expected weekly noise.
"""
import argparse
import pathlib
import re
from datetime import datetime, timedelta

import pandas as pd
import warnings
warnings.filterwarnings("ignore")

BASE_DIR = pathlib.Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "raw"
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

import sys
sys.path.insert(0, str(BASE_DIR))
from build_excel import _resolve_macro_market

_DATE_RE = re.compile(r"_raw_(\d{4}-\d{2}-\d{2})")


def find_latest_two_weeks():
    """Return (curr_files, prior_files, curr_anchor, prior_anchor)."""
    files = sorted(RAW_DIR.glob("*_raw_*.csv"))
    buckets = {}
    for f in files:
        m = _DATE_RE.search(f.stem)
        if not m:
            continue
        d = datetime.strptime(m.group(1), "%Y-%m-%d").date()
        wk = d - timedelta(days=(d.weekday() - 5) % 7)
        buckets.setdefault(wk, []).append(f)
    if len(buckets) < 2:
        return [], [], None, None
    weeks = sorted(buckets, reverse=True)
    return buckets[weeks[0]], buckets[weeks[1]], weeks[0], weeks[1]


def load_dedup(files):
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df["scrape_date"] = pd.to_datetime(df["scrape_date"])
    df = (df.sort_values("scrape_date")
            .groupby(["reit", "unit_id"], as_index=False)
            .last())
    df["macro_market"] = df["market"].apply(_resolve_macro_market)

    # Apply AMH bare-percent fix (matches build_excel.py logic)
    bare_mask = (
        (df["reit"] == "AMH")
        & df["concession_raw"].fillna("").astype(str).str.match(r"^\s*\d+\s*%\s*off\s*$", case=False)
    )
    if bare_mask.any():
        for c in ["has_concession", "concession_hardness", "concession_raw",
                  "concession_type", "concession_value",
                  "concession_pct_lease_value", "concession_pct_lease_term",
                  "effective_monthly_rent"]:
            if c in df.columns:
                df.loc[bare_mask, c] = None
        df["has_concession"] = df["has_concession"].fillna(False).astype(bool)

    # NER coalesce: no-concession units get NER = gross rent.
    fill = (~df["has_concession"].astype(bool)) & df["effective_monthly_rent"].isna() & df["rent"].notna()
    df.loc[fill, "effective_monthly_rent"] = df.loc[fill, "rent"]
    return df


def scan(curr, prior):
    """Return a DataFrame of (reit, market) rows with agg vs matched deltas."""
    # Compute all aggregates by reit x macro_market
    def agg(df):
        return df.groupby(["reit", "macro_market"]).agg(
            units=("unit_id", "count"),
            avg_rent=("rent", "mean"),
            avg_ner=("effective_monthly_rent", "mean"),
            conc_rate=("has_concession", "mean"),
            ner_coverage=("effective_monthly_rent", lambda x: x.notna().mean()),
        )

    ca = agg(curr).add_suffix("_cur")
    pa = agg(prior).add_suffix("_pr")
    agg_df = ca.join(pa, how="outer").reset_index()
    agg_df["units_cur"] = agg_df["units_cur"].fillna(0).astype(int)
    agg_df["units_pr"] = agg_df["units_pr"].fillna(0).astype(int)
    agg_df["rent_wow_pct"] = (agg_df["avg_rent_cur"] / agg_df["avg_rent_pr"] - 1) * 100
    agg_df["ner_wow_pct"] = (agg_df["avg_ner_cur"] / agg_df["avg_ner_pr"] - 1) * 100
    agg_df["conc_wow_ppts"] = (agg_df["conc_rate_cur"] - agg_df["conc_rate_pr"]) * 100

    # Matched-unit rent and NER moves
    curr2 = curr[["reit", "unit_id", "macro_market", "rent",
                  "effective_monthly_rent", "has_concession"]].rename(
        columns={"rent": "rent_cur", "effective_monthly_rent": "ner_cur",
                 "has_concession": "hc_cur"})
    prior2 = prior[["reit", "unit_id", "rent",
                    "effective_monthly_rent", "has_concession"]].rename(
        columns={"rent": "rent_pr", "effective_monthly_rent": "ner_pr",
                 "has_concession": "hc_pr"})
    m = curr2.merge(prior2, on=["reit", "unit_id"], how="inner")

    # Matched rent: drop rows where rent is null either side
    mr = m.dropna(subset=["rent_cur", "rent_pr"])
    matched_rent = mr.groupby(["reit", "macro_market"]).apply(
        lambda g: pd.Series({
            "matched_units": len(g),
            "matched_rent_avg_cur": g["rent_cur"].mean(),
            "matched_rent_avg_pr": g["rent_pr"].mean(),
            "matched_rent_wow_pct": (g["rent_cur"].mean() / g["rent_pr"].mean() - 1) * 100
            if g["rent_pr"].mean() else None,
            "matched_rent_median_wow_pct": ((g["rent_cur"] / g["rent_pr"] - 1) * 100).median(),
        })
    ).reset_index()

    # Matched NER: only units with NER on BOTH sides
    mn = m.dropna(subset=["ner_cur", "ner_pr"])
    matched_ner = mn.groupby(["reit", "macro_market"]).apply(
        lambda g: pd.Series({
            "matched_ner_n": len(g),
            "matched_ner_avg_cur": g["ner_cur"].mean(),
            "matched_ner_avg_pr": g["ner_pr"].mean(),
            "matched_ner_wow_pct": (g["ner_cur"].mean() / g["ner_pr"].mean() - 1) * 100
            if g["ner_pr"].mean() else None,
        })
    ).reset_index()

    # Concession flips
    flips = m.groupby(["reit", "macro_market"]).apply(
        lambda g: pd.Series({
            "new_conc": int(((g["hc_pr"] == False) & (g["hc_cur"] == True)).sum()),
            "lost_conc": int(((g["hc_pr"] == True) & (g["hc_cur"] == False)).sum()),
            "both_conc": int(((g["hc_pr"] == True) & (g["hc_cur"] == True)).sum()),
        })
    ).reset_index()

    out = (agg_df
           .merge(matched_rent, on=["reit", "macro_market"], how="left")
           .merge(matched_ner, on=["reit", "macro_market"], how="left")
           .merge(flips, on=["reit", "macro_market"], how="left"))
    return out


def flag_suspicious(out, min_units=30):
    """Return subset of rows with suspicious moves."""
    suspects = []

    for _, r in out.iterrows():
        # Skip tiny markets — noise dominates
        if pd.isna(r.get("matched_units")) or r["matched_units"] < min_units:
            # Still flag if the AGG change is dramatic
            if pd.notna(r.get("rent_wow_pct")) and abs(r["rent_wow_pct"]) >= 5:
                suspects.append({
                    **r.to_dict(),
                    "flag": f"TINY_MKT_LARGE_RENT_MOVE ({r['rent_wow_pct']:+.1f}%)",
                })
            continue

        flags = []
        # 1. Aggregate vs matched rent divergence > 1.5 ppts
        if (pd.notna(r.get("rent_wow_pct")) and pd.notna(r.get("matched_rent_wow_pct"))):
            diff = abs(r["rent_wow_pct"] - r["matched_rent_wow_pct"])
            if diff >= 1.5:
                flags.append(
                    f"RENT_MIX_SHIFT (agg {r['rent_wow_pct']:+.2f}% vs matched {r['matched_rent_wow_pct']:+.2f}%)"
                )

        # 2. Large matched rent move (>3% WoW)
        if pd.notna(r.get("matched_rent_wow_pct")) and abs(r["matched_rent_wow_pct"]) >= 3:
            flags.append(f"MATCHED_RENT_MOVE ({r['matched_rent_wow_pct']:+.2f}%)")

        # 3. Very large matched NER move (>5% WoW) with n >= 30
        if (pd.notna(r.get("matched_ner_wow_pct"))
                and pd.notna(r.get("matched_ner_n"))
                and r["matched_ner_n"] >= 30
                and abs(r["matched_ner_wow_pct"]) >= 5):
            flags.append(
                f"MATCHED_NER_MOVE ({r['matched_ner_wow_pct']:+.2f}%, n={int(r['matched_ner_n'])})"
            )

        # 4. Big concession flip (>30% of matched units changed conc status)
        total_match = r["matched_units"] or 1
        flip_share = (r.get("new_conc", 0) + r.get("lost_conc", 0)) / total_match
        if flip_share >= 0.30:
            flags.append(
                f"BIG_CONC_FLIP ({int(r['new_conc'])} new + {int(r['lost_conc'])} lost / {int(total_match)} matched)"
            )

        if flags:
            suspects.append({**r.to_dict(), "flag": " | ".join(flags)})

    return pd.DataFrame(suspects)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-units", type=int, default=30,
                    help="Minimum matched units to run matched-unit checks")
    ap.add_argument("--reit", type=str, default=None, help="Filter to one REIT")
    args = ap.parse_args()

    cur_files, prior_files, curr_wk, prior_wk = find_latest_two_weeks()
    if not cur_files or not prior_files:
        print("[Scan] Not enough data for WoW comparison.")
        return

    print(f"[Scan] Current week: {curr_wk}   ({len(cur_files)} files)")
    print(f"[Scan] Prior week:   {prior_wk}  ({len(prior_files)} files)")

    curr = load_dedup(cur_files)
    prior = load_dedup(prior_files)
    print(f"[Scan] Loaded {len(curr):,} current, {len(prior):,} prior rows.")

    if args.reit:
        curr = curr[curr["reit"] == args.reit.upper()]
        prior = prior[prior["reit"] == args.reit.upper()]

    out = scan(curr, prior)
    suspects = flag_suspicious(out, min_units=args.min_units)

    if suspects.empty:
        print("[Scan] No suspicious moves flagged.")
    else:
        print(f"[Scan] {len(suspects)} flagged rows:\n")
        for reit, grp in suspects.groupby("reit"):
            print(f"===== {reit} =====")
            for _, r in grp.sort_values("macro_market").iterrows():
                mkt = r["macro_market"]
                mu = int(r.get("matched_units") or 0)
                rent_wow = r.get("rent_wow_pct", float("nan"))
                mr = r.get("matched_rent_wow_pct", float("nan"))
                ner_n_raw = r.get("matched_ner_n", 0)
                ner_n = 0 if pd.isna(ner_n_raw) else int(ner_n_raw)
                mner = r.get("matched_ner_wow_pct", float("nan"))
                rent_wow_s = f"{rent_wow:+.2f}%" if pd.notna(rent_wow) else "  n/a "
                mr_s = f"{mr:+.2f}%" if pd.notna(mr) else "  n/a "
                mner_s = f"{mner:+.2f}%" if pd.notna(mner) else "  n/a "
                print(f"  {mkt:<22}  units={mu:>4}  ner_n={ner_n:>3}  "
                      f"rent_agg={rent_wow_s}  rent_matched={mr_s}  "
                      f"ner_matched={mner_s}")
                print(f"     -> {r['flag']}")
            print()

    # Write full CSV
    out_path = LOG_DIR / f"wow_market_scan_{curr_wk.isoformat()}.csv"
    out.to_csv(out_path, index=False)
    print(f"[Scan] Full data: {out_path}")


if __name__ == "__main__":
    main()
