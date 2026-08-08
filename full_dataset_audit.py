"""Full-dataset audit across every REIT x week in data/raw.

Written after the INVH `rent` defect (scraper attached neighbouring homes'
prices to listings, so rent was statistically independent of the home's own
size). That bug was invisible to the existing integrity audit because every
aggregate looked stable -- it was stable *garbage*. These checks target the
bug CLASS, not the single instance.

Checks
  A  corr(rent, sqft) / corr(rent, beds) per REIT x week
       Near-zero correlation => rent is not attached to the right entity.
       AMH acts as the SFR control for INVH.
  B  unit_id attribute stability across consecutive weeks
       Same unit_id whose sqft/beds changes => ID collision / mis-keying.
  C  rent-psf level breaks week over week (step changes)
  D  duplicate unit_id within a single week
  E  null rates on rent / sqft / beds
  F  rent distribution sanity (implausible tails)
  G  concession parse coverage (has_concession but unparseable value)
"""

import glob
import os
import re
from collections import defaultdict

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(BASE, "data", "raw")

REITS = ["amh", "avb", "cpt", "eqr", "ess", "invh", "maa", "udr"]

# Saturday-anchored week, matching the rest of the pipeline.
def week_of(d: pd.Timestamp) -> str:
    return (d - pd.Timedelta(days=(d.weekday() - 5) % 7)).strftime("%Y-%m-%d")


def load(reit: str) -> pd.DataFrame:
    frames = []
    for f in sorted(glob.glob(os.path.join(RAW, f"{reit}_raw_*.csv"))):
        m = re.search(r"_raw_(\d{4}-\d{2}-\d{2})", f)
        if not m:
            continue
        try:
            d = pd.read_csv(f, low_memory=False)
        except Exception:
            continue
        if d.empty:
            continue
        d["_week"] = week_of(pd.Timestamp(m.group(1)))
        frames.append(d)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    # Same dedup key the rebuild uses.
    if {"unit_id", "_week"} <= set(df.columns):
        df = df.drop_duplicates(subset=["unit_id", "_week"], keep="last")
    return df


findings = []          # (severity, check, message)
corr_table = []        # rows for the A summary


def add(sev, check, msg):
    findings.append((sev, check, msg))


for reit in REITS:
    df = load(reit)
    if df.empty:
        add("WARN", "load", f"{reit.upper()}: no raw files found")
        continue

    for col in ("rent", "sqft", "beds"):
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

    weeks = sorted(df["_week"].unique())

    # ---- E: null rates -------------------------------------------------
    for c in ("rent", "sqft", "beds"):
        nullpct = df[c].isna().mean() * 100
        if nullpct > 25:
            add("WARN", "null_rate",
                f"{reit.upper()}: {c} null in {nullpct:.0f}% of rows overall")

    # ---- A: correlation per week --------------------------------------
    for wk in weeks:
        d = df[(df["_week"] == wk)].dropna(subset=["rent", "sqft"])
        d = d[(d["sqft"] > 300) & (d["rent"] > 300)]
        if len(d) < 150:
            continue
        cs = np.corrcoef(d["sqft"], d["rent"])[0, 1]
        psf = (d["rent"] / d["sqft"]).mean()

        # Pooled correlation is confounded by geographic mix: a REIT spread
        # across expensive coastal and cheap Sun Belt markets shows a low
        # pooled figure even when rent tracks size perfectly *within* each
        # market. UDR pooled 0.22 -> 0.55 within-market is the worked example.
        # So judge on the within-market (market-demeaned) correlation.
        g = d.groupby("market")
        dd = d.assign(_r=d["rent"] - g["rent"].transform("mean"),
                      _s=d["sqft"] - g["sqft"].transform("mean"))
        dd = dd[g["rent"].transform("size") >= 30]
        cw = np.corrcoef(dd["_s"], dd["_r"])[0, 1] if len(dd) > 100 else np.nan

        corr_table.append(dict(reit=reit.upper(), week=wk, n=len(d),
                               corr_sqft=cs, corr_within=cw, psf=psf))

        # Observed healthy range across 7 clean REITs: 0.55 - 0.80.
        # Broken INVH sat at 0.21 even within market.
        if not np.isnan(cw) and cw < 0.40:
            add("FAIL", "entity_mismatch",
                f"{reit.upper()} {wk}: within-market corr(rent,sqft)={cw:.3f} "
                f"(pooled {cs:.3f}, n={len(d)}) -- rent may be attached to the "
                f"wrong listing")

    # ---- C: psf level breaks ------------------------------------------
    sub = [r for r in corr_table if r["reit"] == reit.upper()]
    for a, b in zip(sub, sub[1:]):
        if a["psf"] and b["psf"]:
            chg = b["psf"] / a["psf"] - 1
            if abs(chg) > 0.05:
                add("FAIL" if abs(chg) > 0.15 else "WARN", "psf_break",
                    f"{reit.upper()} {a['week']} -> {b['week']}: "
                    f"rent-psf {a['psf']:.3f} -> {b['psf']:.3f} ({chg*100:+.1f}%)")

    # ---- D: duplicate unit_id within a week ---------------------------
    if "unit_id" in df.columns:
        for wk in weeks:
            d = df[df["_week"] == wk]
            dup = d["unit_id"].duplicated().sum()
            if dup > 0:
                add("WARN", "dup_unit_id",
                    f"{reit.upper()} {wk}: {dup} duplicate unit_id rows")

    # ---- B: attribute stability for the same unit_id ------------------
    if "unit_id" in df.columns:
        flips = 0
        checked = 0
        for a_wk, b_wk in zip(weeks, weeks[1:]):
            A = df[df["_week"] == a_wk][["unit_id", "sqft", "beds"]]
            B = df[df["_week"] == b_wk][["unit_id", "sqft", "beds"]]
            m = A.merge(B, on="unit_id", suffixes=("_a", "_b")).dropna(
                subset=["sqft_a", "sqft_b"])
            if m.empty:
                continue
            checked += len(m)
            flips += (m["sqft_a"] != m["sqft_b"]).sum()
        if checked:
            pct = flips / checked * 100
            if pct > 2:
                add("FAIL" if pct > 10 else "WARN", "id_collision",
                    f"{reit.upper()}: same unit_id changed sqft in {pct:.1f}% "
                    f"of {checked:,} consecutive-week matches")

    # ---- F: rent tails -------------------------------------------------
    r = df["rent"].dropna()
    if len(r):
        if (r < 400).sum() > 0.01 * len(r):
            add("WARN", "rent_tail",
                f"{reit.upper()}: {(r<400).sum()} rows with rent < $400")
        if (r > 20000).sum() > 0:
            add("WARN", "rent_tail",
                f"{reit.upper()}: {(r>20000).sum()} rows with rent > $20,000")

    # ---- G: concession parse coverage ---------------------------------
    if "has_concession" in df.columns and "concession_value" in df.columns:
        hc = df["has_concession"].fillna(False).astype(bool)
        if hc.sum():
            unparsed = (hc & df["concession_value"].isna()).sum() / hc.sum() * 100
            if unparsed > 20:
                add("WARN", "conc_parse",
                    f"{reit.upper()}: {unparsed:.0f}% of concession rows have "
                    f"no parsed value")


# ------------------------------------------------------------------ report
ct = pd.DataFrame(corr_table)

print("=" * 78)
print("CHECK A - WITHIN-MARKET corr(rent, sqft) by REIT x week  [low => wrong entity]")
print("=" * 78)
if not ct.empty:
    piv = ct.pivot_table(index="week", columns="reit", values="corr_within")
    print(piv.round(3).to_string())
    print()
    print("Per-REIT mean WITHIN-MARKET corr(rent,sqft):")
    print(ct.groupby("reit")["corr_within"].mean().round(3).sort_values().to_string())

print()
print("=" * 78)
print("FINDINGS")
print("=" * 78)
order = {"FAIL": 0, "WARN": 1}
findings.sort(key=lambda x: (order.get(x[0], 9), x[1]))
nf = sum(1 for f in findings if f[0] == "FAIL")
nw = sum(1 for f in findings if f[0] == "WARN")
print(f"{nf} FAIL, {nw} WARN\n")
by = defaultdict(list)
for sev, check, msg in findings:
    by[(sev, check)].append(msg)
for (sev, check), msgs in by.items():
    print(f"[{sev}] {check} ({len(msgs)})")
    for m in msgs[:14]:
        print(f"   - {m}")
    if len(msgs) > 14:
        print(f"   ... and {len(msgs)-14} more")
    print()
