# Derived analysis series

CSV exports of the computed series behind the analysis workbook. Regenerated
by `build_excel.py` on every weekly run and committed by `weekly_run.py`
(STEP 7), so the repo carries the numbers behind the charts, not just the raw
scrapes.

## Format

Long format, one row per (date, series):

| column | meaning |
|---|---|
| `date` | Saturday-anchored week |
| `series` | REIT ticker, or macro market, depending on the file |
| `value` | the computed figure |
| `interpolated` | `True` = not observed (see below) |

## interpolated

Four weeks were never scraped — 2026-06-27, 07-04, 07-11 and 07-25 — because
the weekly run did not fire. Listings are a live snapshot, so there is nothing
to back-fill from; those points are linearly interpolated between the nearest
observed weeks on each side.

`interpolated` is also `True` where a single REIT has no value in a week the
others do (scraper miss, coverage gap, or a scraper methodology break).

**Never cite an `interpolated=True` row as an observation.** Filter them out
for any analysis that depends on measured values.

Interpolation never extrapolates: a series with no observation on one side of
a gap is omitted rather than guessed.

## Files

- `same_prop_avg_rent` / `same_prop_avg_ner` — count-weighted same-property
  levels ($) by REIT
- `index_*` — chain-linked same-property indices (base 100)
- `charts_concession_rate` — share of listings carrying a concession, by REIT
- `markets_<reit>_*` — per-REIT index by macro market
- `market_comparison_<market>_*` — cross-REIT index within one market

Same-property = units present in BOTH consecutive weeks, matched on `unit_id`.
Indices are chain-linked on weekly factors, so a missing factor does not shift
the level.
