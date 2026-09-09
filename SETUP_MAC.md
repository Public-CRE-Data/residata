# Moving the pipeline to a new machine (macOS)

Everything that matters lives in this repo. What does **not** transfer via
`git clone`, and why, is the point of this document. Work through it in
order — step 3 in particular must happen before any commit.

## 1. Prerequisites

```bash
xcode-select --install                      # git + build tools
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
brew install python@3.11 gh
brew install --cask google-chrome           # EQR scraper prefers system Chrome (Cloudflare)
```

## 2. Clone

```bash
cd ~
git clone https://github.com/Public-CRE-Data/residata.git reit-rental-scraper
cd reit-rental-scraper
```

## 3. Commit identity — DO THIS BEFORE ANYTHING ELSE

The publishing identity is stored in `.git/config`, which is **not** part of
a clone. A fresh clone has no identity, and if a *global* one with a personal
name is set, every push to this public repo will carry it.

```bash
git config user.name  "CRE Data"
git config user.email "noreply@public-cre-data.github.io"
git config --get user.name      # must print: CRE Data
```

Note the absence of `--global`: the identity is scoped to this repo only.

## 4. GitHub authentication

```bash
gh auth login        # HTTPS, browser flow; this also configures the credential helper
git push --dry-run   # must not prompt or fail
```

## 5. Python environment

```bash
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
python3 -m playwright install chromium
```

## 6. Prove the environment before scheduling anything

Rebuild from the committed data — this touches every dependency and every
data path without scraping:

```bash
python3 rebuild_summary_history.py     # ~10-15 min at 20+ weeks
python3 build_excel.py                 # writes REIT_Rental_Analysis_<date>.xlsx + data/derived/
python3 build_charts.py
```

If all three finish, the environment is complete.

## 7. Files that are NOT in the repo

Copy these from the old machine by hand (AirDrop, USB, cloud drive):

| What | Why not in git | Needed? |
|---|---|---|
| `render_charts.py`, `embed_charts.py` | never committed | **Yes** — generate the hero-chart PNGs. Already portable (`Path(__file__)`); no edits needed. |
| `build_workbook.py`, `build_workbook_v2.py`, `check_other.py` | contain machine-specific absolute paths | Optional legacy helpers. Replace the hardcoded `C:/Users/...` root with `os.path.dirname(os.path.abspath(__file__))` after copying. |
| `REIT_Rental_Analysis_*.xlsx`, `output/*.xlsx` | `*.xlsx` is gitignored | Optional. Historical dated snapshots; the *current* workbook regenerates from data (step 6). |
| `output/*.png` | build artefacts | Optional; regenerate with `render_charts.py` |
| `residata_cache/` | ~300 MB cache | No — regenerates |
| `logs/` | gitignored | No |

### If `~/reit-rental-scraper` already exists as a plain file copy

A folder copied from the old machine (no `.git` inside) blocks `git clone`.
Keep the copy — it holds gitignored snapshots and caches — and give it a
repository instead:

```bash
git clone https://github.com/Public-CRE-Data/residata.git /tmp/residata-clone
mv /tmp/residata-clone/.git ~/reit-rental-scraper/.git && rm -rf /tmp/residata-clone
cd ~/reit-rental-scraper
git status          # review before discarding: the copy may hold uncommitted work
```

Then do step 3 before any commit.

## 8. Claude Code memory (if you use it)

Claude Code keeps per-project memory under
`~/.claude/projects/<slug>/memory/`, where `<slug>` is derived from the
working-directory path — so it changes across machines. Copy the `memory/`
folder from the old machine into the new slug's directory (open Claude Code
in the project once to create it, then copy the files in).

## 9. Schedule the weekly run

```bash
./install_launchd.sh
launchctl list | grep reit-weekly       # confirm registered
```

Saturday 23:00 local, same as the Windows task. launchd fires a missed run on
next wake automatically; it cannot fire if the machine was shut down.

## 10. Retire the old machine's schedule

If the Windows laptop stays powered on, **both** schedulers will scrape the
same week and produce duplicate files. Disable the Windows task before the
first Saturday:

```powershell
Disable-ScheduledTask -TaskName "REIT_Weekly_Scrape"
```

## Verify the first scheduled week

After the first Saturday, confirm the run happened and covered all 8 REITs:

```bash
tail -30 logs/weekly_$(date +%F).log
python3 -c "import pandas as pd; h=pd.read_csv('data/summary/summary_history.csv'); print(sorted(h.scrape_date.unique())[-2:])"
```
