#!/usr/bin/env bash
# weekly_run.sh — macOS/Linux entry point for the weekly pipeline.
# Counterpart to weekly_run.bat (Windows). Invoked by launchd; see
# install_launchd.sh and SETUP_MAC.md.
set -euo pipefail
cd "$(dirname "$0")"

# launchd runs with a minimal PATH; make Homebrew Python resolvable.
export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"

exec python3 weekly_run.py
