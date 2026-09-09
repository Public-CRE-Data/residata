#!/usr/bin/env bash
# install_launchd.sh — register the weekly scrape with macOS launchd.
#
# The macOS equivalent of the Windows "REIT_Weekly_Scrape" scheduled task.
# Saturday 23:00 local, matching the Windows schedule so the Saturday week
# anchor is unchanged.
#
# launchd behaviour worth knowing (it differs from Task Scheduler):
#   * A StartCalendarInterval job that was missed because the machine was
#     asleep runs as soon as it wakes. This is the behaviour we had to turn
#     on by hand on Windows (StartWhenAvailable); launchd does it by default.
#   * A missed job is NOT run if the machine was fully shut down at the
#     time. Nothing on any OS can fix that.
#   * Jobs run whether or not the laptop is on battery.
#
# Usage:  ./install_launchd.sh           install / reinstall
#         ./install_launchd.sh --remove  unload and delete
set -euo pipefail

LABEL="io.public-cre-data.reit-weekly-scrape"
REPO="$(cd "$(dirname "$0")" && pwd)"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
LOGDIR="$REPO/logs"

if [[ "${1:-}" == "--remove" ]]; then
  launchctl bootout "gui/$(id -u)" "$PLIST" 2>/dev/null || true
  rm -f "$PLIST"
  echo "removed $LABEL"
  exit 0
fi

mkdir -p "$LOGDIR" "$HOME/Library/LaunchAgents"

# The repo path is discovered at install time and written in absolutely,
# because launchd does not expand ~ or $HOME inside a plist.
cat > "$PLIST" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>            <string>$LABEL</string>
  <key>ProgramArguments</key>
  <array>
    <string>/bin/bash</string>
    <string>$REPO/weekly_run.sh</string>
  </array>
  <key>WorkingDirectory</key> <string>$REPO</string>
  <key>StartCalendarInterval</key>
  <dict>
    <key>Weekday</key> <integer>6</integer>   <!-- 0=Sun ... 6=Sat -->
    <key>Hour</key>    <integer>23</integer>
    <key>Minute</key>  <integer>0</integer>
  </dict>
  <key>StandardOutPath</key>  <string>$LOGDIR/launchd.out.log</string>
  <key>StandardErrorPath</key><string>$LOGDIR/launchd.err.log</string>
  <key>EnvironmentVariables</key>
  <dict>
    <key>PATH</key>
    <string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin</string>
  </dict>
</dict>
</plist>
PLIST

# Reload cleanly if already present.
launchctl bootout "gui/$(id -u)" "$PLIST" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$PLIST"

echo "installed: $PLIST"
echo "schedule : Saturday 23:00 local (missed runs fire on next wake)"
echo "verify   : launchctl list | grep $LABEL"
echo "test now : launchctl kickstart -k gui/$(id -u)/$LABEL"
