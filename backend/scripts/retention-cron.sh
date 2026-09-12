#!/usr/bin/env bash
# PS-14 — Data retention cron wrapper
# Run via crontab: 0 2 * * * /path/to/scripts/retention-cron.sh
# Or via systemd timer (see ps14-retention.timer)
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="${PROJECT_DIR}/logs"

mkdir -p "$LOG_DIR"

LOGFILE="${LOG_DIR}/retention-$(date +%Y%m%d).log"

echo "=== Data retention run: $(date -Iseconds) ===" >> "$LOGFILE"

cd "$PROJECT_DIR"

# If running inside Docker
if command -v docker &> /dev/null && docker compose ps --status running 2>/dev/null | grep -q front; then
  echo "Running retention via Docker..." >> "$LOGFILE"
  docker compose exec -T retention python scripts/data_retention.py >> "$LOGFILE" 2>&1
else
  # Running on host (venv)
  if [[ -f .venv/Scripts/python.exe ]]; then
    .venv/Scripts/python.exe scripts/data_retention.py >> "$LOGFILE" 2>&1
  elif [[ -f .venv/bin/python ]]; then
    .venv/bin/python scripts/data_retention.py >> "$LOGFILE" 2>&1
  else
    python scripts/data_retention.py >> "$LOGFILE" 2>&1
  fi
fi

EXIT_CODE=$?
echo "Exit code: $EXIT_CODE" >> "$LOGFILE"
echo "" >> "$LOGFILE"

# Rotate logs older than 30 days
find "$LOG_DIR" -name "retention-*.log" -mtime +30 -delete 2>/dev/null || true

exit $EXIT_CODE
