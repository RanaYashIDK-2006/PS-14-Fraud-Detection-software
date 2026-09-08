#!/usr/bin/env bash
# Dockerized retrain: run the FULL training pipeline inside the compose
# `train` profile container, then — ONLY if it exits 0 (every step green,
# including the OOD recall gate) — rebuild the image with the fresh
# artifacts and restart the risk service so it serves them.
#
#   bash scripts/docker_retrain.sh      # or: make retrain
#
# Env:
#   NO_RESTART=1    train only; do not rebuild/restart risk (CI staging)
set -euo pipefail

if ! command -v docker >/dev/null 2>&1; then
  echo "docker is required - install Docker and start the daemon first (see HOSTING.md)" >&2
  exit 1
fi

echo "== 1/3 training pipeline inside the compose 'train' container =="
# --rm: the container is removed after the run; a non-zero exit (a failed
# suite, a failed retune, or the OOD recall gate) stops the script here and
# leaves the old artifacts/image untouched.
docker compose --profile train run --rm train

if [ "${NO_RESTART:-0}" = "1" ]; then
  echo "== gate passed; NO_RESTART=1 - skipping rebuild/restart =="
  exit 0
fi

echo
echo "== 2/3 gate passed - rebuilding the image with fresh artifacts =="
docker compose build risk

echo
echo "== 3/3 restarting the risk service on the new image =="
docker compose up -d risk

echo
echo "retrain complete. Verify:"
echo "  docker compose ps                     # risk should show (healthy) on the new image"
echo "  curl -s localhost:8003/health"
echo "  grep -E 'OOD recall gate|PIPELINE COMPLETE' <(docker compose logs train 2>/dev/null) || true"
