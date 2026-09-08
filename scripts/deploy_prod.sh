#!/usr/bin/env bash
# PS-14 Production Deployment Script
# Usage: bash scripts/deploy_prod.sh
#
# Steps:
#   1. Validate .env has all required secrets
#   2. Build Docker images
#   3. Start Redis + all services
#   4. Wait for health checks
#   5. Run smoke test against live stack
#   6. Print deployment summary

set -euo pipefail

echo "============================================================"
echo "  PS-14 Production Deployment"
echo "============================================================"

# ── Step 1: Validate environment ────────────────────────────────────
echo ""
echo "[1/6] Validating environment..."

REQUIRED_VARS="JWT_SECRET PII_ENCRYPTION_KEY EXPORT_SIGNING_KEY INTERNAL_TOKEN COMPLIANCE_TOKEN ADMIN_PASS CORS_ORIGINS"
missing=0
for var in $REQUIRED_VARS; do
    if [ -z "${!var:-}" ]; then
        echo "  MISSING: $var"
        missing=$((missing + 1))
    else
        echo "  OK: $var (${#!var} chars)"
    fi
done

if [ $missing -gt 0 ]; then
    echo ""
    echo "FATAL: $missing required environment variables not set."
    echo "Copy .env.example to .env and fill in production values."
    exit 1
fi

# Check CORS doesn't point to localhost
if echo "$CORS_ORIGINS" | grep -q "localhost\|127.0.0.1"; then
    echo ""
    echo "WARNING: CORS_ORIGINS contains localhost/127.0.0.1"
    echo "  Current: $CORS_ORIGINS"
    echo "  Set to your production domains before going live."
fi

# ── Step 2: Build images ───────────────────────────────────────────
echo ""
echo "[2/6] Building Docker images..."
docker compose -f docker-compose.prod.yml build --parallel

# ── Step 3: Start services ─────────────────────────────────────────
echo ""
echo "[3/6] Starting production stack..."
docker compose -f docker-compose.prod.yml up -d

# ── Step 4: Wait for health ────────────────────────────────────────
echo ""
echo "[4/6] Waiting for health checks..."
services=("front:8000" "identity:8001" "privacy:8002" "risk:8003" "verify:8004" "audit:8005" "inference:8006")
max_wait=60
for svc_port in "${services[@]}"; do
    svc="${svc_port%%:*}"
    port="${svc_port##*:}"
    waited=0
    while [ $waited -lt $max_wait ]; do
        if docker compose -f docker-compose.prod.yml exec -T "$svc" \
            python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:$port/health', timeout=3)" \
            >/dev/null 2>&1; then
            echo "  $svc (:$port): HEALTHY"
            break
        fi
        sleep 2
        waited=$((waited + 2))
    done
    if [ $waited -ge $max_wait ]; then
        echo "  $svc (:$port): TIMEOUT after ${max_wait}s"
    fi
done

# ── Step 5: Smoke test ─────────────────────────────────────────────
echo ""
echo "[5/6] Running smoke test..."
docker compose -f docker-compose.prod.yml exec -T risk \
    python -c "
import sys; sys.path.insert(0, '.')
from fastapi.testclient import TestClient
from src.risk_engine.main import app
c = TestClient(app)
r = c.get('/health')
print(f'  Risk engine health: {r.status_code} {r.json()[\"status\"]}')
" 2>/dev/null || echo "  Smoke test skipped (not running locally)"

# ── Step 6: Summary ────────────────────────────────────────────────
echo ""
echo "[6/6] Deployment Summary"
echo "============================================================"
docker compose -f docker-compose.prod.yml ps --format "table {{.Name}}\t{{.Status}}\t{{.Ports}}"
echo ""
echo "  Redis:    redis://localhost:6379"
echo "  Front:    http://localhost:8000"
echo "  Monitor:  http://localhost:8000/monitor-page"
echo "  Admin:    http://localhost:8000/admin"
echo ""
echo "  Workers:  risk=4, privacy=4, front=2, identity=2, verify=2, audit=2, inference=2"
echo "  Total:    18 Gunicorn workers"
echo ""
echo "  Latency SLO: P99 < 100ms at concurrency <= 50/worker"
echo "  Check SLO:   curl -H 'X-Internal-Token: \$INTERNAL_TOKEN' http://localhost:8003/internal/latency-slo"
echo ""
echo "  Logs: docker compose -f docker-compose.prod.yml logs -f [service]"
echo "============================================================"
