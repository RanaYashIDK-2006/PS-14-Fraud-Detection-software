#!/usr/bin/env bash
# PS-14 — Production deployment script
# Validates config → runs tests → builds image → deploys → verifies health
#
# Usage:
#   bash scripts/deploy.sh                    # full deploy
#   bash scripts/deploy.sh --skip-tests       # skip test suite
#   bash scripts/deploy.sh --dry-run          # validate only, no deploy
#   bash scripts/deploy.sh --docker           # deploy with Docker Compose
#   bash scripts/deploy.sh --compose-prod     # deploy with production compose (PostgreSQL)
#
set -euo pipefail

SKIP_TESTS=false
DRY_RUN=false
MODE="docker"  # docker | compose-prod

while [[ $# -gt 0 ]]; do
  case $1 in
    --skip-tests) SKIP_TESTS=true; shift ;;
    --dry-run) DRY_RUN=true; shift ;;
    --docker) MODE="docker"; shift ;;
    --compose-prod) MODE="compose-prod"; shift ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log()  { echo -e "${GREEN}[DEPLOY]${NC} $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
err()  { echo -e "${RED}[ERROR]${NC} $*" >&2; }

# ── Step 1: Validate environment ──────────────────────────────────────────
log "Step 1/6: Validating environment..."

if [[ ! -f .env ]]; then
  err ".env file not found. Copy .env.example to .env and set all secrets."
  err "  cp .env.example .env"
  exit 1
fi

# Source .env for validation
set -a
source .env
set +a

MISSING=()
[[ -z "${JWT_SECRET:-}" ]]         && MISSING+=("JWT_SECRET")
[[ -z "${PII_ENCRYPTION_KEY:-}" ]] && MISSING+=("PII_ENCRYPTION_KEY")
[[ -z "${EXPORT_SIGNING_KEY:-}" ]] && MISSING+=("EXPORT_SIGNING_KEY")
[[ -z "${INTERNAL_TOKEN:-}" ]]     && MISSING+=("INTERNAL_TOKEN")
[[ -z "${COMPLIANCE_TOKEN:-}" ]]   && MISSING+=("COMPLIANCE_TOKEN")
[[ -z "${ADMIN_PASS:-}" ]]         && MISSING+=("ADMIN_PASS")

if [[ ${#MISSING[@]} -gt 0 ]]; then
  err "Missing required secrets: ${MISSING[*]}"
  err "Generate with: python -c \"import secrets; print(secrets.token_urlsafe(32))\""
  exit 1
fi

# Check secrets aren't dev defaults
for SECRET in "$JWT_SECRET" "$PII_ENCRYPTION_KEY" "$EXPORT_SIGNING_KEY" "$INTERNAL_TOKEN" "$COMPLIANCE_TOKEN" "$ADMIN_PASS"; do
  if [[ "$SECRET" == *"replace-in"* ]] || [[ "$SECRET" == *"dev-"* ]]; then
    err "Found dev default secret. Generate real secrets for production."
    exit 1
  fi
done

log "  ✓ All secrets present and non-default"

# Check CORS isn't localhost-only in prod
if [[ "${CORS_ORIGINS:-}" == *"localhost"* ]] && [[ "${PS14_MODE:-}" == "production" ]]; then
  err "CORS_ORIGINS contains localhost in production mode"
  exit 1
fi

log "  ✓ Configuration valid"

# ── Step 2: Run tests ─────────────────────────────────────────────────────
if [[ "$SKIP_TESTS" == "false" ]]; then
  log "Step 2/6: Running test suite..."

  python src/generate_synthetic_data.py
  python src/train_compare.py

  SUITES=(
    smoke_test risk_engine_test verification_test audit_test
    drift_test feedback_test k_anonymity_test pipeline_test
    tune_test ood_gate_test rules_gate_test federated_test
    resilience_test security_test pseudonym_separation_test
    feature_compatibility_test production_gate_test
  )

  FAILED=()
  for SUITE in "${SUITES[@]}"; do
    if ! python "scripts/${SUITE}.py" > /dev/null 2>&1; then
      FAILED+=("$SUITE")
    fi
  done

  if [[ ${#FAILED[@]} -gt 0 ]]; then
    err "Failed suites: ${FAILED[*]}"
    err "Fix failures before deploying."
    exit 1
  fi

  log "  ✓ All ${#SUITES[@]} test suites passed"
else
  log "Step 2/6: Skipped (--skip-tests)"
fi

# ── Step 3: Security scan ─────────────────────────────────────────────────
log "Step 3/6: Running security scan..."

if python scripts/security_scan.py --full > /dev/null 2>&1; then
  log "  ✓ Security scan passed"
else
  warn "  Security scan found issues — review before deploying to production"
  if [[ "$DRY_RUN" == "true" ]]; then
    exit 1
  fi
fi

# ── Step 4: Build Docker image ────────────────────────────────────────────
log "Step 4/6: Building Docker image..."

if [[ "$DRY_RUN" == "true" ]]; then
  log "  DRY RUN — skipping build"
else
  docker build -t ps14:$(date +%Y%m%d-%H%M%S) -t ps14:latest .
  log "  ✓ Docker image built"
fi

# ── Step 5: Deploy ────────────────────────────────────────────────────────
log "Step 5/6: Deploying..."

if [[ "$DRY_RUN" == "true" ]]; then
  log "  DRY RUN — skipping deploy"
elif [[ "$MODE" == "compose-prod" ]]; then
  if [[ ! -f docker-compose.prod.yml ]]; then
    err "docker-compose.prod.yml not found"
    exit 1
  fi
  docker compose -f docker-compose.prod.yml --env-file .env up -d --build
  log "  ✓ Production stack deployed (PostgreSQL)"
else
  docker compose up -d --build
  log "  ✓ Dev stack deployed (SQLite)"
fi

# ── Step 6: Verify health ─────────────────────────────────────────────────
log "Step 6/6: Verifying health..."

if [[ "$DRY_RUN" == "true" ]]; then
  log "  DRY RUN — skipping health check"
  log "=== Deployment validation complete (dry run) ==="
  exit 0
fi

sleep 10

HEALTHY=0
TOTAL=6
for PORT in 8000 8001 8002 8003 8004 8005; do
  if curl -sf "http://localhost:${PORT}/health" > /dev/null 2>&1; then
    HEALTHY=$((HEALTHY + 1))
  fi
done

if [[ $HEALTHY -eq $TOTAL ]]; then
  log "  ✓ All $TOTAL services healthy"
else
  warn "  $HEALTHY/$TOTAL services healthy"
  docker compose logs --tail=20
fi

log "=== Deployment complete ==="
log ""
log "  Front page:   http://localhost:8000"
log "  Alerts UI:    http://localhost:8004"
log "  Compliance:   http://localhost:8005"
log ""
log "  Logs:         docker compose logs -f"
log "  Status:       docker compose ps"
log "  Stop:         docker compose down"
log "  Retrain:      bash scripts/docker_retrain.sh"
