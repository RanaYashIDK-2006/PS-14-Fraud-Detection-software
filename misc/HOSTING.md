# Hosting PS-14

The project ships as **six services behind one Docker Compose stack** —
the front page (8000) plus the five services (8001–8005). Every service is a
stateless uvicorn app except for the four SQLite stores, which live on a
shared named volume. The front page shows live status for all five services
and links to the two browser UIs.

---

## 1. Quick start (Docker)

Requires Docker with Compose v2.

```bash
# from the project root
cp .env.example .env        # then EDIT the three secrets (step 4)
docker compose up --build -d
```

Open **http://localhost:8000** — the front page. Service URLs:

| Service | URL |
|---|---|
| Front page (status + links) | http://localhost:8000 |
| Identity Service | http://localhost:8001 |
| Privacy Layer (internal only) | http://localhost:8002 |
| Risk Engine (internal only) | http://localhost:8003 |
| **Alerts & verification UI** | http://localhost:8004 |
| **Compliance viewer** | http://localhost:8005 |

```bash
docker compose ps            # health status of all six
docker compose logs -f risk  # follow one service
docker compose down          # stop (keeps the dbdata volume)
docker compose down -v       # stop AND wipe the databases
```

## 2. Running without Docker (local dev)

Same as before — the one-command Windows stack now includes the front page:

```powershell
powershell -ExecutionPolicy Bypass -File .freebuff/start_stack.ps1
# -> front :8000 + identity :8001 + privacy :8002 + risk :8003 + verify :8004 + audit :8005
```

or run any service directly:

```bash
uvicorn src.front_service.main:app --port 8000        # landing page
uvicorn src.verification_service.main:app --port 8004 # alerts UI
```

## 3. Hosting on your LAN

The UIs build their service links from the **browser's hostname** (not a
hardcoded `127.0.0.1`), so they work from any machine on the network:

1. Find the host's LAN IP: `ipconfig` (Windows) / `ip addr` (Linux).
2. Open firewall ports 8000–8005 (Docker: `docker compose up` already maps
   them; allow them inbound).
3. From another machine open `http://<host-ip>:8000` — status tiles and the
   alerts/compliance links all follow automatically.

## 4. Secrets (change these before any non-local deployment)

The dev defaults are **not** for production. Set at least these, plus
your own access-document passphrase:

```bash
# .env (never commit it)
JWT_SECRET=<long random string>            # rotates Fernet PII keys + export signatures — coordinate
INTERNAL_TOKEN=<random service token>      # guards every /internal/* endpoint
COMPLIANCE_TOKEN=<random passphrase>       # guards the compliance viewer
ADMIN_USER=admin                           # front-page admin username
ADMIN_PASS=<your own admin passphrase>     # the ONLY way into the admin area
ACCESS_DOC_KEY=<your own passphrase>       # opens docs/ACCESS.md.enc
```

**Front-page admin area** (`http://<host>:8000` → 🔒 Admin): login with
`ADMIN_USER`/`ADMIN_PASS`. `ADMIN_PASS` is used ONCE to create the admin
record (`db/admin.json`); afterwards only a stored scrypt hash is verified
and the essentials (credentials, break-glass, service map) are decrypted
from an AES-256-GCM blob keyed by the passphrase — the passphrase itself is
never stored in plaintext. Rotate it from the admin panel (revokes all
sessions) or by deleting `db/admin.json` and re-bootstrapping.

- Rotating `JWT_SECRET` re-derives the PII encryption key — existing DB-1
  rows become unreadable. Rotate it only as a coordinated, planned event.
- The access document itself is encrypted (`python scripts/access_doc.py
  view`); keep its passphrase in the OS credential store or a secret manager,
  not in the chat or a shared drive.

## 5. Production deployment (VPS / cloud)

Two options for production:

### Option A: Dev compose + Caddy (simpler)

Keep the SQLite shared volume, add Caddy for auto-HTTPS:

```bash
cp .env.example .env
# Edit .env with real secrets
# Set DOMAIN=ps14.example.com

docker compose up --build -d
```

The included `Caddyfile` maps subdomains:
- `ps14.example.com` → front page (8000)
- `alerts.ps14.example.com` → verification UI (8004)
- `compliance.ps14.example.com` → audit trail (8005)
- `auth.ps14.example.com` → identity API (8001)

### Option B: Production compose with PostgreSQL (recommended)

Separate PostgreSQL databases, per-store credentials, internal-only network:

```bash
cp .env.example .env
# Edit .env: generate ALL secrets + PostgreSQL credentials
# Set DOMAIN=ps14.example.com
# Set PS14_MODE=production

docker compose -f docker-compose.prod.yml --env-file .env up --build -d
```

This gives you:
- 4 separate PostgreSQL 16 databases with per-store credentials
- Internal Docker network (services can't reach the internet)
- Caddy auto-HTTPS with Let's Encrypt
- Data retention profile for scheduled cleanup
- Retrain profile for model updates

### Option C: One-command deploy

```bash
bash scripts/deploy.sh              # validate + test + build + deploy
bash scripts/deploy.sh --dry-run    # validate only
bash scripts/deploy.sh --compose-prod  # use production compose
```

The deploy script validates secrets, runs the test suite, builds the
Docker image, deploys, and verifies all 6 services are healthy.

## 6. Data & backups

- All four stores (`identity.db`, `features.db`, `risk.db`, `audit.db`)
  live in the `dbdata` volume at `/app/db`.
- Back up by stopping the stack, then copying the volume:

```bash
docker compose down
docker run --rm -v ps14_dbdata:/data -v "$PWD/backups":/backup \
  alpine sh -c "cp /data/*.db /backup/"
docker compose up -d
```

- `models/artifacts/` (the trained models + rules) is baked into the image —
  retrain (`python scripts/training_pipeline.py --full`) and rebuild the
  image to ship new models.

## 7. Production hardening checklist

### Secrets & Config
- [ ] Replace all dev secrets in `.env` (6 required: JWT_SECRET, PII_ENCRYPTION_KEY, EXPORT_SIGNING_KEY, INTERNAL_TOKEN, COMPLIANCE_TOKEN, ADMIN_PASS)
- [ ] Set `PS14_MODE=production` — fail-closed startup validation
- [ ] Set `CORS_ORIGINS` to your actual domains (no localhost)
- [ ] Use a secrets manager (HashiCorp Vault, AWS Secrets Manager) instead of `.env`

### Database
- [ ] Move each SQLite store to its own Postgres with per-store credentials (`docker-compose.prod.yml`)
- [ ] Enable TLS for PostgreSQL connections
- [ ] Set up automated backups (pg_dump + off-site storage)
- [ ] Test restore procedure

### Network
- [ ] Put Caddy/reverse proxy in front (auto-HTTPS)
- [ ] Firewall: only expose 80/443 externally
- [ ] Rate limiter + WAF in front of auth endpoints
- [ ] Enable fail2ban for brute-force protection

### Service Security
- [ ] Replace shared tokens with per-service mTLS + short-lived RBAC credentials
- [ ] Encrypt volumes at rest
- [ ] PII keys in KMS/HSM (not derived from JWT_SECRET)

### Monitoring
- [ ] Schedule audit chain verification: `bash scripts/retention-cron.sh`
- [ ] Set up systemd timers (see `scripts/ps14-retention.timer`, `scripts/ps14-audit-chain.timer`)
- [ ] Monitor service health via `/health` endpoints
- [ ] Alert on audit chain failures

### CI/CD
- [ ] GitHub Actions: `.github/workflows/ci-cd.yml` runs tests → security → build → deploy
- [ ] `.github/workflows/security-scan.yml` runs weekly security scans
- [ ] Tag releases with `v*` to trigger production deploy

## 7b. Validating the stack end-to-end (clean machine, Docker required)

```bash
docker compose up --build -d
# wait for health (first build + artifact load takes a minute or two):
docker compose ps --format "table {{.Name}}\t{{.Status}}"   # expect 6 running (healthy)
```

Then run the register-to-audit walkthrough against the containers (the
host ports 8001–8005 map to them, so the script works unchanged):

```bash
bash scripts/live_walkthrough.sh
```

Expected results:
- Decisions: scenario **A → ALLOW (low, ~1)**, **B → STEP_UP (medium, ~43)**, **C → VERIFY (high, 100)** with the reason codes shown.
- Verification: 1 alert → dispute → case id, 0 open alerts after.
- Audit: 13+ hash-chained events, export re-verified (`EXPORT VERIFIED`), chain `OK`.
- Front page: `http://<host>:8000` → **5 of 5 services online** (the tile
  strip tracks the five backend services; `docker compose ps` shows all
  **six** containers healthy — the front page doesn't count itself).

Assertions that ONLY a real compose run covers (the container-network
layer): service-name DNS (`http://identity:8001`, …) resolving inside the
compose network, and the `dbdata` volume being shared by all four stores.

## 7c. Dockerized retrain (in-container, gate-gated restart)

`scripts/docker_retrain.sh` (or `make retrain`) runs the FULL training
pipeline inside a compose `train` profile container and rebuilds/restarts
**only the risk service** when every step passes:

```bash
bash scripts/docker_retrain.sh
# 1/3  docker compose --profile train run --rm train
#        -> python scripts/training_pipeline.py --full (retrain -> retune ->
#           apply rules.yaml -> all 11 suites incl. the OOD recall gate)
# 2/3  docker compose build risk     # bake the fresh artifacts into the image
# 3/3  docker compose up -d risk     # restart risk so it serves them
```

How it stays safe:
- `./models` and `./data` are **bind-mounted** into the train container, so
  fresh artifacts (and the tuner's score cache) land on the HOST and survive
  the container's removal (`--rm`).
- The pipeline **fails fast**: a failed suite, a failed retune, or the OOD
  recall gate dropping below its floor exits non-zero, which stops the
  script BEFORE any rebuild — the running risk service is never touched by
  a bad model.
- `NO_RESTART=1 bash scripts/docker_retrain.sh` (or `make train-only`) runs
  the pipeline only — for CI staging where a later job does the rebuild.
- Only `risk` is recreated; the other five containers keep their image and
  state (they don't read the model artifacts).

First run is the slow one (retrain + per-event score compute ~10 min);
re-runs reuse the bind-mounted tuner cache.

## 8. Data retention & scheduling

The data retention script enforces per-store retention periods:
- DB-2 (features): 90-day retention for feature vectors
- DB-3 (risk): 180-day retention for risk scores without outcomes
- DB-4 (audit): 365-day retention via redaction-in-place

### Scheduling options

**Option A: Crontab (Linux)**
```bash
# Add to crontab:
0 2 * * * /path/to/scripts/retention-cron.sh
```

**Option B: systemd timer**
```bash
# Copy timer files:
sudo cp scripts/ps14-retention.service /etc/systemd/system/
sudo cp scripts/ps14-retention.timer /etc/systemd/system/
sudo cp scripts/ps14-audit-chain.service /etc/systemd/system/
sudo cp scripts/ps14-audit-chain.timer /etc/systemd/system/

# Enable:
sudo systemctl enable --now ps14-retention.timer
sudo systemctl enable --now ps14-audit-chain.timer

# Check status:
systemctl list-timers | grep ps14
```

**Option C: Docker maintenance profile**
```bash
docker compose --profile maintenance run --rm retention
```

Audit chain integrity is verified hourly by the timer, which calls
`GET /audit/integrity` and fails loudly if the chain is broken.

## 9. CI/CD pipeline

Two GitHub Actions workflows:

| Workflow | Trigger | What it does |
|---|---|---|
| `ci-cd.yml` | push/PR to main | Test suite (17 suites) → Security scan → Docker build → Integration test |
| `security-scan.yml` | push/PR + weekly | Security scanner + Bandit + TruffleHog |

**Deployment:** Tag a release (`git tag v1.0.0 && git push --tags`) to
trigger the deploy job. Requires a `production` environment in GitHub
Settings → Environments.

**Gates:** The pipeline blocks deployment if:
- Any test suite fails
- Security scan finds critical/high issues
- Docker build fails
- Integration test fails

## 10. Troubleshooting

| Symptom | Fix |
|---|---|
| `docker compose up` fails on port bind | Another process holds 8000–8005 — `docker compose down`, then retry, or change the left side of `ports:` |
| Front page shows services down | The `/status` check pings container names — confirm `docker compose ps` shows all healthy; if you run services outside Docker, set `IDENTITY_URL`/… env to the host URLs |
| Alerts UI can't reach identity | The UI uses `window.location.hostname:8001` — on a proxy setup, point the browser at the mapped hostname, or set `IDENTITY_URL` accordingly |
| DB schema errors after an upgrade | SQLAlchemy `create_all` doesn't add columns to existing tables — a dev reset is `docker compose down -v` (destroys demo data only) |
