# Admin Console — Staging Deployment Procedure (Phase 111)

Staging runbook for bringing up the fraud-ops admin console with TOTP MFA, the
live monitor, and flagged-transaction investigation. Follow the steps in order.

**Production rule:** never seed a known password or TOTP secret. The bootstrap
passphrase below is a *staging-only* value; in production the first admin is
created interactively and the TOTP secret is generated on the admin's own
device/secret at enrollment time — it is never committed, logged, or printed.

--------------------------------------------------
## 1. Install dependencies
--------------------------------------------------

- Create/refresh the virtualenv: `python -m venv .venv`
- Install: `./.venv/Scripts/pip install -r requirements.txt` (Windows) or
  `./.venv/bin/pip install -r requirements.txt` (POSIX)
- Confirm: `./.venv/Scripts/python.exe -c "import fastapi, uvicorn"`

--------------------------------------------------
## 2. Configure environment
--------------------------------------------------

- Copy `.env.example` → `.env` (or create `.env` at the repo root).
- Required for admin console:
  - `ADMIN_PASS` — **bootstrap only**. Used for the very first login when
    `db/admin.json` does not exist. After the first successful login the
    scrypt hash is stored and `ADMIN_PASS` is no longer consulted; delete
    `db/admin.json` to re-bootstrap.
  - `ACCESS_DOC_KEY` — Fernet key for `docs/ACCESS.md.enc` (unchanged).
- Do **not** change `JWT_SECRET` on an existing deployment: it derives
  `fernet_key` and changing it breaks decryption of existing DB-1 PII.
- Services do not auto-load `.env`; the front service merges it itself for
  `ADMIN_PASS`. Other services need real env vars or accept dev defaults.

--------------------------------------------------
## 3. Initialize databases
--------------------------------------------------

- Four stores live under `db/`: `features.db` (DB-2), `risk.db` (DB-3),
  `audit.db` (DB-4), `identity.db` (DB-1), plus `admin.json`.
- Tables are created on first import of each service module
  (SQLAlchemy `create_all` + audit writer triggers). Fresh staging: just start
  the stack once. Upgrading an existing staging DB with new columns: delete
  `db/features.db` and `db/risk.db` (demo state is throwaway; `audit.db` and
  `identity.db` keep their data).
- Verify: `ls db/` shows all four stores after first start.

--------------------------------------------------
## 4. Start API + frontend
--------------------------------------------------

- Windows staging: `powershell -NoProfile -ExecutionPolicy Bypass -File .freebuff/start_stack.ps1`
  (idempotent — stops existing :8000–:8005 listeners first).
- Manual: start each service module with uvicorn on its port —
  front 8000, risk 8001, privacy 8002, verification 8003, audit 8004,
  compliance 8005 (see `.freebuff/run.md` for the canonical list).
- Health check: `curl http://localhost:8000/status` aggregates all five
  `/health` endpoints server-side (do not fetch services cross-origin from
  the page).

--------------------------------------------------
## 5. Create the first admin
--------------------------------------------------

- Browse to `http://localhost:8000/admin`.
- On first run (`db/admin.json` absent), log in with username `admin` and the
  bootstrap `ADMIN_PASS` from `.env`. The service stores only: username,
  scrypt passphrase hash, timestamps, login count, and an AES-256-GCM
  essentials blob (wrap key derived from the passphrase).
- After this login `ADMIN_PASS` is inert; subsequent logins verify only the
  stored hash.

--------------------------------------------------
## 6. Enroll TOTP
--------------------------------------------------

- In the console: **Access Control → TOTP → Set up** (`POST /admin/totp/setup`).
- The server generates a fresh random secret per admin (no shared or
  hardcoded secret anywhere in the repo), persists it encrypted in the
  essentials blob, and returns:
  - the **TOTP enrollment URI** (`otpauth://totp/...`), and
  - the secret for manual entry into any authenticator app.
- **QR code:** the URI can be rendered by any standard authenticator; this
  build does not ship a QR library (deliberate — no extra dependency), so the
  admin enters the URI/secret manually.
- Verify with a 6-digit code from the authenticator:
  `POST /admin/totp/verify`. On success, 8 one-time **recovery codes** are
  issued — store them offline.
- Rotation (`POST /admin/totp/rotate`) invalidates the old
  secret; disable (`/admin/totp/disable`) requires an active code.
- While MFA is enabled, `/admin/totp/current-code` returns **410** (refuses to
  replay the secret).

--------------------------------------------------
## 7. Authenticate with MFA
--------------------------------------------------

- Log out, then log in again: password first, then a TOTP code. A missing or
  wrong code is rejected; three consecutive failures trigger a 60-second
  lockout (HTTP 429 with seconds-remaining).
- Session cookie: `admin_session` — HttpOnly, SameSite=Strict; cookie-based
  requests must also send `X-Requested-With: XMLHttpRequest` (403 otherwise).
  Bearer-token clients are CSRF-exempt.

--------------------------------------------------
## 8. Verify the live monitor
--------------------------------------------------

- Open **Live Monitor** (`/admin` → Live, backed by `GET /admin/api/live`).
- Auth is enforced server-side (401 unauthenticated — monitor is not public).
- Switch windows (5m / 15m / 1h) and pause/resume; while paused the displayed
  rows stay stable. Metrics without a source render **N/A**, never 0.
- Confirm identity fields show both the governance identity
  (`model_id=altman_native`, `release_id=release-altman_native_E_hardneg_...`)
  and the attested runtime identity from `model/release_manifest.json`.

--------------------------------------------------
## 9. Investigate a flagged transaction
--------------------------------------------------

- Open `/admin/transactions`, search by event_id / decision / band / score
  range / time range / degraded / data-quality state (limits: ≤200 per page).
- Open a result → `/admin/transactions/{event_id}` detail view:
  - **DecisionTrace** (9 stages: input validation → idempotency → drift →
    runtime state → feature enforcement → inference/rules → decision band →
    persistence → audit), each with status/reason where the backend provides
    one; otherwise `REASON_NOT_AVAILABLE`.
  - **Audit trail** for the event with prev-hash/event-hash/verification from
    the authoritative verifier. The preserved Phase 109 fork (seqs
    731/735/740/745/750) reports as *Valid (quarantined), strict break* —
    never as clean.
- No PII beyond pseudonymous identifiers (fraud_id/event_id) is exposed; no
  secrets, tokens, or credential material appear in any response.

--------------------------------------------------
## 10. Post-deploy checks
--------------------------------------------------

- Admin actions are audited (login, MFA success/failure, logout,
  investigation views, session revocation, MFA enrollment/rotation/disable).
- Run the phase suite: `./.venv/Scripts/python.exe backend/scripts/phase111_admin_console_test.py`
- Final state must remain `SYSTEM_READY_PENDING_ELIGIBLE_DATASET` /
  `BLOCKED_PENDING_ELIGIBLE_DATASET` / `PROMOTION_GATE_REQUIRED`.
