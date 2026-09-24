# PS-14 Authoritative Contracts (Phase 110)

One source of truth for the shapes that previously drifted across tests
and prose.  Where an older test or document disagrees, the code cited
below is authoritative and the stale text was reconciled by Phase 110.

## 1. Health response (`GET /health`, risk engine)

Produced by `HealthReport.to_dict()` (`src/monitoring/observability.py`)
plus service-added fields (`service`, `started_at`, `release_attested`,
pool/security blocks).  Canonical keys:

| key | values | meaning |
|---|---|---|
| `status` | `ok` / `degraded` / `not_ready` / `dead` | derived from liveness/readiness/model readiness |
| `liveness` | `alive` / `dead` | process alive |
| `readiness` | `ready` / `not_ready` | can process requests |
| `model_readiness` | `loaded` / `not_loaded` | verified model present; **never** `loaded` when the model is missing, unverifiable, or drifted |
| `runtime_state` | `READY` / `MODEL_NOT_READY` / `INCONSISTENT` / `DRIFTED` / `FAILED` / `STARTING` / `STOPPED` | Phase 49 runtime attestation state |
| `db`, `audit_chain` | status strings | connectivity / chain verdicts |
| `release_id`, `model_id`, `feature_version` | strings | attested identity (see §3/§4); unattested boots report honest fallbacks |
| `uptime_seconds`, `details` | numbers/objects | diagnostics |

There is **no `model` key**.  The legacy `model: ok|error` field was
removed when `HealthReport` was introduced (Phase 70/72); tests 49/50/51
and the README claim were reconciled to `model_readiness` in Phase 110.
Consumers: `model_readiness == "loaded"` replaces `model == "ok"`.

## 2. Runtime states

`READY` gates normal ML inference.  `MODEL_NOT_READY` / `INCONSISTENT` /
`DRIFTED` / `FAILED` fail closed to rules-only degraded decisions tagged
`RUNTIME_RELEASE_UNVERIFIED`.  A missing release manifest is legacy/dev
mode: the model loads, `release_attested: false` — health never claims a
verified identity it does not have.  Verification failure leaves no
attestation and no model.

## 3. Model identity (two deliberate layers)

| layer | value | where |
|---|---|---|
| GOVERNANCE | `model_id = altman_native` | `real_world_evaluation_protocol`, `model_contract_reconciliation`, RWV/promotion machinery |
| DEPLOYED (attested) | `model_version = altman_native_E_hardneg_cert_20260904` | `release_manifest.json` `model_id`/`model_version`; `/health.model_id`; `/internal/release-attestation` |

Both name the same certified artifact set (`models/production/altman_native/`).

## 4. Release identity (two deliberate layers)

| layer | value | where |
|---|---|---|
| GOVERNANCE | `release-altman_native_E_hardneg_cert_20260904` | RWV/promotion/protocol constants |
| DEPLOYED (grandfathered manifest format) | `legacy-altman_native_E_hardneg_cert_20260904`, `gate_verdict: LEGACY_ATTESTED` | `release_manifest.json`; `/health.release_id` |

The `legacy-` prefix is metadata about the reconstructed manifest format
(Phase 50), not a second model.  Verification is hash/HMAC-bound against
disk; neither id is an alias for any feature or contract value.

## 5. Feature version

`v1` everywhere (`ML_FEATURE_VERSION`, manifest `feature_version`/
`schema_version`, `/health.feature_version`, observability defaults).
The historical string `altman_native_v1` is **stale and rejected**; it
survives only as a test-fixture literal in
phase67/68/70/71/72/train_altman_native (self-consistent, non-authoritative)
and as `STALE_FEATURE_VERSIONS` in `src/monitoring/manifest_contract.py`.

## 6. Release-manifest schema (authoritative contract)

`src/monitoring/manifest_contract.py` is the single source:
`CANONICAL_*` constants + `validate_release_manifest(manifest,
artifact_dir=None)`.  Required bindings: `release_id`, `model_id`,
`feature_version`, `gate_verdict`, `artifact_hash`, `artifact_files`
(per-file SHA-256, disk-checkable), `preprocessing_hash` (== scaler
hash), `rule_hash`, `evaluation_record_hash`, `training_config_hash`.
Runtime verification (`verify_release_for_load`) is unchanged and strict.

## 7. Audit-chain hygiene (CI-enforced)

`backend/scripts/audit_hygiene_check.py`, run by
`scripts/security_ci_gate.py` (step 1) and the CI Security Scan job
(`.github/workflows/ci-cd.yml`).  Deterministic, zero parameters, no
bypass.  Rules:

- Direct test-fixture event types are **frozen** at their 2026-09-24
  snapshot counts (23 types; e.g. `lifecycle_test_*` 75 total,
  `queue_test` 720, `rapid_test` 4800, `security_test` 540).  A shared
  chain containing any fixture row must match exactly; growth = leak,
  loss = tampering.  Fixture rows dated after the cutover
  (`2026-09-24 08:00:00`) fail.  `fraud_id LIKE 'CHAIN-%'` frozen at 75.
- A chain with **zero** fixture rows (fresh/isolated environment) passes.
- Legitimate application event types (`decision`/`OUTCOME_*`/`admin_*`/
  `runtime_*`…) are deliberately not frozen — they grow with real usage.
- Duplicate `seq` or `event_id` fails.
- Chain walk = Phase 109 `evaluate_chain`: every row strictly verified;
  breaks allowed only when they are exactly the five frozen,
  evidence-matched historical forks (or zero breaks).

## 8. Historical fork treatment

Sequences {731, 735, 740, 745, 750} (2026-09-19, cross-process stale
max-seq read before the Phase 109 `BEGIN IMMEDIATE` writer repair) are
**preserved evidence**: never deleted, never re-hashed, never declared
strictly valid (`strict_ok` stays `False`).  They are covered only by
immutable `AuditForkFinding` records whose evidence hash binds the exact
row content; any drift on those rows fails closed.  The strict
`verify_chain` authority is untouched.

## 9. Test-fixture isolation

Test fixture writers must use private temp DB storage, never the shared
production DB-4:

| suite | fixture types | isolation |
|---|---|---|
| phase76 | `lifecycle_test_*`, `test_drain_event`, `crash_survival_test` | `DB_DIR = mkdtemp` at import (Phase 109) |
| phase77 | `rapid_test`, `queue_test`, `perf_*`, `lock_test`, `*_test` family (~15k rows/run) | `DB_DIR = mkdtemp` at import (Phase 110) |
| phase79 | `security_test` (20/run) | `DB_DIR = mkdtemp` + temp-table bootstrap (Phase 110) |
| phase109 | `legacy_race`, `phase109_repro`, `p109_probe` | worker subprocesses bind temp `DB_DIR` |

Suites that exercise **live services** (security tests, admin console
use, outcome pipelines) produce legitimate application events through
the real request path — those are real operations, not fixture leakage,
and are not frozen.
