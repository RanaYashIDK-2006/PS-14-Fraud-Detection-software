# PS-14 — Privacy-First AI Fraud Detection System (Prototype)

Phase 2 prototype of the architecture described in the PS-14 design doc.
Core principle: *detect fraud without unnecessarily knowing who the person
is. AI recommends — verification decides.*

## Current status (Phase 2 build order)

- [x] **Synthetic dataset generator** — `src/generate_synthetic_data.py`
  (10k transactions, ~1.5% fraud, section-16 feature shape; raw amounts and
  PII discarded at the generator — only derived, purpose-limited features
  persist).
- [x] **Model training & comparison** — `src/train_compare.py`
  (Logistic Regression, Random Forest, XGBoost, Isolation Forest; time-based
  70/15/15 split; PR-AUC / ROC-AUC / recall@1%FPR / F1 / FPR / FNR).
- [x] **Identity Service** — `src/identity_service/` (FastAPI, DB-1):
  register/login, CSPRNG Fraud ID generation, `pseudonym_mapping` (the only
  real↔pseudonym bridge), column-encrypted PII, break-glass resolution
  logging, JWTs carrying only the `fraud_id` claim.
- [x] **Privacy Layer** — `src/privacy_layer/` (FastAPI, DB-2):
  `/internal/ingest-transaction` strips raw PII/amounts/devices and persists
  only the derived §16 feature vector, keyed by `fraud_id`.
- [x] **Risk Engine** — `src/risk_engine/` (FastAPI, DB-3): stacked fusion of
  the four model outputs + a declarative, PR-reviewed rule engine
  (`rules.yaml`) → 0–100 score, low/medium/high band routing
  (allow / step-up / verify), and §11 category-level reason codes.
- [x] **Verification flow** — `src/verification_service/` (FastAPI, static UI):
  alerts with category-level reasons, “This was me / wasn’t me” buttons,
  guided recovery with a case ID, and `verification_outcomes` in DB-3.
  One-click demo entry (register + sign in + seed) and a pre-filled
  compliance passphrase mean the whole flow — alert → verify → audit trail —
  is reachable without typing any credentials. The compliance trail is
  compact and paged (reason chips, "Show older events" via `?offset=`), so
  it stays scannable as the hash chain accumulates. Verified end-to-end over
  live HTTP across all four services.
- [x] **Audit trail** — `src/audit_service/` (FastAPI, DB-4): append-only,
  hash-chained `audit_events` (UPDATE/DELETE blocked by storage triggers),
  wired into risk scoring and verification outcomes, plus a
  compliance-role viewer with chain-integrity verification, served as a
  static UI at `http://127.0.0.1:8005/` (passphrase-gated, per-event hash
  linkage, compact reason chips with a "no flags" placeholder on flag-free
  scores, chain-wide summary chips (e.g. "no flags ×26"), event-type filter
  chips, and "Show older events" paging). The
  Verification UI's compliance view adds a live chain-status card (entries,
  genesis, latest chain entry) and a latest-case card (case ID → event →
  score → chain entry), so the register-to-audit trail is visible
  end-to-end in the browser.
- [x] **Feedback loop** — verification outcomes become labeled training rows
  (`scripts/export_feedback.py`), merged via `src/train_compare.py
  --feedback` and retrained (see `models/feedback_loop_report.md`).
- [x] **Federated learning (stretch)** — 3 simulated institutions on disjoint
  account shards, FedAvg over worker processes that exchange only weights
  (`scripts/federated_sim.py`; see `models/federated_report.md`).

## Performance (Honest — Verified, No Data Leakage)

Performance measured against the **Kaggle ULB Credit Card Fraud** dataset
(284,807 transactions, 492 fraud) and a **1.2M PaySim synthetic** dataset.
All features are derived from transaction data only — no label-dependent
features.

### Kaggle ULB (real fraud data, 16 features)

| Metric | Value | Methodology |
|---|---|---| 
| ROC-AUC | 0.966 | `src/train_compare.py`, time-split eval |
| Recall@1%FPR | 91.8% | Same |
| PR-AUC | 0.877 | Same |
| Precision (threshold 0.43) | 48.0% | Optimal ensemble (0.75 XGB + 0.25 RF) |
| Recall (threshold 0.43) | 97.8% | 4 frauds missed out of 492 |
| False Positive Rate | 1.996% | 192/9,618 legit flagged |

### PaySim 1.2M (synthetic, 16 features, in-process only)

| Metric | Value | Caveat |
|---|---|---|
| ROC-AUC | 0.664 | Synthetic features (no real auth/device data) |
| Inference speed | ~200K samples/sec | Batch sklearn, not HTTP latency |

**⚠️ Previous benchmark claims of ROC-AUC 1.0000 and 752K TPS were based
on data leakage (5 features derived from the label). The corrected
benchmark uses only transaction-derived features.** See
`scripts/benchmark_comparison.py` for methodology.

### Live HTTP stack (via `load_test.py`)

| Metric | Value | Methodology |
|---|---|---|
| Latency | ~15-25ms | p50, single concurrent request |
| Throughput | ~40-60 TPS | Measured via `load_test.py -n 200 -c 10` |

**NOT measured:** End-to-end latency including audit chain writes,
database persistence, and concurrent load. The HTTP stack adds ~10-15ms
overhead beyond pure model inference.

**Industry comparisons** in `scripts/benchmark_comparison.py` are approximate
and use different datasets/protocols — not directly comparable.

## Validation Phases (Phases 14–25)

A rigorous, adversarial audit of the system's real-world readiness.
Every conclusion is evidence-classified; firewalls are verified.

| Phase | Title | Classification | Key Finding |
|---|---|---|---|
| **14** | Chip supervision transfer | `NO_ROBUST_WIN` | Chip supervision **hurt**: C2 chip recall 32.3% vs C0 57.7% (−25.4pp) |
| **15** | Distribution-shift forensics | `CASE_B_PARTIALLY_REPRESENTED` | Channel composition shift PROVEN; 95% of 2017 patterns have weak/no historical analogue |
| **16** | Data-inventory boundary | `OUTCOME_C — NONE` | 51 datasets inventoried, 1 development-eligible (IBM, synthetic); boundary is protocol-constrained |
| **17** | IBM generator audit | `SYNTHETIC_REGIME_ARTIFACT` | Abrupt step change at 2017 affecting the whole population; real-world validity UNVERIFIED |
| **18** | System readiness gate | `CONDITIONALLY_DEPLOYABLE` | Security/monitoring/rollback PASS; 3 fraud-rate features with unverified label availability |
| **19** | Real-world data gate | `DATA_ACQUISITION_REQUIRED` | Every dataset in the environment is synthetic; no ground-truth labels; production prevalence unknown |
| **20** | Decision-time remediation | `CLEAN_PROD_COMPAT_CANDIDATE` | P20_45feat: 45 features, all decision-time valid, parity + leakage + causality PASS |
| **21** | Real-world validation gate | `INSUFFICIENT_REAL_WORLD_EVIDENCE` | No legitimate real-world transaction dataset available; promotion blocked pending data acquisition |
| **22** | Executable validation harness | `DATA_ACQUISITION_BLOCKED` | Complete CLI runner built (`phase22.run_real_world_validation`); READY but blocked by data absence |
| **23** | Data acquisition gate | `CONDITIONALLY_AVAILABLE` | 114 sources evaluated; best candidate Kaggle fraudTrain (1.85M rows); provenance unclear, no channel info, no label timing |
| **23B** | Frozen Kaggle external eval | `CONDITIONAL_EXTERNAL_EVALUATION_COMPLETE` | P20_45feat on Kaggle (1.85M rows, 0.58% fraud): ROC-AUC 0.47 (degraded features); E_hardneg unlabeled; production promotion BLOCKED |
| **24** | Conditional external eval | `CONDITIONAL_EXTERNAL_EVALUATION_COMPLETE` | E_hardneg on Kaggle (555K test rows): ROC-AUC 0.435, recall 26.8%, FPR 44.9%; 5 of 12 promotion gates failed; production promotion BLOCKED |
| **25** | Leakage audit & corrections | `NO_LEAKAGE — TOOLING VERIFIED` | Fixed Phase 24's import-path + tz-comparison + index-order bugs; maximal causal reconstruction (47/48 features); no target/temporal/feature leakage; promotion BLOCKED |

### Current Model Status

| Model | Features | Decision-Time Valid | Real-World Evidence | Status |
|---|---|---|---|---|
| **E_hardneg** (incumbent) | 48 | 3 fraud-rate features UNVERIFIED | None (synthetic only) | DEPLOYED, UNTOUCHED |
| **P20_45feat** (candidate) | 45 | All features valid | None (synthetic only) | ELIGIBLE for real-world validation |

### Real-World Validation Status

```text
PHASE22_RUNNER = READY
PHASE23_ACQUISITION = CONDITIONALLY_AVAILABLE_WITH_CAVEATS
PHASE23B_EVAL = CONDITIONAL_EXTERNAL_EVALUATION_COMPLETE
PHASE24_EVAL = CONDITIONAL_EXTERNAL_EVALUATION_COMPLETE (E_hardneg: ROC-AUC 0.435)
PHASE25_AUDIT = NO_LEAKAGE (post-reconstruction re-eval: ROC-AUC 0.595, recall 18.5%, FPR 9.9%)
REAL_WORLD_VALIDATION = BLOCKED (provenance unclear)
PROMOTION = BLOCKED
NEXT_ACTION = Continue authorized real-world data acquisition; wire Phase 25's
              reconstruction-diagnostics + leakage-audit outputs into promotion_guard.py's
              FEATURE_AVAILABILITY/FEATURE_PARITY gates (still Phase 24's hardcoded FAIL stubs)
```

**Honest interpretation:** E_hardneg is certified on the synthetic IBM corpus
(recall 99.67%, FPR 10.99% on the untouched 2018–2020 window) and remains
deployed as the best available model. The 2017 regime shift is strongly
supported as a generator artifact, so these numbers must not be read as
real-world performance. Production promotion of P20_45feat is blocked pending
real-world data — the next milestone is data acquisition, not more modeling.

The Phase 22 executable validation harness is READY:
```bash
python -m phase22.run_real_world_validation --mode prepare
python -m phase22.run_real_world_validation --data DATASET --metadata METADATA
```

Phase 23 evaluated 114 data sources. The best candidate (Kaggle fraudTrain)
has unclear provenance and missing channel/label-timing metadata.

Phase 23B executed a frozen conditional evaluation of P20_45feat on the
Kaggle dataset (1.85M rows, 0.58% fraud). With degraded features (33/45
required historical context unavailable), P20 scored ROC-AUC 0.47 with
all alerts firing. E_hardneg could not be evaluated because its
label-dependent features require verified label timing.

Phase 24 executed a full conditional external evaluation of E_hardneg
on the Kaggle test set (555K rows, 2,145 fraud). With 20/48 features
reconstructed exactly and 3 fraud-rate features computed causally from
training data, E_hardneg scored ROC-AUC 0.435, recall 26.8%, FPR 44.9%.
This confirms the model does not generalize beyond synthetic IBM data.
5 of 12 promotion gates failed (provenance, governance, latency,
feature availability, feature parity). Production promotion remains
BLOCKED — the next milestone is acquiring legitimate real-world data
with verified provenance and label governance.

Phase 24 evaluation is fully executable:
```bash
python experiments/phase24_conditional_external_eval/run_evaluation.py
python experiments/phase24_conditional_external_eval/run_evaluation.py --dry-run
```

Phase 25 corrected Phase 24's defects and closed the remaining audit gap:
fixed the zip import-path bug (the runner now works from an extracted zip),
rebuilt feature reconstruction maximally causally (47/48 features available;
only `err` is unreconstructable — no error/decline code in Kaggle), and ran
the leakage audit: no target, temporal, or feature leakage; temporal check
(now bug-fixed) finds 0 future timestamps. The cold-start audit verified
the rate features are healthy in the shipped train-aggregate design
(user 24.8% / merchant 1.3% / city 21.6% degenerate; median 1,054 tx/card
history) — the widely-cited "76% degenerate" figure applies only to a
running-rate design that was NOT evaluated. Metrics were unchanged
(ROC-AUC 0.595, recall 18.5%, FPR 9.9%); promotion remains BLOCKED and real-
world data acquisition remains the next milestone.

## Setup

Requires Python 3.10+.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate   |   macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

### Demo accounts (for testing)

| Account | Email | Password | Role |
|---|---|---|---|
| Alice | `alice@test.com` | `AlicePass123!` | user |
| Bob | `bob@test.com` | `BobSecure456!` | user |
| Carol | `carol@test.com` | `CarolSafe789!` | user |
| Dave | `dave@test.com` | `DaveTest012!` | user |
| Eve | `eve@test.com` | `EveDemo345!` | user |

Admin access:
- **Front page**: http://127.0.0.1:8000 → click 🔒 Admin
- **Admin passphrase**: from `.env` `ADMIN_PASS`
- **Compliance passphrase**: from `.env` `COMPLIANCE_TOKEN`

## Run

### ML pipeline

```bash
python src/generate_synthetic_data.py          # → data/transactions.csv
python src/train_compare.py                    # → models/artifacts/ (metrics, EVALUATION.md, model files)
```

### Feedback loop (§7: verification outcome → labeled data → retraining)

Confirmed/disputed outcomes in DB-3 are exported into a labeled training
pool (joined to the stored feature vectors in DB-2, `confirmed` → 0,
`disputed` → 1) and merged into the synthetic set for retraining. The pool
is **accumulative and versioned**: every export rebuilds it from all DB-3
outcomes (idempotent, no double counting) and writes a dated snapshot
`data/feedback_snapshots/feedback_YYYYMMDD_HHMMSS.csv`, so any point in the
pool's history is reproducible:

```bash
python scripts/export_feedback.py                    # → data/feedback_labeled.csv + dated snapshot
python src/train_compare.py --feedback-date latest  # consume the newest snapshot
python src/train_compare.py --feedback-date 20260816   # reproduce training from any date
python src/train_compare.py --feedback data/feedback_labeled.csv  # explicit file (unchanged)
```

The trainer records the consumed snapshot's path and a content hash in
`metadata.json` (`feedback_source`, `feedback_sha256`), so trained artifacts
are attributable to an exact pool state. `scripts/training_pipeline.py`
consumes the latest snapshot by default (falling back to the working pool
file only when no snapshots exist).

**Section-6 gate + retrain-queue trigger.** An export is **gated** (exit
code 2, nothing written) while the pool has fewer than `--min-outcomes`
REAL verified outcomes (default 10; simulated rows never satisfy it;
`--force` bypasses) — the pool must be big enough to train on. Once the
pool's real **disputed** events reach `--disputed-threshold` (default 5),
the export appends a hash-chained `retrain_trigger` event to the DB-4 audit
trail (mirroring the drift monitor's `drift_alert`) — the signal that
enough fraud-positive feedback has accumulated to justify a retraining
round. A successful export that leaves the pool unchanged writes no new
snapshot. A gated export leaves the latest snapshot intact, so the
pipeline keeps consuming the last valid pool.

`--simulate N` appends a mechanics-demo wave sampled from an existing
labeled CSV (archetype `feedback_sim`) — circular by construction, for
validating the pipeline at volume, not as evidence of real-world gain.
Retrained artifacts carry the feedback count in `model_version` (e.g.
`seed42-transactions.csv+2fb`). See `models/feedback_loop_report.md` for the
comparison.

### Training pipeline (§6/§7, one command)

Retrain → retune → apply → test as a single reproducible command:

```bash
python scripts/training_pipeline.py                       # retrain (with the feedback pool),
                                                          # sweep severity_scale, write rules.yaml,
                                                          # re-run the risk-engine tests
python scripts/training_pipeline.py --skip-train          # retune + apply + test on current artifacts
python scripts/training_pipeline.py --full                # also run the other six test suites
python scripts/training_pipeline.py --rules <staging.yaml> --skip-test   # dry-run apply elsewhere
```

Deterministic by construction (fixed seed; the sweep is pure numpy), so the
same inputs reproduce the same artifacts, the same recommended scale, and
the same green test run. The tuned `severity_scale` is written into
`src/risk_engine/rules.yaml` atomically with the previous file kept as
`rules.yaml.bak`; a scale-only change does NOT invalidate the tuner's
per-event score cache (the key hashes the per-rule severities/conditions,
not the scale), so re-runs stay ~0.1s. `--rules` targets a staging copy for
dry-runs without touching the live config.

### Federated learning (§19, stretch)

Three simulated institutions on account-disjoint shards of one synthetic
population. Each institution is a separate worker process that trains a
balanced logistic regression (or, with `--mlp`, a one-hidden-layer ReLU
MLP) locally and exchanges ONLY weight vectors with the coordinator, which
averages them with FedAvg (weighted by local size). Raw data, features, and
labels never cross the boundary. Compares local-only vs federated vs a
centralized oracle:

```bash
python scripts/federated_test.py   # smoke test, 43 checks
python scripts/federated_sim.py    # full run -> models/federated_report.md
python scripts/federated_sim.py --mlp   # + LR vs MLP on the same shards -> models/federated_mlp_report.md
python scripts/federated_sim.py --dp-sweep 0.5,1,2,4,8   # DP privacy-utility trade-off
python scripts/federated_sim.py --hetero-sweep   # size x fraud-skew map -> models/federated_heterogeneity_report.md
```

**MLP comparison** (`--mlp`, `--hidden 16`, `--mlp-lr 0.1`): the same
account-disjoint shards, same rounds and local epochs, run once with LR and
once with the MLP (`src/federated/mlp.py` — hand-rolled for the same
params/epochs contract as `LocalLR`; aggregation is per-layer FedAvg via
`fedavg_params`, and the DP path uses a flat-vector variant,
`dp_aggregate_flat`). The report compares FedAvg convergence and the
**separability benefit** (fed MLP − fed LR PR-AUC). The honest finding on
this dataset: the 6 engineered per-event features already linearize the
fraud patterns (AND-of-flags interactions like CNP testing / ATO become
linear flag combinations), so the MLP's nonlinear capacity buys nothing —
it converges and tracks its own centralized bound, but the gap is ~0
(±0.01). The benefit needs raw inputs with genuinely nonlinear structure;
the point of the comparison is the FL mechanics, not SOTA.

**Heterogeneity map** (`--hetero-sweep`): the same population re-sharded
into the 3 institutions with different **size weights × fraud-rate skews**
(`split_institutions_skewed` — each account is drawn into institution i with
weight `w_i·(1 + (skew_i−1)·u)`, where u is the account's own fraud rate
normalized to [0,1], so skew>1 pulls fraud-heavy accounts in and skew<1
repels them; account-disjointness always holds). Each config reports the FL
gain (fed − local macro PR-AUC) and the per-institution **drag**
(local_i − fed_i on i's own test — positive = the global model is worse for
that institution than its own local model). The map's findings: FL helps
most when data is concentrated (size-skew +0.097) or a clean dominant exists
(clean dominant +0.143) — small institutions jump from local PR-AUC
~0.57–0.72 to ~1.00 under FedAvg; the only positive drag appears at the
clean dominant's own test (+0.008) where its clean patterns are marginally
diluted; and when the dominant already holds the fraud (fraud dominant) FL
adds ~nothing (+0.002). Institutions whose test split has no fraud are
marked n/a and excluded from the macro (a 0.0000 there would be a false
'disaster').

**Differential privacy** (`--dp-sweep` / `--dp-epsilon`): client-level DP
FedAvg (McMahan-style) — the coordinator clips each per-client update delta
to a norm bound S and adds Gaussian noise calibrated by **RDP composition**
over all rounds (`src/federated/dp.py`). S is auto-calibrated to the clean
run's median update norm; each epsilon is averaged over `--dp-seeds` noise
draws (± std), and the trade-off vs the clean baseline lands in
`models/federated_dp_report.md`.

Honest caveats (also in the reports): naive FedAvg without differential
privacy leaks information via weight updates; the DP variant is CENTRAL
(server-side noise — production pairs it with secure aggregation); with
only 3 institutions the DP budget is shared across a few contributors, so
even ε = 8 costs ~2/3 of the clean PR-AUC and ε ≤ 4 leaves the model
indistinguishable from random (that is the quantified price); LR and the
small MLP are the weight-averageable architectures (RF/XGB/ISO are not);
per-institution local standardization; the oracle is the theoretical bound
computed out-of-band; synthetic data is cleanly separable so absolute
scores are inflated.

### Drift monitoring (§6: population stability index)

Week-over-week feature-distribution checks against the distribution the
models were trained on. The baseline is stored once as bin edges + expected
proportions (never the raw training rows); each check compares only the
current window — drift triggers a retraining review, it does not require
retaining raw data:

```bash
python scripts/drift_monitor.py build-baseline                # → data/drift_baseline.json
python scripts/drift_monitor.py check                         # last 7 days from DB-2 (live path)
python scripts/drift_monitor.py check --window-csv window.csv # explicit window (offline/CI)
```

Exit codes: 0 = ok, 1 = warn (PSI ≥ 0.10), 2 = alert (PSI ≥ 0.25),
3 = insufficient data. An alert-level breach automatically appends a
hash-chained `drift_alert` event to the DB-4 audit trail (feature PSI only,
pseudonymous) — the compliance viewer shows it like any other event.

### k-anonymity gate on exports (threat scenario D)

Re-identification defense for *released* feature-store datasets: an attacker
with auxiliary knowledge of a person's transaction behavior (amount, time,
device, payee pattern) can match it against a released pseudonymous export
and defeat pseudonymization. The gate rejects any release where a profile is
uniquely identifiable — every combination of quasi-identifiers must occur in
≥ k rows drawn from ≥ k *distinct accounts* (k rows all from one account
still reveal that account):

```bash
python scripts/k_anonymity_check.py --csv data/transactions.csv --k 5
python scripts/k_anonymity_check.py --db --days 30                 # live DB-2 export
python scripts/k_anonymity_check.py --csv release.csv --k 5 \
    --report kanon_report.json --suppress safe_release.csv
```

Quasi-identifiers default to the store's coarse, purpose-limited behavioral
attributes (`txn_amount_bucket` + the boolean flags; §16 — no new raw data
is collected to protect the release). Exit codes: 0 = k-anonymous, 1 =
REJECT (uniquely identifiable profiles), 2 = input error. `--suppress`
removes the violating rows into a k-anonymous release (publish less, do not
collect more); the exit code still reflects the original dataset, so
re-check the release. The current training set fails k=5 (103 long-tail rows
in `extreme_relative_to_avg` flag combinations) and passes only after
suppression — a real finding about what is safe to release.


### Services

Each service owns a physically separate SQLite store (`db/identity.db` =
DB-1, `db/features.db` = DB-2) — no shared credentials, no cross-queries.
Production swaps the SQLite engines for per-store PostgreSQL with its own
network segment and KMS keys (section 2).

**Front page:** `src/front_service` (port 8000) serves the project landing
page with live per-service status tiles (health, latency, uptime since
restart, plus the audit chain's integrity as a footer badge), quick access
to the alerts and compliance UIs, and the architecture/access overview —
`GET /status` aggregates all five health checks server-side (no CORS
needed), with the O(chain) integrity recompute cached on a 30s TTL. The
page polls every 8s, backing off to 30s after three consecutive network
failures (a dead host isn't hammered) and snapping back to 8s on the first
success — a healthy all-down report keeps the fast cadence so recovery is
spotted promptly. The offline path is covered twice:
`scripts/front_service_test.py` forces it hermetically (unreachable URLs),
and `scripts/front_offline_check.py run` stops the real five backends,
asserts the all-down shape + badge, and restores the stack (opt-in — never
part of the pipeline). Two corner entry points sit top-right: **👤 Sign in**
logs in an account holder against the Identity Service's JWT (`/auth/login`,
password never stored here — only the pseudonymous token is kept in the
tab's sessionStorage, with an "Open the alerts UI" link for the next step)
and **🔒 Admin** is the passphrase-gated admin area: login with
`ADMIN_USER`/`ADMIN_PASS` (bootstrap passphrase from `.env`; only a scrypt
hash is stored afterwards) to view the essential access payload — service
map, credentials (masked by default), break-glass — which is persisted
AES-256-GCM encrypted under a key derived from the passphrase
(`db/admin.json`, minimal fields only; rotate via the panel). The admin
area supports **TOTP 2FA** (RFC 6238, 30-second codes): once enabled, login
requires both the passphrase and a 6-digit authenticator code. TOTP setup
is inline — generate a Base32 secret, scan it with any authenticator app,
verify the code to enable. Disable requires a valid code.

A hidden **Admin Database Explorer** (double-click the footer to reveal)
provides read-only SQL access to all five databases (identity, features,
risk, verify, audit). It requires admin JWT + passphrase, blocks all
write operations (INSERT/UPDATE/DELETE/DROP), caps results at 500 rows,
and audit-logs every query to the hash chain. Click a table name to
auto-fill a `SELECT *` query.

The UIs build their service links from the browser's hostname, so the same
stack works on localhost, LAN, or a VPS behind a proxy (see `HOSTING.md`).

```bash
# Front page (landing + status)   → http://127.0.0.1:8000
uvicorn src.front_service.main:app --port 8000

# Identity Service (user-facing)  → http://127.0.0.1:8001
uvicorn src.identity_service.main:app --port 8001

# Privacy Layer (internal only)   → http://127.0.0.1:8002
uvicorn src.privacy_layer.main:app --port 8002

# Risk Engine (internal only)     → http://127.0.0.1:8003
uvicorn src.risk_engine.main:app --port 8003

# Verification Service (user-facing UI) → http://127.0.0.1:8004
uvicorn src.verification_service.main:app --port 8004

# Audit Service (compliance role)  → http://127.0.0.1:8005

Windows one-command stack: `powershell -ExecutionPolicy Bypass -File
.freebuff/start_stack.ps1` — idempotent (stops python/uvicorn listeners on
ports 8000–8005 and restarts all six); also the recovery path when a
service has died. The demo button surfaces its seed summary ("Seeded 7
events · baseline committed 5/6 · 1 high-risk alert") — 5/6 is the typical
count on a fresh account: the birth event steps up on its own brand-new
device and is excluded from the baseline; re-seeds can exclude more (new
demo device stays unknown until an allowed event), and the summary always
shows the real count. After a decision, the verification UI's **History**
toggle lists the caller's resolved cases (`GET /cases`, joined with the
risk score) and the **feedback-pool** status (`GET /feedback-status`:
confirmed/disputed counts + the retrain trigger threshold),
auto-refreshing when a new decision lands while open. Each case card
expands to show why it was flagged (reason chips) — `/cases` joins the
reason codes/texts through to history — and a summary chip row
(`N cases · confirmed · disputed`) matches the compliance trail's compact
treatment. The whole screen is one fetch: the UI loads `GET /overview`
(alerts + cases + feedback + chain status in a single BFF payload) instead
of chaining `/alerts` + `/cases` + `/feedback-status` + `/chain-status`;
opening History re-renders from that cached payload with zero extra calls. The
alerts screen also **auto-refreshes every 30s while it's open** (silent — no
spinner, errors keep the last good state; a generation token makes the newest
load win), so new alerts appear without clicking Refresh.

**API client is generated from the OpenAPI schema** (`src/verification_service/static/api.js`):
index.html fetches each service's `/openapi.json` at load time and builds a
callable per endpoint (`api.identity.login(...)`, `api.verify.confirmAlert(...)`,
`api.verify.complianceEvents({limit, offset})`, ...) — a new backend endpoint is
automatically callable with no hand-written fetch wrapper. The FastAPI apps set
`generate_unique_id_function` so `operationId == handler name` (stable, readable
helper names); auth is per-operation (bearer default, compliance for the trail
view, none for register/login/chain-status).
uvicorn src.audit_service.main:app --port 8005
```

Endpoints (section 14):

| Service | Endpoint | Notes |
|---|---|---|
| Identity | `POST /auth/register` | creates user + auth credential + `pseudonym_mapping` |
| Identity | `POST /auth/login` | returns JWT with `fraud_id` claim only (15 min) |
| Identity | `GET /me/fraud-id` | Bearer; caller's own ID; access logged |
| Identity | `POST /internal/resolve-fraud-id` | internal token; break-glass, always logged, returns masked PII |
| Privacy | `POST /internal/ingest-transaction` | internal token; returns §16 feature vector; stores derived features only — **does not** update the behavioral baseline; idempotent on event_id (replays return stored vector without re-auditing) |
| Privacy | `POST /internal/commit-baseline` | internal token; applies a stored vector to the profile (median, typical hours, device) — called ONLY for allowed/confirmed events |
| Risk | `POST /internal/evaluate` | internal token; fuses models + rules → score, band, decision, reason codes, `ml_score` (calibrated probability), `odds`, `degraded` (fail-open flag), `uncertainty` (model variance, disagreement, confidence level), `feature_version`, `rule_version`; enforces `velocity_limits` from the §16 vector first; idempotent on event_id |
| Risk | `POST /internal/attribution` | internal token; SHAP-style attribution — per-model + per-feature contributions + uncertainty (analyst only, nothing persisted) |
| Verify | `GET /alerts` | Bearer; high-risk alerts for the caller (unresolved) |
| Verify | `POST /alerts/{event_id}/confirm` | Bearer; `this_was_me` \| `this_wasnt_me` → outcome + case ID |
| Verify | `GET /alerts/{event_id}/reason` | Bearer; §11 category-level explanation |
| Verify | `POST /demo/seed` | Bearer; DEV ONLY — chains Privacy→Risk for demo alerts; commits allowed events to the baseline |
| Verify | `GET /cases` | Bearer; resolved verification cases, newest first, joined with risk score + reason codes/texts |
| Verify | `GET /feedback-status` | Bearer; feedback-pool counts + the retrain trigger threshold (§6) |
| Verify | `GET /overview` | Bearer; **BFF aggregate** — alerts + cases + feedback + chain status in one round trip; the UI renders everything from this. Served from a short-lived in-process cache (10s TTL), invalidated on every confirm/demo-seed write |
| Verify | `GET /compliance/events` | compliance token; pseudonymous decision trail (scores, reason codes, outcomes, case IDs), proxied from Audit |
| Verify | `GET /compliance/integrity` | compliance token; hash-chain integrity (ok, n_entries, first_bad_seq) |
| Verify | `GET /compliance/overview` | compliance token; **BFF aggregate** — integrity + latest case (joined with its score) + paged events + chain-wide aggregates in one call; the compliance view loads from this |
| Verify | `GET /investigator/cases` | compliance token; list investigator cases filtered by status/priority, sorted by priority descending |
| Verify | `POST /investigator/cases` | compliance token; create investigator case with auto-computed priority (score × confidence_weight) |
| Verify | `POST /investigator/cases/{case_id}/transition` | compliance token; transition case state (NEW→REVIEWING→USER_VERIFICATION→CONFIRMED_*/CLOSED), audit-logged |
| Audit | `GET /audit/events` | internal token (compliance role); pseudonymous trail, reads logged |
| Audit | `GET /audit/integrity` | internal token; recomputes the whole chain, reports first bad link |
| Audit | `GET /compliance/events` · `/compliance/integrity` · `/audit/overview` | compliance passphrase / internal token; same trail/integrity plus the BFF aggregate for the browser viewers |
| Audit | `GET /` | static compliance viewer UI (per-event hash linkage + integrity banner) |
| Front | `POST /admin/login` | public; admin passphrase (+ optional TOTP code) → JWT + essentials |
| Front | `POST /admin/db-query` | admin JWT + passphrase; read-only SQL against any of the five databases |
| Front | `GET /admin/db-tables` | admin JWT + passphrase; list tables + row counts for a given database |
| Front | `GET /admin/totp/setup` | admin JWT; generate a pending TOTP secret (Base32) |
| Front | `POST /admin/totp/verify` | admin JWT; verify TOTP code and enable 2FA |
| Front | `POST /admin/totp/disable` | admin JWT; disable 2FA (requires valid code) |
| Front | `GET /admin/totp/status` | admin JWT; check whether TOTP is enabled |
| Front | `POST /admin/rotate` | admin JWT; change the admin passphrase (revokes all sessions) |

#### Data-store access matrix

Every store is owned by exactly one service; every read passes through that
service's API with one of three credentials. Production replaces the shared
tokens with per-service mTLS + short-lived RBAC credentials (section 3) and
the shared DB-3 prototype connection with a single owning-service boundary.

| Store | DB file | Owned by | Holds | Passage (exact endpoint) | Credential | Who can use it |
|---|---|---|---|---|---|---|
| DB-1 | `db/identity.db` | Identity Service (:8001) | PII, auth credentials, `pseudonym_mapping` (fraud_id ↔ user) | `POST /auth/register` · `POST /auth/login` | none | anyone creating / owning an account |
| DB-1 | “ | “ | “ | `GET /me/fraud-id` | Bearer JWT (15 min) | the account holder — own pseudonym only |
| DB-1 | “ | “ | “ | `POST /internal/resolve-fraud-id` (break-glass, always logged) | `X-Internal-Token` | Identity Service / authorised ops — the **only** passage that links a pseudonym back to a person (returns masked PII) |
| DB-2 | `db/features.db` | Privacy Layer (:8002) | pseudonymous feature vectors, behavioural profiles, device fingerprints — no PII, no raw amounts | `POST /internal/ingest-transaction` · `POST /internal/commit-baseline` · `POST /internal/demo-age-account` | `X-Internal-Token` | service-to-service (Privacy, Verification) |
| DB-2 | “ | “ | “ | `GET /internal/device-graph` (also proxied as `GET /compliance/device-graph` on :8004) | `X-Internal-Token` / `X-Compliance-Token` (proxy) | compliance role (via the proxy) |
| DB-3 | `db/risk.db` | Risk Engine (:8003); Verification Service shares it in the prototype (§13: `verification_outcomes` sits in DB-3) | `risk_scores` (pseudonymous) + `verification_outcomes` | `POST /internal/evaluate` · `POST /internal/attribution` | `X-Internal-Token` | Risk Engine; analysts (attribution, nothing persisted) |
| DB-3 | “ | “ | “ | `GET /alerts` · `POST /alerts/{id}/confirm` · `GET /alerts/{id}/reason` · `GET /cases` · `GET /history` · `GET /overview` · `GET /chain-status` | Bearer JWT | the account holder — own rows only |
| DB-3 | “ | “ | “ | `POST /demo/seed` | Bearer JWT | DEV ONLY — demo seeding |
| DB-4 | `db/audit.db` | Audit Service (:8005) | append-only, hash-chained `audit_events` (the whole decision trail) | `GET /audit/events` · `GET /audit/integrity` · `GET /audit/overview` · `GET /audit/export` | `X-Internal-Token` (+ `X-Audit-Actor` for reads) | compliance role (service-to-service); `/audit/export` is HMAC-signed so an external regulator can verify independently |
| DB-4 | “ | “ | “ | `GET /compliance/events` · `GET /compliance/integrity` (and the :8004 `GET /compliance/*` proxies) | `X-Compliance-Token` (compliance passphrase) + `X-Analyst-ID` header | compliance-role viewer (browser, pseudonymous trail); each read logged with analyst identity |
| All | `db/*.db` | Front Service (:8000) | read-only SQL access to all five databases | `POST /admin/db-query` · `GET /admin/db-tables` | Admin JWT + admin passphrase (+ TOTP if enabled) | admin only; allowlist queries only (no arbitrary SQL), SAFE_COUNT_TABLES filter, 500-row cap, every query audit-logged |

**Three human passages, one pseudonym↔person passage:** the account holder
reaches their own rows via **JWT**; the compliance role reaches the
pseudonymous decision trail via the **compliance passphrase** (each
analyst identified by `X-Analyst-ID` header, logged to audit chain);
and only **break-glass** (`/internal/resolve-fraud-id`, internal token,
always logged) can map a fraud_id to a person. Nothing else reads PII or
raw amounts anywhere in the system.

Dev secrets live in `src/settings.py` (env-overridable; see `.env.example`).

**Presentation deck:** `docs/PS-14_fraud_detection.pptx` (16 slides, dark
theme, speaker notes) — regenerate with `python scripts/build_presentation.py`.

**Private access document (encrypted at rest):** `docs/ACCESS.md.enc` is a
Fernet-encrypted copy of all private access info — credentials, service map,
the data-store access matrix, and quick procedures. Open it with
`python scripts/access_doc.py view` (the passphrase lives in the gitignored
`.env` as `ACCESS_DOC_KEY`, so no prompt is needed locally); rotate with
`python scripts/access_doc.py rekey`. The plaintext master `docs/ACCESS.md`
is gitignored — never commit it.

### Smoke test

Exercises the full flow and asserts the DB-separation guarantee from
section 2 (DB-1 has no feature tables; DB-2 has no PII, raw amounts, or
device IDs):

```bash
python scripts/smoke_test.py        # identity + privacy, 35 checks
python scripts/risk_engine_test.py  # risk engine, 37 checks
python scripts/verification_test.py # verification flow, 30 checks (incl. compliance viewer)
python scripts/audit_test.py        # audit chain, 33 checks (incl. tamper)
python scripts/verify_export.py export.json   # regulator-side export verifier
python scripts/drift_test.py        # PSI drift monitor, 17 checks (incl. hash-chained alert)
python scripts/k_anonymity_test.py  # threat-D k-anonymity gate, 23 checks
python scripts/federated_test.py    # federated learning + DP + MLP + hetero map, 43 checks
python scripts/front_service_test.py # front page + admin + offline status, 33 checks
python scripts/security_test.py     # CORS + rate-limiting + headers + key separation, 9 checks
python scripts/resilience_test.py    # resilience beyond ML: DB/audit/privacy failure paths
python scripts/ood_gate_test.py      # OOD recall gate validation
python scripts/rules_gate_test.py    # rules backtesting gate validation
python scripts/pseudonym_separation_test.py  # DB-2 compromise doesn't reveal identity
python scripts/feature_compatibility_test.py # training/serving feature skew prevention
python scripts/production_gate_test.py       # fail-closed startup in production mode
```

Full test suite (17 suites, all passing):

```bash
python scripts/regression_suite.py          # fast mode (11 suites, ~30s)
python scripts/regression_suite.py --full   # full mode (17 suites, ~95s)
#   risk_engine_test.py     # 35 checks: scoring, bands, rules, attribution, DB-3
#   smoke_test.py           # full pipeline: register → ingest → evaluate → audit
#   k_anonymity_test.py     # k-anonymity gate on exports
#   pseudonym_separation_test.py  # identity/fraud DB isolation
#   ood_gate_test.py        # out-of-distribution recall gate
#   rules_gate_test.py      # rules.yaml validation gate
#   pipeline_test.py        # training pipeline mechanics
#   security_test.py        # CORS, rate-limiting, headers, key separation
#   verification_test.py    # verification flow, compliance UI, external scripts
#   audit_test.py           # audit chain, hash integrity, compliance UI
#   front_service_test.py   # front page, admin, offline status
#   resilience_test.py      # DB/audit/privacy failure paths
#   feedback_test.py        # feedback loop, snapshots, gate trigger
#   tuner_test.py           # operating-point tuning, cache mechanics
#   drift_test.py           # PSI drift monitor, alerts
#   federated_test.py       # federated learning, DP, heterogeneity
#   feature_compatibility_test.py  # training/serving feature skew
```

Additional tooling:

```bash
python scripts/load_test.py -n 500 -c 20   # performance/load testing
python scripts/data_retention.py --dry-run  # data retention enforcement (preview)
python scripts/fairness_review.py           # bias analysis across account segments
python scripts/security_scan.py --full     # security scan (custom SAST + dependency check)
python scripts/quality_scorecard.py        # consolidated metrics dashboard
python scripts/penetration_test.py         # offensive security: 42 vectors (all blocked when services running)
python scripts/benchmark_comparison.py     # 1.2M dataset benchmark vs industry
python scripts/compare_ml_systems.py       # PS-14 vs 8 industry systems (rankings, privacy, explainability)
python scripts/ps14_client.py              # easy API client (CLI + Python)
```

## Hosting

One command, anywhere: `docker compose up --build -d` runs all six services
(front :8000 + the five) with a shared data volume, health checks, and
internal service-name URLs — the landing page is at http://localhost:8000.
Full guide: **`HOSTING.md`** (LAN, VPS + Caddy reverse proxy, secrets,
backups, hardening checklist). Without Docker, `.freebuff/start_stack.ps1`
boots the same six locally.

### Security Hardening

For production deployment, see **`docs/SECURITY_HARDENING_GUIDE.md`** which covers:
- Pre-deployment checklist
- Secrets management and rotation
- Network security (TLS, mTLS, CORS)
- Authentication and authorization
- Data protection (encryption, backups)
- Container security
- Monitoring and logging
- Incident response
- Compliance (DPDP Act, RBI guidelines)

Run security scans before deployment:
```bash
python scripts/security_scan.py --full  # Security scan (custom SAST — not a comprehensive audit)
python scripts/security_test.py         # Runtime security tests (9 checks)
python scripts/penetration_test.py      # Offensive security (42 vectors, all blocked when services running)
python scripts/security_ci_gate.py      # CI gate: fails on ANY finding
```

**⚠️ Security Note**: The security tests (`security_scan.py`, `penetration_test.py`, `security_ci_gate.py`) are self-authored test suites, not a third-party audit. A professional penetration test is recommended before handling real financial data.

**Dockerized retrain:** `bash scripts/docker_retrain.sh` (or `make retrain`)
runs `training_pipeline.py --full` inside a compose `train` profile
container (`./models` + `./data` bind-mounted), then rebuilds and restarts
**only the risk service** — and only when every step passes, including the
OOD recall gate. `make train-only` / `NO_RESTART=1` skips the rebuild.

### Cross-domain evaluation (4 real datasets)

PS-14 was evaluated against 4 real public datasets mapped to the §16 feature
space, proving that domain-specific training is mandatory:

| Dataset | Source | Rows | Positive | Rate |
|---|---|---|---|---|
| Kaggle ULB Credit Card Fraud | Dal Pozzolo et al. | 284,807 | 492 fraud | 0.17% |
| UCI Credit Card Default | Yeh & Lien, 2009 | 30,000 | 6,636 default | 22.1% |
| UCI Bank Marketing | UCI #222 | 41,188 | 4,640 subscribe | 11.3% |
| UCI Diabetes Readmit | UCI #296 | 101,766 | 11,357 readmit | 11.2% |

**Cross-domain ROC-AUC matrix** (train → test):

| Train \ Test | Kaggle | Default | Bank | Diabetes |
|---|---|---|---|---|
| Kaggle | **0.94** | 0.36 | 0.44 | 0.60 |
| Default | 0.71 | **0.74** | 0.56 | 0.54 |
| Bank | 0.53 | 0.53 | **0.94** | 0.56 |
| Diabetes | 0.35 | 0.36 | 0.38 | **0.61** |

**Key finding:** Domain-specific training always wins (diagonal = max in each
column). Cross-domain transfer drops 17–58% because PCA features and
payment-behavior features encode fundamentally different signals.

```bash
python scripts/real_datasets_expanded.py   # 4×4 cross-domain evaluation
python scripts/meta_ensemble.py            # 6-strategy meta-ensemble comparison
python scripts/train_compare_uci.py        # UCI vs Kaggle head-to-head
```

### Meta-ensemble strategies

Six strategies were compared for combining domain-specific models:

| Strategy | Avg ROC-AUC | vs Single-Domain |
|---|---|---|
| Single-domain (best) | **0.825** | — |
| Universal meta-learner | 0.798 | -2.7% |
| Weighted average | 0.777 | -5.8% |
| Simple average | 0.773 | -6.3% |
| Best-of (max) | 0.771 | -6.5% |
| Meta-learner (LOO) | 0.396 | -52.0% |

**Verdict:** If you know the domain → use the domain-specific model. If the
domain is unknown or mixed → use the universal meta-learner (only -2.7% drop).

### Model uncertainty (§2.1)

`/internal/evaluate` and `/internal/attribution` now return an `uncertainty`
dict with `model_variance` (variance of the four base learner outputs),
`model_disagreement` (max − min spread), and a derived `confidence` level
(`"high"` if variance < 0.01, `"medium"` < 0.05, `"low"` otherwise). Low-
confidence + high-score cases naturally surface for human review via the
investigator case priority calculation. Raw per-model outputs are never
exposed to end users — the public surface keeps §11 category-level reason
codes. The uncertainty fields are audit-logged on every `score_generated`
event so analysts can retroactively see which decisions the model was
uncertain about.

### Idempotency / replay protection (§2.2)

Both `POST /internal/ingest-transaction` and `POST /internal/evaluate` are
idempotent on `event_id`: a duplicate request returns the stored result
with `"idempotent_replay": true` without re-auditing, so the hash chain
never double-counts a replayed event. Expired idempotency records are
bounded by the natural retention window (feature vectors after 90 days,
risk scores after 180 days — see Data retention below).

### Staged model rollout (§2.3)

`src/risk_engine/model_registry.py` provides a file-based registry
(`models/artifacts/registry.json`) for staged rollout: **shadow** mode runs
the candidate alongside production, logging predictions separately without
affecting decisions; **canary** mode routes a configurable % of traffic to
the candidate; **auto-rollback** triggers on latency p95 > 500ms, error
rate > 2%, or FPR spike. The registry keeps the last-known-good model
for instant revert. Direct deploy (`mode: "direct"`) remains the
default — shadow/canary is opt-in config, not a required detour.

### Model & feature versioning per decision (§2.4)

Every `score_generated` audit event now carries `feature_version` (currently
`"v1"` — bumped when `ML_FEATURES` changes) and `rule_version` (SHA-256
hash of `rules.yaml`, first 12 chars). An investigator can answer "which
model, which feature schema, which rule set produced this decision" from
the audit trail alone, without cross-referencing `models/artifacts/`.

### Investigator case workflow (§2.5)

A case-state machine for medium/high-risk events:
`NEW → REVIEWING → USER_VERIFICATION → CONFIRMED_SUSPICIOUS /
CONFIRMED_LEGITIMATE → CLOSED`. Each transition is audited to the hash
chain. Investigators see pseudonymous ID + score + confidence + reason
cases + prior related alerts — never real identity outside break-glass.
Priority is auto-computed as `risk_score × confidence_weight` (low-
confidence gets 1.5× boost for human review). Endpoints: `GET
/investigator/cases` (filter by status/priority), `POST
/investigator/cases` (create), `POST /investigator/cases/{id}/transition`
(state change).

### Data retention policy (§2.9)

`scripts/data_retention.py` enforces per-store retention periods:
- **DB-2 (features.db):** 90-day retention for feature vectors; behavioral
  profiles kept indefinitely (they ARE the account baseline).
- **DB-3 (risk.db):** 180-day retention for risk scores without outcomes;
  verification outcomes kept indefinitely (training data).
- **DB-4 (audit.db):** 365-day retention via redaction-in-place (append-only
  triggers temporarily disabled, payloads replaced with redacted stubs,
  triggers re-created — the hash chain stays unbroken).

Run with `--dry-run` for safe preview. Schedule as a cron/systemd timer
in production.

### Fairness / bias review (§2.10)

`scripts/fairness_review.py` segments risk scores by account tenure,
device count, and time pattern to detect disparities. It also analyzes
the §16 feature set for plausible proxy variables (hour_of_day → work
schedule, unusual_location_flag → geography, account_tenure_days → age)
and recommends whether each is necessary. No new sensitive attributes are
collected — the review works entirely from features the system already
has.

### Performance / load testing (§2.7)

`scripts/load_test.py` measures avg/p50/p95/p99/max latency and
throughput for `/internal/evaluate` and `/internal/ingest-transaction` with
configurable request count and concurrency. Run against a live stack:

```bash
python scripts/load_test.py -n 500 -c 20    # 500 requests, 20 workers
python scripts/load_test.py -n 100 --json    # JSON output for CI
```

### Resilience beyond ML failure (§2.11)

`scripts/resilience_test.py` verifies safe failure for DB unavailability,
audit-service unavailability, and Privacy Layer unavailability — each
should fail safely (no silent security downgrade, no ambiguous state),
matching the existing ML fail-open pattern. Also confirms security
middleware (CORS, rate-limiting, headers) is present on all 5 services,
idempotency checks exist at ingest and evaluate, model uncertainty is
computed, feature/rule versions are stamped per decision, and the
investigator case workflow is functional.

## Layout

```
src/
  settings.py                   # shared config (DB paths, secrets) - env-overridable
  generate_synthetic_data.py    # synthetic transaction generator (§16 shape)
  train_compare.py              # four-model comparison + artifact export
  identity_service/             # DB-1: PII, auth, pseudonym_mapping (FastAPI)
  privacy_layer/
    features.py                 # shared derivation: buckets, escalation, ML_FEATURES
    main.py                     # DB-2: ingest-transaction (FastAPI)
  risk_engine/
    main.py                     # DB-3: /internal/evaluate (FastAPI)
    fusion.py                   # stacked meta-learner over the 4 models
    model_registry.py           # staged rollout: shadow/canary/rollback
    rules_engine.py             # declarative rule evaluator
    rules.yaml                  # PR-reviewed rules config (§12)
    reason_codes.py             # §11 category-level explanation texts
  verification_service/
    main.py                     # alerts / confirm / reason + demo seed
                                # + /compliance/* proxy + investigator case workflow
    models.py                   # VerificationOutcome + InvestigatorCase (DB-3, §13)
    static/index.html           # self-contained verification UI (no build)
                                # incl. compliance trail view + integrity
  audit_service/
    main.py                     # compliance viewer + integrity checks
    writer.py                   # hash-chaining append path (shared)
    models.py                   # audit_events + audit_access_log (DB-4, §13)
  drift_monitor/
    psi.py                      # PSI math, baseline binning, thresholds (§6)
  k_anonymity/
    checker.py                  # threat-D k-anonymity + suppression (§16)
data/
  transactions.csv              # numeric feature matrix + label (ML input)
  events_sample.jsonl           # §16 "live event" shape, label=null
models/artifacts/
  metrics_comparison.csv / recall_by_archetype.csv / EVALUATION.md / *.joblib / metadata.json
db/                             # SQLite stores (gitignored): identity.db, features.db, risk.db
scripts/
  smoke_test.py                 # identity + privacy end-to-end + separation
  risk_engine_test.py           # bands, reason codes, DB-3 separation
  verification_test.py          # four-service verification flow
  audit_test.py                 # hash chain, tamper detection, append-only
  verify_export.py              # regulator-side verifier for the signed export
  drift_monitor.py              # PSI baseline build + window check + alerts (§6)
  drift_test.py                 # PSI math, drift detection, alert/audit
  k_anonymity_check.py          # k-anonymity gate on exports + suppression
  k_anonymity_test.py           # threat-D checks (unique profile, accounts)
  tune_operating_point.py       # cost-weighted tuning: global severity-scale sweep
                                #   and constrained per-rule rebalancing (§6)
  tune_test.py                  # tuner score-cache + per-rule search (23 checks)
  training_pipeline.py          # one-command retrain -> retune -> apply -> test (§6/§7)
  pipeline_test.py              # pipeline apply/backup + train-argv wiring (16 checks)
  feedback_test.py              # versioned feedback pool: snapshots, resolution,
                                #   gate + retrain-queue trigger (25 checks)
  live_walkthrough.py           # live HTTP walkthrough (servers must be up)
  live_curl_walkthrough.sh      # self-contained curl walkthrough (boots 3 services
                                #   on a clean db/walkthrough, registers + seeds +
                                #   ages, prints the allow/step-up/verify decisions)
  live_walkthrough.sh           # register-to-audit walkthrough (boots ALL FIVE
                                #   services, adds the verification loop, prints the
                                #   hash-verified audit trail + independent export check)
  load_test.py                  # performance/load testing: latency, throughput, error rates
  data_retention.py             # per-store retention enforcement (90d/180d/365d)
  fairness_review.py            # bias analysis across account segments + proxy variable review
  resilience_test.py            # resilience beyond ML: DB/audit/privacy failure paths
  security_test.py              # CORS, rate-limiting, headers, key separation tests
  front_service_test.py         # front page + admin + offline status tests
  penetration_test.py           # offensive security: auth bypass, SQLi, JWT forging, IDOR, allowlist tests
  pseudonym_separation_test.py  # proves DB-2 compromise doesn't reveal identity
  feature_compatibility_test.py # prevents training/serving feature skew
  production_gate_test.py       # fail-closed startup validation in production mode
  regression_suite.py           # unified test runner for CI/CD
  quality_scorecard.py          # consolidated metrics dashboard
  real_datasets_expanded.py     # 4-dataset cross-domain evaluation
  meta_ensemble.py              # 6-strategy meta-ensemble comparison
  train_compare_uci.py          # UCI vs Kaggle head-to-head training
  chameleon_rules_test.py       # chameleon fraud rule effectiveness
  fp_reduction_test.py          # false positive reduction analysis
  analyze_fp_rules.py           # FP rule analysis tool
  tune_parameters.py            # severity_scale sweep tool
  tune_rules.py                 # rule threshold tuning tool
  import_uci_default.py         # UCI dataset → PS-16 feature mapper
  deploy.sh                     # one-command production deployment
  retention-cron.sh             # cron wrapper for data retention
```

The model registry (`src/risk_engine/model_registry.py`) supports staged
rollout (shadow → canary → rollback) with auto-rollback on breach.
The investigator case workflow endpoints are in `src/verification_service/main.py`
with the `InvestigatorCase` model in `src/verification_service/models.py`.

### Production deployment

Three deployment options (see `HOSTING.md` for full guide):

```bash
# Option 1: Dev (SQLite, localhost)
docker compose up --build -d

# Option 2: Production (PostgreSQL, Caddy auto-HTTPS)
cp .env.example .env   # generate ALL secrets
docker compose -f docker-compose.prod.yml --env-file .env up --build -d

# Option 3: One-command deploy (validates + tests + builds + deploys)
bash scripts/deploy.sh              # full deploy
bash scripts/deploy.sh --dry-run    # validate only
bash scripts/deploy.sh --compose-prod  # use production compose
```

Production features:
- **Auto-HTTPS** via Caddy with Let's Encrypt (`Caddyfile`)
- **Per-store PostgreSQL** with separate credentials (`docker-compose.prod.yml`)
- **Internal-only network** (services can't reach the internet)
- **Fail-closed startup** when `PS14_MODE=production`
- **Automated data retention** (`scripts/retention-cron.sh` + systemd timers)
- **Hourly audit chain verification** (`scripts/ps14-audit-chain.timer`)
- **CI/CD gates** (`.github/workflows/ci-cd.yml`) block deployment on failures

### CI/CD pipeline status

Both workflows (`.github/workflows/ci-cd.yml`, `.github/workflows/security-scan.yml`) run on
every push to `main`. As of September 11, 2026 the pipeline is **fully green end-to-end** for
the first time — all five CI/CD stages pass: Test Suite (22/22 fast regression checks on a
clean checkout), Security Scan (bandit medium+ clean, dependency audit, penetration test),
Docker Build (image builds, runs as non-root, `/health` answers), Integration Test (the live
register-to-audit walkthrough boots all five services over real HTTP, then the compose stack
health-checks and the audit chain verifies), and Deploy (tag-gated).

Getting here surfaced and fixed real defects, not CI-only quirks:

- **Regression runner `NameError`** — the failing-check detail block referenced an undefined
  variable, so the first failing check crashed the runner instead of reporting which check
  failed. All failure detail was being silently swallowed.
- **Export-signing key was random per process** — `export_signing_key_str` never read the
  `EXPORT_SIGNING_KEY` env var, so any service started detached (Docker, CI, no-`.env`) signed
  compliance exports with an unrepeatable key; signatures were unverifiable by design.
  `load_dotenv_and_patch` also looked up lowercase field names in `os.environ`, which only ever
  worked on Windows (case-insensitive environ) — Linux containers silently missed every patch.
- **Fresh-checkout fragility** — several test suites assumed local artifacts a clone doesn't
  have (`db/*.db`, `data/creditcard.csv`, the production manifest); they now SKIP with a clear
  message when their inputs are absent and still run fully where the artifacts exist.
- **Penetration-test gate semantics** — "service unreachable" was counted as FAIL, so the
  security job could never pass without a live stack; unreachable is now BLOCKED (reported,
  non-gating) and only a demonstrated vulnerability fails the gate. The elevated-user test
  also sent a payload that FastAPI body-validation rejected (422) before the auth check —
  the attack was never reaching the code it claimed to test.
- **Docker health-check startup** — the CI smoke container runs the front service with
  `DB_DIR=/tmp/db`, which nothing created; `SessionStore` now creates its parent directory.
  The integration walkthrough also needed shared secrets exported to its children (on CI each
  child generated random tokens and every internal call 403'd) and free ports, so it runs
  before compose starts.

Run the same checks locally:
```bash
python scripts/regression_suite.py --fast   # 22 checks, ~60s
bandit -r src/ --severity-level medium      # gate: zero medium+ findings
python scripts/penetration_test.py          # 0 FAIL required (BLOCKED = service down)
bash scripts/live_walkthrough.sh            # full register-to-audit over HTTP
```

### Security hardening

Browser-standard security headers on all services:
- **CSP**: Development: `default-src 'self'; script-src 'self' 'unsafe-inline'; frame-ancestors 'none'; object-src 'none'; upgrade-insecure-requests`
- **CSP**: Production: nonce-based (`script-src 'self' 'nonce-{random}'`) — inline scripts blocked
- **No inline scripts**: All `<script>` blocks externalized to `/static/app.js` files (verification, audit, front services)
- **HSTS**: `max-age=31536000; includeSubDomains; preload`
- **X-Content-Type-Options**: `nosniff`
- **X-Frame-Options**: `DENY`
- **Referrer-Policy**: `strict-origin-when-cross-origin`
- **Permissions-Policy**: camera, microphone, geolocation, payment, USB all disabled
- **Cache-Control**: `no-store, no-cache, must-revalidate, private`
- **CSP meta tags** in all 3 HTML files as fallback
- **CSP violation reporting** endpoint: `POST /csp-report`
- **Nonce generation**: per-request cryptographically random nonces in production mode (`PS14_MODE=production`)

Authentication & authorization:
- **Argon2id** password hashing (user) + **scrypt** (admin)
- **Brute-force protection**: 5-attempt admin lockout, 10-attempt user lockout (15min)
- **JWT tokens**: 15-minute expiry, fraud_id only (no PII), constant-time comparison
- **RBAC roles**: user, analyst, admin with role-gated endpoints
- **TOTP 2FA** on admin account (RFC 6238, 30-second codes)
- **Per-analyst compliance credentials**: each analyst has unique `analyst_id` + passphrase, logged to audit chain on every compliance read (`src/audit_service/analyst_store.py`)

Input validation:
- **Pydantic Field constraints** on all request models (min/max length, patterns)
- **SQL injection eliminated**: arbitrary SQL replaced with allowlist of 4 pre-defined parameterized queries (`recent_audit`, `unresolved_alerts`, `recent_scores`, `recent_features`)
- **DB-tables restricted**: `row_count` and `table_list` only expose non-PII tables via `SAFE_COUNT_TABLES` and `PII_TABLE_NAMES` filters
- **No stack traces** in error responses (generic messages only)
- **No internal endpoints** in OpenAPI schemas (`include_in_schema=False`)

Run the penetration test to verify:
```bash
python scripts/penetration_test.py   # attack vectors — PASS/FAIL/BLOCKED per test
python scripts/security_ci_gate.py   # CI gate: fails on ANY finding
```

### Automated regression gates

Unified test runner for CI/CD:
```bash
python scripts/regression_suite.py   # runs all 17 test suites
```

Separate suites:
- **Fast** (11 suites, ~30s): pseudonym_separation, production_gate, smoke_test, risk_engine, drift_monitor, k_anonymity, ood_gate, rules_gate, tuner, pipeline, feedback_loop
- **Full** (+6, ~95s): security_test, verification_flow, audit_chain, front_service, resilience, federated_learning

## Live curl walkthrough (allow / step-up / verify)

The checked-in script `scripts/live_curl_walkthrough.sh` is the fastest way
to see the whole flow over real HTTP — no pre-started servers needed. It
boots Identity (:8001), Privacy (:8002) and Risk (:8003) on a dedicated,
gitignored `db/walkthrough/` store, registers a throwaway pseudonymous user,
seeds six normal transactions, **ages** the account to ~60 days (backdates
DB-2 rows so the time-derived features reflect a real history — a demo
harness; streaming events would carry real timestamps), and evaluates three
scenarios via plain `curl`:

```
bash scripts/live_curl_walkthrough.sh
```

```
scenario               decision   band     score   reasons
A                      ALLOW      low      2       none
B                      STEP_UP    medium   64      NEW_DEVICE,LOCATION_UNUSUAL,RECIPIENT_UNUSUAL
C                      VERIFY     high     100     AMOUNT_UNUSUAL,NEW_DEVICE,UNUSUAL_TIME,
                                                   LOCATION_UNUSUAL,RECIPIENT_UNUSUAL,
                                                   AUTH_ANOMALY,BEHAVIOR_DEVIATION
```

It also prints the §16 derived vector the Privacy Layer actually stored for
scenario A (no raw amount, no device id, no exact geo). Ports 8001–8003 must
be free — stop the dev stack first; the services it starts are stopped again
on exit (trap).

## Live register-to-audit walkthrough (all five services)

`scripts/live_walkthrough.sh` is the full pipeline version: it boots **all
five** services — Identity (:8001), Privacy (:8002), Risk (:8003),
Verification (:8004), Audit (:8005) — on the same clean `db/walkthrough/`
store, and runs the whole register-to-audit chain over plain `curl`:

```
bash scripts/live_walkthrough.sh
```

1. register a throwaway user → pseudonymous `fraud_id`, log in → JWT,
2. seed six normal transactions + age the account to ~60 days (as above),
3. evaluate the three scenarios → the allow / step-up / verify table,
4. **verification loop** — list the high-risk alert (`/alerts` with the
   JWT), dispute it (`this_wasnt_me`) → case id + recovery flow; the
   blocked event never enters the behavioral baseline,
5. **hash-verified audit trail** — the pseudonymous decision trail for the
   user (`feature_ingested` × 9, `score_generated` × 3,
   `verification_resolved` × 1, each with its `entry_hash`), the full-chain
   compliance export re-verified **independently** by
   `scripts/verify_export.py` (HMAC signature + per-entry hash linkage
   recomputed from genesis), and the Audit Service's own integrity verdict.

Ports 8001–8005 must be free by default; set `PS14_*_PORT`
(`PS14_IDENTITY_PORT` … `PS14_AUDIT_PORT`) to run alongside the dev stack
on other ports — the flow is self-contained (direct curl + the shared local
DB), so the running services are untouched. Services started by the script
are stopped again on exit (trap).

## 50-case batch test (live scoring)

`scripts/batch_cases.py` runs 50 distinct scoring scenarios through the
LIVE Risk Engine (`/internal/evaluate`) and checks the invariants that must
hold regardless of model weights:

```
python scripts/batch_cases.py [--recheck N] [--seed 7]
```

* **5 named real-world scenarios** (new device at the usual location;
  old device 250 miles away; new device + $1,500; usual device + $1,200;
  different location, small amount) printed as their own table;
* **45 seeded perturbations** across the whole §16 feature space
  (amount ratio, velocity, tenure, mule-ring, spend caps, auth anomalies);
* invariants: valid score/band/reason shape for all 50, **determinism**
  (re-evaluating a vector yields the same decision), **monotonicity**
  (adding a red flag never lowers the score), and the **audit chain grows
  by exactly the number of evaluations and still verifies from genesis**.

Requires the live stack (:8003 + :8005) — it is a live check, not part of
the pipeline. Exit 0 = all invariants held; the decision table and the
scenario bands are printed for reading. A clean run writes a last-run
summary to `DB_DIR/batch_report.json` (gitignored) which the front page
serves at `GET /batch` and renders as the **"Last 50-case batch run"**
panel — run time/seed, the allow / step-up / verify distribution with
bars, chain growth and integrity, and the five named scenarios — refreshed
on load and every 60s, hidden until a report exists.

## Notes

- **Risk Engine composition** (§5): the four model outputs are fused by a
  class-balanced logistic stacker fit on validation (`stacker.joblib`), and
  the final score is `100 * max(ml_fusion, rule_score)` — the strongest
  signal wins, so hard red flags can't be diluted by a confident ML pass.
  Critical rules (`level: critical`) floor the score at 80. The fusion beats
  every solo model (current test split: PR-AUC 0.9375 vs best solo 0.9340).
- **Cold-start false positives** (fixed): the generator and the live Privacy
  Layer both clamped time-derived features to a 1-day floor, so training
  never saw young accounts and every brand-new account's normal transaction
  scored ~0.7. The floor is removed (real sub-day tenure, minutes-to-hours
  first swipes, and zero-known-device accounts are all represented in
  training) and the live clamp is gone, so a fresh account's normal first
  event scores ~0.41 (step-up, not verify) and a known-device event ~0.02.
- **Operating point** (§6): `severity_scale: 0.45` in `rules.yaml` is tuned
  by `scripts/tune_operating_point.py` with a cost-weighted objective
  (C_FN/C_FP = 20; step-up friction 0.15–0.25). Per-event scores are
  scale-invariant, so the tuner computes them ONCE and caches them on disk
  (`models/cache/`, gitignored), keyed on the dataset content + model
  artifacts + the per-rule severity/condition state: a re-run (any cost
  ratio) hits the cache and finishes in ~0.1s instead of ~3 min;
  retraining, regenerating the data, or editing a rule's severity/condition
  invalidates it, while a `severity_scale`-only write (e.g. by
  `scripts/training_pipeline.py`) deliberately does not. The post-cold-start-fix
  retune cuts legit challenges from 1.77% (25 events) to 0.14% (2) on the
  test split while holding fraud recall at 92.3% (48/52) and lowering total
  cost 118.2 → 70.6. The residual 2 legit challenges are ML-driven (above
  any rule scaling). See `models/tuning_report.md`. `--per-rule`

- **Link analysis / device graphs**: three cross-account features
  (`shared_device_accounts`, `shared_recipient_accounts`, `mule_ring_score`)
  are derived at ingest from DB-2 (counts of OTHER accounts on the device /
  recipient) and synthesized in training by a new mule-ring archetype (2–4
  accounts sharing one device and a converging sink recipient, moderate
  amounts so amount features see nothing). A declarative `RULE_MULE_RING`
  (reason code `MULE_RING`) fires on shared device + shared/new recipient
  or a heavily shared device alone. Trained, tuned, and verified live: a
  three-account ring trips score 100 → VERIFY while a family phone shared by
  two legit accounts stays low (device-only sharing contributes 1/4 to the
  composite).
- **Calibration + odds**: the stacker output is mapped to the true fraud
  probability by **Platt scaling** (`PlattCalibration` in
  `src/risk_engine/calibration.py`) fit on an out-of-archetype pool
  blended with the production model's own validation predictions — the
  isotonic fit on near-separable data produced a step function, so it was
  replaced (`models/artifacts/calibrator.joblib`); `ml_score` is therefore a
  calibrated probability and the response carries `odds = p/(1-p)` (clamped
  at 9999 when p == 1) so thresholds carry business meaning across model
  versions. The score remains `100 × max(ml, rule)`.
- **Fail-safe degraded mode**: if ML fusion raises or the circuit breaker
  (3 consecutive failures → open 30s, then half-open probe) is open, the
  evaluation falls back to the declarative rules with a fail-safe floor
  (score ≥ 31 = STEP_UP), tags the decision `degraded: true` with
  reason code `ML_UNAVAILABLE`, and records it in DB-3 + the audit chain —
  ML failure never silently produces ALLOW. A degraded ML never 500s the
  request.
- **Rules backtesting**: `scripts/simulate_rules.py` replays a candidate
  rule config (or a `--severity-scale`) over the labeled training history
  with the live calibrated fusion, reporting recall / FPR / decision-mix
  deltas and the exact events whose allow/step-up/verify decision would
  change — the "what-if" step before editing `rules.yaml`.
- **Velocity / spend limits** (pre-scoring enforcement, section 7): the
  Risk Engine evaluates per-account (max 10 txns / 24h hard cap, 3× median
  daily spend soft cap) and per-device (max 20 txns / 24h across accounts)
  caps configured in `rules.yaml` `velocity_limits`, derived entirely from
  the §16 vector the Privacy Layer builds (`account_daily_spend_ratio`,
  `device_daily_count` — raw amounts never reach the Risk Engine). Hard
  caps force score 100 → VERIFY regardless of ML+rules; soft caps floor at
  step-up. The 24h windows are anchored at the event's own timestamp, so
  backdated history (demo seed, replays) counts correctly.
- **SHAP-style attribution** (internal analyst tool):
  `POST /internal/attribution` explains the fused probability with exact
  per-model contributions (logistic stacker, logit space) and per-feature
  perturbation contributions against the training median — no `shap`
  dependency, nothing persisted, and never exposed to end users (the public
  surface keeps §11 category-level reason codes).
- **Device-graph viewer** (compliance role): `GET
  /compliance/device-graph` on the Verification Service proxies the Privacy
  Layer's `/internal/device-graph` behind the compliance passphrase. Given
  a hashed device or recipient token it returns the pseudonymous accounts
  sharing it (event counts, first/last seen) — the mule-ring signature. The
  compliance UI includes a lookup card with a one-click "use demo device"
  quick-fill after a demo seed.
  generalizes the one-multiplier sweep to a constrained greedy search over
  individual rule severities (grid bounds `--min-mult`/`--max-mult`,
  default 0.5–1.5), caching the fired-rule mask per event so re-runs stay
  ~0.1s; cost-neutral tie-drift multipliers are pruned so only
  cost-proven rebalances are recommended. On the same test split it finds a
  strictly better point the global scale cannot reach: 5 softened rules →
  legit challenges 2 → 1 (0.07%) with fraud recall held at 92.3% and cost
  71.35 → 67.50 (see `models/tuning_report_per_rule.md`).
- **Baseline discipline** (blocked events never pollute): ingest stores the
  §16 vector but **does not** touch the account's behavioral baseline.
  `POST /internal/commit-baseline` applies a stored vector to the profile
  (median EMA in ratio space — `median × (0.9 + 0.1 × ratio)` — so the raw
  amount is never stored; plus typical hours and device registration), and
  is called ONLY for **allowed** events (by the flow owner) or **confirmed**
  events (by the Verification Service on `this_was_me`). Blocked (step-up /
  verify) or **disputed** events never enter the baseline: a 4500 attack no
  longer inflates the median (later normal events keep ratio ~1.0, not 0.32)
  and an attacker's device never becomes "known". Feature inputs (freq,
  recent-window escalation) still count all attempts — only the stored
  baseline is gated. The walkthrough re-verifies this live.
- **Demo seed warm-up** (`/demo/seed`, dev-only — the UI's "Simulate demo
  activity"): before evaluating the attack, the seed first ingests six
  normal transactions spread over the past ~7 weeks and AGES the account in
  DB-2 via the Privacy Layer's dev-only `POST /internal/demo-age-account`
  (backdates `created_at` across [8, 55] days and the profile birth to 60
  days — the service-side version of the curl walkthrough's aging step).
  The preview therefore shows exactly ONE alert (the attack, scored against
  a mature history) instead of cold-start false positives from a brand-new
  account. Each demo account gets its own device id: the fingerprint store
  is global (one row per hashed device), so a constant demo device shared
  across accounts would be "known" only to its first owner and every
  warm-up event would look like a brand-new device.
- **Audit trail** (§4/§13): `entry_hash = sha256(prev_entry_hash + canonical(payload))`,
  first entry linked to a documented genesis hash. Append-only is enforced
  at the storage layer (SQLite triggers abort UPDATE/DELETE), every
  compliance read is logged to `audit_access_log`, and `/audit/integrity`
  recomputes the chain and reports the first bad link — including payloads
  corrupted into non-JSON. Every privacy-sensitive action appends to the
  same chain: `score_generated` (Risk Engine), `verification_resolved`
  (Verification Service), `feature_ingested` (Privacy Layer — event id
  only, never the §16 features), `fraud_id_resolved` (Identity Service
  break-glass — actor + reason only; the PII stays in DB-1 and never
  touches the chain), and `drift_alert` (§6 drift monitor). The verification
  UI footer and the alerts screen show live chain-integrity status (polling
  `GET /chain-status` every 60s plus an immediate refresh after a demo seed
  or confirm; each poll is itself logged to `audit_access_log` as
  `verification-ui-footer`), and the compliance view auto-refreshes its
  integrity banner + trail every 20s while open. `GET /audit/export` dumps the full chain as a
  single signed document for an external regulator: HMAC-SHA256 over the
  canonical body (dev key from `export_signing_key`; production would use
  KMS/HSM asymmetric signing with the public half published), plus the
  integrity report. `scripts/verify_export.py` validates a copy
  independently - signature (authenticity) AND chain recomputation from
  genesis (integrity, not trusting the service's own report) - so a
  doctored export is rejected even if the service's `integrity` field still
  claims ok.
- Four fraud archetypes in the synthetic data: boiling-frog escalation,
  one-off impulse fraud, account-takeover bursts, and card-not-present
  testing (small, rapid txns to new merchants - deliberately invisible to
  amount-based signals), plus the mule-ring pattern. Each event is tagged
  with its `archetype`, and the trainer reports recall per pattern
  (`models/artifacts/recall_by_archetype.csv`, `EVALUATION.md`).
- **Chameleon fraud detection** (§12 rules): 6 deterministic rules in
  `rules.yaml` catch "chameleon frauds" — transactions that spread weak
  signals across many features instead of having one strong anomalous spike.
  These rules combine multiple moderate §16 signals (new device + unusual
  recipient + unusual time, etc.) and fire at severity 0.45–0.65. They
  complement the ML model, which misses ~14 out of 492 frauds (2.8%) on
  the Kaggle dataset because no single feature exceeds 3σ.
- **False positive reduction**: a micro-transaction discount in the Risk
  Engine reduces FPs by 23% (13→10) with only 0.2% fraud loss. The
  discount targets Amount < 0.1 with ML score 0.50–0.70 and no critical
  rules — the exact FP pattern where 10/13 FPs are micro-transactions with
  genuinely anomalous feature profiles.
- **Rules optimization**: `RULE_AMOUNT_HIGH` threshold raised from 1.3× to
  1.8× average (cuts 1,713 FPs); `RULE_UNUSUAL_LOCATION` and
  `RULE_UNUSUAL_RECIPIENT` severity reduced from 0.30 to 0.20. Net result:
  same fraud recall, 29% fewer false positives, 23% fewer challenges to
  legitimate users.
- **Label-preserving feature noise** (`--noise`, default 0.15): raw events
  are perturbed before feature derivation (flag flips, cross-profile hour
  redraws, amount drift, auth jitter) so archetypes are not crisply
  separable - the models must learn structure instead of memorizing
  signatures. `--noise 0` restores the crisp generator. Noise draws from its
  OWN rng stream (`seed + 1`), so the row set and archetype mix are
  bit-identical across noise levels and comparisons stay apples-to-apples.
  Cost: cold-start FPR on held-out legit accounts rises with noise (2.7% at
  noise 0 -> 4.5% at 0.10 -> 20.3% at 0.15, F1-threshold basis), while the
  pinned live OOD vectors gain real discrimination (s3 $1,500 ml 0.013 ->
  0.079) - pick the level for your FPR budget.
- **Honest generalization metrics**: the time-split test is optimistic -
  every fraud archetype also appears in training, so same-pattern
  near-duplicates leak across the split. `src/train_compare.py` therefore
  also runs a **leave-one-archetype-out evaluation** (the full ensemble is
  RETRAINED with each fraud archetype excluded and recall is measured on
  the never-seen pattern) plus a **cold-start probe** (whole legit accounts
  held out entirely, FPR on their rows). Results:
  `models/artifacts/generalization_by_archetype.csv` and the
  "Generalization" section of `EVALUATION.md`; the operating-point tuner
  appends the same table (plus per-archetype recall at the chosen scale) to
  `models/tuning_report.md`. The **pinned OOD set** (`data/ood_scenarios.csv`
  - the five live walkthrough vectors) is scored with the shipped artifacts
  every retrain as a regression check. `--no-group-eval` skips all of it.
- **OOD recall gate**: the retrain FAILS (exit 1, pipeline stops before
  retune/apply) when a gated held-out archetype's `recall_at_1pct_fpr`
  drops below a floor - default floor 0.60 on `ato,mule` (the weak
  archetypes), configurable via `--ood-recall-floor` / `--ood-gate-archetypes`,
  overridable with `--no-ood-gate`. The metric is the ranking-based
  recall@1%FPR, NOT `recall_f1` (unreliable at low fold fraud prevalence -
  the ato fold's validation is ~0.2% fraud). Gate results ride in
  `metadata.json` `ood_gate`.
- ~1/3 of rare-archetype accounts start late in the timeline so every
  pattern is represented in the out-of-time test split (raises the test
  fraud rate above the train rate - expected, by design).
- Feature derivation logic is shared (`privacy_layer/features.py`) between
  the generator, the trainer, and the live ingest path, so training features
  and production features stay consistent.
- Identifiers are CSPRNG-random 16-char base32-style IDs — not hashes of any
  PII — fitting the doc's `VARCHAR(16)` pseudonym column (80 bits; a true
  128-bit ID would need ~26 chars, a schema change flagged in review).
- PII at rest is encrypted (Fernet/AES-256, KMS envelope in production);
  login uses a blind index on the email so plaintext is never queried.
- The fraud-id->identity bridge is internal-only and every resolution is
  logged with actor + reason (break-glass).
