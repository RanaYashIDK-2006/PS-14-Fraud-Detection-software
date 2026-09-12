# PS-14 — Check #28: Model Release, Rollback & Disaster-Recovery Audit

**MODEL RECOVERY VERDICT: FAIL** — with an important nuance: the rollback
*mechanism* (`model_governance.rollback`) now has been **actually exercised and
proven** (5/5 scenarios, byte-identical recovery, ~0.24 s rollback), but it is
**not wired into the real deployment path**, the active artifact set is a mix of
≥2 training runs that no single model record describes, and the governance record
for the deployed model still has material gaps. Evidence is live-measured in a
sandbox (`scripts/model_release_drill.py`, real engine, real artifacts, real
hashes; the `models/production/` directory was never modified) plus a real
backup/restore run (`scripts/backup_restore_test.py`, 69/69 PASS).

Evidence: `reports/model_release_drill.json`.

---

## 1. Artifact integrity — FAIL (record exists, record incomplete, record wrong in one place)

The deployed model (`altman_lean_15feat_20260830_200346`) has a governance record
(`models/model_records/`), and every artifact hash in it verifies. But the record
is not a complete release record:

| Required field | Recorded? | Finding |
| -------------- | --------- | ------- |
| Unique model version | YES | `altman_lean_15feat_20260830_200346` |
| Artifact hashes | YES | all 14 recorded files hash-match; but the record also **hash-lists stale artifacts** (`logistic_regression.joblib`, `model.joblib`, `random_forest.joblib`, `scaler.joblib`, `drift_reference.*`) from an older era, giving a "verified" set that includes files no release should carry |
| Training data hash | YES | `features_all.parquet` `4ea90501…` (matches check #23) |
| Feature-schema version | PARTIAL | `"altman_lean_v1"` free-form string, not a registry |
| Configuration version | **NO** | engine weight constant `{xgb 0.34, lgb 0.33, cb 0.33}` **contradicts the manifest** (`{xgb 0.5, lgb 0.5}`); neither the manifest nor the record states the 0.34/0.33/0.33 combination that actually runs |
| Code/commit version | **NO** | `"models/production + src/risk_engine/altman_ensemble.py"` — a description, not a hash (project has no git) |
| Hyperparameters | **NO** | `"could not introspect: int() … not 'method'"` — recorded as a failure note, not values |
| Training/validation/final-test periods | **NO** | all three empty |
| **Threshold** | **NO** | `selected_threshold: null` (re-confirms checks #16/#19: the deployed model decides with no locked, validated threshold) |

**NEW high-severity finding — the active artifact set mixes ≥2 training runs the
manifest never declared.** The lean manifest says `lean: true`, `xgb_lgb_ensemble_lean`,
weights 0.5/0.5, and its builder (`retrain_15feat.py`) trains **only XGB + LGB**.
But `models/production/` also contains a **44 MB `cb_production.joblib`** (written
25 min *after* the lean manifest, Aug 30 20:28) whose CatBoost model was trained
with **anonymous feature names `'0'..'14'`** — from a different run
(`train_altman_fullscale.py`/`train_altman_production.py` family). The engine
loads it whenever present and blends it at weight 0.33. Measured on one input:
**as-deployed (with cb + calibrator) = 0.0918** vs manifest-declared 0.5·xgb+0.5·lgb
= **0.7804** — the undeclared third member plus the engine's different weight
constant silently re-scale every decision. The governance record hash-covers the
cb, so the mixed set *verifies as intact* even though no record describes it as a
member of this ensemble. A `cb_production.joblib.bak` (1.2 MB, earlier cb) shows
the replacement happened over an explicit backup.

## 2. Release validation gates — FAIL (nothing gates an artifact deploy)

The artifact deploy path that produced the active model (`retrain_15feat.py`)
writes straight into `models/production/` with **no gate run**: no leakage check,
no threshold verification, no feature-parity check, no canary — and it is the
script whose manifest is now live. The governance *record* for the deployed model
documents the gate results of a *different* model (the audit 25-feature XGB):
`production_parity: PARTIAL→FAIL`, `untouched_test: UNVERIFIED` for this artifact,
`approval: UNVERIFIED`. A higher-AUC artifact is not required to clear any gate
before landing in the serving directory.

## 3. Atomic deployment — PARTIAL (fail-safe present, atomicity absent)

There is no staging/activation protocol: artifacts are overwritten in place,
sequentially. **Tested:** an interrupted deploy (new LGB bytes + old manifest,
the state a crash mid-deploy leaves) is **refused at load in 0.011 s** — the
governance record-backed integrity verify (engine `verify_integrity=True`, what
the app enables) raises rather than serve the mixed set. So a *partial upload
cannot become active* once a record exists. But the flip side is unrecoverable by
design: overwriting destroys the previous known-good artifact set (no release
archive is kept), so an interrupted or bad deploy leaves the system degraded
(rules-only) until a human restores files — there is no "keep previous model"
option at the file level.

## 4. Rollback — mechanism VALIDATED (drill), NOT WIRED (production)

`model_governance.rollback()` now has a real, passing exercise record (it copies
from a verified-good archive, then re-verifies in place). Five drills, each
activated a deliberately broken model over a good copy, detected the fault,
rolled back, and confirmed **byte-identical** canary outputs afterward:

| Drill | Detection | Detector | Rollback time | Reload | Post-rollback outputs = baseline |
| ----- | --------- | -------- | ------------: | -----: | :-------------------------------: |
| Corrupted artifact | 0.011–load error | engine load/predict raised | 0.252 s | 1.49 s | YES |
| Incompatible feature schema (3-feat LGB) | load-time | predict raised (`X has 15 features…`) | 0.240 s | 1.47 s | YES |
| Bad model (random-noise member) | canary | raw-mean shift +0.164 vs 0.833 | 0.244 s | 1.49 s | YES |
| Abnormal score distribution (member collapse to 0.001) | canary | raw-mean shift −0.330 | 0.242 s | 1.47 s | YES |
| Severe latency (250 ms/member) | canary | p95 254.6 ms vs 100 ms SLO | 0.233 s | 1.46 s | YES |

Honest caveat measured during the drill: replacing one member with a *constant
high* model was **not** detectable by score-mean canary — because the deployed
trio itself saturates near 1.0 raw on most inputs (baseline raw mean 0.833 on a
mixed normal/attack canary), a constant-high member is statistically invisible.
Score-based canaries on this model are insensitive to member-level high-side
damage (a monitoring design consequence of the saturation pathology).

**In production, none of this is automatic:** nothing keeps a release archive of
the previous good set, nothing triggers rollback on a canary SLO breach, and
activation requires a service restart (the engine loads at lifespan). The drill
proves the *building block*; the *system* still relies on manual file surgery.

## 5. Rollback correctness — PASS (as drilled)

After each rollback: correct artifact set serving (hash-verified), same feature
list (15, record schema check), same score outputs to 15 decimal places, latency
restored to baseline, and the drill's audit record logged each
activate→detect→rollback→verify step with the bad and good model ids. (Real
deployment-level audit tagging of a rollback event does not exist yet — no
deployment event type is written to the audit chain by any deploy path.)

## 6. Database/config migration safety — FAIL (no migration story)

Nothing in the release path tests a schema/config change: the engine loads
whatever files are in place, and the only compatibility gate is the record-backed
hash/schema verify (which covers *this* model, not a *changed* one). A deploy
expecting a new feature schema would fail at predict time (drill #2 = exactly that
failure, detected only because the canary raised) — i.e., detection happens after
activation, not before, and rollback then depends on the manual archive that does
not exist. Config (weights, threshold) is not versioned with the model: the live
engine runs a weight constant that contradicts the live manifest (see §1) and no
record flags it.

## 7. Disaster recovery — mostly PASS (measured)

| Scenario | Result | Evidence |
| -------- | ------ | -------- |
| Model storage unavailable | PASS (loud, safe) | Engine load raises `FileNotFoundError`; **no silent fallback** to stale/other artifacts. The app lifespan catches this and serves rules-only with `ML_UNAVAILABLE` tagging (main.py L157-170) — a degraded decision is never untagged |
| Recovery from known-good archive | PASS | 1.63 s to restore + verify + byte-identical outputs |
| Application restart (cold engine) | PASS | fresh load + record verify 1.53 s, outputs identical (full stack boot ≈ 6 s, check #26) |
| Temp-state loss (tracker restart) | PASS (with caveat) | entity fraud-rate tracker returns to baseline 0.001 after restart until re-seeded from DB-2 at startup — model scores unaffected, the two fraud-rate features cold-start (and they are frozen at baseline anyway in the real call path, check #27) |
| Database failure | PASS (prior evidence) | check #26: DB-4 exclusive lock during scoring → 15/15 OK; risk service down → explicit connection error, never a silent decision |
| Backup restoration | **PASS — actually executed** | `backup_restore_test.py`: 69/69 PASS — copies → backup with checksums → `PRAGMA integrity_check` → row-count compare → corrupts copies → restores → verifies checksums/integrity/row counts → **audit chain re-verified across all 1,392 restored entries** → originals untouched |

## 8. Recovery objectives — RTO small, RPO unbounded (measured)

- **RTO (model):** ~1.5–2.5 s per incident (rollback copy 0.24 s + engine reload
  1.5 s; + ~6 s for a full service restart if required). Satisfies any sane
  recovery SLA. Note: RTO depends on a *known-good archive existing*, which the
  deploy path does not create.
- **RPO (data/config):** **no scheduled backup is running.** The retention
  service/timer (`ps14-retention.service`/`.timer`, daily 02:00) targets a Linux
  systemd deployment at `/opt/ps14`; in this environment the only existing backup,
  `db/audit.db.bak`, is **4 days stale**: 1,028 events vs 1,392 live — restoring it
  would lose 364 audit events (~26%). For an append-only compliance chain that is
  a real compliance exposure. Identity/features/risk DBs have no backup artifact
  at all (demo state is throwaway by design, but DB-1 identity is not).

## 9. No silent downgrade — PASS (at load), with one live exception

With a governance record present, the engine refuses (raises) on hash mismatch,
schema mismatch, or partial activation — demonstrated in 0.011 s (atomicity) and
in every corrupt drill. **The live exception is §1:** the engine today serves a
composition (XGB+LGB+undeclared CB at 0.34/0.33/0.33 + cross-model calibrator)
that **no validated model record describes** — an unvalidated, undocumented
configuration serving silently, which is precisely the class of downgrade this
check forbids.

## Failure matrix & verdict

| Failure Scenario | Detection | Automatic Response | Recovery Time | Data/Alert Impact | Result |
| ---------------- | --------- | ------------------ | ------------: | ----------------- | ------ |
| Corrupted model artifact | load error / hash verify | refuse to serve (record) | ~1.7 s | none (rules-only until restored) | PASS* |
| Incompatible feature schema | predict error / canary | refuse/rollback in drill | ~1.7 s | none (degraded) | PASS* |
| Bad model (noise) | canary raw-mean shift | rollback in drill | ~1.7 s | mis-scored traffic during window | PASS* |
| Abnormal score distribution | canary raw-mean shift | rollback in drill | ~1.7 s | mis-scored traffic during window | PASS* |
| Severe latency increase | canary p95 vs SLO | rollback in drill | ~1.7 s | slow decisions | PASS* |
| Constant-high member | **NOT detected** (saturation) | none | — | silent score bias | **FAIL** |
| Partial/atomicity breach | 0.011 s (record verify) | refuse to serve | ~1.5 s | none | PASS |
| Model storage unavailable | load error | rules-only degrade (tagged) | 1.6 s restore | degraded (tagged) | PASS |
| DB failure | (check #26) | fail-safe/no silent allow | varies | varies | PASS |
| Temp state loss | n/a (by design) | cold-start re-seed | — | fraud-rate feats cold | PARTIAL |
| Backup restoration | 69/69 restore checks | manual/scripted | 12 s test | RPO unbounded | PASS* |
| Undeclared/conflicting config member | **not detected by any gate** | none (serving today) | — | every decision scaled silently | **FAIL** |
| Release archive of previous good | **does not exist** | none | — | rollback manual-only | **FAIL** |

\* PASS **as drilled in sandbox only** — the same mechanisms are not wired into
the real deploy path (no archive, no auto-trigger, no deploy audit event).

**MODEL RECOVERY VERDICT: FAIL**

The recovery *building blocks* are real and now proven: record-backed load-time
integrity (0.011 s refusal), a working rollback primitive (5/5, byte-identical),
a fail-safe degrade path, cold restart in ~1.5 s, and a passing backup/restore
procedure (69/69 including the 1,392-entry audit chain). What fails is the
*release system around them*: deploys are ungated direct overwrites that keep no
archive of the previous good model, rollback has no automatic trigger and no
deployment audit trail, RPO is unbounded (only stale manual backups exist), and —
decisively — **the artifact set serving right now is a mix of at least two
training runs the governing manifest never declared, blended at weights no record
states**. The single highest-value fix: make the deploy path archive the previous
verified set, write the manifest/weights/threshold as one versioned config the
engine reads, and drop or re-certify the undeclared `cb_production.joblib`.

**Artifacts:** `scripts/model_release_drill.py`, `reports/model_release_drill.json`,
`reports/PS14_MODEL_RECOVERY_AUDIT_REPORT.md`. No `src/` files were modified; the
real `models/production/` directory was untouched (all drills ran in a sandbox
copy). `smoke`/`risk_engine` suites re-run below.
