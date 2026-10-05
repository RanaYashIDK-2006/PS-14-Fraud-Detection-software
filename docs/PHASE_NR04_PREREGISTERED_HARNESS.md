# NR-04 — Preregistered Benchmark Harness

**Phase:** NR-04 (Gate 2 prerequisite, gate 1 deliverable) · **Date:** 2026-10-04
**Scope:** harness engineering and validation only. No final scientific benchmark was
executed; no model was optimized; no threshold was tuned; no dataset was selected by
observed results; no reviewer-owned decision was resolved by assumption; no blocked
dataset was obtained or fabricated; no reviewer approval is claimed anywhere.

---

## A. Objective

Build and verify the deterministic execution harness required to run the
preregistered benchmark (protocol 0.1.2-draft, NR-03 frozen semantics) once the
remaining reviewer decisions and external dependencies resolve — translating
FROZEN semantics into executable, provenance-complete evaluation procedures
that refuse to silently value PENDING REVIEW or BLOCKED states, with evidence-
first MEASURED semantics, a seven-experiment registry, and a complete status
matrix. Deliverable: `docs/PHASE_NR04_PREREGISTERED_HARNESS.md` (this file) +
the minimum code needed for the harness.

---

## B. Baseline

| Item | Value at NR-04 start |
|---|---|
| Git SHA | `5a5ff55cc03df73318fdf31a16e3665f159a4720` (`5a5ff55`) — the user-requested commit of Phases 104…NR-03 made immediately before this phase; working tree clean except hook-appended ledger lines |
| Prior phase statuses | 104 PASS WITH LIMITATIONS · 105 PASS WITH LIMITATIONS · 106 PASS · NR-01 PASS WITH LIMITATIONS · NR-02 PASS WITH LIMITATIONS · NR-03 PASS WITH LIMITATIONS |
| Protocol | `docs/evaluation/PRE_REGISTERED_EVALUATION_PROTOCOL.md` v0.1.2-draft, **DRAFT/NOT APPROVED**, marker literal count **11** (verified before and after this phase) |
| Evidence ledger | 41 records at 2026-10-04T12:00:21+05:30 (grows ~every 10 min via a client-side hook — cited as timestamped observations) |
| Enforcement at start | `claim_evidence_check` PASS (22 claims); `eval_record_test` 22/22; battery (after NR-01) = 81 rows |
| Reviewer gates at start | all six gates closed: protocol unsigned, seed policy/aggregation unapproved, bootstrap count unapproved, ECE binning unapproved, paired-CI spec unapproved |
| Environment | stack 6/6 `/health` → 200 before battery; `models/` + `data/` content digest `a98862a59f202703edcdc4caed8e3fa6b82ac52124896a3ab22bb08f16fefa32` (re-verified unchanged at phase end) |
| NR-03 design baseline | 161-cell semantics matrix (23 fields × 7 experiments), nine-row failed-run policy, 14 provenance items, three bootstrap implementations, paired-CI not implemented |

Source documents read this phase (not relied on from prose alone): the protocol,
NR-01/NR-02/NR-03 reports, Phase 105 baseline, evidence schema, baseline
reproduction manual, Phase 106 roadmap + requirement status, research-gate
readiness. Implementations inspected directly: `calibration_test.py`,
`eval_ulb.py`, `cross_dataset_eval.py`, `train_compare.py`, `evaluate.py`,
`ibm_train.py`, `metric_definitions.py` (incl. `bootstrap_ci`), `eval_record.py`,
`claim_evidence_check.py`, `public_feature_contract.json`, the battery script,
and the CI workflow.

---

## C. Harness Contract

One authoritative contract: `CONTRACT_FIELDS` in
`backend/scripts/prereg_harness.py` — 24 keys, each with a documented
deterministic source; `validate_contract()` rejects missing fields and version
drift with `CONFIGURATION ERROR` diagnostics that name the source.

| Contract field | Deterministic source |
|---|---|
| `protocol_version` | pinned `0.1.2-draft`; must equal the protocol file (test-asserted) |
| `experiment_id` | experiment registry (NR-03 §C transcription in this module) |
| `dataset_name` / `dataset_path` / `dataset_expected_sha256` / `dataset_role` | dataset registry with hashes pinned by protocol §9 / NR-03 §D / BASELINE_REPRODUCTION §2; hash recomputed over raw bytes at run time |
| `split` | preregistered split spec `{name, axis, role, …}` (identity is PENDING under M-1 until signed) |
| `population` | experiment-registry `Population` field / preregistered population statement |
| `seed` | `SeedPlan` for the replicate (proposal 42–46 NOT approved) |
| `model_identity` / `artifact_paths` | frozen-candidate label + files hashed by `eval_record.artifact_hashes` |
| `feature_contract_path` | contract file; identity recorded as `feature_schema_version = <name>@<sha256[:12]>` |
| `preprocessing_identity` / `preprocessing_fit_population` | NR-03 §H vocabulary (`standardscaler_fit_on_train`, `embedded_in_artifacts`, `none_fixed_artifacts`, …); population must be `train` |
| `calibration_config` | `{procedure: platt, fit_population: validation}` (protocol §5.2) |
| `threshold_config` | `ThresholdSpec` (§14 four states) |
| `metric_definitions_version` / `evidence_schema_version` | pinned to `metric_definitions` v1.0 and `eval_record` v1.1 — drift is a configuration error |
| `bootstrap_config` | explicit `{n_bootstrap, seed, ci}` — approval tracked by `BOOTSTRAP_COUNT_APPROVED` |
| `statistical_test_config` / `acceptance_criterion_config` | paired-CI state (PD-PAIRED) / MMD rule ids from protocol §3 + NR-03 §K |
| `execution_command` | exact invocation string → evidence `command` (v1.1) |
| `data_tier` / `deviations` | protocol §9 tiers; dated deviation list (empty list is valid) |
| Git SHA | **captured by the harness from the repository** (`eval_record.git_commit()`), never caller-typed |

No field duplicates an evidence-schema field: everything that already exists in
schema v1.1 is written there (`evaluation_config` carries the free-form items —
experiment id, protocol linkage, deviations, split, guard state, attempt);
the contract is the *input* view, the record is the *evidence* view.

---

## D. Experiment Registry

All **seven** NR-03 planned experiments are represented explicitly
(`EXPERIMENTS` = E1 ensemble/criteria 1–2 · E2 gating/3–4 · E3 external
transfer/5–6 · E4 external calibration · E5 subgroup §6 · E6 business §7 ·
E7 baselines), each with purpose, criteria, and the 23 matrix fields carrying
status (`FROZEN`/`PENDING REVIEW`/`BLOCKED`/`NOT APPLICABLE`), a value only
when FROZEN, and no value whatsoever when PENDING (the test suite fails if a
PENDING cell carries a value — “do not invent values for
PENDING REVIEW cells” is machine-checked).

**Transcription fidelity:** `prereg_harness_test.py` parses the actual table
out of `docs/PHASE_NR03_EVALUATION_SEMANTICS.md` on every run and compares it
cell-for-cell with the harness registry, then recomputes the totals: **161
cells: FROZEN 87 · PENDING REVIEW 62 · BLOCKED 0 · NOT APPLICABLE 12** — which
matches NR-03’s stated totals line exactly. (An early hand-transcription by
this phase put `Final-test population` at F F F F; the parser corrected it to
the document’s P P F F before anything depended on it — the document table is
the source of truth, and the test now prevents recurrence.)

Dataset values are given only where NR-03 froze them: E3/E4 =
`kaggle_fraud_test` + pinned sha256; E1/E2/E5/E6/E7 datasets remain PENDING
(D-1/D-1b/§1 — no dataset invented for unresolved criteria).

---

## E. Frozen/Pending/Blocked Guard

`guard(experiment_id, contract, tier=…)` returns a `Decision` with a strict
precedence: **`CONFIGURATION ERROR` > `BLOCKED` > `PROTOCOL VIOLATION` >
`DO NOT EXECUTE — PENDING REVIEW` > `CLEAR TO EXECUTE`** — three distinct
states, never merged; `PENDING REVIEW` and `BLOCKED` have no path to
`MEASURED` anywhere in the module.

The reviewer-decision registry holds **22 entries: the 11 protocol markers +
11 sign-off/implementation items**, each mapping to the experiments it gates:

- **Guard minimums (§6) all present:** seed aggregation (`PD-SEED-AGG`),
  bootstrap replicate count (`M-3`), ECE binning (`M-2`), paired-difference CI
  (`PD-PAIRED`), criteria 1–4 dataset identity (`PD-D1`, `PD-D1B`), external
  confirmatory status (`PD-T8`), external split boundaries (`PD-T9`), plus every
  remaining marker (`M-0`…`M-10`) and `PD-SIGNOFF` (protocol unsigned) and
  `PD-FREEZE` (no freeze record exists yet — gated to final tiers).
- **Current behavior:** `guard` for **every one of E1…E7** returns
  `DO NOT EXECUTE — PENDING REVIEW` listing the pending ids (verified in the
  suite for all seven).
- **No casual override:** the only flip points are module-level constants
  (`PROTOCOL_SIGNED_OFF`, `SEED_POLICY_APPROVED`, `SEED_AGGREGATION_APPROVED`,
  `BOOTSTRAP_COUNT_APPROVED`, `ECE_BINNING_APPROVED`,
  `PAIRED_CI_SPEC_APPROVED`), documented as reviewer-authorized code changes
  paired with a protocol change-history entry. The CLI has **no execute
  subcommand at all** (a test asserts `execute` is rejected), hence no flag
  can override the guard.
- **Labeled development override:** `execute(..., mode="development")` may
  proceed past *pending* decisions for testing only (config errors, blocked
  datasets, and protocol violations are refused in **both** modes). Every such
  run writes `preregistered=false`, a `NON-PREREGISTERED development run`
  warning, and the pending ids into its evidence record, and its result status
  can never be `MEASURED` (tested).

---

## F. Seed Handling

`SeedPlan` implements the NR-03 §F interface without deciding the pending
policy: explicit unique-integer seed lists (duplicates refused), deterministic
propagation (each seed is a contract field, part of `config_digest`, part of
its own evidence record), per-seed execution (`run_seed_plan`), per-seed
evidence records, failed-seed recording (a crashed seed is retained as a
`FAILED` record and the loop continues), and deterministic rerun of an
identical seed/configuration (same digest, distinct `evaluation_id` via the
attempt salt — §U). The proposed five-seed policy 42–46 is carried as
`PROPOSED_SEEDS` with `SEED_POLICY_APPROVED = False` and
`SEED_AGGREGATION_APPROVED = False` — the proposal is **not** treated as
approved anywhere; `SeedPlan.pending_decisions()` surfaces `M-7` +
`PD-SEED-AGG` until both flip. The suite verifies seeds `(42,43,44)` execute
individually with all three records present (no post-hoc exclusion, §9.9) and
that seed 43 crashing is preserved as `FAILED`.

---

## G. Failed-Run Handling

The frozen nine-row policy is encoded as `FAILURE_CLASSES` (exactly 9 rows,
asserted): crash → implementation / rerun allowed; invalid output →
implementation/data; feature-contract violation → invalid experiment,
`NOT ESTABLISHED`, rerun only after contract fix with a new experiment ID;
cannot-produce-evidence → evidence failure, one identical-config rerun then
`NOT ESTABLISHED`; missing data → data quality, no rerun; NaN/Inf →
implementation/data; resource limit → implementation/infra; reproducibility
check failure → evidence failure; metric outside its mathematical domain →
data quality, no rerun (split-design issue back to the reviewer).

`record_failed_run()` persists every failed attempt with the full identity
required by §8: experiment ID, seed, configuration digest, Git SHA, timestamp,
failure class, diagnostic, and `rerun_permitted`. Records are append-only —
the suite asserts two identical-config failed attempts both remain in the
ledger with distinct ids, equal config digests, and the class/diagnostic/rerun
fields intact; a runner crash inside `execute()` takes the same path.
Rerun identity is enforced by construction: the rerun recomputes
`config_digest` over the same frozen contract, and a digest mismatch is a
different configuration (not a rerun).

---

## H. Evidence-First Execution

`execute()` runs the §9 order: (1) validate contract/protocol fields →
`CONFIGURATION ERROR`; (2) guard → pending/blocked; final-test protection →
`PROTOCOL VIOLATION` (refusals **never call the runner** — asserted); (3)
execute the injected runner; (4) calculate metrics through the authoritative
path; (5) generate the v1.1 evidence record; (6) append + re-read it from the
ledger; (7) validate it (`validate_evidence_record`: schema, `COMPLETED`,
dataset hash, git commit, command, seed/`seed_not_applicable`, and **all 14
provenance items**); (8) only then return `MEASURED`. Any failure in steps 5–7
returns `EVIDENCE FAILURE` with status `NOT ESTABLISHED`,
`failure_class = evidence failure`, `rerun_permitted = true`, and the
diagnostic “result cannot become a measured benchmark claim” — proven both by
an unwritable-ledger fixture and by a deliberately truncated record. Because
every experiment is currently pending, `MEASURED` is unreachable in
preregistered mode today (by design), and the suite asserts that too.

---

## I. Provenance Population

All **14** NR-03 §O items are mapped onto existing schema-v1.1 fields
(`PROVENANCE_ITEMS` — no duplicate infrastructure) and
`provenance_report()` reports POPULATED/MISSING per item. The suite executes a
fixture run and asserts **14/14 POPULATED**, specifically verifying the newly
populated items: `preprocessing_version` (`standardscaler_fit_on_train`) and
`feature_schema_version` (`public_feature_contract.json@<sha256[:12]>` —
contract file name + hash per the §H vocabulary), plus experiment id,
protocol linkage (`evaluation_config.protocol_version = 0.1.2-draft`),
deviations (explicit list), seed, dataset hash, split, command, git SHA,
model/config identity, threshold state, metric-definition identity, and
artifact identity. The harness additionally requires artifact identity to be
real (non-empty `artifact_file_hashes`) for evidence validation — stricter
than the schema’s null-with-warning allowance, per §27’s “all 14 populated”.

---

## J. Bootstrap Handling

`BOOTSTRAP_IMPLEMENTATIONS` inventories all three implementations with their
callers: `metric_definitions.bootstrap_ci` (default **1000**, percentile,
class-stratified, seed 42; called by `FullEvaluationMetrics`,
`evaluate.py --bootstrap`, `eval_record_test`), `eval_ulb.bootstrap_ci`
(default **200**; called only by `eval_ulb.py` lines 135–136), and the
protocol §2 proposal (**2000**, not implemented, marker). An implicit default
cannot determine a preregistered result: `resolve_bootstrap()` requires an
explicit `n_bootstrap` **and** `BOOTSTRAP_COUNT_APPROVED` for preregistered
mode — and passing an explicit count is *explicitly not* approval (the refusal
text says so; tested with 200/1000/2000). In development mode the default is
allowed only with `explicit: false` + a warning naming the pending decision.
`BOOTSTRAP_COUNT_APPROVED = None` today; the mechanism is ready to accept the
reviewer-approved value once resolved.

---

## K. Paired-CI Status

**Interface/schema boundary only — no estimator, no interval** (§12’s
conservative branch, justified by NR-02 §H listing “paired-difference CI
construction” as an open statistical-review item). `paired_difference_ci()`
validates the schema contract (paired rows of identical length, explicit
`n_bootstrap` and `seed`, no implicit defaults) and then raises
`PendingReview` while `PAIRED_CI_SPEC_APPROVED = False` — producing no CI
whatsoever; the docstring records the frozen structural context (percentile,
class-stratified, shared-row pairing, 95% from NR-03 §J) and names the missing
decision (T-5/§2 marker). E1/E2/E3 carry `PD-PAIRED` in their pending sets
(asserted), so the affected experiments are explicitly blocked rather than
silently given an unapproved interval. Tests use synthetic arrays only.

---

## L. Metric Handling

One authoritative path: `compute_preregistered_metrics()`.

| Metric | Implementation used | Pin verified by the suite |
|---|---|---|
| ROC-AUC | `metric_definitions.roc_auc` | equals `sklearn.roc_auc_score` on fixtures |
| PR-AUC | `metric_definitions.pr_auc` | equals `average_precision_score` (average-precision pin, not trapezoidal) |
| recall@1%FPR | `eval_ulb.recall_at_fpr` (ranking variant: prepend (0,0) when needed, `searchsorted`, first FPR ≥ 0.01) | equals a literal transcription of `eval_ulb.py:24–30` |
| Brier | `calibration_test.compute_brier_score` | equals `mean((p−y)²)` |
| ECE | `calibration_test.compute_ece` (10 equal-width) | returned as `provisional: true` with `binning_status = PENDING REVIEW (M-2)` — never silent |
| Threshold metrics | `metric_definitions.confusion_counts` | recall/precision/FPR recomputed on fixtures |
| MMD | `MMD_RULES` (C1/C3/C5 numbers unchanged from §K) + `evaluate_mmd` mechanics (equality = MET; interval touching the excluded value does not exclude it); marked pending M-5 in output | — |

Metric-domain failures raise `MetricDomainError` mapped to the §G rows
(NaN/Inf → implementation/data; single-class partition → data quality with no
rerun) — tested. The `cross_dataset_eval.py` variant of recall@1%FPR
(`side='right'` then `idx−1`) is a **different implementation** and is
documented as non-interchangeable in the pipeline registry; the harness metric
path uses the NR-03-pinned `eval_ulb` semantics only.

---

## M. Threshold Freeze

`ThresholdSpec` distinguishes the four §14 states: **frozen**
(`source=frozen`), **validation-selected** (`source=validation`,
`selected_on=validation`), **exploratory**, and **pending**. Refusals:
`source ∈ {test, external, holdout}` → `PROTOCOL VIOLATION` (parity with
`evaluate.refuse_test_tuning` — the parity is itself tested against the real
function); `selected_on ∈ {final_test, external}` → `PROTOCOL VIOLATION`
(threshold selected from final-test observations); exploratory threshold on a
final tier → `PROTOCOL VIOLATION`; pending threshold on a confirmatory run →
`PENDING REVIEW`. The harness contains **no threshold optimizer at all** — a
test asserts no `*optim*` symbol exists in the module, so automatic
optimization against final-test results is structurally impossible here.

---

## N. Dataset Validation

`DATASETS` registers five present datasets with the NR-03 §D hashes (ULB,
IBM v2, Kaggle fraudTest + fraudTrain, synthetic) plus BAF marked BLOCKED, and
`validate_dataset()` checks identity, hash, and label availability in this
order, never substituting:

- BAF / missing file → **`BLOCKED — DATASET UNAVAILABLE`** (blocker text names
  NR-16 acquisition/license) — tested;
- hash mismatch vs pin → **`DATASET IDENTITY MISMATCH`** — tested with a
  fixture file;
- declared label column absent from the CSV header → blocked under the
  unavailable status (no silent schema downgrade) — tested; header parsing
  uses `csv.reader` because the real ULB header is quote-wrapped (`"Class"`).

Live verification: `validate_dataset("ulb")` re-hashes the real 150 MB file
and matches its pin inside the suite. **Limitation:** the 2.35 GB IBM v2 file
is deliberately *not* re-hashed in the suite (runtime); its pin is registered
and will be verified on first real use. Unknown dataset names without a path
are `CONFIGURATION ERROR`, not a block.

---

## O. Split Validation

`validate_split()` proves the requested split against its declaration: split
identity present; axis ∈ {temporal, entity, random}; **pairwise partition
disjointness** (train/test and validation/test overlap →
**`BLOCKED — SPLIT VALIDATION`**); chronology when `axis=temporal`
(unprovable without timestamps, or `max(train) > min(test)`, → blocked);
entity separation when `axis=entity` (shared entities → blocked); and
external-set role match (a development-role split claiming a final-external
role → blocked). All five failure modes are exercised by fixtures. Because
the §1 split-axis marker is pending for every dataset, preregistered split
*identity* stays behind `M-1`; the structural prover is ready for the signed
definition and is used by the guard on every requested run.

---

## P. Final-Test Protection

`final_test_protection()` returns violation diagnostics for: test/external-
selected thresholds (incl. via `validate_threshold`); undeclared or
non-`train` preprocessing fit population; non-`validation` calibration fit on
final tiers; seed sets differing from the pre-declared set (selective seed
removal/addition, §9.9); dataset differing from the preregistered dataset
(post-hoc selection, §9.4); and **final-tier execution without a freeze record
in the ledger** (§9.2). Guard precedence puts `PROTOCOL VIOLATION` above
pending, and a violated run is refused in *both* modes with **no record
appended** (asserted) — it can never become a confirmatory `MEASURED` result.
All five diagnostics are fixture-tested, including the freeze-present case
that clears the freeze diagnostic while the seed-removal check still trips.

---

## Q. Full-Matrix Reporting

`build_matrix()` (CLI: `prereg_harness.py matrix`) emits the §9.11 tensor
**experiment × dataset × seed × contract × condition × transfer direction**
with every axis slot explicit — unresolved axes are single labeled `PENDING`
slots (no invented seed values, no invented datasets), E2 enumerates its 7
pre-defined conditions, E3/E4 carry an extra `21+7_full_bank_contract` slot,
E5/E6 mark the seed axis `NOT APPLICABLE`, and E3/E4 mark transfer direction.
Current output: **15 cells — MEASURED 0 · BLOCKED 2 · PENDING REVIEW 13 ·
NOT ESTABLISHED 0 · FAILED 0 · NOT APPLICABLE 0**, plus the full 7×23 design
matrix. The suite asserts: every cell carries one of the six statuses, no cell
is omitted (counts sum to the total), no cell is zero-filled, blocked cells
are never collapsed into pending or missing, and `measured_cells == 0`. This
is the canonical reporting surface; cells become `MEASURED` only via
validated preregistered ledger records.

---

## R. Determinism Tests

All nine §19 cases pass (in `prereg_harness_test.py`, against safe fixtures):

| Case | Expected | Observed |
|---|---|---|
| Same configuration, same seed | identical deterministic config + equivalent output | equal `config_digest`, metrics equal to 1e-12, ids still unique |
| Same configuration, repeated | no unexplained semantic difference | identical status/metrics/code across repeats |
| Different seed | seed identity changes, frozen config identical | digests differ with seed, equal with `include_seed=False`; own outputs |
| Different dataset hash | execution blocked | `DATASET IDENTITY MISMATCH`, runner never invoked |
| Missing provenance | execution blocked | `CONFIGURATION ERROR` naming the field; runner never invoked |
| Pending decision | execution blocked | preregistered run → `DO NOT EXECUTE — PENDING REVIEW`, runner never invoked |
| Failed evidence generation | result not `MEASURED` | `EVIDENCE FAILURE` → `NOT ESTABLISHED`, measured=False, rerun permitted |
| Feature-contract mismatch | blocked | tampered contract → `BLOCKED — FEATURE CONTRACT` (N-02 invariant) |
| Final-test selection attempt | protocol violation | `PROTOCOL VIOLATION`, no record appended, measured=False |

---

## S. Synthetic/Fixture Tests

The suite (31 tests) exercises metric calculation, bootstrap plumbing,
failed-run handling, evidence generation, matrix generation, provenance
population, and every protocol guard on synthetic arrays and temp files only.
Fixtures are labeled three ways: contracts carry `fixture_test: true`,
execution uses `mode="development"` (records stamped `preregistered=false` +
NON-PREREGISTERED warning), and the command string says `(FIXTURE)`. All
fixture runs write to **temp ledgers**; the final test asserts the canonical
`eval_ledger.jsonl` contains **zero** harness-authored records and zero
fixture records, and that no `prereg_*` model identifier exists in it — fixture
values cannot enter the real evidence ledger as benchmark measurements, and
no fixture number appears anywhere in this report as a scientific result.

---

## T. Pipeline Integration

`PIPELINES` registers all six relevant canonical paths with role, record
behavior, and integration notes:

- **calibration_test** — harness imports the pinned `compute_brier_score` /
  `compute_ece` (single authoritative Brier/ECE path);
- **eval_ulb** — NR-04 refactor: the script body moved under `def main():`
  with an `if __name__ == "__main__"` guard (the file previously executed
  training at import). Behavior when run as a script is unchanged; **all
  cited line numbers are preserved** (guard replaced the blank line 52;
  `recall_at_fpr` at :24, `searchsorted` at :29, scalers at :82/:85 —
  asserted); the harness imports the pinned ranking `recall_at_fpr` and the
  200-replicate `bootstrap_ci` directly;
- **cross_dataset_eval** — registered; its distinct recall@1%FPR variant
  documented as non-interchangeable (NR-03 §I pin belongs to eval_ulb);
  its try/except-wrapped ledger append (NR-01 flag) recorded;
- **train_compare** — registered with `outdir_required: true` (harness-invoked
  runs must pin `--outdir` so production artifacts are never overwritten);
- **evaluate** — registered; `refuse_test_tuning` parity-tested; its
  output-dir-local ledger behavior documented (canonical ledger remains the
  harness’s single source);
- **ibm_train** — registered with a **documented incompatibility** (§21):
  producing a ledger record requires re-execution, which would overwrite
  `reports/ibm_train/training_report.json` — the artifact pinned by claims
  C-004…C-009 — and takes tens of minutes on 1.2M rows. **The R-04 remainder
  (IBM v2 reproduction as ledger records) is therefore deferred, not done
  this phase**; the integration point (attach a v1.1 record post-run with a
  pinned outdir) is specified in the registry entry. No scientific procedure
  was rewritten.

Import-safety is asserted live: importing `calibration_test`, `eval_ulb`,
`cross_dataset_eval`, and `evaluate` appends nothing to any ledger and runs
no training.

---

## U. Ledger Interaction

Harness execution uses the existing `EvaluationLedger` on the canonical path
`reports/evaluation_runs/eval_ledger.jsonl` (fixtures use temp paths). Every
attempt is distinguishable by experiment id, seed, timestamp, artifact
hashes, and git SHA, and repeated executions of identical commands are
preserved as distinct records: an **attempt salt** (`execution_attempt`,
counted from prior records with the same `config_digest`) prevents the
second-level evaluation-id from colliding on same-second identical reruns
(the suite hits this deliberately and asserts distinct ids + distinct
`model_hash` for changed artifact bytes under the same command). Distinct
artifact versions are never treated as duplicates; failed evidence never
creates a falsely valid measured record (validation happens against the
re-read ledger line, not the in-memory object). No ledger record was deleted
or rewritten by this phase.

---

## V. Claims Enforcement

After all code/doc changes: `claim_evidence_check` → **PASS (22 claims)**,
rc=0; `eval_record_test` → **22/22**, rc=0. No registry entry, README line, or
managed-region row was touched by this phase; no fixture value became a
claim; historical claims remain historical. This report contains **no
measured performance numbers** — only structural counts (statuses, cells,
test tallies) and identifiers that are verifiable by rerunning the suites.
The new suite is wired into the battery after `eval_record_test`
(`.freebuff/p114_battery.sh` line 86, 81 → 82 rows).

---

## W. No-Final-Experiment Verification

- **No final benchmark executed:** the matrix reports `measured_cells = 0`;
  the canonical ledger contains zero `prereg_*` records and zero
  harness-authored records; guard returns `DO NOT EXECUTE — PENDING REVIEW`
  for all seven experiments in preregistered mode.
- **No final-test result generated:** no freeze record exists, and final-tier
  execution without one is a `PROTOCOL VIOLATION` (tested).
- **No performance claim added:** claims registry untouched; checker passes
  with the same 22 claims; this report carries no metric values.
- **No threshold optimized:** no optimizer exists (symbol scan asserted);
  `ThresholdSpec` values in fixtures are literals, never searched.
- **No dataset selected by observed results:** datasets in the registry come
  from protocol/NR-03 pins; E1/E2/E5/E6/E7 datasets remain PENDING.
- **Artifacts untouched by this phase’s code:** no harness/test code writes
  to `models/` or `data/`. Registered datasets (`creditcard.csv`, IBM v2,
  Kaggle splits, `transactions.csv`) were **not modified** (mtime scan of the
  battery window shows only suite-output files: `data/fed/*` simulation
  outputs, `*_results.json`/`*_report.txt` suite reports — pre-existing
  battery behavior). One production file’s mtime changed during the battery:
  `models/production/altman_native/xgb_native.joblib` was tampered and
  restored by `phase50_legacy_attestation_test.py` §23 (its own
  copy2 → tamper → detect-DRIFTED → copy2-restore cycle); the suite passed,
  including its “restore → no drift” check, i.e. the restored bytes match the
  attested hash — **content unchanged, mtime only**. `release_manifest.json`
  was re-signed by the same pre-existing attestation/lifecycle suites (fresh
  nonce/signature — this also happens on every battery run, incl. Phase 105
  and NR-01). All other model artifacts retain their pre-phase mtimes.

No accidental substantive experiment occurred (only hashes, imports, fixture
runs on synthetic arrays, and suite executions); nothing to reclassify as
exploratory.

---

## X. Gate 1 Impact

NR-04 delivers the harness half of Gate-1/Gate-2 readiness: the R-07
“independent benchmark harness” now exists as tested machinery (contract,
registry, guards, evidence-first execution, matrix) with a suite in the CI
battery. What did **not** change: the protocol is still DRAFT/NOT APPROVED
(11 markers + 11 sign-off items), the four-role sign-off block is still
`_pending_`, and the GitHub-runner verification of enforcement remains
unproven (N-01, needs push authorization). Research gate: **PARTIAL**
(unchanged) — the harness is now ready to consume signed decisions rather
than being a missing prerequisite. The roadmap’s NR-04 acceptance line “every
harness run yields a DEMONSTRATED-grade record” is demonstrated on fixtures
(full 14-item provenance, validated against the re-read ledger record);
“contract-count check passes” is now true (N-02 fixed + enforced);
“ibm_train results re-produced as ledger records” is **not** done (§T, §Y).

---

## Y. Remaining Blockers

1. **External reviewers** — statistical (bootstrap count/MMDs/multiplicity/
paired-CI construction/ECE binning), domain (operating point, analyst
capacity, S1 threshold carry), lead-researcher scope (§0), governance; plus
the 4-role protocol sign-off. Until then every experiment stays PENDING.
2. **Dataset dependencies** — BAF (NR-16 acquisition/license) blocks the
second-dataset replication track; IEEE-CIS credentials; bank data blocks the
21+7 contract track (already a BLOCKED cell in the matrix).
3. **GitHub runner verification** (N-01) — needs push/run authorization;
locally proven only.
4. **R-04 remainder (IBM v2 ledger reproduction)** — deferred with a
documented incompatibility (§T): needs a pinned outdir + record attachment
plan so claims C-004…C-009 artifacts are not overwritten. Owner decision.
5. **Final-test freeze record** — not created by design (nothing is ready to
freeze); `PD-FREEZE` is currently a static registry entry gated to final
tiers, so once a real freeze *is* created, that entry must be updated to
check the ledger dynamically — recorded here so it is not forgotten.
6. **Full real-dataset identity checks** — IBM v2’s 2.35 GB hash is not
re-verified inside the suite (runtime); it is verified on first real use.

---

## Z. Next Requirement

**NR-05 — Temporal-split evaluation** (Gate 2): splits pre-declared in the
frozen-test design, no test-set inspection before threshold freeze, results
as ledger records — builds directly on this harness. Identified only —
**not started.**

---

## AA. Scope Verification

- **Files changed this phase (code/config):** new
  `backend/scripts/prereg_harness.py`, new `backend/scripts/prereg_harness_test.py`;
  `backend/scripts/eval_ulb.py` (main-guard refactor only — line-preserving,
  behavior unchanged when run as a script);
  `backend/research/public_feature_contract.json` (+7 lines: N-02 restoration
  of the declared 14th feature `failed_auth_proxy` from the original phase-65
  contract, with `available_in: []` so it can never be silently used);
  `.freebuff/p114_battery.sh` (+suite); this report.
- **Protocol:** untouched this phase — version 0.1.2-draft, marker literal
  count **11 → 11**, no Change-history row needed (nothing in the protocol
  changed).
- **Model artifacts / datasets / secrets:** registered datasets byte-untouched;
  model-artifact content unchanged (the aggregate mtime digest moved only via
  the battery’s pre-existing tamper/restore + manifest re-sign cycles and
  suite-output files under `data/` — detailed in §W); no `.env`/`ACCESS`/admin
  files touched; no training executed; threshold 0.018758 untouched.
- **Reviewer decisions:** none resolved by assumption — all six approval
  gates remain closed; no approval claimed anywhere.
- **Checks run after edits:** `prereg_harness_test` 31/31; `eval_record_test`
  22/22; `claim_evidence_check` PASS (22); marker count 11; `py_compile` on
  all changed/added python files; full battery (below).

---

## AB. Final Status

```
PASS WITH LIMITATIONS
```

**Final verification (§29), all after the last edit:**

| Check | Result |
|---|---|
| `git status` / `git diff --stat` | 4 tracked files changed (`eval_ulb.py` main-guard, `public_feature_contract.json` +7, hook-ledger +3 lines, hook `calibration_metrics.json`), 3 new files (`prereg_harness.py`, `prereg_harness_test.py`, this report) + 3 hook record sidecars; nothing else |
| Protocol not substantively altered | `git diff` on the protocol = **empty**; version 0.1.2-draft; markers **11 → 11** |
| No model artifact changed (content) | production xgb: mtime-only tamper/restore verified by the suite’s own hash check (§W); `models/artifacts/*` mtimes unchanged; other production files untouched |
| No dataset changed | registered datasets byte-untouched (mtime scan, §W) |
| No environment secret changed | no `.env`/ACCESS/admin files in the change set |
| No final experiment ran | matrix `measured_cells = 0`; canonical ledger 44 records, **0 harness-authored, 0 fixture**, no `prereg_*` id; all 7 guards refuse |
| Harness/unit tests | `prereg_harness_test` **31/31 PASS** (rc=0) |
| `eval_record_test` | **22/22 PASS** (rc=0) |
| `claim_evidence_check` | **PASS — 22 claims** (rc=0) |
| Deterministic fixture behavior | all 9 §19 cases pass (§R) |
| Pending-decision guards | all 7 experiments `DO NOT EXECUTE — PENDING REVIEW`; 22 pending entries; all 6 approval gates closed |
| Blocked-dataset guards | BAF → `BLOCKED — DATASET UNAVAILABLE`; hash/contract/split failures all blocked (§N/§O) |
| Evidence-failure behavior | `EVIDENCE FAILURE` → `NOT ESTABLISHED`, never MEASURED (§H) |
| Full battery (extended 81 → 82 rows) | **82 PASS / 0 FAIL** with stack 6/6 healthy, including the new `prereg_harness_test` row |
| Final Git SHA | `5a5ff55cc03df73318fdf31a16e3665f159a4720` (`5a5ff55`) — unchanged; **nothing committed** (no commit was requested for this phase) |
| Working-tree state | modified: `backend/research/public_feature_contract.json`, `backend/scripts/eval_ulb.py`, `reports/evaluation_runs/eval_ledger.jsonl` + `reports/calibration_test/calibration_metrics.json` (hook-generated, 41 → 44 records during the phase); untracked: the 2 harness code files, this report, 3 hook sidecars |

**Limitations (why not PASS):**

1. **R-04 remainder not closed:** `ibm_train` is registered but was not
   re-executed — doing so would overwrite the claim-pinned
   `training_report.json` (C-004…C-009); reproduction needs a pinned-outdir
   plan (§T/§Y). The roadmap acceptance line “ibm_train results re-produced
   as ledger records” is therefore **not** met by this phase.
2. **`PD-FREEZE` is static:** it blocks all final-tier runs until a freeze
   record exists; once a real freeze is created it must be updated to check
   the ledger dynamically (§Y.5).
3. **IBM v2 hash not re-verified inside the suite** (2.35 GB runtime cost);
   registered and verified on first real use (§N).
4. **GitHub-runner verification** of the extended battery remains blocked
   (push/run authorization — carried from NR-01).
5. The new suite runs in the **local battery** only; it was not added to the
   CI workflow (the committed workflow still runs the two Phase-105 evidence
   steps).

No reviewer decision was resolved by assumption; no approval is claimed; no
scientific methodology, MMD, marker, or threshold changed; no final benchmark
ran; no new performance claim exists. Research gate: **PARTIAL** (unchanged).
**NR-05 (temporal-split evaluation) is the next requirement — not started.**

