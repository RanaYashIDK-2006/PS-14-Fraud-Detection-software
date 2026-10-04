# NR-01 — Evidence Enforcement Verification & Battery Coverage

**Phase:** NR-01 (Gate 1, first substantive phase after Phase 106) · **Date:** 2026-10-03
**Scope:** evidence and reproducibility verification only. No model, dataset, feature,
threshold, methodology, preregistration, or scientific-conclusion change was made.

---

## A. Objective

Establish that the repository's evidence-enforcement system actually:

1. detects unsupported quantitative claims (not merely executes);
2. preserves legitimate historical/example/metadata content under a defined policy;
3. covers the canonical evaluation pipelines;
4. behaves correctly around CI/commit coupling;
5. is covered by the local regression battery (the G-11 remainder from Phase 104);
6. has an honestly recorded GitHub-runner status.

This report is the single NR-01 deliverable.

---

## B. Baseline

| Item | Value at NR-01 start |
|---|---|
| Git SHA | `a0b600ba2edd917498d3a9551c19e10225930e4b` (`a0b600b`) — unchanged |
| Phase 105 status | PASS WITH LIMITATIONS (evidence baseline built) |
| Phase 106 status | PASS (roadmap reconciled; NR-01 = next requirement) |
| Working tree | 13 modified tracked files (Phase 105 set) + untracked Phase 104/106 docs, `claim_evidence_check.py`, evidence artifacts |
| Evidence ledger | 11 records at Phase 106 read; **17–19 during NR-01** (grows ~every 10 min via a client-side hook — always cite as a timestamped observation, see §I) |
| Claims registry | 22 entries (`docs/evaluation/claims_registry.jsonl`) |
| Battery (Phase 105) | 79/79 effective; **did not include** `claim_evidence_check` or `eval_record_test` (G-11 remainder) |
| CI | two evidence steps present in the **working-tree** `.github/workflows/ci-cd.yml` (uncommitted); committed workflow lacks them; never run on GitHub runner |
| Stack | ports 8000–8005 all `/health` → 200 before the battery run |

---

## C. Enforcement Surface (verified from the actual implementation)

All behavior below was read from `backend/scripts/claim_evidence_check.py` and
`backend/scripts/eval_record.py`, not from Phase 104/105 prose.

### Mechanisms

1. **Registry integrity** (`run_check` → `verify_evidence`): every entry in
   `docs/evaluation/claims_registry.jsonl` must resolve against real evidence.
2. **Document coverage** (`scan_surfaces`): inside managed regions, every quantitative
   **table row** must carry a `<!-- claim:C-NNN -->` annotation that is registered,
   whose documented file matches, and whose value matches `documented_value ± value_tolerance`.
3. **Region structure**: nested regions, unclosed regions, end-without-start → FAIL.
4. **Registry structure**: duplicate/invalid claim ids, invalid classification, missing
   claim text, missing `documented_in`, missing annotation in `documented_in`, unparseable
   JSONL (`REGISTRY_PARSE_ERROR`) → FAIL.

### Enforced files/regions (exactly two surfaces)

| Surface | Managed region |
|---|---|
| `README.md` | lines 25–61 ("Key Results": in-domain, external-transfer tables) |
| `docs/PHASE_105_EVIDENCE_BASELINE.md` | lines 150–159 (historical headline table) |

No other file is scanned. **There are no git hooks** (`.git/hooks/` contains only
sample files). CI runs the checker as a test-job step
(`.github/workflows/ci-cd.yml` lines 56–63, working tree only).

### What counts as valid evidence

| Evidence type | Rule |
|---|---|
| `ledger` | record must exist in `reports/evaluation_runs/eval_ledger.jsonl`; `json_path` must resolve; value must match within tolerance. For **DEMONSTRATED**: `dataset.sha256`, `git_commit`, `status=COMPLETED`, `command` (v1.1), and `seed` or `evaluation_config.seed_not_applicable` are all required |
| `artifact` | file must exist, parse, and contain `json_path`. For **DEMONSTRATED**: `provenance.dataset_sha256`, `provenance.split`, `provenance.producer`, and `provenance.seed` (or `seed_not_applicable`) required |
| `suite` | log must contain the `expect` marker; can support at most **SELF-TESTED** (DEMONSTRATED with suite evidence → FAIL) |
| `none` | only for **NOT ESTABLISHED**, and requires a `reason` |
| `blocked` | only for **BLOCKED**, and requires a `blocker` |

### Provenance field representation

- **dataset identity** → `dataset.sha256` (SHA-256 over raw file bytes, `dataset_fingerprint()`)
- **split identity** → `evaluation_config.split` (records) or `provenance.split` (artifacts)
- **seed** → top-level `seed`, or `evaluation_config.seed_not_applicable` for deterministic runs
- **command** → top-level `command` (schema v1.1; legacy v1.0 records have `None` and cannot back DEMONSTRATED claims)
- **Git SHA** → `git_commit` (captured by `git rev-parse HEAD` at run time)
- **artifact identity** → `artifact_file_hashes` + aggregate `model_hash`; ledger identity → `evaluation_id`

---

## D. Claim Classification

| Class | Examples | Enforced? |
|---|---|---|
| **A. Current benchmark claims** | C-001…C-003 (ULB, DEMONSTRATED, ledger), C-004…C-010 (IBM/23B, SELF-TESTED, artifact), C-013…C-015 (external, artifact/ledger), C-016/C-017 (calibration, ledger), C-018 (leakage suite) | **Evidence enforced for all 22**; **row-value enforced only where annotated inside a managed region** (C-001…C-015 are; C-016/C-017/C-018 are annotated only *outside* the regions — see §G FN-2) |
| **B. Historical claims** | C-011/C-012 (0.435/44.9%, inside README region), C-101…C-104 (0.966/0.877/97.8%/1.996%, inside PHASE_105 region) | Enforced: `NOT ESTABLISHED` requires an explicit reason and forbids evidence-backed types; values are row-checked |
| **C. Methodological/example values** | prose like "seed 42", "n=1470", "--ibm-rows 200000", formulas | Exempt by design: prose/non-table lines are never scanned (verified by FP probes, §F) |
| **D. Test fixtures** | self-test fixtures (0.5123, C-901 coupling fixture) | Isolated in temp dirs; never touch the real registry/ledger (verified — live check unchanged after self-test) |
| **E. Metadata** | dates, versions, schema versions, row counts, ledger ids, config limits | Exempt: not bare-numeric cells inside a region, or outside regions entirely (§F) |
| **F. Evidence artifacts** | ledger records, `record_*.json` sidecars, `reports/**/*.json` artifacts | Not scanned as claims; they are the *targets* of verification (hash/parse/value) |

---

## E. Test Matrix (expected vs observed)

Mechanism: `claim_evidence_check.py --self-test`, **31 cases, 31 PASS, 0 FAIL**
(extended this phase from the original 7). Every required matrix row:

| Case | Expected | Observed |
|---|---|---|
| Valid measured claim with valid evidence | PASS | PASS (ledger fixture + artifact-with-provenance fixture + live 22-claim check) |
| Quantitative claim with no evidence | FAIL | FAIL — "unregistered quantitative row" + "DEMONSTRATED with no evidence" |
| Quantitative claim with malformed evidence | FAIL | FAIL — unknown evidence type; unparseable registry (`REGISTRY_PARSE_ERROR`); artifact with invalid schema (json_path absent) |
| Quantitative claim with incomplete provenance | FAIL | FAIL — four separate cases: missing `dataset.sha256`, `git_commit`, `command`, `seed` |
| Quantitative claim with wrong provenance | Explicitly defined | FAIL when evidence value belongs to another experiment; FAIL when `json_path` absent; **consistent doc+registry forgery passes by design** (consistency, not authorship — registry is a tracked file, so a forged edit is visible in review) |
| Historical claim | Expected policy result | PASS when `NOT ESTABLISHED` + reason (C-002/C-101 fixtures); FAIL when edited without registry (scenario E) |
| Methodological example | Expected policy result | PASS — prose with numbers inside a region is not scanned |
| Test fixture | Expected policy result | PASS — fixtures live in temp dirs; FP probe row outside region passes |
| Metadata number | Expected policy result | PASS — dates/versions/sizes/limits outside regions pass; headers/separators/non-bare cells inside regions pass |
| Evidence artifact with valid schema | PASS | PASS (artifact evidence + full provenance fixture) |
| Evidence artifact with invalid schema | FAIL | FAIL (missing file; json_path absent; value mismatch) |
| Stale/mismatched evidence | FAIL or defined status | FAIL (documented-value mismatch = stale documentation); ledger evidence is immutable so a superseded *artifact* on disk does not invalidate a ledger-pinned claim (defined status, §G FN-7) |

The checker does **not** merely execute: **22 of the 31 cases** assert a *specific*
rejection message for the unsupported case it must reject (the other 9 assert the
absence of errors for legitimately-passing content).

---

## F. False-Positive Audit

| Probe | Result |
|---|---|
| Date row, version row, dataset-size row, config-limit row, example row, fixture row, historical row — all **outside** a managed region | PASS (not scanned — by design) |
| Table header (`\| Phase 23B \|`), separator, prose line containing numbers, non-bare cell (`1.2M`) — **inside** a managed region | PASS (not quantitative claims) |
| Live README + PHASE_105 surfaces (22 claims, both regions) | PASS — zero false positives |
| Real-content temp copy of the full registry+ledger+artifacts baseline | PASS |
| Battery rerun of the checker after all edits | PASS (row 68 of the battery) |

**Conclusion:** no legitimate date/version/size/limit/example/fixture/metadata value
was blocked in any tested configuration. Bare-numeric metadata rows *inside* a managed
region would be treated as claims (by design — the region is the claim surface); no
such row exists in either live surface.

---

## G. False-Positive-Negative Audit — discovered gaps and dispositions

| # | Gap (class searched) | Disposition |
|---|---|---|
| FN-1 | **Prose metric claims inside managed regions are not scanned** (README L34 historical-tuple warning blockquote) | **Intentionally excluded, documented reason:** prose-scanning would flood on metadata (seed/row counts/ledger ids in the same prose); the same historical values ARE registered and row-enforced in the PHASE_105 managed table (C-101…C-104) |
| FN-2 | **C-016/C-017/C-018 are annotated only outside managed regions** (README L88, L118, L126, L576, L1159) — evidence is verified, but row values are never cross-checked | **Intentionally excluded for NR-01;** remediation belongs to **NR-41** (README rewrite should place every quantitative claim inside a managed surface). Editing L88's "0.0009" would not be caught today |
| FN-3 | **README "Claims & Evidence" index table (L113–132)** — quantitative rows outside any managed region | **Intentionally excluded:** it is a classification index whose cells carry evidence pointers inline; registering its ~18 rows is an evidence-registration task, not a checker defect; growth path = extend the region under NR-41 |
| FN-4 | **README cross-domain matrix (L918–921) and meta-ensemble table (L939–944)** — bare ROC-AUC-like performance values with no annotations | **Blocked by an explicit external limitation for evidence upgrade + surface exclusion documented:** no committed artifact contains these values (`data/meta_ensemble_results.json` absent; `misc/reports/cross_domain_*.json` hold different experiments); regeneration requires downloading the 4 public datasets. Registering them unevidenced would manufacture claims — forbidden. Tracked for NR-04/NR-41 |
| FN-5 | README dataset table (L909–912) and phase-history table (L238–267) | **Intentionally excluded:** dataset metadata and historical audit log, not benchmark claims |
| FN-6 | **`fraud_report.py` success path hardcoded the historical tuple** (`ROC-AUC=0.966 \| PR-AUC=0.877 \| Recall@1%FPR=0.918`) in the file-output block (line 357), while Phase 105 had already computed `_roc/_pr/_r1` for stdout/metrics.json | **Fixed now (smallest correction):** line 357 now formats the computed values. Verified: `py_compile` OK, AST confirms `_roc/_pr/_r1` module-scope assignments precede the use, BLOCKED guard still exits rc=2 (live re-run). Residual hardcoded *narrative* stats in the same unreachable path (e.g. "45% of fraud…") are **deferred to NR-28**, which already owns `fraud_report`'s disposition — computing them would be substantive new analysis, not a minimal correction |
| FN-7 | **Current-artifact divergence:** ledger-pinned claim C-016/C-017 (Brier 0.0009280, record `…72d714bd81b3`, calibrator bytes `21b37c…`) vs the calibrator now on disk (bytes `6c885d…`, measured by later records at Brier 0.0107337); `reports/calibration_test/calibration_metrics.json` has been overwritten accordingly | **Documented, not silently rewritten (hard rule).** The claim remains ledger-true (immutable record preserved). Root cause: `train_compare.py` defaults `--outdir models/artifacts`, so the Phase 105 reproduction run rewrote the production artifact set at 09:00:47 UTC. Checker behavior is *by design*: ledger evidence proves the execution, not the current bytes. Artifact-retention/distribution policy = **G-40/NR-44**; README wording clarification = **NR-41**. No claim was edited this phase |
| FN-8 | Hardcoded metric literals in executable code (repo-wide search for 0.966/0.877/97.8/1.996) | Remaining hits: self-test fixtures (intentional, class D) and `phase71_architecture_gap_audit.py` audit-fixture tuple (historical audit data, class B/E). **`compare_ml_systems.py` clean** (Phase 105 de-hardcode holds). `fraud_report.py` hit fixed (FN-6) |
| FN-9 | Metrics in other docs (`misc/docs/CLAIMS_MATRIX.md`, PHASE_104/106 audits) | **Intentionally excluded by design:** audit/historical documents; the enforcement docstring reserves surface growth over time |

---

## H. Canonical Pipeline Coverage

### H.1 Calibration evaluation — `backend/scripts/calibration_test.py`
- Generates artifact `reports/calibration_test/calibration_metrics.json` + ledger record + `record_*.json` sidecar: **YES**.
- Schema/provenance: dataset sha ✓, git ✓, command ✓, seed via `seed_not_applicable` ✓, split=`validation` ✓.
- Claim mapping: C-016/C-017 → ledger `…72d714bd81b3` (value within tolerance) ✓.
- Missing evidence → checker FAIL (scenario F tested). **Caveat:** the artifact file is *overwritten* every run (hook-driven, §I) — claims correctly bind to the ledger, not the file (FN-7).

### H.2 ULB evaluation — `backend/scripts/eval_ulb.py`
- Ledger record + sidecar + `reports/ulb_results.json` (carries `evaluation_id`): **YES**.
- Provenance: dataset `data/creditcard.csv` sha ✓, seed=42 ✓, command ✓, split=`random_stratified_80_20_test_size=0.2_random_state=42` ✓, git ✓.
- Claim mapping: C-001/C-002/C-003 → `…e58d68c2bf88` ✓; README region rows cross-checked ✓.

### H.3 Cross-dataset evaluation — `backend/scripts/cross_dataset_eval.py`
- Ledger record + sidecar: **YES**; seed via `seed_not_applicable` ✓, command ✓, warning about proxy mapping recorded.
- Claim mapping: C-015 → `…e2a14ae94bef` ✓ (README row 0.873 ↔ evidence 0.8726 within tolerance).
- **Weakness (documented):** record generation is wrapped in `try/except Exception → print("evaluation record skipped")` — a record failure does not fail the run. Enforcement catches this later only if a registry claim references the missing record. Same pattern in `train_compare.py`.

### H.4 Synthetic fused — `backend/src/train_compare.py`
- Ledger record: **YES** — the Phase 105 `gate_rows` fix holds: the append block now runs *after* OOD-gate evaluation and before gate-failure exit, so passing and failing runs both record (code read, lines 888–935).
- Provenance: seed ✓, command ✓, split `time_70_15_15` ✓, threshold_source=`validation` ✓, OOD-gate rows embedded ✓.
- **No registry claim currently consumes a train_compare record** — the synthetic fused baseline is ledgered but not displayed in a managed surface (Scenario C in the wild: evidence without claim — allowed by definition).
- **Side effect (finding FN-7):** default `--outdir models/artifacts` rewrites production artifacts during evaluation.

### H.5 Other pipelines
- `backend/scripts/evaluate.py`: generates a full v1.1 record but writes it to `args.output_dir/eval_ledger.jsonl` — an **output-dir-local ledger**, not the canonical `reports/evaluation_runs/eval_ledger.jsonl`. Not consumed by any claim. Documented; canonical-ness is by convention, not code.
- `backend/scripts/fraud_report.py`: **cannot currently produce evidence** — BLOCKED rc=2 live-verified (missing `xgb_kaggle.joblib`/`scaler_kaggle.joblib`, no producer script). Marked explicitly rather than fabricated (per §5). Hardcoded tuple fixed (FN-6).
- `backend/scripts/result_matrix.py`: read-only consumer of records (no generation).

**Legacy record note:** record `…19002de2f24e` (Sept 14) is schema v1.0 — no `command`.
No registry claim references it; if one tried, DEMONSTRATED rules would correctly reject.

---

## I. Ledger Duplicate-Growth Analysis

Observed directly during NR-01 (ledger **11 → 19** and still growing; counts below are
a timestamped observation at the 17-record snapshot):

- **Mechanism:** a client-side test hook (configured outside the repository — no in-repo
  config found) re-runs `calibration_test.py` at ~10-minute cadence (re-runs observed at
  09:44, 09:54, 10:05, 10:16, 10:27, 10:28 UTC during this phase).
- **Are they duplicates?** calibration records (12 at the 17-record snapshot, growing)
  form **two content groups** (hash over
  metrics+config+model_hash+dataset+seed+git+command+status+warnings):
  - group A (2 records, Brier 0.0009280) — pre-rewrite artifact bytes `21b37c…`;
  - group B (10 records at the snapshot, growing — all later executions, Brier 0.0107337)
    — artifact bytes `6c885d…` (rewritten by the
    Phase 105 `train_compare` run at 09:00:47 UTC, FN-7).
  Within a group, records are content-identical **re-executions** — each genuinely ran
  the script; they are not fabricated or copied.
- **Distinguishability:** every record has a distinct `evaluation_id` (UTC-second +
  content digest) and distinct `timestamp_utc`; `artifact_file_hashes` separate the two
  content groups; the ledger is append-only (no record rewritten).
- **Impact on reproducibility/audit:** none — a reader can group by content hash or
  cite the pinned id (C-016/C-017 cite `…72d714bd81b3`, which is immutable). Growth is
  linear in hook firings and bounded by hook cadence.
- **Conclusion: the current schema already provides enough information** to distinguish
  repeated executions (id + timestamp + artifact hashes + command). **No change made** —
  dedup machinery would be infrastructure without traceability benefit (§6 permits
  documenting sufficiency; that is the finding).

---

## J. Claims Registry Integrity

All 22 entries classified and verified (`claim_evidence_check` live → PASS, exit 0):

| Classification | Count | IDs | Evidence resolves? |
|---|---|---|---|
| DEMONSTRATED | 8 | C-001,002,003,013,014,015,016,017 | ✓ ledger/artifact with full provenance |
| SELF-TESTED | 8 | C-004…C-010, C-018 | ✓ artifact/suite |
| NOT ESTABLISHED | 6 | C-011,012, C-101…C-104 | ✓ explicit reasons (no evidence by policy) |
| BLOCKED | 0 | — | — |
| **invalid/stale/mismatched/unsupported/fabricated** | **0** | — | — |

- Valid claims map to valid evidence (live resolution of all ledger ids/artifact paths —
  4 distinct artifact/suite paths, all exist).
- Unsupported claims cannot silently appear as measured: classification rules forbid
  `none` evidence for DEMONSTRATED, suite evidence for DEMONSTRATED, missing provenance.
- Stale documentation detectable: value-mismatch check (scenario E).
- Historical records distinguishable: classification + `evidence.type=none` + reason.
- No fabricated evidence found.
- **Discrepancy documented before repair (none required repair):** (1) C-016/17/18
  annotated outside managed surfaces (FN-2); (2) ledger-pinned calibration claim vs
  current artifact bytes (FN-7) — claim itself unchanged and ledger-true; (3) a second,
  legacy registry `misc/reports/CLAIMS_REGISTRY.json` (S1–S6 security claims,
  `last_verified: 2026-08-27`, referenced by README L678) is **not consumed** by the
  checker — historical security inventory, stale dates noted, deliberately untouched.

---

## K. Commit Coupling

All scenarios run against **real repository content** (temp copies of README +
PHASE_105 + registry + ledger + all referenced artifacts):

| Scenario | Expected | Observed | Exact first error (if any) |
|---|---|---|---|
| Baseline (unmodified copies) | PASS | **PASS** | — |
| A — claim + evidence together | accepted | **PASS** | — |
| B — new quantitative claim without evidence | rejected | **FAIL** | `README.md:37: annotation claim:C-901 is not in the registry` |
| C — evidence without claim | explicitly defined | **PASS by definition** (evidence may exist unclaimed; the converse is enforced) | — |
| D — claim and evidence changed in the same change set | recognized | **PASS** (checker validates the end state; a consistent set passes) | — |
| E — historical result modified | not allowed to masquerade | **FAIL** (doc edited, registry unchanged): `row for C-101 does not contain the registered value 0.966`; **consistent doc+registry edit passes by design** — detection is via the tracked registry diff in review, documented as defined behavior | |
| F — evidence referenced but unavailable | rejection/BLOCKED | **FAIL** | `C-001: ledger record eval-20261003T084934+0000-e58d68c2bf88 not found` |

**CI-coupling note:** the checker validates the *checked-out state*. On the current
remote, the two CI steps and all evidence files are uncommitted — a runner checkout of
the remote HEAD would neither run the steps nor contain the evidence. Commit coupling
is therefore **locally demonstrated, remote-unverified** (§L).

---

## L. GitHub Runner Status

```
GITHUB RUNNER VERIFICATION: BLOCKED — PUSH/RUN AUTHORIZATION REQUIRED
LOCAL VERIFICATION:          VERIFIED (live check, 31-case self-test, 81/81 battery)
GITHUB RUNNER VERIFIED:      NOTHING
```

Exact limitations:

1. No GitHub CLI (`gh: command not found`) — no workflow dispatch path.
2. `git remote -v` → `origin github.com/RanaYashIDK-2006/PS-14-Fraud-Detection-software.git`,
   read access works (`ls-remote` OK), but **write access was not tested and no push was
   made** — pushing would require committing the entire uncommitted Phase 104→NR-01
   worktree (13 modified + ~30 untracked files, including every evidence file the
   commit-coupling check needs), which is beyond this phase's authorization
   (repository convention: no commit workflow; no push requested).
3. The committed workflow does not contain the two evidence steps (working-tree diff
   only), so a remote run today would not exercise enforcement regardless.

No CI result was fabricated or inferred. `docs/evaluation/PRE_REGISTERED_EVALUATION_PROTOCOL.md`
and all claims were left untouched.

---

## M. Regression Results

| Command | Result | Count | Failures | Reruns | Notes |
|---|---|---|---|---|---|
| `bash .freebuff/p114_battery.sh` (stack up, 8000–8005 healthy) | **81 PASS / 0 FAIL**, `BATTERY END 2026-10-03T16:01:59+05:30` | 81 | 0 | 0 | includes the **two new rows**: `claim_evidence_check` (row 68) and `eval_record_test` (row 69) — G-11 remainder closed. No environmental failures (no service restart occurred) |
| `claim_evidence_check.py --self-test` | **31/31 PASS**, rc=0 | 31 | 0 | 1 fixture repair | initial extension run showed 29/30: the suite-evidence *fixture* forgot to update its surface row (test-fixture defect, not a checker defect) — fixed, then 30/30, then 31/31 after adding the invalid-schema case |
| `claim_evidence_check.py` (live) | **PASS — 22 claims verified**, rc=0 | 22 claims | 0 | run 3× (baseline, mid-phase, final) | stable across all hook-driven ledger appends |
| `eval_record_test.py` | **22/22 PASS**, rc=0 (standalone) + battery row PASS | 22 | 0 | 0 | |
| `fraud_report.py` (after FN-6 fix) | rc=**2 BLOCKED** (expected) | — | — | — | BLOCKED guard unaffected by the line-357 fix; `py_compile` OK; AST scope check OK; no battery suite imports `fraud_report` (verified) |

Evidence for the battery run lives in `.freebuff/p114rq_battery_results.tsv` /
`.freebuff/p114rq_battery.log` / `.freebuff/nr01_battery_driver.out` (the tally is
reproduced above so the single-deliverable rule holds).

---

## N. Scope Verification

Confirmed **NOT** altered by this phase:

- model architecture, training data, evaluation datasets, feature definitions,
  thresholds, evaluation methodology, preregistration criteria, scientific conclusions,
  performance claims, deployment architecture, governance scope;
- `backend/src/privacy_layer/`, `backend/src/risk_engine/`, `models/`, `data/` — untouched;
- README claim text and `claims_registry.jsonl` — **not edited** (the calibration
  divergence was documented, not rewritten);
- no new performance claim introduced (the observed 0.0107337 artifact measurement is
  recorded as a factual ledger-backed observation about artifact state, not as a
  replacement claim, ranking, or classification change).

Files changed by NR-01 (exactly three, all mechanical):

1. `backend/scripts/claim_evidence_check.py` — `self_test()` extended 7 → 31 cases
   (enforcement logic `run_check`/`verify_evidence`/`scan_surfaces` unchanged).
2. `.freebuff/p114_battery.sh` — 2 `run` lines added (the two evidence suites).
3. `backend/scripts/fraud_report.py` — one line: hardcoded tuple → computed
   `_roc/_pr/_r1` (FN-6).

Plus this report (the required deliverable). Hook-driven side effects outside this
phase's control: ledger appends + `record_*.json` sidecars + `calibration_metrics.json`
overwrites (§I).

---

## O. Gate 1 Readiness

**NR-01 materially advances Gate 1 but does not complete it.**

- **G-11 → CLOSED:** the battery now covers `claim_evidence_check` + `eval_record_test`;
  81/81 demonstrated with the stack up.
- **G-01 → remains PARTIAL:** local enforcement is now proven with a 31-case matrix
  (valid/invalid/provenance/historical/fixture/metadata/coupling), but the GitHub-runner
  execution and evidence-commit coupling on the remote remain blocked (N-01 carried as-is).
- **N-01 (commit coupling):** locally demonstrated across scenarios A–F; remote behavior
  still unverified — unchanged status, now precisely characterized.
- Enforcement reality check: the mechanism **rejects the specific unsupported cases it
  is intended to reject**, with defined exemptions; known FN classes are dispositioned
  (§G) rather than assumed absent.
- Gate 1 still requires **NR-02** (protocol sign-off) and **NR-03** (metric definitions
  v1.1 + frozen-test design). Current research gate remains **PARTIAL**.

---

## P. Next Requirement

**NR-02 — Pre-registration factual correction, contract-scope decision & sign-off**
(apply the four Phase 106 §H corrections, choose the evaluated contract track(s),
resolve the 11 approval markers, leave DRAFT). Identified only — **not started.**

---

## Q. Limitations / Blockers (carried forward only, genuine)

1. **GitHub runner verification blocked** — push/run authorization required (NR-01
   remainder; also NR-01's own external dependency).
2. **Calibration artifact divergence (FN-7)** — ledger-claim vs current artifact bytes;
   remediation = NR-41 (README wording) + NR-44 (artifact retention/distribution policy);
   `train_compare` default outdir side effect noted for NR-28/NR-44 review.
3. **FN-2/FN-3/FN-4** — claims annotated outside managed surfaces, README index table,
   and the artifact-less cross-domain/meta-ensemble tables remain outside enforcement
   (documented reasons; NR-41/NR-04 paths).
4. **Record-generation failures are non-fatal** in `cross_dataset_eval`/`train_compare`
   (`try/except → print`) — detected only downstream at check time; acceptable for now,
   noted for NR-04 harness design.
5. **Hook-driven ledger growth** continues (~10-min cadence); sufficient schema to
   distinguish runs (§I) — ledger counts must always be cited as timestamped observations.
6. **`evaluate.py` writes an output-dir-local ledger**, not the canonical one; no claim
   depends on it.
7. **Legacy `misc/reports/CLAIMS_REGISTRY.json`** stale since 2026-08-27; not consumed
   by enforcement; left untouched (candidate for NR-28 consolidation).

---

## R. Final Status

```
PASS WITH LIMITATIONS
```

Limitations are exactly those in §Q: GitHub-runner verification is explicitly recorded
as BLOCKED (permitted by the acceptance criteria when unauthorized), and the discovered
false negatives are dispositioned with documented reasons rather than silently ignored.
All other acceptance criteria — enforcement documented from actual implementation,
missing/malformed/incomplete/mismatched provenance rejected, historical/example/
fixture/metadata behavior verified, FP+FN audits performed, canonical pipelines audited,
registry integrity checked, ledger growth understood, commit coupling tested, regressions
green (81/81), no scientific change, Gate 1 readiness updated, NR-02 identified, no new
governance phase invented — are met.
