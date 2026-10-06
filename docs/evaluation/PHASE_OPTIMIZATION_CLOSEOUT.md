# PS-14 — Phase 5 (Prototype Optimization) Closeout

**Phase identity:** Phase 5 — Prototype Optimization (Track A engineering
optimization; Track B model-level deliberately not performed). Started and
completed 2026-10-06. Companion document:
`docs/evaluation/PROTOTYPE_OPTIMIZATION_REPORT.md` (full evidence, 20
sections).

## Commit / SHA ledger

| Item | SHA |
|---|---|
| Starting SHA (measured code for every Phase 5 run) | `ab5a6d0a6baf2c743de9c05fd4277b602478ea60` |
| Source + evidence commit | recorded post-push below |
| Docs commit (report + this closeout) | recorded post-push below |
| Timer-records commit (records-only) | recorded post-push below |
| CI record follow-up commit | recorded post-push below |

*Ledger completed in the post-push follow-up commit, as in Phase 4A.*

## Changed files

**Source / tooling (5 + 2):**

- `backend/src/risk_engine/altman_native_ensemble.py` — +40 lines: new
  `predict_combined_many(rows)` batched inference (pure addition; `predict`,
  `predict_many`, `map_raw_to_native` byte-identical to HEAD) with an exact
  float64 widening cast so results are bit-identical to `predict()`.
- `backend/src/risk_engine/main.py` — `/internal/evaluate-batch` uses the
  batched path; loop now iterates `feature_dicts_allowed` (fixes score/block
  mis-pairing under BLOCK_INFERENCE); truthful docstring. +30/−? lines.
- `backend/scripts/risk_engine_test.py` — +58 lines: Phase 5
  batch-equivalence block (exact loop-vs-many scores/uncertainties; 32-event
  single-vs-batch endpoint identity).
- `backend/scripts/benchmark_e2e.py` — repaired (missing `fraud_id`, wrong
  batch payload shape, stale `FeatureVector` names, invalid bench fraud_id
  pattern, no status check on section 4); runs end-to-end.
- `backend/scripts/review_package_check.py` — R4 sha-pin updated for the
  sanctioned ensemble addition (pin follows the sanctioned source change).
- `scripts/large_scale_runner.py` — `--threads`, `--scores-out`,
  `prediction_sha256`/`member_sha256`, `batch_wall_ms`, parallel model load.
- `scripts/large_scale_cpu_sampler.py` — `--threads`/`--run-out`
  passthroughs (Phase 3 outputs never clobbered).

**Evidence (tracked, `misc/reports/`):** `phase5_t0_run.json`,
`phase5_t16_run.json`, `phase5_50m_t0_run.json`, `phase5_50m_t16_run.json`,
`phase5_smoke_t0.json`, `phase5_smoke_t16.json`, `phase5_cpu_t0.json`,
`phase5_cpu_t16.json`, `phase5_cpu_50m_t0.json`, `phase5_cpu_50m_t16.json`,
`phase5_profile.json`, `phase5_threads.json`, `phase5_single_event.json`,
`phase5_memload.json`, `phase5_model_load.json`, `phase5_equivalence.json`,
`phase5_e2e_bench.txt`, `phase5_battery_results.tsv` (18 files, all 1–16 KB).

**Documents:** `docs/evaluation/PROTOTYPE_OPTIMIZATION_REPORT.md`,
`docs/evaluation/PHASE_OPTIMIZATION_CLOSEOUT.md`.

**Not included in source commits (separate records-only commit):** 17
hook-generated `reports/evaluation_runs/record_eval-*.json` files,
`reports/evaluation_runs/eval_ledger.jsonl`,
`reports/calibration_test/calibration_metrics.json` — timer-generated
calibration records, documented per brief §21 and distinguished from
source/model changes. No `.env`, no secrets, no large benchmark artifacts
(NPZ score dumps stay gitignored).

## Optimization experiments

| # | Experiment | Outcome |
|---|---|---|
| 1 | Runner `--threads` per-member model threading | Accepted — 1.900× (5M), 1.665× (50M), 1.541× (500k), bit-identical |
| 2 | Parallel model load (ThreadPool 4) | Accepted — fresh-process 2.095 → 1.461 s (1.43×) |
| 3 | `predict_combined_many` + batched `/internal/evaluate-batch` | Accepted — 30.9× engine, 198× HTTP per-event; exact after float64 fix |
| 4 | Instrumentation (hashes, `batch_wall_ms`, sampler passthroughs) | Accepted — enables equivalence evidence |
| 5 | `benchmark_e2e.py` repair | Accepted — demonstrability (was pre-broken) |
| 6 | Batch-equivalence tests + R4 pin update | Accepted — regression protection |
| 7 | Data-path extraction variants (prealloc, column_stack) | Rejected — slower (22.8 vs 64.3/97.8 ms), RSS equal (~57–58 MB) |
| 8 | Row-level (single-event) threading | Rejected — slower (1.62 ms @16t vs 1.24 ms; overlap 0.245 vs 0.117 s) |
| 9 | Entity-tracker optimisation | Not an opportunity — 0.001 ms warm, constant baseline on production path |
| 10 | Member concurrency without thread scaling | Subsumed into #1 (1.26× alone, not double-counted) |
| 11 | Ensemble re-selection / weight tuning (Track B) | Not attempted by design — scientific experiment, out of scope |

## Accepted changes

Experiments 1–6 (table above) plus the latent BLOCK_INFERENCE score-pairing
fix inside the batch endpoint. Acceptance basis: measurable benefit (brief
§15) with prediction equivalence proven by SHA-256 (ensemble + all three
members) at 500k/5M/50M, unchanged ROC-AUC/PR-AUC/Brier/confusion, and all 84
unique suites green. No scientific contract, model artifact, threshold, gate,
or calibration state changed.

## Rejected changes

Experiments 7–10 (table above; full records with measurements and reasons in
report §8). None changed scientific behaviour — nothing rejected was ever
applied to a running path. Also rejected as *claims*: the discarded 46×
startup figure (page-cache confound; honest value 1.43×).

## Measured improvements

- **Wall time:** 1.900× at 5M (38.565 → 20.294 s) and 1.665× at 50M
  (413.221 → 248.215 s) same-session; 2.015× vs Phase 3's recorded 50M
  (500.15 s); 1.541× at 500k — gain holds across sizes (not startup noise).
- **Throughput:** 142,078 → 246,378 rows/s (5M, +90.0% vs Phase 3 recorded);
  121,001 → 201,438 rows/s (50M same-session).
- **Inference stage:** 2.231× (5M), 1.771× (50M); per-batch wall mean 2.042×
  (5M), 1.680× (50M).
- **API:** `/internal/evaluate-batch` 30.9× engine-level; 6,538 vs 33 eval/s
  per-event HTTP (198×, in-process).
- **Startup:** 1.43× faster fresh-process model load (2.095 → 1.461 s).
- **Memory:** 5M peak RSS −14.1% (1,213.2 → 1,041.7 MB); 50M flat (+1.3%).
- **Equivalence:** prediction + member SHA-256 equal across thread settings at
  every size; 50M ROC-AUC identical to Phase 3
  (0.5731960569465072).

Costs recorded alongside: CPU-seconds +31% (5M) / +46% (50M); store-tail p50
slightly worse at t16 (4.94 → 6.62 ms).

## Negative results

- Data-path extraction variants: no time or RSS benefit → rejected.
- Row-level threading: slower → rejected; single-event latency **not
  improved** (limitation).
- Entity tracker: not on the live hot path → no opportunity.
- Memory at 50M: **flat (+1.3%)** — memory optimization `NOT ESTABLISHED`.
- Discarded invalid 46× startup measurement (page-cache confound).
- Cross-session variance ±10–17% between Phase 3 records and Phase 5 reruns —
  same-session pairs used as primary evidence.
- Preserved from before: mean-fusion ensemble did not outperform the best
  individual model (5M: xgb 0.5816 > ensemble 0.5626; 50M: xgb 0.5742 >
  ensemble 0.5732) — kept, not "fixed".
- Found-but-not-fixed pre-existing defects: entity seeding always fails
  (`m.TransactionFeature` missing from risk_engine models); `security_test`
  rate-limit case is order/state-dependent; `benchmark_e2e.py` was stale
  (repaired — provenance recorded).

## Tests

| Check | Result |
|---|---|
| `regression_suite.py --fast` | 23/23 PASS (~55 s) |
| Full battery `bash .freebuff/p114_battery.sh` (live stack) | **84/84 unique suites PASS**, 0 FAIL, `BATTERY_EXIT=0`, ended 2026-10-06T21:21:56+05:30 (TSV: 87 lines = 3 duplicate appends; 84 unique names, all PASS) |
| `risk_engine_test.py` (incl. new Phase 5 block; final-state rerun) | ALL CHECKS PASSED |
| `security_test.py` / `sql_injection_test.py` (live) | 16/16 / 50/50 |
| `claim_evidence_check` / `eval_record_test` / `review_resolution_check` / `review_package_check` | PASS (22) / 22/22 / PASS (14/14) / PASS (13/13) |
| `check_freeze.py` | rc=1, **78 problems — expected unchanged** pre-freeze state (77 placeholders + missing FREEZE_RECORD); deliberately not "fixed" |
| `benchmark_e2e.py` | runs end-to-end after repair |

Battery history: run 1 (services down) 81/84 — 3× `WinError 10061`
(`security_test`, `sql_injection_test`, `security_ci_gate`); stack started →
all three pass standalone; run 2 killed by client restart at 67/84 with 0
failures; run 3 complete and green.

## Security verification

- No security-relevant surface changed: no new dependencies; no crypto, auth,
  validation, or PII logic touched; NaN/Inf → 422 enforcement intact and
  exercised; BLOCK_INFERENCE behaviour preserved (batch pairing actually
  corrected).
- `bandit -r backend/src --severity-level medium`: exit 0, no findings.
- `secret_hygiene_test.py`: 16/16 (final-state rerun).
- Diff reviewed: no secrets, no `.env`, no large files; score NPZ dumps and
  logs remain gitignored; tracked additions are 1–16 KB JSON/TSV/TXT evidence
  plus two documents.
- Supabase `service_role` rotation remains **`OWNER ACTION REQUIRED —
  PROVIDER CREDENTIAL ROTATION`** — untouched, not claimed complete.

## CI status

- Pending at document time; verified via the GitHub REST API immediately
  after push and recorded in the post-push follow-up commit below.

## Evidence classification

All Phase 5 performance results: **`SELF-TESTED`** — local benchmarks on the
`SYNTHETIC — SCALE/ENGINEERING` workload; never upgraded to `DEMONSTRATED`
because a local run passed. Freeze rc=1/78 expected-state: `DEMONSTRATED`.
Model quality / real-world effectiveness / live-server performance: `NOT
ESTABLISHED`. Track B / calibration / gating alternatives: `NOT APPLICABLE`
(not performed by design). Supabase rotation: `OWNER ACTION REQUIRED`
(external). Full table: report Annex A.

## Unresolved external dependencies

1. Supabase `service_role` rotation — owner-only action (Phase 4A carry-over).
2. Live-server HTTP performance — in-process numbers only; belongs to Phase 6.
3. Institutional/real-world validation — blocked until eligible authorized
   data exists (Track N unchanged).
4. Model research (ensemble re-selection, single-event request-path overhead)
   — open, out of this phase's scope.
5. Pre-existing defects recorded in report §15.8 (entity seeding,
   `security_test` ordering) — deliberately not fixed here.

## Final decision

# `OPTIMIZATION SUCCESSFUL WITH LIMITATIONS`

Material measurable improvements were demonstrated on the preserved baseline
(1.900×/1.665× same-session wall, 30.9×/198× batched inference, 1.43×
startup, −14.1% RSS at 5M) with bit-identical predictions and all 84 unique
suites green — but single-event latency is unimproved, 50M memory is flat,
CPU-seconds rose 31–46%, the HTTP benchmark is in-process, and no
model-level/gating/calibration experiment was performed. Next phase:
**PHASE 6 — PROTOTYPE VALIDATION & PACKAGING**.

---

## Post-push record (appended after CI verification)

*Filled in by the post-push follow-up commit: final SHAs + CI run outcome.*
