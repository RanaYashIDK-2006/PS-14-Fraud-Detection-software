# PS-14 — Phase 5: Prototype Optimization

**Status:** complete — 2026-10-06
**Starting Git SHA:** `ab5a6d0a6baf2c743de9c05fd4277b602478ea60`
**Final decision:** `OPTIMIZATION SUCCESSFUL WITH LIMITATIONS`

> **Phase completion statement.** What measurable improvement did the prototype
> optimization phase actually achieve, under what workload and evidence
> conditions, and what remains unestablished?
>
> Under the synthetic `synthetic_50m` benchmark (chronological partitions,
> batch 250,000 rows, 48-feature `altman_native_v2` contract, model
> `altman_native_v2_20260904_115703`, measured code at git `ab5a6d0`),
> enabling per-member model threading in the existing runner made the offline
> inference path **1.900× faster at 5M rows and 1.665× faster at 50M rows
> within a single session**, with **bit-identical predictions** (equal
> `prediction_sha256` and per-member hashes at every workload size) and
> unchanged metrics (ROC-AUC/PR-AUC/Brier/confusion identical to each other
> and, at 50M, identical to the Phase 3 baseline). The batch inference
> endpoint `/internal/evaluate-batch`, which previously looped per event,
> gained a true batched path: **30.9× at the engine level and 198× per-event
> HTTP throughput (33 → 6,538 eval/s)** versus single-event calls in the
> in-process latency benchmark. Model load at startup is **1.43× faster**
> with a parallel loader. Peak RSS at 5M rows fell **−14.1%**. All **84 unique
> repository test suites pass**, the freeze checker still reports its expected
> 78 pre-freeze problems, and prediction equivalence is asserted in the test
> suite itself.
>
> What remains unestablished: **single-event latency did not improve**
> (threaded single-row inference was measured *slower* and rejected);
> **memory at 50M rows is unchanged (+1.3%)** — every data-path/dtype/streaming
> candidate measured was neutral or slower, so no memory optimization was
> accepted; the wall-clock gain **costs +31…+46% CPU-seconds** (parallel trade,
> not free); the HTTP benchmark runs in-process (TestClient), not against a
> live server fleet; **no model-level (Track B) or gating experiment was
> performed** — the scientific behaviour of the system is unchanged by
> construction. All results are `SELF-TESTED` local benchmarks on synthetic
> data; real-world performance and effectiveness remain `NOT ESTABLISHED`.

---

## 1. Objective

Optimize the existing PS-14 fraud-detection research prototype for measurable
engineering performance and demonstrable prototype usability, while preserving
the scientific evidence already established — answering whether the prototype
can be made materially faster, more memory-efficient, more reliable, and/or
easier to demonstrate **without changing the scientific meaning of the system**.
This is an optimization phase, not a validation/governance phase: no metric was
invented, no negative finding replaced, no evidence standard weakened, and the
Research Plan was not altered.

Track A (engineering) was profiled and measured first; Track B (model-level)
was considered only afterwards and deliberately left unchanged (§12).

## 2. Repository baseline

**Starting state:** git `ab5a6d0a6baf2c743de9c05fd4277b602478ea60`
(`chore(evidence): append hook-driven calibration records (records-only)`), the
Phase 4A head — clean working tree, `HEAD == origin/main`.

**Structure (as actually implemented, not the template in the brief):**

- `backend/src/` — five FastAPI services: `privacy_layer` (ingest, feature
  construction, 24h velocity windows), `risk_engine` (inference, fusion,
  decisioning, velocity-limit enforcement), `verification`,
  `audit_service` (DB-4 hash chain), plus `front_service` (landing/admin,
  port 8000).
- `backend/src/risk_engine/altman_native_ensemble.py` — the production engine:
  XGBoost + LightGBM + CatBoost mean fusion over 48 native features.
- `scripts/large_scale_runner.py`, `scripts/large_scale_cpu_sampler.py` — the
  Phase 3 offline benchmark instrument (reused, not rebuilt).
- `models/production/altman_native/` — production model artifacts.
- `docs/RESEARCH_PLAN.md`, `docs/evaluation/*` — the evidence chain.
- `backend/scripts/*_test.py` (10+ plain-assert suites), `regression_suite.py`,
  `.freebuff/p114_battery.sh` (84-check battery), `.github/workflows` (CI).

**Required preconditions inspected before modifying anything:**
`docs/RESEARCH_PLAN.md`; `LARGE_SCALE_MODEL_TEST_REPORT.md` +
`PHASE_LARGE_SCALE_MODEL_TEST_CLOSEOUT.md`; `STRIX_SECURITY_ASSESSMENT.md` +
`PHASE_STRIX_SECURITY_CLOSEOUT.md`; `CREDENTIAL_ROTATION_AND_PII_MIGRATION.md`
+ `PHASE_CREDENTIAL_ROTATION_CLOSEOUT.md` +
`PHASE4A_CREDENTIAL_ROTATION_EVIDENCE.json`; the production model/feature
manifest; the native 48-feature contract; preprocessing/feature-construction
code; inference, ensemble, calibration and gating code; the existing
profiling/benchmark scripts; and the test/CI configuration. Implementation was
inspected directly rather than relying on documentation — that is how the
`/internal/evaluate-batch` docstring/behaviour mismatch (§6) and the stale
`benchmark_e2e.py` (§6) were found.

**Model/feature identity at baseline (preserved — full manifest in §17):**
`altman_native_v2_20260904_115703`, `xgb_lgb_cb_native`, weights
xgb .34 / lgb .33 / cb .33, `n_features: 48`,
`feature_schema_version: altman_native_v2`, production manifest sha
`e6da1a43…b2485`, threshold `0.7847116291110687`, `calibrator_loaded: false`.
No model file, weight, threshold, gate, scaler, or feature was modified by this
phase — the diff touches only the files listed in §7 and §17.

## 3. Phase 3 baseline

The Phase 3 artifacts were read, never rewritten: `misc/reports/phase3_cpu.json`
and `misc/reports/phase3_inference_50m.json` remain byte-identical to HEAD; all
Phase 5 runs write to new `misc/reports/phase5_*` files. Verified values (they
match the brief's known numbers exactly):

| Quantity (5,000,000 rows) | Phase 3 recorded value |
|---|---|
| Sampler wall | 48.17 s (~103,794 rows/s) |
| Runner inner wall | 35.19 s (142,078 rows/s) |
| CPU mean / p50 / p95 / max | 309.2% / 390.6% / 465.6% / 482.6% |
| Peak RSS | 1,144.1 MB |
| Stage seconds | infer 30.32, validate 0.82, scale 0.68, load 0.28, fuse 0.06, store 0.10 |
| Store-tail latency p50/p95/p99 | 4.13 / 10.23 / 11.40 ms |

Phase 3 50M baseline: **500.15 s, 99,971 rows/s**, infer 464.62 s, ROC-AUC
0.5731960569465072. The Phase 3 latency claims are **store-tail only** by their
original definition; Phase 5 keeps that definition for comparability and
additionally instruments full per-batch wall time (`batch_wall_ms`) so batching
improvements are visible (§10). Nothing in this phase overwrites or reinterprets
the baseline.

## 4. Profiling methodology

- Stage attribution already existed (runner `stage_seconds` + the Phase 3 CPU
  sampler), so per the brief it was **reused rather than rebuilt**: load /
  validate / scale / infer / fuse / store are separated on every run.
- Before editing source, targeted probes measured the candidate areas, each
  recorded to `misc/reports/phase5_*.json` (tracked evidence):
  `phase5_profile.json` (member timing, sequential vs concurrent),
  `phase5_threads.json` (thread scaling, bit-identity across counts),
  `phase5_single_event.json` (row-level latency and threading),
  `phase5_memload.json` (data-extraction variants: time + transient RSS),
  `phase5_model_load.json` (fresh-process sequential vs parallel load),
  `phase5_equivalence.json` (batched vs looped inference, stage-wise).
- CPU/RSS sampling wraps the runner process via the existing
  `scripts/large_scale_cpu_sampler.py` (extended with `--threads`/`--run-out`
  passthroughs so Phase 3's committed outputs were never clobbered), 0.5 s
  interval.
- Equivalence was checked the strong way: content hashes of the full score
  vector and of each member's scores, not just summary metrics (§11).
- All measurements are on the synthetic `synthetic_50m` benchmark —
  `SYNTHETIC — SCALE/ENGINEERING` work (§19 boundary).

## 5. Bottleneck findings

Measured attribution (no claimed bottleneck without measurement):

1. **Model inference dominates:** infer is 30.32 s of the 35.19 s Phase 3 inner
   wall (**86%**); validate+scale+load+fuse+store together ≈ 1.94 s. The
   inference stage is ~99% the three member models.
2. **Member costs (1,153,615-row slice):** sequential members 6.19 s total —
   LightGBM 4.58 s (dominant), XGBoost 1.50 s, CatBoost 0.12 s. Running the
   three members concurrently (threads=4 each) gives 4.90 s = **1.26×**, all
   outputs bit-identical.
3. **Thread scaling is strong at batch level** (same slice, bit-identical at
   every count): xgb 5.285 s (t1) → 0.698 s (t16); lgb 17.737 → 2.096 s;
   cb 0.401 → 0.082 s. Config comparison: sequential-4t 6.556 s →
   concurrent-8t 3.355 s (**1.95×**) → concurrent-16t 2.970 s (**2.21×**).
4. **The batch endpoint was not batched:** `/internal/evaluate-batch` looped
   `predict_combined` per event — a 30.9× engineering miss masked by its own
   docstring claiming batched inference (§6).
5. **Not bottlenecks (measured):** data extraction (~23 ms/batch and already
   the fastest variant tested), feature-row mapping (`map_raw_to_native`
   ≈ 0.012 ms warm), entity fraud-rate lookups (`get_rates` 0.001 ms warm;
   production call path supplies no user/merchant/city ids so the features are
   the constant baseline 0.001 anyway), model load at steady state
   (0.3–3.7 s per process, one-time).
6. **Single-row inference is overhead-bound:** ~1.24–1.31 ms/event at 1–4
   threads, and *worse* at 16 threads (1.62 ms) — row-level parallelism cannot
   pay off (§8).

## 6. Engineering optimizations attempted

Complete inventory (accepted → §7, rejected/measured-negative → §8):

| # | Attempt | Track | Outcome |
|---|---|---|---|
| 1 | Per-member model threading in the runner (`--threads`) | CPU/throughput | **ACCEPTED** (§7.1) |
| 2 | Parallel model load at startup | Startup overhead | **ACCEPTED** (§7.2) |
| 3 | True batched inference for `/internal/evaluate-batch` (`predict_combined_many`) | Inference/ensemble | **ACCEPTED** (§7.3) |
| 4 | Instrumentation: `--scores-out`, prediction/member SHA-256, `batch_wall_ms`, sampler `--threads`/`--run-out` | Measurement enablement | **ACCEPTED** (§7.5) |
| 5 | Repair of `benchmark_e2e.py` (3 pre-existing breaks + 2 of its own) | Demonstrability | **ACCEPTED** (§7.4) |
| 6 | Batch-equivalence block in `risk_engine_test.py`; R4 sha-pin update | Regression protection | **ACCEPTED** (§7.6) |
| 7 | Data-path extraction variants (prealloc, column_stack vs pandas) | Memory/data path | **REJECTED** — no gain (§8.1) |
| 8 | Row-level (single-event) threading | Inference | **REJECTED** — slower (§8.2) |
| 9 | Entity-tracker lookup optimisation | Feature construction | **NOT AN OPPORTUNITY** (§8.3) |
| 10 | Concurrency without thread scaling | Ensemble execution | **SUBSUMED** — 1.26× alone (§8.4) |
| 11 | Ensemble re-selection / weight tuning | Track B — model | **NOT ATTEMPTED** by design (§12) |

## 7. Engineering optimizations accepted

### 7.1 Runner thread control (`scripts/large_scale_runner.py`, +47 lines)

`--threads N` (default `0` = Phase 3 baseline behaviour) sets `n_jobs` on
XGBoost/LightGBM and `thread_count` on CatBoost (fitted CatBoost models reject
`set_params`; the value must be passed at predict time). Default 0 is
deliberate: single-event serving must not accidentally enable row threading
(§8.2).

### 7.2 Parallel model load (same file)

ThreadPool (4 workers) loading the three members: fresh-process load
**2.095 s → 1.461 s = 1.43×** (`phase5_model_load.json`, three repeats each).
An earlier in-process 46× figure was a page-cache confound, was discarded, and
is not claimed anywhere.

### 7.3 True batched inference (`backend/src/risk_engine/main.py`,
`altman_native_ensemble.py`)

- New `predict_combined_many(rows)` (+40 lines, pure addition after
  `predict_many`): one vectorised float32 matrix, one scaler transform, one
  `predict_proba` per member, vectorised weighted aggregation with the same
  per-row rounding as `predict()`.
- **Exactness fix found by measurement:** `xgb.predict_proba` returns
  **float32**, and under NumPy weak-scalar promotion `0.34 * float32_array`
  stays float32 — the first weighted term was rounded before the float64 terms
  joined, diverging from `predict()` by 1.49e-9 (caught by the new test, not
  tolerated). An exact float64 widening cast of *every* member output before
  weighting makes the batched path **bit-identical** (`max_d = 0.0`).
- `/internal/evaluate-batch` now preallocates score arrays and calls the
  batched path (exception fallback preserves the original loop behaviour).
- **Latent bug fixed en route:** the old loop inferred over *all* feature
  dicts — including events blocked by `enforce_before_inference`
  (BLOCK_INFERENCE) — then `zip`ped scores onto the allowed list,
  mis-pairing scores whenever a block occurred mid-batch. The loop now
  iterates `feature_dicts_allowed` (identical behaviour when no blocks occur;
  correct pairing when they do). Single-event `/evaluate` was already correct
  and is untouched.
- Docstring now truthful. Engine-level result: 200 events 0.2713 s looped →
  0.0088 s batched = **30.91×**, scores bit-identical.

### 7.4 Demonstrability: repaired `backend/scripts/benchmark_e2e.py`

Pre-existing stale in three ways (missing required `fraud_id`; batch payload
wrapped as `{"events": …}` against a bare-list request model; stale
`FeatureVector` field names — it would 422 before this phase's changes), plus
two of its own (bench ID violating the `F[A-Z2-9]{15}` pattern; a section
timing 422-rejected requests without checking status). All five fixed; the
benchmark now runs end-to-end (§10).

### 7.5 Measurement enablement (`scripts/large_scale_runner.py`,
`scripts/large_scale_cpu_sampler.py`)

`--scores-out` NPZ dump (kept in gitignored scratch — only hashes are
committed), `prediction_sha256` + per-member `member_sha256`, full per-batch
wall statistics, `threads` in result and manifest, sampler passthroughs
(`--threads`, `--run-out`) so Phase 5 outputs land in new files. Acceptance
rationale: equivalence evidence (§11) is impossible without content hashes.

### 7.6 Regression protection (`backend/scripts/risk_engine_test.py`,
`review_package_check.py`)

New Phase 5 batch-equivalence block: `predict_combined_many` exists; its
scores/uncertainties match the per-event loop **exactly** (max_d 0.0);
outputs finite; 32 events through single `/evaluate` vs `/internal/evaluate-batch`
produce identical score/band/decision/codes/ml_weighted/degraded fields. Check
R4 pins the ensemble file's sha256 — the sanctioned addition changed it, so
the pin was updated after confirming the diff is a **pure 40-line addition**
(one hunk; `map_raw_to_native`, `predict`, `predict_many` byte-identical to
HEAD); review-package check passes 13/13 afterwards.

## 8. Engineering optimizations rejected

Each record: what was attempted → why it looked promising → measurement →
result → reason for rejection → did it change scientific behaviour? (none did —
nothing in this table was ever applied to a running path).

### 8.1 Data-path extraction variants — REJECTED

- **Attempted:** three strategies for building the float32 inference matrix
  (pandas path as-is; preallocated row fill; `np.column_stack`).
- **Why promising:** preallocation/column_stack are the classic copy-avoidance
  wins for dataframe→matrix conversion in the largest non-inference stage.
- **Measurement:** `phase5_memload.json`, identical inputs, bit-identical
  outputs: pandas **22.8 ms/batch**, prealloc 64.3 ms, column_stack 97.8 ms;
  transient RSS ~57–58 MB **for all three**.
- **Result:** every alternative was *slower* than the existing path and none
  reduced RSS.
- **Reason:** no measurable benefit (§15 acceptance rule) — the candidate with
  no gain is rejected/documented as neutral.
- **Scientific behaviour:** unchanged (nothing applied).

### 8.2 Row-level (single-event) threading — REJECTED

- **Attempted:** `--threads` applied to single-row predict; a threadpool
  overlapping successive per-call inferences.
- **Why promising:** the same knobs gave 2.2× at batch level (§5.3).
- **Measurement:** `phase5_single_event.json`/`phase5_threads.json` —
  ~1.24–1.31 ms/event at 1–4 threads, **1.62 ms at 16 threads** (slower);
  overlap wrapper 0.117 s sequential vs **0.245 s threaded**.
- **Result:** threading *costs* more than it saves on 1-row matrices.
- **Reason:** would be a regression for the live single-event path — exactly
  the brief's "improvement that changes nothing measurable must be rejected".
- **Scientific behaviour:** unchanged (nothing applied); single-event latency
  remains unimproved and is recorded as a limitation (§19).

### 8.3 Entity-tracker lookup optimisation — NOT AN OPPORTUNITY

- **Attempted:** profiling the fraud-rate entity tracker as a feature-
  construction candidate.
- **Why promising:** it does I/O-shaped work (Redis probe) on the event path.
- **Measurement:** `get_rates` 0.001 ms warm (in-process map; Redis absent,
  `redis_available()` false), cold first call 1.35 ms, one-time 96 ms import,
  `map_with_tracker` 5.2 ms first call only — and
  `misc/reports/feature_integrity_audit.json` records the production call path
  supplies **no user/merchant/city ids**, so the features are the constant
  baseline 0.001 regardless.
- **Result:** not on any live hot path.
- **Reason:** nothing to optimise; an edit here would be change for its own
  sake.
- **Scientific behaviour:** unchanged.

### 8.4 Member concurrency without thread scaling — SUBSUMED

- **Attempted:** running members concurrently at fixed 4 threads.
- **Measurement:** 6.19 → 4.90 s (**1.26×**) alone, but 2.21× combined with
  thread scaling (§5.3).
- **Reason:** kept as part of attempt #1 rather than claimed separately —
  reported here so the isolated 1.26× is not double-counted.

## 9. Memory results

| Workload | Baseline peak RSS | Optimized peak RSS | Δ | Predictions equivalent? |
|---|---|---|---|---|
| 5,000,000 rows (sampler RSS) | 1,213.2 MB (t0) / 1,144.1 MB (Phase 3) | **1,041.7 MB** | **−14.1%** / −8.9% | yes — equal SHA |
| 5,000,000 rows (runner `ru_maxrss`) | 997.0 MB | 891.8 MB | −10.6% | yes — equal SHA |
| 50,000,000 rows (sampler RSS) | 3,816.9 MB | 3,865.2 MB | **+1.3% (flat)** | yes — equal SHA |

- The 5M reduction comes from threading changing the intermediate-copy
  pattern; no lossy transform, dtype downcast of stored data, or sampling was
  used to beautify a memory number.
- **At 50M, memory is flat** — no material memory optimisation was achieved at
  scale, and every candidate that might have done so was rejected on
  measurement (§8.1, §8.2). Streaming/column-projection/reuse-of-buffers
  candidates were tested in the form of the extraction variants (§8.1) and
  showed neither time nor RSS benefit; larger structural streaming changes
  were not attempted because the measured data path was not a bottleneck
  (§5.5) and the brief forbids changes without measurable benefit.
- Row count, workload, and equivalence are recorded per row of the table
  above; evidence: `phase5_cpu_t0/t16.json`, `phase5_cpu_50m_t0/t16.json`,
  run JSONs.

## 10. CPU / throughput results

**Multi-size wall-clock (same-session pairs, one variable apart):**

| Workload | t0 wall | t16 wall | Speedup | t0 rows/s | t16 rows/s |
|---|---:|---:|---:|---:|---:|
| 500,000 (smoke) | 6.496 s | 4.215 s | **1.541×** | 76,975 | 118,613 |
| 5,000,000 | 38.565 s | **20.294 s** | **1.900×** | 129,651 | 246,378 |
| 50,000,000 | 413.221 s | **248.215 s** | **1.665×** | 121,001 | 201,438 |

- Infer stage: 5M 34.206 → 15.329 s (**2.231×**); 50M 373.448 → 210.824 s
  (**1.771×**). Full per-batch wall mean: 5M 1,102.2 → 539.7 ms (**2.042×**);
  50M 1,208.7 → 719.5 ms (1.680×).
- The gain holds across two orders of magnitude of workload → **not a fixed
  startup effect**. 25M was not run: three sizes satisfy the multi-size
  requirement; a fourth adds session noise, not information. No 25M/50M
  figure was fabricated anywhere.
- Against Phase 3's *recorded* values (cross-session context only — the
  environment is not locked): 5M t16 = 1.734× inner / 1.521× sampled;
  50M t16 = **2.015×**. The Phase 5 t0 rerun is 9.6% slower than Phase 3's
  recorded inner wall (38.6 vs 35.2 s) — session variance; that is precisely
  why the same-session pairs are the primary evidence.

**CPU:**

| | 5M t0 | 5M t16 | 50M t0 | 50M t16 |
|---|---:|---:|---:|---:|
| CPU mean | 316.0% | 784.6% | 327.3% | 792.7% |
| CPU p50 | 393.7% | 1,119.5% | 393.7% | 1,047.7% |
| CPU p95 | 443.8% | 1,504.9% | 439.1% | 1,359.2% |
| CPU max | 471.9% | 1,538.3% | 699.5% | 1,444.2% |
| CPU-seconds (wall × mean) | 121.9 s·core | 159.2 (+30.6%) | 1,352 s·core | 1,968 (+45.5%) |

The baseline is capped near 4 cores by its 4-thread defaults; t16 averages
~8 cores of 16 logical. **The speedup is partly purchased with CPU: +31%
(5M) / +46% (50M) CPU-seconds for 1.90× / 1.67× wall time** — a
throughput-for-cores trade, not an efficiency win; on a fully saturated host
t0 could win (§19). Peak CPU stays within 16 logical cores (max 1,538%).

**Latency/demonstrability** (`phase5_e2e_bench.txt`, repaired
`benchmark_e2e.py`, in-process TestClient, quiet machine):

| Path | Result |
|---|---|
| Privacy ingest | p50 27.2 ms, 37 ingest/s |
| Single `/evaluate` | p50 29.9 ms, **33 eval/s** |
| `/internal/evaluate-batch` (50 events) | total p50 7.7 ms, 0.2 ms/event, **6,538 eval/s** |
| Full pipeline (Privacy → Risk) | p50 56.2 ms end-to-end |
| 10 concurrent singles | 291 eval/s |

Batch vs single: **198× per-event throughput**. An earlier run under 50M
CPU contention measured 979 vs 34 eval/s (28.8×) — both recorded; the
contended run shows the number degrades under load rather than being asserted.
The single-event ~30 ms is dominated by validation/persistence/audit overhead,
not inference (~1.2 ms) — which is why engine-level batching, not threading,
is the demonstrability win. Store-tail latency (Phase 3 definition) is
*slightly worse* at t16 (5M p50 4.94 → 6.62 ms) — thread wake/sync on tiny
post-store windows while the same batch's full wall improved 2.04×; recorded,
not hidden.

## 11. Chunking / streaming equivalence (§6 of the brief)

The brief's caution (the historical per-chunk entity-factorization failure)
was taken seriously:

- **No chunking/batching semantics changed.** Partition order (chronological),
  batch size (250,000), row order, feature construction, and the stateless
  per-batch runner design are untouched — the diff is confined to model-thread
  configuration, a batched inference call inside the *same* request scope,
  load ordering, and instrumentation. No global-vs-local mapping was
  introduced, so the failure mode class cannot arise from this phase.
- **Entity IDs / categorical mappings / historical features / temporal
  features / labels / feature values:** produced by unchanged code and
  validated empirically — non-inference stage times are session-noise-equal;
  `missing_feature_partitions: 0`, `nonfinite_values: 0`.
- **Predictions:** `prediction_sha256` **and** each member's hash are equal
  across thread settings at every size:
  - 5M: `468cfe8d600d0b89e985fd4ee6c42d483561e33dc0d8f315e7cf9d818d092e81`
    (xgb `3c0d3631…`, lgb `f6d3bdf9…`, cb `2563c89e…`)
  - 50M: `ee1e7050f07363382448dcf5698d68f34e4f6d59879406eaf5076124e56da2e5`
    (xgb `6e607f98…`, lgb `4fbf1181…`, cb `66ea37e4…`)
  - 500k: `90a49b0bc33d1d1d6dbcb79cc9cd1c9976e46c198a5c6a6d8a36ec2c37697b94`
- **Aggregate statistics:** ROC-AUC, PR-AUC, Brier, precision/recall/F1, alert
  rate, and the confusion matrix are pairwise identical between t0 and t16 at
  every size; 50M ROC-AUC equals the Phase 3 recorded value exactly
  (0.5731960569465072).
- **Batched vs looped inference:** asserted bit-exact in the test suite,
  including the float32-promotion fix found *because* the equivalence test
  refused a 1.49e-9 divergence rather than tolerating it. Single-vs-batch
  endpoint identity passes for score, band, decision, decision codes,
  `ml_weighted`, and `degraded` across 32 events.

## 12. Model-level experiments (if any)

**None performed — by design.** After Track A was measured (§5–§10), no
evidence pointed at model configuration as the bottleneck (aggregation is
microseconds; member inference is the cost), and changing weights/members/
thresholds would be a *scientific experiment* (brief §9), not an optimization —
doing it here to move a number would violate the brief.

The repository's preserved negative finding was re-observed in the Phase 5
runs' `member_roc_auc` and is **kept, not "fixed"**:

- 5M slice: xgb 0.5816 > ensemble 0.5626 (lgb 0.5688, cb 0.5431)
- 50M: xgb 0.5742 > ensemble 0.5732 (lgb 0.5664, cb 0.5585)

"The existing mean-fusion ensemble did not automatically outperform the
strongest individual model" stands. Alternative aggregation, computational
cost comparison of alternatives, and prediction agreement vs alternatives:
`NOT APPLICABLE` — no alternative was implemented. Ensemble re-selection
remains open as model research (§19).

## 13. Calibration results (if any)

**None performed.** Calibration was not touched: `src/risk_engine/calibration.py`,
the Platt artifacts, and `calibrator_loaded: false` for the production native
engine are as Phase 3 left them; `models/artifacts/calibrator.joblib` remains
for the mapped-ensemble path only. The IBM Platt-calibration inversion issue
is untouched and unregressed. Ranking/probability semantics: unchanged by
construction; empirically, Brier is byte-equal between t0/t16 (5M 0.00141;
50M 0.001248) and ROC-AUC/PR-AUC are identical (§11) — no calibration
experiment was run because no engineering finding warranted one (brief §10).

## 14. Gating results (if any)

**No gating change was attempted or applied.** Gating surfaces — velocity
limits (`src/risk_engine/limits.py`), drift fallback (`drift_detector` +
`should_fallback`), `enforce_before_inference` BLOCK_INFERENCE, threshold
0.7847116291110687, `rules.yaml` — are untouched by this phase's diff.
Consequently there is no coverage/rejection-rate/false-positive movement to
report: optimization cannot "win" by rejecting difficult cases because
**nothing in the decision path changed** (brief §11).

One gating *observation* (behaviour, not change): during the contended
latency benchmark the drift guardrail engaged mid-run — constant synthetic
bench vectors are out-of-distribution, so PSI flagged CRITICAL and the engine
correctly fell back to rules-only (the clean-run evidence file
`phase5_e2e_bench.txt` shows no engagement). Guardrail fallback worked as
designed; latency timings are unaffected because they time the request, not
the score source. Class B or real-world gating effectiveness is **`NOT
ESTABLISHED`** — unchanged from prior phases; synthetic data here is
engineering evidence only, never institutional validation (brief §12).

## 15. Negative findings

Explicitly preserved (nothing deleted because it makes the report less
impressive):

1. **Rejected optimizations** — extraction variants (no gain, no RSS win),
   row-level threading (slower), entity-tracker (not an opportunity),
   concurrency-only (subsumed): full records in §8. None changed scientific
   behaviour; none were applied.
2. **Memory at scale did not improve:** 50M peak RSS **+1.3%** (flat) — the
   memory-optimization goal of the brief is `NOT ESTABLISHED`, not
   substituted by the 5M −14.1%.
3. **Single-event latency did not improve** and is bounded by
   validation/persistence/audit overhead (~30 ms in-process), not inference.
4. **CPU-seconds increased** +31% (5M) / +46% (50M) — the wall win is partly
   a core-count trade (§10).
5. **Store-tail p50 latency slightly worse at t16** (5M 4.94 → 6.62 ms) —
   recorded alongside the 2.04× full-batch improvement.
6. **Discarded invalid measurement:** an early in-process "46× faster model
   load" was a page-cache confound; the honest fresh-process figure is 1.43×
   (§7.2). The discarded number appears nowhere in the deliverables.
7. **Cross-session baseline variance:** Phase 5 t0 at 5M was 9.6% slower than
   Phase 3's recorded inner wall (38.6 vs 35.2 s); at 50M, t0 was 17% *faster*
   than Phase 3's recorded wall (413 vs 500 s). The environment is unlocked —
   same-session pairs are therefore the primary evidence, cross-session ratios
   are labelled as context (§10).
8. **Found-but-not-fixed pre-existing defects** (recorded, out of Phase 5
   scope, none security-critical):
   - `_seed_entity_tracker` references `m.TransactionFeature`, which lives in
     privacy-layer models, not `src.risk_engine.models` — entity seeding
     always fails at startup ("seeding skipped"). Benign today because the
     production path supplies no entity ids (§8.3), but it is dead code.
   - `security_test`'s rate-limit case is order/state-dependent (per-IP 60 s
     window + accumulated failures): against a freshly started service it once
     reported "never blocked" while the limiter demonstrably worked (manual
     burst → 429). It passes in battery order and in the final reruns; this
     phase touches no auth/limiting code.
   - `benchmark_e2e.py` was already broken before this phase (found and
     repaired — §7.4, listed here for provenance).
9. **Battery process history:** run 1 was 81/84 with 3 connection-refused
   failures (services down); run 2 was killed by a client restart at 67/84
   with 0 failures; run 3 is the complete green record (§16).

## 16. Security and regression results

**Security boundary preserved (brief §13):** no new dependencies; no crypto,
auth, permission, validation, or PII logic altered — the diff is computational
(`main.py` batch path, ensemble addition, scripts). `enforce_before_inference`
BLOCK_INFERENCE behaviour is intact and its batch pairing actually corrected
(§7.3); NaN/Inf → 422 enforcement untouched and exercised by the test suite;
no credentials/tokens/secrets in code, artifacts, config, docs, or logs (the
benchmark's dev-only placeholder env defaults were pre-existing and are not
production credentials).

| Verification | Command | Result |
|---|---|---|
| Fast regression set | `regression_suite.py --fast` | **23/23 PASS** (~55 s) |
| Full battery (live stack) | `bash .freebuff/p114_battery.sh` → `misc/reports/phase5_battery_results.tsv` | **84/84 unique suites PASS**, 0 FAIL, `BATTERY_EXIT=0`, ended 2026-10-06T21:21:56+05:30 (87 TSV lines = 3 duplicate appends) |
| Risk-engine suite (incl. new Phase 5 block) | `risk_engine_test.py` | **ALL CHECKS PASSED** (final state rerun) |
| Security tests (live) | `security_test.py` / `sql_injection_test.py` | **16/16 / 50/50** |
| Bandit | `bandit -r backend/src --severity-level medium` | **exit 0**, no findings |
| Secret hygiene | `secret_hygiene_test.py` | **16/16** (final state rerun) |
| Claim-evidence / eval-record / review-resolution / review-package | checkers | **PASS (22) / 22/22 / PASS (14/14) / PASS (13/13)** |
| Freeze checker | `check_freeze.py` | **rc=1, 78 problems** — expected unchanged pre-freeze state (77 placeholders + missing FREEZE_RECORD); deliberately not "fixed" |
| PII rotation test | in fast suite (`pii_key_rotation`) | PASS |
| Latency benchmark | `benchmark_e2e.py` | runs end-to-end after repair (§10) |

`Supabase service_role` rotation remains exactly
**`OWNER ACTION REQUIRED — PROVIDER CREDENTIAL ROTATION`** — this phase did
not touch it and does not claim completion (brief §13). CI status is recorded
in `PHASE_OPTIMIZATION_CLOSEOUT.md` after the push (post-push follow-up
commit, Phase 4A pattern).

## 17. Reproducibility information

**Measured code:** git `ab5a6d0a6baf2c743de9c05fd4277b602478ea60` (every run's
`experiment_manifest.git_sha`; the Phase 5 code changes post-date all runs, so
the measured baseline/optimized runs are both on the recorded SHA — the only
in-run variable is `threads`).

**Environment (from manifests):** Windows-11 (10.0.26200-SP0), Python
3.12.10, 12th Gen Intel Core i7-12650H (16 logical cores), 15.6 GB RAM;
CPU sampler 0.5 s interval; no GPU.

**Benchmark identity:** `benchmark_id: synthetic_50m`, combined sha
`c2db7142e24dd8d9c2806ef8a1ec0c23e62cbf2d8fcc4b4b614396922b47edf1`,
schema hash `fb1d8e6a…d96`, generator 1.0.0, seed **20261005**, 17 chronological
partitions (5M = 5,000,000 rows, 6,028 fraud, prevalence 0.0012056), batch
250,000, workers 1, split order chronological (year_month ascending).

**Model/artifact identity:** `altman_native_v2_20260904_115703`,
`xgb_lgb_cb_native`, weights .34/.33/.33, 48 features,
`feature_schema_version: altman_native_v2`, production manifest sha
`e6da1a433ec618576aa12208190d35df3b53e9349b1bb6e54c16465e324b2485`,
feature list sha `657459ac9526c77f6f9006b7ba18a41338a54937e12177cd16eaf944d59aed8f`,
scaler `b98fadf3…8611`, xgb `a1cdebd1…0bbc`, threshold
0.7847116291110687, `calibrator_loaded: false`.

**Benchmark commands:**

```
python scripts/large_scale_cpu_sampler.py --rows 5000000  --run-out misc/reports/phase5_t0_run.json        --threads 0
python scripts/large_scale_cpu_sampler.py --rows 5000000  --run-out misc/reports/phase5_t16_run.json       --threads 16
python scripts/large_scale_cpu_sampler.py --rows 50000000 --run-out misc/reports/phase5_50m_t0_run.json    --threads 0
python scripts/large_scale_cpu_sampler.py --rows 50000000 --run-out misc/reports/phase5_50m_t16_run.json   --threads 16
python scripts/large_scale_cpu_sampler.py --rows 500000   --run-out misc/reports/phase5_smoke_t{0,16}.json --threads {0,16}
python backend/scripts/benchmark_e2e.py > misc/reports/phase5_e2e_bench.txt     # from backend/, PYTHONPATH=.
python backend/scripts/regression_suite.py --fast
bash .freebuff/p114_battery.sh
```

**Tracked evidence files (this phase):** `misc/reports/phase5_t0_run.json`,
`phase5_t16_run.json`, `phase5_50m_t0_run.json`, `phase5_50m_t16_run.json`,
`phase5_smoke_t0/t16.json` (runs), `phase5_cpu_t0/t16.json`,
`phase5_cpu_50m_t0/t16.json` (CPU/RSS), `phase5_profile.json`,
`phase5_threads.json`, `phase5_single_event.json`, `phase5_memload.json`,
`phase5_model_load.json`, `phase5_equivalence.json` (profiling),
`phase5_e2e_bench.txt` (HTTP benchmark), `phase5_battery_results.tsv`
(84-suite record). NPZ score dumps and probe scripts stay in gitignored
scratch — only hashes are committed. Run windows: 5M t0 14:38:47–14:39:37Z,
t16 14:39:39–14:40:10Z (back-to-back); 50M t16 16:08:16–16:14:32Z,
2026-10-06.

## 18. Baseline-vs-optimized table

Primary workload — 5,000,000 rows, `synthetic_50m`, same session, one variable
apart. Model metrics are omitted because no model-level experiment was
performed (§12). Unavailable values are not populated.

| Metric | Phase 3 baseline | Optimized | Change | Workload | Evidence |
|---|---:|---:|---:|---|---|
| Runtime (runner wall) | 35.19 s | 20.294 s | **−42.3% (1.900×)** | 5M synthetic_50m | `phase5_t16_run.json` |
| Rows/sec | 142,078 | 246,378 | **+90.0%** | 5M synthetic_50m | `phase5_t16_run.json` |
| Peak RSS | 1,144.1 MB | 1,041.7 MB | **−8.9%** | 5M synthetic_50m | `phase5_cpu_t16.json` |
| Mean CPU | 309.2% | 784.6% | +154% (cores traded for wall) | 5M synthetic_50m | `phase5_cpu_t16.json` |
| p95 CPU | 465.6% | 1,504.9% | +223% (within 16 logical cores) | 5M synthetic_50m | `phase5_cpu_t16.json` |
| Startup time (fresh-process model load) | 2.095 s (sequential) | 1.461 s (parallel) | **−30.3% (1.43×)** | fresh process, 3 repeats | `phase5_model_load.json` |
| Prediction equivalence | not recorded in Phase 3 | equal SHA-256 t0 vs t16 (ensemble + all 3 members); metrics equal to Phase 3 | **bit-identical** | 500k / 5M / 50M | `phase5_*_run.json` |

Companion workload — 50,000,000 rows (multi-size check, same session):

| Metric | Phase 3 baseline | Optimized | Change | Workload | Evidence |
|---|---:|---:|---:|---|---|
| Runtime (runner wall) | 500.15 s | 248.215 s | **−50.4% (2.015× vs recorded; 1.665× same-session)** | 50M synthetic_50m | `phase5_50m_t16_run.json` |
| Rows/sec | 99,971 | 201,438 | **+101.5%** | 50M synthetic_50m | `phase5_50m_t16_run.json` |
| Peak RSS | not recorded in Phase 3 | 3,865.2 MB | flat vs same-session t0 (3,816.9 MB, +1.3%) | 50M synthetic_50m | `phase5_cpu_50m_t16.json` |
| Mean CPU | not recorded in Phase 3 | 792.7% | — (same-session t0: 327.3%) | 50M synthetic_50m | `phase5_cpu_50m_t16.json` |
| p95 CPU | not recorded in Phase 3 | 1,359.2% | — (same-session t0: 439.1%) | 50M synthetic_50m | `phase5_cpu_50m_t16.json` |
| Prediction equivalence | ROC-AUC 0.5731960569465072 recorded | equal SHA t0 vs t16; ROC-AUC identical to Phase 3 | **bit-identical** | 50M synthetic_50m | `phase5_50m_*_run.json` |

Startup time at 50M: same-session t0/t16 load stages 3.596 / 3.688 s — noise;
the measured 1.43× startup gain is workload-independent (fresh-process probe)
and is reported once, above. Additional API-level improvement (not part of the
runner baseline): `/internal/evaluate-batch` 30.9× engine-level /
198× per-event HTTP vs single-event (§10), `phase5_e2e_bench.txt`.

## 19. Limitations

1. **Single-event latency unchanged** — row threading measured slower and
   rejected (§8.2); the ~30 ms request path is dominated by
   validation/persistence/audit overhead, untackled here.
2. **Memory at 50M unchanged (+1.3%)** — no accepted memory optimization
   exists to claim at scale (§9).
3. **CPU-seconds +31…+46%** for the wall win; a saturated-host comparison was
   not run, so t0 could win there (§10).
4. **HTTP benchmark is in-process (TestClient)** — live-server, multi-instance
   latency/throughput is `NOT ESTABLISHED`; belongs to Phase 6 packaging.
5. **Synthetic workload only** — `SYNTHETIC — SCALE/ENGINEERING`; real-world,
   institutional, effectiveness and generalization claims remain
   `NOT ESTABLISHED` (Track N/native institutional validation stays blocked
   until eligible authorized data exists).
6. **Cross-session baseline variance** (±10–17%) — same-session pairs are the
   evidence; recorded Phase 3 comparisons are context, not proof.
7. **Track B / calibration / gating experiments not performed** — by design
   (§12–§14); ensemble re-selection and any model-level work remain open as
   model research.
8. **Test-order sensitivity** in `security_test`'s rate-limit case and the
   always-failing entity seeding — pre-existing, recorded, not fixed (§15.8).
9. **Supabase credential rotation** still open — external, owner-only (§16).

## 20. Final decision

**Chosen outcome:** measurable, reproducible improvements were demonstrated on
the preserved baseline with all regression checks green (§16) — 1.900×/1.665×
same-session wall (1.541× at 500k → not a startup effect), 30.9×/198× batched
inference, 1.43× startup, −14.1% RSS at 5M, bit-identical predictions at every
size, Phase 3 baseline and negative findings intact — **but important
limitations remain**: single-event latency not improved (row threading is a
measured regression), memory at 50M flat, CPU-seconds up 31–46%, HTTP numbers
in-process, synthetic-only workload, no model-level/gating/calibration
experiment, external rotation open.

# `OPTIMIZATION SUCCESSFUL WITH LIMITATIONS`

The more positive `OPTIMIZATION SUCCESSFUL` is not used precisely because
those limitations are real and load-bearing; `NO MATERIAL IMPROVEMENT`,
`OPTIMIZATION REJECTED`, `OPTIMIZATION BLOCKED` and `REQUIRES FURTHER MODEL
RESEARCH` are excluded by the measurements above (improvements are material,
nothing regressed, no environment blocked the work, and the remaining costs
are engineering trades, not model-methodology walls).

**Stop condition honoured:** no institutional validation begun, no real-world
effectiveness claimed, no dataset fabricated, no evaluation rule changed to
manufacture a better metric, no new security/governance phase added (no
concrete security regression was found — the two defects recorded are
pre-existing and non-security). The intended next phase is **PHASE 6 —
PROTOTYPE VALIDATION & PACKAGING**.

---

## Annex A. Evidence classification

| Result | Classification |
|---|---|
| Thread speedups (500k/5M/50M), bit-identical predictions & member hashes | `SELF-TESTED` — local benchmark, `SYNTHETIC — SCALE/ENGINEERING` |
| Batched inference 30.9× engine / 198× HTTP | `SELF-TESTED` (in-process TestClient; live fleet `NOT ESTABLISHED`) |
| Startup 1.43×; 5M RSS −14.1% | `SELF-TESTED` |
| Rejected candidates (extraction, row threading, entity tracker) | `SELF-TESTED` (negative results) |
| Full battery 84/84, fast set 23/23, checkers, bandit, hygiene | `SELF-TESTED` |
| Freeze checker rc=1 / 78 problems | `DEMONSTRATED` (expected-state record) |
| Model quality / real-world effectiveness | `NOT ESTABLISHED` (unchanged; no model change) |
| Memory improvement at 50M | `NOT ESTABLISHED` (measured flat, +1.3%) |
| Track B / calibration / gating alternatives | `NOT APPLICABLE` (not performed by design) |
| Supabase credential rotation | `OWNER ACTION REQUIRED` (external, from Phase 4A) |

Nothing local is upgraded to `DEMONSTRATED` merely because it passed a local
benchmark.
