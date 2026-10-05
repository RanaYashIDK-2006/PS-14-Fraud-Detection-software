# Phase 3 Closeout — Large-Scale Model Test on the Validated 50M Benchmark

**Final status: `LARGE-SCALE BASELINE ESTABLISHED WITH LIMITATIONS`**

**Date:** 2026-10-05
**Repository:** PS-14-Fraud-Detection-software
**Git SHA:** `5a5ff55cc03df73318fdf31a16e3665f159a4720` (140 commits, 0 tags)
**Primary deliverable:** `docs/evaluation/LARGE_SCALE_MODEL_TEST_REPORT.md` (22 sections)

> Every model-performance number in this phase is labelled **`SYNTHETIC 50M — SCALE EXPERIMENT`**.
> This phase establishes **computational scalability only**. Real-world fraud-detection
> effectiveness, institutional generalisation, production fraud-loss reduction, regulatory
> readiness and independent external validation all remain **`NOT ESTABLISHED`**.

---

## 1. Exact benchmark identity

The exact artifact validated in the prior phase was used. It was **not regenerated and not
modified** — verified before and after the phase.

| field | value |
|---|---|
| benchmark id | `synthetic_50m` |
| rows | **50,000,000** |
| partitions | 166 monthly Parquet partitions, zstd |
| combined partition hash | `c2db7142e24dd8d9c2806ef8a1ec0c23e62cbf2d8fcc4b4b614396922b47edf1` |
| schema hash | `fb1d8e6a19ffa5837a28747bd928a6c40d4111a6c36eb6d89049bde620e09d96` |
| generator version | `1.0.0` |
| seed | `20261005` |
| source dataset | `data/credit_card_transactions-ibm_v2.csv` |
| source SHA-256 | `b01fa323c98522f8c710c7f7242581860c97c50183f1c5fa9e772e5c674a7f15` |
| manifest | `data/synthetic_50m/manifest.json` |
| fraud rows / prevalence | 61,109 / 0.00122218 |
| partition span | `year_month=2002-09` … `year_month=2016-05` |
| on-disk size | 3.7 GB (NVMe SSD) |
| prior decision | **`VALIDATED WITH LIMITATIONS`** — §16 suitability explicitly permits storage, data-pipeline, throughput and training-scalability testing |

The combined hash was re-derived after the fault-injection tests and is **unchanged**.

## 2. Exact model identity

Evaluated **exactly as it exists**. No optimisation, no architecture change, no threshold change.

| field | value |
|---|---|
| model version | `altman_native_v2_20260904_115703` |
| model type | `xgb_lgb_cb_native` |
| family | gradient-boosted tree ensemble on the native 48-feature contract |
| members | `xgb`, `lgb`, `cb` |
| fusion | **mean, 0.34 / 0.33 / 0.33 — the NR-05 configuration, preserved unchanged** |
| features | 48, contract order = `manifest.json` `features` = `feature_list.json` = `ALTMAN_NATIVE_FEATURES` (all three identical) |
| preprocessing | `RobustScaler(48)` fit on training rows |
| calibration | **none** — `calibrator.joblib` absent, `self.calibrator is None`, confirmed empirically |
| locked threshold | `0.7847116291110687` |
| gating | none |
| fallback | degraded ML → rules-only (fail-open); **never triggered at this scale** |
| model_dir | `models/production/altman_native` |
| training source / rows / seed | IBM v2, 193,027 rows, seed 42 |
| git SHA at training | `6a6f371c8690380cd318cf504a7c82a6658f1a3e` |

Artifact hashes (all verified unchanged this phase):

| artifact | SHA-256 |
|---|---|
| `xgb_native.joblib` | `a1cdebdfe01b709a5e0d9480078e75565521bd6cf5aa7eca06771a828d170bbc` |
| `lgb_native.joblib` | `d59aebcb08d6df05dd9640e1f88c1454aeb0045f8676ee723588b3d401a5c87f` |
| `cb_native.joblib` | `22b8377bc1ff4b6fd07e78630c5b9a8c8729829f286f7e4378948053692cc73f` |
| `scaler_native.joblib` | `b98fadf339f77d09219dabeff94d4b621fd80009e644e5cdbcdec183b1008611` |
| `feature_list.json` | `657459ac9526c77f6f9006b7ba18a41338a54937e12177cd16eaf944d59aed8f` |
| `models/production/manifest.json` | `e6da1a433ec618576aa12208190d35df3b53e9349b1bb6e54c16465e324b2485` |

## 3. Experiment manifest identity

Bound to every reported metric; embedded in each JSON artifact. Timestamps are **not** the
identity.

| field | value |
|---|---|
| git SHA | `5a5ff55cc03df73318fdf31a16e3665f159a4720` |
| benchmark combined SHA-256 | `c2db7142…47edf1` |
| benchmark schema hash | `fb1d8e6a…e09d96` |
| generator / seed | `1.0.0` / `20261005` |
| model version | `altman_native_v2_20260904_115703` |
| preprocessing version | `RobustScaler(48)`, scaler hash `b98fadf3…` |
| calibration version | **none** |
| threshold identifier | `locked_threshold=0.7847116291110687` |
| feature-contract hash | `feature_list.json` `657459ac…`; `native_features.py` `5760a203…` |
| ensemble code hash | `altman_native_ensemble.py` `55f4ad52…` |
| Python | 3.12.10 |
| dependencies | xgboost 3.4.1, lightgbm 4.7.0, catboost 1.2.10, scikit-learn 1.9.0, numpy 2.5.2, pandas 3.0.5, pyarrow 25.0.1, scipy 1.18.0 |
| CPU | 12th Gen Intel(R) Core(TM) i7-12650H |
| GPU | RTX 3050 A Laptop, 4 GB VRAM — **not used** |
| RAM / storage | 15.6 GB / SAMSUNG MZVL2512HCJQ-00BH1 NVMe SSD 477 GB |
| workers / batch size | 1 / 250,000 |
| chunking probe size | 25,000 (2,000 chunks) |
| random seeds | model 42; data 20261005 |

## 4. Tests performed

| § | test | status |
|---|---|---|
| 1–5 | preconditions, benchmark and model identity, experiment manifest, dataset/partition/chunking semantics | **PASS** |
| 6 | 50M inference baseline, stage-separated end-to-end, latency percentiles, peak RAM, storage, failed batches | **PASS** (two full runs) |
| 6.4 | CPU utilisation by `psutil` process-tree sampling | **PASS** (shorter instrumented pass — scope-limited) |
| 7 | training scalability ladder at fixed canonical hyperparameters | **PARTIAL** — 1M/2M/5M/10M measured, 25M/50M `BLOCKED BY COMPUTE` |
| 8 | ensemble vs each member vs existing baseline model, all splits, ROC-AUC / PR-AUC / Brier / ECE / confusion / precision / recall / F1 / alert rate / recall at fixed alert volume | **PASS** |
| 10 | evaluation-split methodology, entity-disjoint via benchmark `split`, temporal cut at `2013-01` | **PASS** |
| 11 | chunking-equivalence design with explicit FP tolerance | **PASS** |
| 12 | entity-disjoint results | **PASS** |
| 13 | temporal results | **PASS** |
| 14 | chunking-equivalence results (2,000 × 25,000 vs single pass) | **PASS** — delta 0 |
| 15 | failure/recovery on partition **copies** (missing / duplicate / corrupt / rerun / recovery) | **PASS** — 2 silent defects found |
| 16 | reproducibility, two independent full 50M runs | **PASS** — bit-identical |
| 17 | resource-scaling table, measured cells only | **PASS** |
| 20 | repository verification battery and checkers | battery **83 PASS / 1 FAIL / 0 MISSING** of 84 suites (the single FAIL is a pre-existing transient, §9.3); `claim_evidence_check` (22 claims), `eval_record_test` (22/22), `check_freeze_test` (22/22), `review_resolution_check`, `review_package_check` all **rc=0**; freeze checker **rc=1 / 78 problems** exactly as expected; invariant hashes and all 166 benchmark partition bytes re-verified unchanged |

## 5. Tests NOT performed

| test | why not | class |
|---|---|---|
| training at 25M and 50M rows | 4.47 GiB / 8.94 GiB float32 matrices against ~7 GB free; each needs a second full-size scaler copy | `BLOCKED BY COMPUTE` |
| CPU utilisation over the **full** 50M pass | sampled over a shorter instrumented pass only; the full pass was already reproduced twice | `NOT MEASURED` |
| training CPU utilisation | the ladder harness did not instrument CPU | `NOT MEASURED` |
| GPU-accelerated inference or training | 4 GB VRAM; production runs CPU-only and this phase measures the production path | not attempted |
| hyperparameter search or retuning | §7 and §22 forbid optimisation | out of scope |
| alternative ensemble configurations as ablations | §3 permits these only after the baseline; this phase stops at the baseline | out of scope |
| significance testing / confidence intervals on any split comparison | not implemented; §8/§12/§13 report point estimates only | limitation |
| `recall_at_1pct_fpr` | not meaningful — the locked threshold yields FPR far below any fixed band; substituted recall at fixed alert volume | `NOT APPLICABLE` |
| fail-open degraded path (ML failure / open circuit breaker) | no ML failure occurred at 50M | `NOT EXERCISED` |
| calibration stage timing | no calibrator exists in production | `NOT APPLICABLE` |
| fault injection against the **production serving path** | §15 covers the measurement pipeline only | out of scope |
| real-world / institutional effectiveness evaluation | explicitly forbidden in this phase | `NOT ESTABLISHED` |
| security assessment | dedicated Phase 4 workstream | `NOT ESTABLISHED` |

## 6. Quantitative results

### 6.1 Inference scalability (two full 50M passes)

| metric | run 0 | run 1 |
|---|---|---|
| rows / partitions | 50,000,000 / 166 | 50,000,000 / 166 |
| wall seconds | 500.1 | 563.9 |
| throughput | 99,971 rows/s | 88,668 rows/s |
| peak RAM | not recorded | **1,785.1 MB** |
| bytes read | 3,916,485,262 | 3,916,485,262 |
| temporary storage | 0 bytes | 0 bytes |
| failed / retried batches | 0 / 0 (no retry mechanism exists) | 0 / 0 |
| non-finite values | 0 | 0 |
| latency mean / p50 / p95 / p99 | 5.13 / 4.13 / 10.23 / 11.40 ms | 6.23 / 4.89 / 18.32 / 20.47 ms |

Stage separation (run 0, fraction of wall): **inference 92.90%** (464.62 s), validate 2.17%,
scale 1.97%, load 0.59%, fuse 0.15%, threshold 0.01%, store 0.28%. Every non-inference stage
combined is under 7%.

### 6.2 Model-performance baseline — `SYNTHETIC 50M — SCALE EXPERIMENT`

Full benchmark, 50,000,000 rows / 61,109 fraud / prevalence 0.00122218, threshold `0.7847116291110687`:

| model | ROC-AUC | PR-AUC | Brier | ECE | precision | recall | F1 | alerts | alert rate | recall @ 0.1% volume |
|---|---|---|---|---|---|---|---|---|---|---|
| ensemble | 0.573196 | 0.001801 | 0.001248 | 0.000289 | 0.000000 | **0.000000** | 0.000000 | 193 | 3.86e-06 | 0.003436 |
| xgb | 0.574237 | 0.001889 | 0.001234 | 0.000829 | 0.000000 | 0.000000 | 0.000000 | 112 | 2.24e-06 | 0.005155 |
| lgb | 0.566367 | 0.001733 | 0.001239 | 0.000595 | 0.000000 | 0.000000 | 0.000000 | 175 | 3.50e-06 | 0.003584 |
| cb | 0.558480 | 0.001664 | 0.001323 | 0.000908 | 0.000704 | 0.000016 | 0.000032 | 1,420 | 2.84e-05 | 0.002716 |
| baseline_xgb | 0.574237 | 0.001889 | 0.001234 | 0.000829 | 0.000000 | 0.000000 | 0.000000 | 112 | 2.24e-06 | 0.005155 |

Confusion at the locked threshold (ensemble): **tp 0, fp 193, fn 61,109, tn 49,938,698**.

**NR-05 confirmed at 50M scale** — the mean-fusion ensemble is below its best member xgb on
ROC-AUC, PR-AUC, Brier, alert volume and recall at fixed alert volume.

### 6.3 Entity-disjoint and temporal — `SYNTHETIC 50M — SCALE EXPERIMENT`

Split design: the benchmark's `split` column is assigned **per `user_id`**
(`scripts/generate_synthetic_50m.py:319`), so `test` rows are genuinely held-out entities.
Temporal cut: **`year_month = 2013-01`**, eval-late inclusive, train-early exclusive.

| slice | rows | fraud | ensemble AUC | xgb AUC | ensemble recall @ 0.1% |
|---|---|---|---|---|---|
| seen entities (train users) | 24,932,067 | 30,442 | 0.574049 | 0.574692 | 0.003186 |
| **entity-disjoint (test users)** | 25,067,933 | 30,667 | 0.572344 | 0.573786 | 0.003652 |
| train-early (`< 2013-01`) | 37,349,420 | 45,578 | — | — | — |
| **temporal eval-late (`>= 2013-01`)** | 12,650,580 | 15,531 | **0.576828** | 0.569964 | 0.004829 |
| entity-disjoint AND eval-late | 6,340,753 | 7,838 | **0.579387** | 0.572484 | 0.004721 |

The ensemble-vs-xgb ranking **reverses** on the temporal splits. Recorded as an unresolved
observation on synthetic data, **not** as grounds to replace the ensemble or overturn NR-05.

**The locked threshold raises ZERO alerts across the entire 12,650,580-row eval-late period** and
yields recall 0.000000 on every split.

### 6.4 Chunking equivalence

Single pass 193 alerts vs 2,000 chunks × 25,000 = 193 alerts. **Absolute decision delta 0, delta
rate 0.0 — exactly identical** at an explicit FP tolerance of 0.

### 6.5 Failure / recovery (partition copies; original benchmark unmodified)

| # | fault | exit | rows scored | outcome |
|---|---|---|---|---|
| 1 | none (baseline) | 0 | 903,612 | as expected |
| 2 | **one partition deleted** | **0** | **602,408** | **SILENT — DEFECT** |
| 3 | one partition truncated to 4 KiB | 1 | — | as expected (loud) |
| 4 | **one partition duplicated** | **0** | **1,204,816** | **SILENT — DEFECT** |
| 5 | rerun of a completed set, pass 1 | 0 | 903,612 | as expected |
| 6 | rerun of a completed set, pass 2 | 0 | 903,612 | as expected — **bit-identical** |
| 7 | recovery after restore | 0 | 903,612 | as expected — restored to baseline |

Duplicate injection shifted base-case ROC-AUC 0.577918 → 0.580575 and fraud 1,142 → 1,504, and was
reported as a successful run.

### 6.6 CPU utilisation (5M-row instrumented pass)

Mean **309.2% of one logical core** (p50 390.6%, p95 465.6%, max 482.6%; ≈3.1 of 16 logical
cores busy on average, 19.3% of aggregate capacity), process-tree RSS mean 918.6 MB / max
1,144.1 MB. The sampler observed 48.2 s wall over 5,000,000 rows (≈103,794 rows/s including
interpreter and model load); the runner's internal wall was 35.2 s (142,078 rows/s, inference
86.2% of that wall), exit code 0. Evidence: `misc/reports/phase3_cpu.json` and
`misc/reports/phase3_cpu_run.json`. Full-50M CPU remains `NOT MEASURED` (§5).

## 7. Compute limitations

| limit | value | effect |
|---|---|---|
| host RAM | 15.6 GB total, ~7 GB free | caps training at 10M rows; 25M needs 4.47 GiB, 50M needs 8.94 GiB as a float32 matrix alone, each requiring a second full-size scaler copy |
| CPU | i7-12650H, single process, `n_jobs=4` inside the boosters | ~89k–100k rows/s inference; no cluster, no multi-node |
| GPU | RTX 3050 A Laptop, 4 GB VRAM | **unused** — CPU path measured, as production runs it |
| disk | 95.2 GB free at phase start | sufficient; benchmark 3.7 GB |
| Docker | not available | dockerised retrain path not exercised |

**Nothing was fabricated to fill these gaps.** The 25M and 50M training rungs are recorded as
`BLOCKED BY COMPUTE` with the blocking arithmetic, and no completion time is claimed for them.

## 8. Reproducibility result

**`BIT-IDENTICAL` on every scientific output across two independent full 50,000,000-row runs.**

Identical: rows, partitions, fraud rows, prevalence, threshold, ROC-AUC (0.5731960569465072),
PR-AUC (0.0018011156833651009), Brier (0.0012478301068767905), full confusion matrix, precision /
recall / F1, alert rate, all member ROC-AUC / PR-AUC / precision, non-finite count, calibrator flag.

Only wall-clock quantities differed (500.1 s vs 563.9 s; 99,971 vs 88,668 rows/s; p95 10.23 vs
18.32 ms), attributable to concurrent host services. Supporting idempotence evidence: two repeated
three-partition runs were bit-identical.

**`REPRODUCIBILITY FAILURE` is not the outcome.**

## 9. Failures encountered, and fixes made

### 9.1 Defects found in the existing codebase — RECORDED, NOT FIXED

| defect | evidence | disposition |
|---|---|---|
| **`map_raw_to_native` branch 2 is unreachable.** Branch 1 guards on `"mcc" in features`; branch 2 (`_from_native_dict`) *also* requires `"mcc" in features`. A native dict is therefore always re-derived by `_from_raw_native`. | `engine.predict()` vs the runner: max diff **1.264e-01**, 0/2000 exact (though 2000/2000 threshold decisions agreed). `engine.predict()` vs `_from_raw_native` + identical fusion: max diff **8.941e-10**, proving the dispatch. Runner vs the true native pass-through `_from_native_dict`: **max diff 0.0, 2000/2000 — BIT-IDENTICAL**. | **NOT FIXED** — §3 forbids model changes. Recorded in report §3.1 and §20. This is why the batched runner is a valid stand-in. |
| **No completeness assertion in the large-scale runner.** A missing partition and a duplicated partition both produce a well-formed, exit-0 report over the wrong row set. | fault tests 2 and 4 above | **NOT FIXED** — this is the measurement pipeline, not production. Recorded as an operational reliability finding for a later phase. |

### 9.2 Phase 3 tooling defects — FIXED during this phase

| defect | fix |
|---|---|
| `sorted(str(p) for p in ..., key=...)` raised `SyntaxError: Generator expression must be parenthesized` | wrapped the generator in parentheses |
| name collision: `fp` used for both the Parquet path and the false-positive counter → `TypeError: can only concatenate str (not "int") to str` | renamed the loop variable to `fpath` |
| `peak_rss_MB` was `NaN` because `psutil` was absent | installed `psutil 7.2.2` into `.venv`; re-measured in run 1 (1,785.1 MB). Run 0's peak RSS is reported as **not recorded**, not back-filled |
| `NATIVE_DIR` resolves relative to the module (`backend/models/...`), not the repo root | `model_dir=Path("models/production/altman_native")` passed explicitly |
| markdown-table checker produced false positives by pairing backticks across block boundaries | checker rewritten to flag only pipes inside inline code **on table rows** |
| the first CPU-instrumented runner pass **crashed** with `ValueError: Found array with 0 sample(s)` and wrote no runner report: the `--rows` cap was only checked at partition boundaries, so a cap landing exactly on an intra-partition batch boundary produced an empty frame | added an `if n >= cap: break` guard inside the batch loop. **A no-op for every recorded run** — the two 50M runs and the fault tests never hit that boundary, so no reported metric changed |
| the first CPU sample itself was meaningless (0.014% mean, runner crashed): the sampler recreated `psutil.Process` objects every iteration, and `cpu_percent(interval=None)` returns a useless 0.0 baseline on a fresh instance | keep one `Process` object per pid alive across samples (only tree membership is refreshed), record `runner_exit_code`; re-ran the 5M pass → mean 309.2%, runner exit 0. The failed first attempt's output is preserved in `.freebuff/phase3_cpu.log` |

### 9.3 Process issues and flakes

| issue | resolution |
|---|---|
| `phase101_audit_remediation_test` failed **once** during this phase's battery (`Same inputs -> same hash`, 203/204) | pre-existing boundary flake, unrelated to this phase: `resolution_hash` covers `created_at`, so two back-to-back `create_finding` calls that straddle a clock tick hash differently. Passed on **five consecutive immediate re-runs** (204/204) and passed in both earlier batteries the same day. Recorded, not suppressed — the battery tally above reports the failure as it happened |
| the reproducibility run 1 launched with `nohup` appeared to die when its parent shell timed out | confirmed alive via `Win32_Process` command line (PID 13252); waited for its artifact rather than re-launching |
| a shell heredoc truncated the report mid-sentence | detected, truncated back to a clean boundary, remainder re-appended from a file and verified |
| concurrent heavy runs risked exhausting 7 GB free RAM | serialised the 50M inference, eval, fault and ladder passes |

## 10. Did any production artifact change?

**No.** Verified by hash after the phase against the phase-start snapshot
`.freebuff/phase3_invariants.txt`, plus a byte-level re-hash of all 166 benchmark partitions
against `data/synthetic_50m/manifest.json` (spec §20):

| invariant | expected | verified |
|---|---|---|
| original IBM v2 data | `b01fa323c98522f8…` | **unchanged** |
| Research Plan | `eab5a0615801db7e…` | **unchanged** |
| preregistration (`docs/evaluation/PRE_REGISTERED_EVALUATION_PROTOCOL.md`) | `fdde71cbd00e8e34…` | **unchanged** |
| production model manifest | `e6da1a433ec61857…` | **unchanged** |
| native 48-feature contract (`native_features.py`) | `5760a20376306598…` | **unchanged** |
| ensemble implementation (`altman_native_ensemble.py`) | `55f4ad5284ec6f45…` | **unchanged** |
| production model artifacts (xgb/lgb/cb/scaler/feature_list) | §2 hashes | **unchanged** |
| validated 50M benchmark | `c2db7142…47edf1` | **unchanged** |

`xgb_native.joblib` carries a recent mtime from the repository battery; its **content hash matches
the pinned value**, so it is unchanged in substance. Training artifacts from the ladder were written
only to `misc/reports/phase3_train_ladder_artifacts/`; **nothing was written under `models/production`**.

## 11. Did any scientific contract change?

**No.** The Research Plan, preregistration, metric definitions, native 48-feature contract,
production threshold (`0.7847116291110687`), release manifest, reviewer records, statistical and
domain review records, and dataset provenance records are all unchanged.

One **reporting** convention was extended, not a contract: a 15-bin equal-width ECE and a
recall-at-fixed-alert-volume metric were added to the Phase 3 evaluation script. These are
**additional** measurements reported alongside the existing metrics; no existing metric definition
was altered, redefined or removed.

## 12. Files created this phase

| file | role |
|---|---|
| `docs/evaluation/LARGE_SCALE_MODEL_TEST_REPORT.md` | **required deliverable**, 22 sections |
| `docs/evaluation/PHASE_LARGE_SCALE_MODEL_TEST_CLOSEOUT.md` | **required deliverable**, this file |
| `scripts/large_scale_runner.py` | 50M inference baseline runner |
| `scripts/large_scale_eval.py` | one-pass ensemble/member/baseline, split and chunking evaluation |
| `scripts/large_scale_train_ladder.py` | fixed-hyperparameter training scalability ladder |
| `scripts/large_scale_faults.py` | fault injection on partition copies |
| `scripts/large_scale_cpu_sampler.py` | CPU utilisation / RSS sampling wrapper |
| `misc/reports/phase3_inference_50m.json`, `phase3_inference_run1.json` | the two full 50M runs |
| `misc/reports/phase3_eval.json` | split, member and chunking results |
| `misc/reports/phase3_train_ladder.jsonl` | per-rung training measurements |
| `misc/reports/phase3_faults.json` | failure/recovery record |
| `misc/reports/phase3_cpu.json` | CPU utilisation samples (valid re-run after the §9.2 fix) |
| `misc/reports/phase3_cpu_run.json` | runner report of the instrumented 5M pass (exit 0) |
| `.freebuff/verify_fastpath.py` | runner-to-production parity harness (3 comparisons) |

## 13. Final status

# `LARGE-SCALE BASELINE ESTABLISHED WITH LIMITATIONS`

**The current PS-14 production/native prototype processes the full 50,000,000-row validated
benchmark reliably, reproducibly and within 1.8 GB of RAM on a single consumer laptop CPU, and its
architecture remains computationally practical at this scale.** The model was evaluated exactly as
it exists; NR-05 is preserved; no production artifact and no scientific contract was changed.

The limitations are stated in report §1 and §22: training `BLOCKED BY COMPUTE` at 25M and 50M,
two silent data-integrity defects, recall 0.000000 at the locked threshold on every split, CPU
utilisation not measured for the full 50M pass, and no significance testing on any split
comparison.

**Every performance figure is `SYNTHETIC 50M — SCALE EXPERIMENT`.** Real-world effectiveness,
institutional generalisation, regulatory readiness and security assurance all remain
**`NOT ESTABLISHED`**.

**Phase 3 is complete and stops here.** No optimisation, no Strix security testing, no new dataset,
no institutional validation.

Next: **Phase 4 — Strix Security Validation**, then **Phase 5 — Prototype Optimization**. The
three defects carried forward as its concrete inputs are the unreachable branch in
`map_raw_to_native`, the zero-recall locked operating point, and the absent completeness assertion.