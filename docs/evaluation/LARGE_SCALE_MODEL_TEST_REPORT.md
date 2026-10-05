# PS-14 — Phase 3: Large-Scale Model Test on the Validated 50M Benchmark

**Final decision: `LARGE-SCALE BASELINE ESTABLISHED WITH LIMITATIONS`**

**Date:** 2026-10-05
**Repository:** PS-14-Fraud-Detection-software
**Git SHA:** `5a5ff55cc03df73318fdf31a16e3665f159a4720`
**Report scope:** computational scalability, training scalability, inference throughput, latency, memory/storage behaviour, batch processing behaviour, failure/recovery, and reproducibility.

> **EVIDENCE LABEL — applies to every performance number in this report.**
> Every model-performance figure below is measured on a **synthetic** benchmark and is labelled
> **`SYNTHETIC 50M — SCALE EXPERIMENT`**. Nothing in this report establishes real-world fraud
> detection effectiveness, institutional generalisation, production fraud-loss reduction,
> regulatory readiness, or independent external validation. See §18 and §21.

---

## 1. Scope and decision

This phase evaluates the **current** PS-14 production/native prototype at 50,000,000-row scale.
It answers one question first:

> How does the current production/native prototype behave at 50M-row scale?

The model was evaluated exactly as it exists. No optimisation, no architecture change, no
threshold change, no new dataset, no security testing. Alternative configurations are **not**
explored here; per §3 the NR-05 finding is preserved, not replaced.

| decision category | outcome |
|---|---|
| `LARGE-SCALE BASELINE ESTABLISHED` | not selected |
| `LARGE-SCALE BASELINE ESTABLISHED WITH LIMITATIONS` | **SELECTED** |
| `BLOCKED BY COMPUTE` | not selected (see §7 — a partial ladder was measured; only 25M/50M were blocked) |
| `REQUIRES REPAIR` | not selected (two silent data-integrity defects recorded in §15 as operational findings, not model repairs) |
| `REPRODUCIBILITY FAILURE` | not selected (two 50M runs were bit-identical on every scientific output — §16) |

**Why `WITH LIMITATIONS` rather than unqualified `ESTABLISHED`:**

1. Training at 25M and 50M rows is **`BLOCKED BY COMPUTE`** on this host (§7). Only the 1M–10M
   rungs are measured.
2. Two fault classes are detected **silently** with exit code 0 — a missing partition and a
   duplicated partition both produce a successful-looking run over the wrong row set (§15).
3. The **locked production threshold yields recall 0.000000** on this benchmark, on every split
   (§8, §12, §13). This is a finding about the benchmark/threshold pairing, not a defect proven
   here, but it means the headline confusion matrix cannot describe useful behaviour.
4. CPU utilisation was measured over a 5M-row instrumented pass, not the full 50M pass (§6).

---

## 2. Benchmark identity

The exact validated artifact was used. It was **not regenerated and not modified**.

| field | value |
|---|---|
| benchmark id | `synthetic_50m` |
| rows | 50,000,000 |
| partitions | 166 monthly Parquet partitions, zstd |
| combined partition hash | `c2db7142e24dd8d9c2806ef8a1ec0c23e62cbf2d8fcc4b4b614396922b47edf1` |
| schema hash | `fb1d8e6a19ffa5837a28747bd928a6c40d4111a6c36eb6d89049bde620e09d96` |
| generator version | `1.0.0` |
| generation seed | `20261005` |
| source dataset | `data/credit_card_transactions-ibm_v2.csv` |
| source SHA-256 | `b01fa323c98522f8c710c7f7242581860c97c50183f1c5fa9e772e5c674a7f15` |
| fraud rows | 61,109 |
| fraud prevalence | 0.00122218 |
| manifest | `data/synthetic_50m/manifest.json` |
| on-disk size | 3.7 GB (NVMe SSD) |
| partition span | `year_month=2002-09` … `year_month=2016-05` |
| combined hash re-verified after this phase | **YES — unchanged** |

**Prior-phase decision that permits this phase:** `VALIDATED WITH LIMITATIONS`. Its §16
suitability explicitly establishes storage, data-pipeline, throughput and training-scalability
testing, and marks model-performance comparison as `LIMITED`. This phase stays inside that
boundary.

---

## 3. Model identity preserved

The evaluated model is the production/native ensemble **exactly as it exists**. Nothing was
retrained, replaced, or re-fitted for the baseline.

| field | value |
|---|---|
| model version | `altman_native_v2_20260904_115703` |
| model type | `xgb_lgb_cb_native` |
| family | gradient-boosted tree ensemble, native 48-feature contract |
| members | `xgb`, `lgb`, `cb` |
| ensemble weights | `xgb 0.34 / lgb 0.33 / cb 0.33` (**mean fusion — the NR-05 configuration that lost to its best member**) |
| feature count | 48 |
| feature order | `models/production/manifest.json` `features` == `feature_list.json` order == `ALTMAN_NATIVE_FEATURES` (all three identical) |
| preprocessing | `RobustScaler(48)` fit on training rows only |
| calibration | **none** — `AltmanNativeEnsembleEngine` loads `models/production/artifacts/calibrator.joblib`, which does not exist, so `self.calibrator is None` (confirmed empirically in §6) |
| locked threshold | `0.7847116291110687` |
| gating | none |
| fallback behaviour | degraded ML → rules-only (fail-open); **not exercised at this scale** — no ML failure occurred |
| model_dir | `models/production/altman_native` |
| source git sha at training | `6a6f371c8690380cd318cf504a7c82a6658f1a3e` |
| train rows / seed | 193,027 / 42 |

**Artifact hashes (verified this phase, unchanged):**

| artifact | SHA-256 |
|---|---|
| `xgb_native.joblib` | `a1cdebdfe01b709a5e0d9480078e75565521bd6cf5aa7eca06771a828d170bbc` |
| `lgb_native.joblib` | `d59aebcb08d6df05dd9640e1f88c1454aeb0045f8676ee723588b3d401a5c87f` |
| `cb_native.joblib` | `22b8377bc1ff4b6fd07e78630c5b9a8c8729829f286f7e4378948053692cc73f` |
| `scaler_native.joblib` | `b98fadf339f77d09219dabeff94d4b621fd80009e644e5cdbcdec183b1008611` |
| `feature_list.json` | `657459ac9526c77f6f9006b7ba18a41338a54937e12177cd16eaf944d59aed8f` |
| `models/production/manifest.json` | `e6da1a433ec618576aa12208190d35df3b53e9349b1bb6e54c16465e324b2485` |

`xgb_native.joblib` carries a recent mtime (touched by the repository battery) but its content
hash matches the pinned value, so it is unchanged.

### 3.1 Parity of the measurement runner with production

Phase 3 must measure the model, not a reimplementation of it. `scripts/large_scale_runner.py`
reproduces the production contract exactly: native 48-vector in contract order → `astype(float32)`
**before** scaling (the cast order `predict()` requires) → `RobustScaler.transform` →
three `predict_proba[:, 1]` → `0.34/0.33/0.33` weighted mean → `clip[0,1]` → `>= locked_threshold`.

Parity was measured on 2,000 rows (`.freebuff/phase3_fastpath.txt`):

| comparison | max abs diff | exact equal | threshold decisions equal |
|---|---|---|---|
| runner vs `AltmanNativeEnsembleEngine.predict()` | 1.264e-01 | 0/2000 | **2000/2000** |
| runner vs true native pass-through `_from_native_dict` | **0.0** | **2000/2000** | **2000/2000** |
| `engine.predict()` vs `_from_raw_native` + identical fusion | 8.941e-10 | 76/2000 | — |

**Recorded pre-existing defect (NOT fixed — §3 forbids model changes).**
`map_raw_to_native()` in `backend/src/risk_engine/altman_native_ensemble.py` has an **unreachable
branch**: branch 1 guards on `"mcc" in features`, but branch 2 (`_from_native_dict`, the native
pass-through) *also* requires `"mcc" in features`. Therefore a native dict is **always** re-derived
by `_from_raw_native`, and branch 2 can never execute. The third row above proves this
empirically: re-deriving and then applying the identical fusion reproduces `engine.predict()`
to within float noise (8.9e-10).

Consequence: `engine.predict()` and the batched runner differ (max 1.264e-01), though **all
2,000 threshold decisions agreed**. Because the runner is **bit-identical** to the true native
pass-through, it is a valid stand-in for the documented production contract. This defect is a
correctness finding for a later phase, not a blocker for this one.

---

## 4. Experiment manifest and determinism

Timestamps are **not** the experiment identity. Every metric in this report is bound to the
manifest below, which is embedded in each JSON artifact.

| field | value |
|---|---|
| git SHA | `5a5ff55cc03df73318fdf31a16e3665f159a4720` |
| benchmark combined SHA-256 | `c2db7142e24dd8d9c2806ef8a1ec0c23e62cbf2d8fcc4b4b614396922b47edf1` |
| benchmark schema hash | `fb1d8e6a19ffa5837a28747bd928a6c40d4111a6c36eb6d89049bde620e09d96` |
| generator version | `1.0.0` |
| benchmark seed | `20261005` |
| benchmark source SHA-256 | `b01fa323c98522f8c710c7f7242581860c97c50183f1c5fa9e772e5c674a7f15` |
| model version | `altman_native_v2_20260904_115703` |
| preprocessing version | `RobustScaler(48)` (scaler artifact hash above) |
| calibration version | **none — calibrator absent, `calibrator_loaded: false`** |
| threshold identifier | `locked_threshold=0.7847116291110687` |
| feature-contract hash | `feature_list.json` `657459ac9526c77f…`; contract module `native_features.py` `5760a20376306598…` |
| ensemble code hash | `altman_native_ensemble.py` `55f4ad5284ec6f45…` |
| Python | 3.12.10 |
| platform | `Windows-11-10.0.26200-SP0` |
| xgboost / lightgbm / catboost | 3.4.1 / 4.7.0 / 1.2.10 |
| scikit-learn / numpy / pandas / pyarrow | 1.9.0 / 2.5.2 / 3.0.5 / 25.0.1 |
| scipy | 1.18.0 |
| CPU | 12th Gen Intel(R) Core(TM) i7-12650H |
| GPU | NVIDIA RTX 3050 A Laptop, 4 GB VRAM (**not used** — inference and training ran CPU-only) |
| RAM | 15.6 GB |
| storage | SAMSUNG MZVL2512HCJQ-00BH1, NVMe SSD, 477 GB |
| free disk during phase | 95.2 GB |
| workers | 1 (single process, CPU-only) |
| chunk size | 250,000 rows (batch) |
| chunking-equivalence probe size | 25,000 rows |
| random seed | 42 (model); benchmark seed `20261005` (data) |

---

## 5. Dataset handling, partition order and chunking semantics

* **The exact validated artifact was processed.** Labels, features, entity IDs, timestamps,
  partitions, prevalence and categorical mappings were not altered. The fault harness re-verified
  the benchmark's combined hash before and after and recorded it **unmodified**.
* **Partition order: chronological, `year_month` ascending** (2002-09 → 2016-05), identical in the
  runner, the evaluator and the training ladder.
* **Every operation is STATELESS across batches.** This is the decisive property for chunking:

| concern | disposition |
|---|---|
| refactorising categorical / entity IDs per chunk | **never performed** — ids are read pre-materialised from the benchmark; nothing is re-fit |
| historical / rolling aggregates recomputed inside a chunk | **never performed** — the benchmark's historical features were already generated under a strictly past-only rule (`historical_aggregates: strictly past-only (row's own amt/label excluded)`) |
| cross-chunk model state | **none** — scaler, all three boosters and the threshold are loaded once and never updated |
| chunk-order sensitivity | **none** — verified empirically in §14 (delta 0) |

Chunking therefore cannot change feature semantics. This is asserted and then **measured** in §14
rather than assumed.

---

## 6. Baseline scalability experiment (§6)

Runner: `scripts/large_scale_runner.py`. Full 50,000,000 rows, 166 partitions, chronological
order, batch 250,000, single worker, CPU-only.

### 6.1 Data processing

| metric | run 0 | run 1 (reproducibility) |
|---|---|---|
| total rows processed | 50,000,000 | 50,000,000 |
| partitions processed | 166 | 166 |
| total wall time | 500.1 s | 563.9 s |
| throughput | 99,971 rows/s | 88,668 rows/s |
| bytes read | 3,916,485,262 | 3,916,485,262 |
| peak RSS | not recorded (psutil absent) | **1,785.1 MB** |
| temporary storage written | **0 bytes** — fully streaming, no spill files | 0 bytes |
| failed batches | 0 | 0 |
| retried batches | 0 — **no retry mechanism exists** (§15) | 0 |
| non-finite feature values | 0 | 0 |
| missing-feature partitions | 0 | 0 |

Throughput differs between runs because of host contention (other services were running); the
scientific outputs did not differ at all (§16).

### 6.2 Model inference

| metric | run 0 | run 1 |
|---|---|---|
| inference time (stage) | 464.62 s | 523.75 s |
| throughput (end-to-end) | 99,971 rows/s | 88,668 rows/s |
| batch size | 250,000 | 250,000 |
| worker count | 1 | 1 |
| mean latency per batch | 5.13 ms | 6.23 ms |
| p50 latency | 4.13 ms | 4.89 ms |
| p95 latency | 10.23 ms | 18.32 ms |
| p99 latency | 11.40 ms | 20.47 ms |
| peak RAM | not recorded | 1,785.1 MB |
| CPU utilisation | see 6.4 | see 6.4 |

### 6.3 End-to-end stage breakdown (measured separately, not quoted as one number)

Run 0 fractions of the 500.1 s wall:

| stage | seconds | fraction |
|---|---|---|
| load (Parquet → pandas) | 2.93 | 0.59% |
| preprocess + feature validation | 10.87 | 2.17% |
| scale (`RobustScaler.transform`) | 9.85 | 1.97% |
| **inference (3 × `predict_proba`)** | **464.62** | **92.90%** |
| fusion (0.34/0.33/0.33 mean + clip) | 0.74 | 0.15% |
| thresholding vs locked threshold | 0.05 | 0.01% |
| store | 1.41 | 0.28% |

Run 1 (`load 3.30 / validate 13.03 / scale 12.28 / infer 523.75 / fuse 0.91 / threshold 0.07 /
store 1.70`) shows the same shape: **inference dominates at 92.9% of wall time**. Every other
stage combined is under 7%. The system throughput number and the inference number are close here
because inference genuinely dominates — but they are reported separately, as required, and are not
the same measurement.

### 6.4 CPU utilisation — measured, with a stated scope limit

The runner itself reports no CPU figure, so `scripts/large_scale_cpu_sampler.py` wrapped a
runner pass and sampled the process tree with `psutil` every 0.5 s. See
`misc/reports/phase3_cpu.json` for the sampled figures and `§17` for the resource-scaling table.

Measured over a 5,000,000-row instrumented pass (17 partitions, batch 250,000, runner exit code
0):

| metric | value |
|---|---|
| CPU utilisation, mean | **309.2% of one logical core** (p50 390.6%, p95 465.6%, max 482.6%) |
| equivalent busy cores | ≈3.1 of 16 logical cores on average (19.3% of aggregate capacity) |
| process-tree RSS | mean 918.6 MB, max 1,144.1 MB |
| sampler-observed wall | 48.2 s over 5,000,000 rows (≈ 103,794 rows/s, interpreter and model load included) |
| runner-internal wall | 35.2 s (142,078 rows/s; inference 86.2% of that wall) |
| samples | 95 × 0.5 s |

**Scope limit, stated plainly:** this was sampled over a shorter instrumented pass, **not** the
full 50M pass, because the full pass was already reproduced twice for §16. The figure is
therefore evidence about CPU behaviour at this scale and row width, not a certified 50M CPU
utilisation number.

### 6.5 Calibration / gating stage

`calibrator_loaded: false` on both runs. There is **no calibration stage in the end-to-end path**
because the production engine has no calibrator loaded — this is the real production behaviour,
reproduced, not a runner omission. The fail-open degraded path (ML failure or open circuit breaker
→ rules-only) was **never triggered** at this scale and is reported `NOT EXERCISED`.

---

## 7. Training scalability (§7)

Script: `scripts/large_scale_train_ladder.py`. Hyperparameters are copied **verbatim** from the
canonical native retrain (`.freebuff/retrain_native_tuned.py`); **no hyperparameter search was
performed** (§7 forbids unlimited search).

Fixed configuration: XGBoost `n_estimators=400, max_depth=7, lr=0.05, subsample=0.8,
colsample_bytree=0.7, min_child_weight=5, random_state=42, n_jobs=4`; LightGBM identical shape;
CatBoost `iterations=400, depth=7, lr=0.05, l2_leaf_reg=3, random_seed=42,
auto_class_weights="Balanced"`; `RobustScaler` fit on the training rows. `scale_pos_weight=20.0`,
which is the value the canonical script's own default resolves to on this benchmark:
`min((1 − 0.00122218) / 0.00122218, 20.0) = 20.0`.

Subset ladder: **1,000,000 → 2,000,000 → 5,000,000 → 10,000,000 → 25,000,000 → 50,000,000**,
each rung a **chronological prefix** of the benchmark (no resampling, no shuffling), so larger
rungs strictly contain smaller ones and each rung is a genuine scaling point.

**Rung results** (`misc/reports/phase3_train_ladder.jsonl`, one JSON entry per rung):

| rows | status | fraud rows | fit total (s) | wall (s) | fit throughput (rows/s) | peak RSS (MB) | artifacts (B) |
|---|---|---|---|---|---|---|---|
| 1,000,000 | MEASURED | 1,199 | 104.0 | 107.6 | 9,613 | 1,304 | 1,982,203 |
| 2,000,000 | MEASURED | 2,444 | 173.0 | 176.8 | 11,559 | 1,677 | 2,125,490 |
| 5,000,000 | MEASURED | 6,028 | 435.3 | 446.5 | 11,487 | 2,464 | 2,271,509 |
| 10,000,000 | MEASURED | 12,158 | 922.0 | 943.3 | 10,846 | 3,294 | 2,327,329 |
| 25,000,000 | **BLOCKED BY COMPUTE** | — | — | — | — | — | — |
| 50,000,000 | **BLOCKED BY COMPUTE** | — | — | — | — | — | — |

Stage and per-member breakdown of the same rungs:

| rows | load (s) | scale (s) | xgb fit (s) | lgb fit (s) | cb fit (s) | rounds built (each member) |
|---|---|---|---|---|---|---|
| 1,000,000 | 0.53 | 2.74 | 44.8 | 20.4 | 38.8 | 400 |
| 2,000,000 | 0.68 | 2.90 | 72.6 | 31.0 | 69.4 | 400 |
| 5,000,000 | 1.59 | 9.34 | 180.8 | 68.4 | 186.1 | 400 |
| 10,000,000 | 3.20 | 17.69 | 313.2 | 102.5 | 506.3 | 400 |

Every measured rung completed all 400 rounds for all three members; no early stopping is
configured in the canonical hyperparameters, so `convergence_status` reads "completed all 400
rounds per member" at every rung. Temporary storage written per rung equals the artifact bytes
(artifacts staged in `misc/reports/phase3_train_ladder_artifacts/`); the float32 feature matrix
is RAM-resident only and is never spilled to disk.

**`BLOCKED BY COMPUTE` applies to the 25M and 50M rungs only.** The float32 feature matrix alone
is `rows × 48 × 4` bytes: 4.47 GiB (4,800,000,000 bytes) at 25M and 8.94 GiB (9,600,000,000 bytes) at 50M, each requiring a second full-size
copy for the scaler output. On a 15.6 GB host with ~7 GB free this cannot be materialised. The
blocker is **recorded, not fabricated** — those rungs are reported as
`BLOCKED BY COMPUTE` with the arithmetic, and no completion time is claimed for them.

**Ladder interpretation.**

* **Fit throughput scales near-linearly, with mild degradation:** 9,613 rows/s at 1M, a plateau
  around 11.5k at 2M–5M, then 10,846 at 10M. Per member, 1M → 10M (10× the rows) costs 7.0× the
  fit time for xgb, 5.0× for LightGBM (most sub-linear) and 13.0× for CatBoost (super-linear — at
  10M, CatBoost alone is 506.3 s of the 922.0 s fit, 54.9%). No algorithmic breakdown occurred at
  any rung.
* **Peak RSS grows linearly with rows** — 1,304 → 1,677 → 2,464 → 3,294 MB from 1M to 10M, a
  slope of ≈221 MB per million rows, matching the materialised float32 matrix (48 × 4 = 192 B per
  row) plus booster working memory. Training is therefore **O(rows) in RAM**, unlike inference,
  which is O(batch) (§6, §17).
* **Artifact size saturates**: 1.98 MB at 1M → 2.33 MB at 10M. A fixed 400 rounds caps model size
  regardless of data volume.
* **The 25M and 50M rungs are not extrapolated from this trend.** They are recorded
  `BLOCKED BY COMPUTE` above, with the blocking arithmetic and **no estimated completion time**.

---

## 8. Model-performance baseline (§8)

**`SYNTHETIC 50M — SCALE EXPERIMENT`** — no real-world claim attaches to any number below.

Evaluated from one streaming pass (`scripts/large_scale_eval.py`, 50,000,000 rows, 907.3 s,
55,111 rows/s) so every model and every split describes the **same scoring of the same rows**.

Models compared (no new architecture introduced):

| key | model |
|---|---|
| `ensemble` | the current production fusion, 0.34 xgb + 0.33 lgb + 0.33 cb |
| `xgb` / `lgb` / `cb` | each individual production ensemble member |
| `baseline_xgb` | **the existing single-model baseline**, defined as the strongest member reused standalone |

Split identity for every row of every table below: synthetic benchmark `synthetic_50m`;
seed `20261005`; model `altman_native_v2_20260904_115703`; threshold `0.7847116291110687`;
artifact identity as in §3.

### 8.1 Full benchmark (50,000,000 rows / 61,109 fraud / prevalence 0.00122218)

| model | ROC-AUC | PR-AUC | PR lift | Brier | ECE | precision | recall | F1 | alerts | alert rate | recall @ 0.1% alert volume |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ensemble | 0.573196 | 0.001801 | 1.47 | 0.001248 | 0.000289 | 0.000000 | **0.000000** | 0.000000 | 193 | 3.86e-06 | 0.003436 |
| xgb | 0.574237 | 0.001889 | 1.55 | 0.001234 | 0.000829 | 0.000000 | 0.000000 | 0.000000 | 112 | 2.24e-06 | 0.005155 |
| lgb | 0.566367 | 0.001733 | 1.42 | 0.001239 | 0.000595 | 0.000000 | 0.000000 | 0.000000 | 175 | 3.50e-06 | 0.003584 |
| cb | 0.558480 | 0.001664 | 1.36 | 0.001323 | 0.000908 | 0.000704 | 0.000016 | 0.000032 | 1,420 | 2.84e-05 | 0.002716 |
| baseline_xgb | 0.574237 | 0.001889 | 1.55 | 0.001234 | 0.000829 | 0.000000 | 0.000000 | 0.000000 | 112 | 2.24e-06 | 0.005155 |

Confusion matrix at the locked threshold (all 50,000,000 rows):

| model | tp | fp | fn | tn |
|---|---|---|---|---|
| ensemble | **0** | 193 | 61,109 | 49,938,698 |
| xgb | 0 | 112 | 61,109 | 49,938,779 |
| lgb | 0 | 175 | 61,109 | 49,938,716 |
| cb | 1 | 1,419 | 61,108 | 49,937,472 |

**Two findings, both reported as measured.**

1. **NR-05 is confirmed at 50M scale.** The mean-fusion ensemble (0.573196) is **below its best
   member** xgb (0.574237) on the full benchmark, on PR-AUC (0.001801 vs 0.001889), and on recall
   at fixed alert volume (0.003436 vs 0.005155). The 0.34/0.33/0.33 mean fusion is preserved as the
   production configuration and has **not** been replaced, per §3.
2. **The locked threshold is functionally inert on this benchmark.** It raises **193 alerts in
   50,000,000 rows — 0.00000386 of the stream — and catches 0 of 61,109 fraud rows.** Recall is
   0.000000, precision 0.000000, F1 0.000000. Ranking quality is not zero (ROC-AUC 0.573 is above
   the 0.5 line; PR-AUC carries a 1.47× lift over prevalence), so the model *ranks* — but at the
   locked operating point nothing is ever investigated. Even when the operating point is freed and
   the top 0.1% of the stream is alerted (50,000 alerts), recall is only **0.34%** for the ensemble
   and **0.52%** for xgb, against a 0.12% random baseline — a 2.8× / 4.2× lift, nowhere near
   operationally useful.

This is a statement about the **locked threshold on a synthetic benchmark**, not a claim that the
threshold is wrong on real banking data. That question cannot be settled here (§18, §21).

### 8.2 What §8 could not measure

`recall_at_1pct_fpr` is not defined here: the locked threshold produces FPR far below any fixed
operating band, so the framework's recall@1%FPR rule has no meaningful evaluation window. The
substitute reported above is **recall at a fixed alert volume** (top 0.1% of rows), which is
defined at any score distribution and is used consistently across all models and splits.

---

## 9. Interpretation against the known benchmark limitations (§9)

The prior validation report established specific, named limitations. Each is checked here against
what was actually observed — the point is to prevent a synthetic AUC from being read as a capability claim.

| known limitation | observed in this phase | consistent? |
|---|---|---|
| historical fraud-rate features are largely inert | `user_fraud_rate` / `merch_fraud_rate` / `city_fraud_rate` are present but the model cannot exploit them: at fixed alert volume recall is 2.8–4.2× prevalence, not orders of magnitude | **yes** |
| MCC / channel dependence not faithfully reproduced | `mcc` dispatches into the unreachable branch (§3.1); the derived re-derivation path is what `predict()` actually executes | **yes — and it is also a live code defect, recorded in §20** |
| two native features are constant | constant columns cannot separate classes; consistent with ROC-AUC 0.573 rather than the ~0.92 the model shows on the real IBM v2 reference | **yes** |
| entity activity far more uniform than the real reference | uniform activity collapses the velocity/aggregation features (`user_tx_count`, `card_tx_count`, diversity) that carry most of the real signal | **yes** |
| real-vs-synthetic distinguishability experiment is confounded | this phase makes **no** real-vs-synthetic claim at all | **yes** |

**Consequence.** ROC-AUC 0.573196 and PR-AUC 0.001801 are properties of *this benchmark under a
model fitted to a different dataset*. They are **not** an estimate of banking performance, and the
combination of near-inert historical features, uniform entity activity, constant features and
broken channel dependence is sufficient on its own to explain a near-chance ranking result. A weak
synthetic result here does not show the model is weak on real data; a strong one would not have
shown it is strong either. **Neither direction is evidence.**

---

## 10. Evaluation-split methodology (§10)

The benchmark's `split` column is assigned **per user**, at generation time
(`scripts/generate_synthetic_50m.py:319` → `feats["split"] = np.where(state["user_split"][user] == 0, "train", "test")`).
A `test` row therefore belongs to a user that never appears in a `train` row. **This makes the
benchmark's own split genuinely entity-disjoint by user, not a disguised temporal slice.**

| split | definition | rows | fraud rows | prevalence |
|---|---|---|---|---|
| `all_rows` | every row | 50,000,000 | 61,109 | 0.00122218 |
| `seen_entities_train` | `split == "train"` (users seen in training) | 24,932,067 | 30,442 | 0.00122100 |
| `entity_disjoint_test` | `split == "test"` (**held-out users**) | 25,067,933 | 30,667 | 0.00122336 |
| `temporal_train_early` | `year_month < 2013-01` | 37,349,420 | 45,578 | 0.00122031 |
| `temporal_eval_late` | `year_month >= 2013-01` | 12,650,580 | 15,531 | 0.00122769 |
| `entity_disjoint_AND_temporal_late` | held-out users **and** `year_month >= 2013-01` | 6,340,753 | 7,838 | 0.00123613 |

**Exact temporal boundaries: the cut is `year_month = 2013-01`, inclusive of 2013-01 onward for
evaluation, exclusive for training.** The benchmark spans 2002-09 … 2016-05, so train-early
covers 2002-09 … 2012-12 (125 partitions) and eval-late covers 2013-01 … 2016-05 (41 partitions).

Prevalence is stable to four significant figures across every slice (0.0012210 – 0.0012361), so no
split is prevalence-confounded.

---

## 11. Chunking-correctness methodology (§11)

The claim to be tested is not "the runner is fast in batches" but "**chunking does not change
feature semantics**". §5 argued this from the code. §14 measures it.

Method: the same 50,000,000 scored rows are compared under two chunkings of the thresholding and
aggregation step — one single pass, and 2,000 chunks of 25,000 rows. The comparison uses an
**explicit FP tolerance of 0** (absolute count of differing per-row decisions), not a loose
epsilon. Rationale for the strict tolerance: every operation is stateless across batches (§5), so
any non-zero delta would indicate a real defect rather than floating-point noise. The measured
delta is `0`.

| dimension | single pass | chunked |
|---|---|---|
| chunks | 1 | 2,000 |
| rows per chunk | 50,000,000 | 25,000 |
| rows compared | 50,000,000 | 50,000,000 |
| scored inputs | the same §6 run-0 per-row scores — scoring is not re-run per chunk | identical |
| decision rule | `prob >= 0.7847116291110687` | identical |
| state carried across chunks | none — every operation stateless (§5) | none |
| equality criterion | per-row decision equality, **FP tolerance 0** | — |
| outcome | measured in §14 | measured in §14 |
---

## 12. Entity-disjoint results (§10)

**`SYNTHETIC 50M — SCALE EXPERIMENT`** — held-out users, `split == "test"`, 25,067,933 rows /
30,667 fraud / prevalence 0.00122336.

| model | ROC-AUC | PR-AUC | PR lift | Brier | ECE | recall | alerts | recall @ 0.1% alert volume |
|---|---|---|---|---|---|---|---|---|
| ensemble | 0.572344 | 0.001789 | 1.46 | 0.001250 | 0.000289 | **0.000000** | 100 | 0.003652 |
| xgb | **0.573786** | **0.001886** | 1.54 | 0.001236 | 0.000833 | 0.000000 | 60 | **0.005283** |
| lgb | 0.566259 | 0.001733 | 1.42 | 0.001241 | 0.000600 | 0.000000 | 86 | 0.003620 |
| cb | 0.556790 | 0.001648 | 1.35 | 0.001326 | 0.000922 | 0.000000 | 743 | 0.003000 |
| baseline_xgb | 0.573786 | 0.001886 | 1.54 | 0.001236 | 0.000833 | 0.000000 | 60 | 0.005283 |

For contrast, the **seen-entity** slice (24,932,067 rows / 30,442 fraud):

| model | ROC-AUC | PR-AUC | recall @ 0.1% alert volume |
|---|---|---|---|
| ensemble | 0.574049 | 0.001815 | 0.003186 |
| xgb | 0.574692 | 0.001894 | 0.004960 |
| lgb | 0.566476 | 0.001734 | 0.003581 |
| cb | 0.560178 | 0.001681 | 0.002398 |

**Findings.**

* **Entity-disjoint degradation is small but non-zero and not uniform.** Ensemble ROC-AUC falls
  0.574049 → 0.572344 (−0.0017) and xgb 0.574692 → 0.573786 (−0.0009) moving from seen to
  held-out users. The gap is an order of magnitude smaller than the gap this model shows between
  the real IBM v2 reference and the synthetic benchmark (§8) — consistent with §9's finding that
  the benchmark's entity structure is more uniform than the real reference, which removes the
  entity-specific structure a genuine entity-disjoint split would penalise.
* **NR-05 holds on held-out entities.** xgb beats the ensemble on ROC-AUC, PR-AUC, Brier, ECE,
  alert volume and fixed-alert-volume recall. The mean fusion is still the worse configuration.
* **The locked threshold yields recall 0.000000 on held-out entities too** — 100 alerts, 0 true
  positives. This is not an artifact of the entity split.

---

## 13. Temporal results (§10)

**`SYNTHETIC 50M — SCALE EXPERIMENT`** — cut at `year_month = 2013-01` (§10). Evaluation is
train-early (2002-09 … 2012-12, 37,349,420 rows) → eval-late (2013-01 … 2016-05, 12,650,580 rows).

| model | ROC-AUC | PR-AUC | PR lift | Brier | ECE | recall | alerts | recall @ 0.1% alert volume |
|---|---|---|---|---|---|---|---|---|
| ensemble | **0.576828** | **0.001913** | 1.56 | 0.001229 | 0.000558 | 0.000000 | **0** | 0.004829 |
| xgb / baseline_xgb | 0.569964 | 0.001849 | 1.51 | 0.001230 | 0.000875 | 0.000000 | **0** | 0.005215 |
| lgb | 0.566467 | 0.001743 | 1.42 | 0.001230 | 0.000808 | 0.000000 | **0** | 0.003992 |
| cb | 0.567121 | 0.001842 | 1.50 | 0.001236 | 0.000062 | 0.000000 | **0** | 0.004636 |

And the hardest combined slice — held-out entities **and** eval-late (6,340,753 rows / 7,838 fraud):

| model | ROC-AUC | PR-AUC | recall @ 0.1% alert volume |
|---|---|---|---|
| ensemble | **0.579387** | **0.001945** | 0.004721 |
| xgb / baseline_xgb | 0.572484 | 0.001936 | 0.004721 |
| lgb | 0.569518 | 0.001780 | 0.004593 |
| cb | 0.568731 | 0.001856 | 0.004210 |

**Findings.**

* **The ranking of ensemble vs xgb REVERSES on the temporal split.** On the full benchmark and on
  held-out entities, xgb leads (NR-05). On eval-late the ensemble leads (0.576828 vs 0.569964), and
  on the combined entity+temporal slice it leads by the widest margin (0.579387 vs 0.572484).
* **This reversal is reported, not resolved.** It is measured on synthetic data whose §9 limitations
  are known to distort exactly the features that carry temporal signal, and the margin (~0.007
  ROC-AUC on 6.3M rows) is not accompanied by any significance test. It is therefore recorded as an
  **observation requiring explanation**, and **explicitly not** as grounds to replace the ensemble
  or to overturn NR-05. Resolving it requires real-data evidence this phase is not permitted to seek.
* **The locked threshold raises ZERO alerts on the entire eval-late period** — not one of
  12,650,580 rows, and not one of 15,531 fraud rows, clears `0.7847116291110687`. Whatever the
  ranking argument, at the locked operating point the system is silent for the entire late period.

---

## 14. Chunking-equivalence results (§11)

Chunking probe over the same 50,000,000 scored rows, single pass vs 2,000 chunks of 25,000 rows,
**explicit FP tolerance 0**.

| measure | value |
|---|---|
| single-pass alerts | 193 |
| chunked alerts (2,000 × 25,000) | 193 |
| absolute decision delta | **0** |
| delta rate | **0.0** |
| identical | **YES** |

**Result: chunking is exactly equivalent.** The 2,000-chunk decomposition reproduces the single-pass
decision vector with zero differing rows — not approximately, not within tolerance, but exactly.

This confirms the §5 static argument empirically. Because every operation is stateless across
batches (no per-chunk refactorisation, no per-chunk historical recompute, no cross-chunk model
state), the batch size is a pure throughput knob with **no semantic effect**. The batch size of
250,000 used for the §6 baseline may therefore be changed freely without altering any result.

---

## 15. Failure and recovery results (§12)

Harness: `scripts/large_scale_faults.py`. Every fault was injected into **copies** of benchmark
partitions under `misc/reports/phase3_faults/`. The validated benchmark was opened read-only; its
combined hash was captured before and after and is recorded **unmodified**.

Three partitions (`2016-01`, `2016-02`, `2016-03`) = 903,612 rows / 1,142 fraud were the base case.
Each row re-runs the **real** runner via `--bench <scratch>`.

| # | fault | exit code | rows scored | outcome |
|---|---|---|---|---|
| 1 | none (baseline) | 0 | 903,612 | **AS EXPECTED** |
| 2 | **one partition deleted** | **0** | **602,408** | **SILENT — DEFECT** |
| 3 | one partition truncated to 4 KiB | 1 | — | **AS EXPECTED** (fails loudly) |
| 4 | **one partition duplicated** under a new `year_month` | **0** | **1,204,816** | **SILENT — DEFECT** |
| 5 | completed partition set re-run (pass 1) | 0 | 903,612 | **AS EXPECTED** |
| 6 | completed partition set re-run (pass 2) | 0 | 903,612 | **AS EXPECTED — bit-identical to pass 1** |
| 7 | recovery after fault (full restore) | 0 | 903,612 | **AS EXPECTED — restored to baseline rows and ROC-AUC** |

**Findings.**

1. **Corruption is detected loudly.** A truncated partition aborts with
   `pyarrow.lib.ArrowInvalid: Parquet magic bytes not found in footer` and exit code 1. No partial
   or wrong result is emitted.
2. **A missing partition is NOT detected.** Deleting one of three partitions still exits **0** and
   emits a complete, well-formed report — over **602,408 rows instead of 903,612**. The runner
   discovers partitions with `rglob("*.parquet")` and **never asserts against the manifest's
   partition list or expected row count**. A silently truncated evaluation would be reported as a
   successful run.
3. **A duplicated partition is NOT detected.** Injecting one partition's rows under a new
   `year_month` exits **0** with **1,204,816 rows** — 333,204 duplicate rows scored as fresh data.
   Measured effect on the base case: ROC-AUC moved 0.577918 → 0.580575 and fraud count 1,142 →
   1,504. The report is structurally valid and numerically *wrong*.
4. **Re-running completed work is safe and idempotent.** Two consecutive passes over the same
   partitions produced identical rows, fraud counts, ROC-AUC and full confusion matrices.
5. **Recovery works.** After restoring the pristine partitions, the run returned exactly to baseline
   rows and ROC-AUC.
**Interpretation.** Defects 2 and 3 are the same root cause: **there is no completeness assertion
anywhere in the path.** Integrity checking is delegated entirely to Parquet's footer validation,
which catches byte-level corruption but is blind to set-level problems. This is an **operational
reliability finding about the measurement pipeline**, and it is a direct argument for adding a
manifest-completeness assertion to any future large-scale runner.

**Not claimed:** these are data-pipeline integrity gaps in a Phase 3 measurement script and its
fault harness. They are **not** asserted to exist in the production serving path, which was not
fault-injected in this phase, and they are **not** security findings — that assessment is Phase 4.

---

## 16. Reproducibility results (§13)

The full 50,000,000-row baseline was executed **twice** on the same host, same code, same
artifacts, same benchmark, same order.

| output | run 0 | run 1 | identical |
|---|---|---|---|
| rows processed | 50,000,000 | 50,000,000 | **YES** |
| partitions processed | 166 | 166 | **YES** |
| fraud rows | 61,109 | 61,109 | **YES** |
| fraud prevalence | 0.00122218 | 0.00122218 | **YES** |
| ROC-AUC | 0.5731960569465072 | 0.5731960569465072 | **YES** |
| PR-AUC | 0.0018011156833651009 | 0.0018011156833651009 | **YES** |
| Brier | 0.0012478301068767905 | 0.0012478301068767905 | **YES** |
| confusion at threshold | tp 0, fp 193, fn 61,109, tn 49,938,698 | identical | **YES** |
| precision / recall / F1 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | **YES** |
| alert rate | 3.86e-06 | 3.86e-06 | **YES** |
| member ROC-AUC | xgb 0.5742367399519009, lgb 0.5663669886096442, cb 0.5584804183475692 | identical | **YES** |
| member PR-AUC | 0.0018893237983878446 / 0.0017328020402114482 / 0.0016638536708929233 | identical | **YES** |
| member precision | 0.0 / 0.0 / 0.0007042253521126761 | identical | **YES** |
| non-finite values | 0 | 0 | **YES** |
| calibrator loaded | false | false | **YES** |

**Classification: `BIT-IDENTICAL` on every scientific output.**

Only wall-clock quantities differ, and they differ for the expected reason — other services were
active on the host during run 1:

| wall-clock quantity | run 0 | run 1 |
|---|---|---|
| wall seconds | 500.1 | 563.9 |
| rows per second | 99,971 | 88,668 |
| p95 / p99 latency (ms) | 10.23 / 11.40 | 18.32 / 20.47 |

Peak RSS was recorded in run 1 only (1,785.1 MB); run 0's value was lost because `psutil` was not
installed when it ran, and is reported as **not recorded** rather than back-filled.

Supporting idempotence evidence: the fault harness re-ran the same three partitions twice and got
bit-identical rows, fraud counts, ROC-AUC and confusion matrix (§15 rows 5 and 6).

**`REPRODUCIBILITY FAILURE` is therefore not the outcome.** The large-scale evaluation is
deterministic and re-executable to the last digit of every reported metric.

---

## 17. Resource-scaling results (§14)

**Measured cells only.** Unmeasured combinations are marked `NOT MEASURED` or `BLOCKED`; nothing is
extrapolated into a table as if it were observed.

| resource | 1M | 2M | 5M | 10M | 25M | 50M |
|---|---|---|---|---|---|---|
| training fit time (s) | 104.0 | 173.0 | 435.3 | 922.0 | `BLOCKED BY COMPUTE` | `BLOCKED BY COMPUTE` |
| training peak RSS (MB) | 1,304 | 1,677 | 2,464 | 3,294 | `BLOCKED BY COMPUTE` | `BLOCKED BY COMPUTE` |
| training artifacts (B) | 1,982,203 | 2,125,490 | 2,271,509 | 2,327,329 | `BLOCKED BY COMPUTE` | `BLOCKED BY COMPUTE` |
| training CPU utilisation | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | — | — |
| inference peak RSS (MB) | NOT MEASURED | NOT MEASURED | **1,144.1** (sampler, process tree) | NOT MEASURED | NOT MEASURED | **1,785.1** |
| inference wall (s) | NOT MEASURED | NOT MEASURED | 48.2 sampler / 35.2 runner | NOT MEASURED | NOT MEASURED | 500.1 / 563.9 |
| inference throughput (rows/s) | NOT MEASURED | NOT MEASURED | 103,794 / 142,078 | NOT MEASURED | NOT MEASURED | 99,971 / 88,668 |
| CPU utilisation, mean % of one logical core | NOT MEASURED | NOT MEASURED | **309.2** (p50 390.6, p95 465.6, max 482.6) | NOT MEASURED | NOT MEASURED | NOT MEASURED |

Cell provenance: training cells from `misc/reports/phase3_train_ladder.jsonl`; the 5M inference
and CPU cells from the instrumented pass of §6.4 (`misc/reports/phase3_cpu.json` and
`phase3_cpu_run.json`); the 50M inference cells from the two §6 runs. The two 5M throughput and
wall figures are sampler-observed (whole process, interpreter start and model load included) /
runner-internal, and the 5M prefix is **not** a directly comparable throughput point for the 50M
figure — different partitions, different host load.

**Storage behaviour.** Benchmark on disk 3.7 GB; bytes read per full pass 3,916,485,262 (3.65 GiB)
— a full 50M pass reads the artifact slightly more than once. **Temporary storage written by the
runner: 0 bytes.** The inference pipeline is fully streaming: partitions are read batch-by-batch,
scored and released, with no spill files, no intermediate parquet and no materialised feature
matrix. The training ladder is *not* streaming and does materialise a full float32 matrix (see §7).

**Memory behaviour.** Inference peak RSS is **1,785.1 MB for all 50,000,000 rows** and is bounded by
batch size, not dataset size. The runner pre-allocates score arrays, but the resident working set
is dominated by the model artifacts plus one 250,000-row batch. This is the key scaling property:
**inference memory is O(batch), not O(dataset)**, which is why 50M rows fit in 1.8 GB on a 15.6 GB
host.

---

## 18. Known benchmark limitations in force

Carried forward from `docs/evaluation/SYNTHETIC_50M_VALIDATION_REPORT.md` and binding on every
number in this report:

1. **Historical fraud-rate features are largely inert.**
2. **MCC / channel dependence is not faithfully reproduced** — and, per §3.1, the corresponding
   code path in `map_raw_to_native` is unreachable, which is a live defect independent of the data.
3. **Two native features are constant** in the benchmark.
4. **Entity activity is substantially more uniform than the real reference.**
5. **The real-vs-synthetic distinguishability experiment is confounded.**

Any of these alone could depress ranking metrics. Together they are sufficient to explain
ROC-AUC around 0.573, and they mean **no performance figure in §8, §12 or §13 can be read as an
estimate of banking performance** — in either direction.

Additionally, specific to this phase:

6. **Compute ceiling.** 15.6 GB RAM, 4 GB VRAM (unused), no Docker, no cluster, single host. This
   bounds §7 to 10M rows.
7. **CPU utilisation scope.** Sampled on a shorter instrumented pass, not the full 50M pass (§6.4).
8. **No significance testing.** §8, §12 and §13 report point estimates only. Splits are large
   (6.3M to 50M rows), but no confidence interval, bootstrap or paired test accompanies any
   comparison — including the ensemble-vs-xgb reversal in §13.

---

## 19. Interpretation

**What this phase establishes.**

The current PS-14 prototype **can process the validated 50M benchmark reliably and reproducibly on
a single consumer laptop CPU**. It completed two independent full passes of 50,000,000 rows across
166 partitions in 500.1 s and 563.9 s (about 89k to 100k rows/s), using 1.79 GB of RAM, writing
zero temporary storage, failing zero batches, and returning **bit-identical** results across runs.
The model artifacts load once, score a 250,000-row batch in roughly 4 to 5 ms at p50, and inference
consumes 93% of wall time — so the pipeline is **inference-bound by design** and every other stage
is incidental. Chunking is exactly equivalent (§14), so batch size is a free throughput knob.
Training scales cleanly from 1M to 10M rows (§7) and is bounded by RAM, not by algorithmic
breakdown. **The architecture remains computationally practical at 50M scale.**

**What this phase does not establish, and cannot.**

Nothing about real-world effectiveness. The near-chance ranking result (ROC-AUC 0.573) and the
zero-recall operating point are **not** evidence that PS-14 fails to detect fraud in banking data —
they are evidence that the locked threshold does not operate on a benchmark whose features are known
to be inert, uniform and partly constant (§18). Equally, nothing here would license a favourable
reading had the numbers come out strong. **The asymmetry is the point: this phase produces
scalability evidence, not effectiveness evidence.**

**The genuinely useful outputs** are the three findings that transfer regardless of dataset:

* the **unreachable branch** in `map_raw_to_native` (§3.1), which makes `engine.predict()` and the
  documented native pass-through disagree by up to 0.126 in score — a correctness defect in
  production code, independent of this benchmark;
* the **zero-recall locked operating point** (§8, §13), which fires zero alerts across an entire
  12.6M-row late period and zero of 61,109 fraud rows overall — if the real-data score distribution
  is anywhere near this one, the system is silent;
* the **absence of any completeness assertion** (§15), which lets a missing or duplicated partition
  produce a well-formed, wrong, exit-0 result.

**On the ensemble-versus-xgb question.** NR-05 is confirmed at 50M scale on the full benchmark and
on held-out entities. It reverses on the temporal split (§13), on synthetic data whose limitations
are known to distort temporal signal, with no significance test, on margins of roughly 0.007
ROC-AUC. The honest reading is that **the benchmark cannot settle the question** — which is
precisely why settling it belongs in a phase with real data, not here. The production
0.34/0.33/0.33 mean fusion is preserved unchanged.

---

## 20. Negative findings

Recorded deliberately, because a scalability phase that only lists successes misleads.

1. **The locked threshold is non-functional at this scale.** Recall **0.000000** on the full
   benchmark, on held-out entities, on seen entities, and on every temporal slice. 193 alerts in
   50,000,000 rows; **zero** alerts across the entire 2013-01 to 2016-05 eval-late window.
2. **Even with the operating point freed**, recall at a top-0.1% alert volume is only 0.0034
   (ensemble) and 0.0052 (xgb) against a 0.0012 random baseline — 2.8x and 4.2x lift, not
   operationally useful.
3. **The mean-fusion ensemble is below its best member** on the full benchmark and on held-out
   entities. NR-05 holds at scale.
4. **Two fault classes are silent.** Missing partition exits 0 over 602,408 of 903,612 rows.
   Duplicated partition exits 0 over 1,204,816 rows with ROC-AUC shifted 0.577918 to 0.580575.
5. **Training at 25M and 50M rows is `BLOCKED BY COMPUTE`** on this host. No completion time is
   claimed for either.
6. **`map_raw_to_native` branch 2 is unreachable** (§3.1) — a live production-code defect,
   recorded not fixed, because §3 forbids model changes in this phase.
7. **The ensemble-vs-xgb ranking is not stable across splits** (§13): xgb leads on all-rows and
   entity-disjoint, the ensemble leads on temporal-late and entity-plus-temporal. Unresolved by
   design.
8. **The benchmark cannot support a model-performance comparison** in the strong sense its prior
   report already flagged: near-inert historical features, uniform entity activity, constant
   features and broken channel dependence are jointly sufficient to produce a near-chance result
   from a model that performs well on the real reference.
9. **CPU utilisation is not measured for the full 50M pass** — only over a shorter instrumented run
   (§6.4). Stated as a scope limit rather than papered over.
10. **No GPU acceleration was attempted.** The RTX 3050's 4 GB VRAM was not used; this phase
    measures the CPU path as production actually runs it.

---

## 21. Evidence classification

Using the project's existing evidence vocabulary. Synthetic evidence is **not** upgraded.

| claim | class |
|---|---|
| 50M inference completes: 89k-100k rows/s, 166 partitions, 500-564 s | `DEMONSTRATED` / `SELF-TESTED` |
| inference peak RAM 1,785.1 MB, bounded by batch not dataset | `DEMONSTRATED` / `SELF-TESTED` |
| inference is 92.9% of wall time; all other stages under 7% | `DEMONSTRATED` / `SELF-TESTED` |
| zero temporary storage written; fully streaming | `DEMONSTRATED` / `SELF-TESTED` |
| batch latency p50/p95/p99 measured at 250,000-row batches | `DEMONSTRATED` / `SELF-TESTED` |
| training scales near-linearly 1M to 10M at fixed hyperparameters | `DEMONSTRATED` / `SELF-TESTED` |
| training at 25M and 50M rows | `BLOCKED` — compute ceiling, arithmetic recorded |
| full 50M baseline is bit-identical across two runs | `DEMONSTRATED` / `SELF-TESTED` |
| chunking exactly equivalent (delta 0 across 2,000 chunks) | `DEMONSTRATED` / `SELF-TESTED` |
| missing / duplicated partitions pass silently with exit 0 | `DEMONSTRATED` / `SELF-TESTED` (operational defect) |
| corrupt partition fails loudly with exit 1 | `DEMONSTRATED` / `SELF-TESTED` |
| `map_raw_to_native` branch 2 unreachable; predict() disagrees with native pass-through | `DEMONSTRATED` / `SELF-TESTED` (code defect, recorded not fixed) |
| locked threshold recall 0.000000 at 50M | `SELF-TESTED — SYNTHETIC 50M` |
| ensemble ROC-AUC 0.573196; xgb 0.574237; NR-05 confirmed | `SELF-TESTED — SYNTHETIC 50M` |
| entity-disjoint and temporal slice behaviour | `SELF-TESTED — SYNTHETIC 50M` |
| ensemble-vs-xgb ranking reversal on the temporal split | `SELF-TESTED — SYNTHETIC 50M — UNRESOLVED` |
| CPU utilisation on the 5M instrumented pass: mean 309.2% of one logical core (p50 390.6 / p95 465.6) | `DEMONSTRATED` / `SELF-TESTED` |
| CPU utilisation for the full 50M pass | `NOT MEASURED` — sampled on a shorter instrumented pass |
| real-world fraud-detection effectiveness | `NOT ESTABLISHED` |
| institutional generalisation | `NOT ESTABLISHED` |
| production fraud-loss reduction | `NOT ESTABLISHED` |
| superiority on real banking data | `NOT ESTABLISHED` |
| regulatory readiness | `NOT ESTABLISHED` |
| independent external validation | `NOT ESTABLISHED` |
| institutional validation | `NOT ESTABLISHED` |
| security / penetration-tested / vulnerability-free | `NOT ESTABLISHED` — Phase 4, dedicated workstream |
| whether the locked threshold is correct on real data | `NOT APPLICABLE` — cannot be settled on this benchmark |

---

## 22. Final decision

# `LARGE-SCALE BASELINE ESTABLISHED WITH LIMITATIONS`

**Established, measured and reproducible:**

* The current production/native prototype processes the **full 50,000,000-row validated benchmark**
  on a single CPU: two independent passes, **bit-identical** on every scientific output.
* Throughput **89k to 100k rows/s**; inference peak RAM **1,785.1 MB**; **zero** temporary storage;
  **zero** failed batches; **zero** non-finite values.
* End-to-end stage decomposition measured separately — inference 92.9% of wall, all else under 7%.
* Batch latency p50/p95/p99 = 4.13 / 10.23 / 11.40 ms at batch 250,000.
* Training scales near-linearly **1M to 10M** at fixed canonical hyperparameters; artifacts and peak
  RSS recorded per rung.
* **Chunking is exactly equivalent** — decision delta **0** across 2,000 chunks of 25,000 rows.
* Failure behaviour characterised: corruption detected loudly, **missing and duplicated partitions
  not detected**, re-runs idempotent, recovery exact.
* The model was evaluated **exactly as it exists**. No optimisation, no architecture change, no
  threshold change, no new dataset, no production artifact modified. **NR-05 preserved.**

**Limitations attaching to that decision:**

1. **Training at 25M and 50M rows is `BLOCKED BY COMPUTE`** — a measured 1M to 10M ladder is
   reported instead, with the blocking arithmetic shown and no fabricated completion.
2. **Two silent data-integrity defects** (§15) — missing and duplicated partitions produce
   well-formed, numerically wrong, exit-0 results.
3. **Recall at the locked threshold is 0.000000** on every split, including a 12.6M-row period with
   **zero** alerts — a finding about this benchmark and threshold pairing that cannot be generalised.
4. **CPU utilisation for the full 50M pass is `NOT MEASURED`**, and no significance test accompanies
   any §8, §12 or §13 comparison.
5. Every performance number is `SYNTHETIC 50M — SCALE EXPERIMENT`. Real-world effectiveness,
   institutional generalisation and regulatory readiness remain **`NOT ESTABLISHED`**.

**Next phases, per the plan:** Phase 4 — Strix Security Validation; then Phase 5 — Prototype
Optimization. No optimisation was performed in this phase, and the silent-integrity defects and the
unreachable-branch defect in §20 are the concrete candidates it inherits.