
---

## Phase 10C — Production Contract Remediation (2026-09-07)

### Finding 1: Phase 10B user-key bug (corrected)
Phase 10B used merchant names as user identity in its parity harness. The correct user key is the IBM `User` column (`tr_user_id`). This caused 8 features to appear divergent when they were actually identical. After correction, 45/48 features achieve production parity.

### Finding 2: 3 fraud-rate features genuinely unavailable at decision time
`user_fraud_rate`, `merch_fraud_rate`, `city_fraud_rate` require confirmed fraud labels before each decision. Production has no such pipeline:
- `VerificationOutcome` labels are retrospective (after investigator review)
- Risk engine uses `score >= 70` proxy (forbidden — model predictions ≠ confirmed labels)
- No in-repo pipeline feeds confirmed labels to the risk engine before new decisions

### Finding 3: Existing 48-feature model is production-incompatible
PATH_B triggered: new 45-feature production-native candidate required (drop 3 fraud-rate features, retrain).

### Finding 4: Temporal causality violation in production trackers
`UserVelocityTracker` and `EntityFraudRateTracker` accumulate in ingestion order with no timestamp cutoff. A backdated event ingested first changes earlier rows' features. Offline sorts by timestamp (correct). Production must be fixed to use ts-aware state.

### Finding 5: Privacy contract PASS
Merchant identity is forwarded as opaque hash code; no raw strings leak to model features.

### Finding 6: Score parity with 45 non-fraud-rate features
max |score delta| = 1.937e-05, 0 decision disagreements across 300k sampled rows. The model weights are identical; only 3 features differ.

---

## Phase 11A — Integrity Reconciliation (2026-09-07)

### Finding 1: Machine summary reporting bug (Contradiction A resolved)
Phase 11 machine-readable summary reported TEST metrics (recall=0.1175, FPR=0.01101) as "RECALL_AT_SELECTED_THRESHOLD" and "FPR_AT_SELECTED_THRESHOLD". The correct VALIDATION metrics are: recall=0.8127, FPR=0.00998. This is a reporting bug in the summary section, not a computation bug.

### Finding 2: 2017 forward-validation recall collapse is GENUINE (Contradiction B confirmed)
- 2016 recall: 0.8516 (3,048/3,579 fraud caught)
- 2017 recall: 0.2667 (68/255 fraud caught)
- Mean fraud score shift: -0.358 (2017 fraud scores are substantially lower)
- 187 of 255 2017 fraud cases score below the operating threshold
- This is CASE_2_VALID_DEGRADATION — the model does not generalize to 2017 fraud patterns
- TEMPORAL_ROBUSTNESS = FAIL
- CERTIFICATION_STATUS = BLOCKED

### Finding 3: Metrics reconcile perfectly
Locked threshold metrics match within tolerance (max delta 2.8e-05). Frontier recomputation matches. Reproducibility is bit-identical (0.00e+00 delta).

### Finding 4: Root cause of 2017 degradation
2017 fraud cases have fundamentally different score distributions from 2016 fraud. The model was trained on pre-2016 data and validated on 2016-2017 combined. The 2017 fraud population represents a distribution shift that the model cannot handle. This is a genuine generalization failure, not a harness bug.

---

## Phase 11B — Temporal Drift Forensics (2026-09-07)

### Root Cause: CHANNEL COMPOSITION SHIFT (Concept Drift)

The 2016→2017 recall collapse is caused by a fundamental change in fraud channel composition:

- **2016 fraud:** 86% online (3,073/3,579), 8% chip, 6% swipe
- **2017 fraud:** 97% chip (248/255), 3% swipe, **0% online**

The model learned online fraud indicators (is_online AUC=0.867 in 2016) that are useless against chip fraud (is_online AUC=0.438 in 2017 — near random).

### Score Distribution Evidence
- 2016 fraud mean score: 0.904 (well above threshold 0.793)
- 2017 fraud mean score: 0.546 (below threshold)
- Score shift: -0.358 (PROVEN)

### Concept Drift Evidence
- 22 of 45 features show degraded AUC in 2017 vs 2016
- Top 3 degraded features: amt_x_online, is_online, is_online_or_no_state (all -0.43 AUC delta)
- These are the features that discriminated online fraud in 2016

### Channel-Level Evidence
- Online fraud recall: 2016=93.7%, 2017=0% (0 online fraud cases in 2017)
- Chip fraud recall: 2016=30.8%, 2017=26.2% (similar — chip fraud is hard in both years)
- The overall 2016 recall (85%) was carried by online fraud detection

### Coverage Evidence
- 808 novel merchants in 2017 (not in training)
- 31 novel users in 2017
- Coverage shift is a secondary factor, not the primary cause

### Diagnosis
DOMINANT_DRIFT_TYPE = CONCEPT_DRIFT (channel composition shift)
CONFIDENCE = STRONGLY_SUPPORTED

---

## Phase 12 — Chip-Robustness Experiment (2026-09-07)

### Critical Finding: ZERO Chip Fraud in Pre-2013 Training

The pre-2013 training data contains **0 chip fraud cases**. All training fraud was online (11,478) or swipe (5,534). The model literally never saw chip fraud during training.

### Results

| Model | 2016 Rec | 2017 Rec | Chip 2017 | Status |
|-------|----------|----------|-----------|--------|
| P11 baseline | 79.7% | 33.7% | 33.9% | NOT_WINNER |
| D (no online) | 80.6% | 51.0% | 51.2% | POTENTIAL_WINNER |

### Winner: Candidate D (no online features)

- Removes 3 online-dependent features: is_online, amt_x_online, is_online_or_no_state
- 2016 recall slightly improved (80.6% vs 79.7%)
- 2017 recall massively improved (51.0% vs 33.7%, +17.3pp)
- Chip recall 2017: 51.2% (vs 33.9%)
- FPR acceptable: 1.1%
- ROBUST_WIN = TRUE

### Why Other Candidates Failed

- Channel-balanced (B): hurt 2017 recall (19.6% vs 33.7%)
- Chip hard-neg (C): identical to baseline (0 hard negatives — no chip fraud in training to mine)
- Balanced combo (E): same as B (channel weighting dominates)

### Implication

The simplest fix — removing online-dependent features — is the most effective. The model's reliance on online-fraud signal was actively harmful when fraud shifted to chip. Without those features, the model falls back on channel-neutral features (amount, timing, MCC) that generalize better.

---

## Phase 13 — Pre-2016 Chip-Fraud Support Experiment (2026-09-07)

### Data State: CASE C (301 pre-2016 chip fraud cases exist)

But the training set (<=2013) contains **0 chip fraud**. The 301 chip fraud cases are in 2014-2015 (validation/forward-validation periods). The model cannot learn chip-fraud patterns from training data that contains none.

### Results

| Candidate | Chip# | 2016 Rec | 2017 Rec | Chip17 | FPR17 |
|-----------|------:|---------:|---------:|-------:|------:|
| C1 (channel-neutral) | 0 | 0.806 | 0.510 | 0.512 | 0.011 |
| C2 (chip-supported) | 0 | 0.806 | 0.510 | 0.512 | 0.011 |
| C3 (channel-balanced) | 0 | 0.797 | 0.443 | 0.456 | 0.010 |
| C4 (chip-hardneg) | 0 | 0.806 | 0.510 | 0.512 | 0.011 |

### Key Finding

C2/C4 are identical to C1 because the training set (<=2013) has 0 chip fraud — weighting/hard-neg mining has nothing to work with. C3 (channel-balanced) actually HURTS performance.

### Conclusion

**NO_ROBUST_WIN**. The pre-2016 chip fraud support cannot repair the 2016→2017 robustness failure because:
1. Training window (<2014) has zero chip fraud
2. Chip fraud only appears in 2014+ (after training cutoff)
3. Channel-neutral features are the only source of chip-fraud generalization
4. The 51% chip recall ceiling appears to be the limit of channel-neutral transfer
