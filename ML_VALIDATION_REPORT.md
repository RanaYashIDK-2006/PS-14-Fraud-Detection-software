# ML Validation Report

**Generated:** 2026-08-24 17:24 UTC
**Seed:** 42
**Training Data:** data\transactions.csv

## Test Results

- Calibration validation: PASS
- Adversarial testing: PASS
- Leakage structural: PASS
- Temporal correctness: PASS

## Dataset

- **Train:** 6,859 samples
- **Validation:** 1,470 samples
- **Test:** 1,470 samples
- **Test Fraud Rate:** 0.0306

## In-Domain Metrics

| Model | PR-AUC | ROC-AUC | Recall@1%FPR | Precision | F1 |
|-------|--------|---------|--------------|-----------|-----|
| logistic_regression | 0.9552 | 0.9958 | 0.7778 | 1.0000 | 0.7838 |
| fused_ensemble (stacker) | 0.9523 | 0.9913 | 0.9333 | 1.0000 | 0.8000 |
| xgboost | 0.9435 | 0.9959 | 0.9556 | 1.0000 | 0.7838 |
| random_forest | 0.8375 | 0.9732 | 0.8222 | 1.0000 | 0.7324 |
| isolation_forest | 0.5074 | 0.8183 | 0.5111 | 0.5882 | 0.5063 |
| random_baseline | 0.0169 | 0.5000 | 0.0100 | 0.0169 | 0.0333 |

## Calibration

- [x] Calibrator validated on real validation data

## Adversarial Results

- [x] Adversarial cases tested against real model

## Limitations

- Model memorizes synthetic archetypes
- Cross-domain generalization limited
- Adversarial cases may score low (honest limitation)