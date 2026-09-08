# Phase 11 — 45-Feature Production-Native Candidate Report

**Date:** 2026-09-07T15:12:08Z
**Candidate:** P11_45feat
**Feature count:** 45 (dropped ['user_fraud_rate', 'merch_fraud_rate', 'city_fraud_rate'])
**Model hash:** 395467fcab7688c2c8d917ee1e934ef69e4fd85d1843a4cb320dd4a50e0dae3e
**Locked threshold:** 0.793435

## Key Results

- Validation AUC: 0.9910
- Validation PR-AUC: 0.9414
- Validation recall @ locked: 0.8127
- Validation FPR @ locked: 0.00998

- **Test recall (untouched, locked): 0.1175**
- **Test FPR: 0.01101**
- Test precision: 0.5663
- Test AUC: 0.9448
- Test PR-AUC: 0.5548
- Test TP=538 FP=412 FN=4040 TN=37006

## Production Parity

- max feature delta: 1.07e-04
- score max delta: 0.00e+00
- decision disagreements: 0

## Forward Validation

- 2016: recall=0.8516 FPR=0.00951 precision=0.9501 alerts/1k=157.2
- 2017: recall=0.2667 FPR=0.01045 precision=0.2753 alerts/1k=14.2

## Decision

CERTIFICATION_STATUS = PENDING
FINAL_TEST_AUTHORIZED = FALSE

All development gates passed. Candidate requires independent certification.