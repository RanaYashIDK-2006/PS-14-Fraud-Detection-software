# Phase 11B — Temporal Drift Forensic Report

**Date:** 2026-09-07T15:28:52Z
**Candidate:** P11_45feat (45 features)
**Locked threshold:** 0.793435

## Key Findings

- 2016 recall: **0.8516** (3048/3579 fraud caught)
- 2017 recall: **0.2667** (68/255 fraud caught)
- 2017 missed fraud: **187** / 255

## Score Shift

- Mean fraud score 2016: 0.90381
- Mean fraud score 2017: 0.54583
- Shift: -0.357979

## Diagnosis

- Dominant drift type: **CONCEPT_DRIFT**
- Confidence: **STRONGLY_SUPPORTED**

- CONCEPT_DRIFT: 22 features show degraded AUC in 2017 vs 2016 (strength: STRONGLY_SUPPORTED)
- SCORE_DISTRIBUTION_SHIFT: Mean fraud score shifted by -0.357979 (strength: PROVEN)
- COVERAGE_SHIFT: 808 novel merchants, 31 novel users in 2017 (strength: SUPPORTED)

## Temporal Robustness

TEMPORAL_ROBUSTNESS = **FAIL** (confirmed from Phase 11A)
CERTIFICATION_STATUS = **BLOCKED**