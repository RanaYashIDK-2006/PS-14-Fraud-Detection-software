# Model Card

**Generated:** 2026-08-24 17:24 UTC
**Model Version:** seed42-transactions
**Feature Version:** v1

## Intended Use

- Fraud detection for financial transactions
- Risk scoring with calibrated probabilities
- Human-in-the-loop verification for medium/high risk

## Prohibited Use

- Autonomous decision-making without human oversight
- PII-based discrimination

## Training Data

- **Source:** data\transactions.csv
- **Seed:** 42

## Features ({len(features)})

- `amount_ratio`
- `txn_freq_last_24h`
- `txn_time_unusual`
- `new_device_flag`
- `unusual_location_flag`
- `unusual_recipient_flag`
- `failed_auth_count_24h`
- `days_since_last_similar_txn`
- `gradual_escalation_score`
- `known_device_count`
- `account_tenure_days`
- `hour_of_day`
- `is_weekend`
- `shared_device_accounts`
- `shared_recipient_accounts`
- `mule_ring_score`

## Thresholds

| Model | F1 Threshold | 1% FPR Threshold |
|-------|--------------|------------------|
| logistic_regression | 0.5006 | 0.1885 |
| random_forest | 0.4069 | 0.1736 |
| xgboost | 0.5182 | 0.0367 |
| isolation_forest | 0.0693 | 0.0429 |

## Known Failure Modes

- OOD fraud archetypes not in training data
- Cross-domain generalization limited
- Subtle multi-feature fraud may evade detection