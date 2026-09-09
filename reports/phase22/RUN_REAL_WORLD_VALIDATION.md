# How to Run Real-World Validation

## Prerequisites

1. A legitimate real-world transaction dataset (CSV)
2. A metadata JSON file describing the dataset
3. Python 3.10+ with project dependencies

## Quick Start

### Preparation Mode (no data)

```bash
python -m phase22.run_real_world_validation --mode prepare
```

Expected output:
```
RUNNER_STATUS=READY
REAL_WORLD_DATA_AVAILABLE=FALSE
REAL_WORLD_VALIDATION=BLOCKED
PROMOTION=BLOCKED
```

### Dry Run

```bash
python -m phase22.run_real_world_validation --dry-run
```

Verifies models load, configurations load, feature contracts load, threshold manifests load.

### Full Validation (with data)

```bash
python -m phase22.run_real_world_validation \
    --data /approved/path/transactions.csv \
    --metadata /approved/path/metadata.json \
    --output reports/phase22/runtime/
```

## Metadata File Format

Create a `metadata.json` file:

```json
{
  "dataset_name": "my_bank_transactions",
  "source": "Bank XYZ partnership",
  "owner": "Bank XYZ",
  "real_world_status": "REAL_WORLD",
  "authorization_status": "AUTHORIZED",
  "label_definition": "Confirmed fraud by analyst investigation",
  "label_source": "Bank XYZ fraud team",
  "transaction_time_field": "transaction_timestamp",
  "label_available_time_field": "label_timestamp",
  "channel_field": "channel",
  "merchant_id_field": "merchant_id",
  "user_id_field": "customer_id"
}
```

**Critical**: `real_world_status` must be `"REAL_WORLD"` and `authorization_status` must be `"AUTHORIZED"` for validation to proceed.

## What the Runner Validates

1. **Firewall** — Model artifacts unchanged, final test not accessed
2. **Metadata** — Provenance, authorization, label governance
3. **Schema** — Required fields present and compatible
4. **Labels** — Binary definition, no contradictions, timing established
5. **Features** — P20 45-feature reconstruction possible
6. **Models** — Frozen models load correctly
7. **Thresholds** — Locked thresholds, no holdout influence
8. **Temporal** — Chronological evaluation windows
9. **Metrics** — ROC-AUC, PR-AUC, recall, FPR, precision
10. **Uncertainty** — Wilson score confidence intervals
11. **Robustness** — Temporal, channel, entity analysis
12. **Comparison** — Paired P20 vs E_hardneg
13. **Promotion** — All gates evaluated

## Output

Results are written to:
```
reports/phase22/runtime/<dataset_hash>/
    dataset_manifest.json
    gate_results.json
    validation_result.json
    machine_summary.json
```

## Failure Modes

| Failure | Reason | Action |
|---------|--------|--------|
| VALIDATION=BLOCKED | Missing metadata | Provide metadata.json |
| VALIDATION=BLOCKED | LABEL_LATENCY_UNVERIFIED | Add label timing fields |
| VALIDATION=BLOCKED | SCHEMA_INCOMPATIBLE | Map fields to PS-14 contract |
| VALIDATION=BLOCKED | FEATURE_RECONSTRUCTION_FAILED | Ensure feature derivation possible |
| METHODOLOGY_FAILURE | Protected artifact changed | Restore from backup |

## Current Status

```
RUNNER_STATUS=READY
REAL_WORLD_DATA_AVAILABLE=FALSE
NEXT_ACTION=ACQUIRE_LEGITIMATE_REAL_WORLD_DATA
```

The harness is ready. The next milestone is acquiring legitimate real-world data.
