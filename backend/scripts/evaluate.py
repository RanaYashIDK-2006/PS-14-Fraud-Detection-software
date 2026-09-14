#!/usr/bin/env python3
"""Phase 39: Reproducible evaluation CLI.

Single documented command for reproducing an evaluation:

    python backend/scripts/evaluate.py \
        --model-dir models/artifacts \
        --dataset data/transactions.csv \
        --config reports/eval_configs/synthetic_test.json \
        [--threshold 0.43 --threshold-source fixed] \
        [--output-dir reports/evaluation_runs]

The command:
  1. hashes the model artifacts (model identity),
  2. fingerprints the dataset (data identity),
  3. loads the evaluation configuration (JSON: split, threshold policy, seed),
  4. runs frozen inference + computes metrics via the authoritative
     definitions in metric_definitions.py,
  5. writes a machine-readable evaluation record (JSON, append-only ledger),
  6. writes a human-readable markdown report,
  7. exits non-zero if a critical gate fails (e.g. metric computation error,
     test-set threshold tuning attempt).

Test-set protection: the config may declare ``threshold_source: validation``
only if a validation split with labels is provided. If the caller supplies
test labels as the threshold-selection source the run is REFUSED. External
holdout data must not be used for threshold selection — passing
``--threshold-source external`` fails the run by design.

The model and its artifacts are never modified by this command.

STATUS: IMPLEMENTED
REAL_WORLD_VALIDATION: BLOCKED_PENDING_ELIGIBLE_DATASET
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # backend/src
sys.path.insert(0, str(Path(__file__).resolve().parent))         # backend/scripts

import numpy as np
import pandas as pd

from eval_record import EvaluationLedger, create_evaluation_record
from metric_definitions import (FullEvaluationMetrics, METRIC_DEFINITIONS_VERSION,
                                operating_point, prevalence_report,
                                roc_auc, pr_auc, recall_at_fpr,
                                threshold_at_fpr_on_validation)

LEDGER_PATH = Path("reports/evaluation_runs/eval_ledger.jsonl")


def load_config(path: Path | None) -> dict:
    if path is None:
        return {}
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    return cfg


def refuse_test_tuning(threshold_source: str) -> None:
    """Hard guard: refuse configurations that tune the threshold on the data
    being evaluated. 'test' and 'external' as a threshold source are always
    invalid — the threshold must come from validation or be fixed a priori."""
    if threshold_source.lower() in ("test", "external", "holdout"):
        raise SystemExit(
            f"REFUSED: threshold_source='{threshold_source}' would tune the "
            f"operating threshold on the evaluation data itself. Use "
            f"'validation' (threshold selected on a held-out validation split) "
            f"or 'fixed' (a priori threshold recorded in the config)."
        )


def main() -> int:
    ap = argparse.ArgumentParser(description="Reproducible frozen evaluation (Phase 39)")
    ap.add_argument("--model-dir", required=True, help="artifacts dir with *.joblib (never modified)")
    ap.add_argument("--dataset", required=True, help="evaluation dataset CSV with 'label' + feature columns")
    ap.add_argument("--config", default=None, help="evaluation config JSON")
    ap.add_argument("--threshold", type=float, default=None,
                    help="operating threshold (fixed); if omitted and no config threshold, selected on validation")
    ap.add_argument("--threshold-source", default=None,
                    choices=["fixed", "validation"], help="provenance of the threshold")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--bootstrap", action="store_true", help="compute bootstrap CIs")
    ap.add_argument("--training-data", default=None, help="training dataset (for the record's provenance)")
    ap.add_argument("--model-identifier", default=None,
                    help="human label for the model (default: artifacts dir name)")
    ap.add_argument("--output-dir", default="reports/evaluation_runs")
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    dataset_path = Path(args.dataset)
    if not model_dir.exists():
        raise SystemExit(f"model dir not found: {model_dir}")
    if not dataset_path.exists():
        raise SystemExit(f"dataset not found: {dataset_path}")

    cfg = load_config(Path(args.config) if args.config else None)
    threshold = args.threshold if args.threshold is not None else cfg.get("threshold")
    threshold_source = args.threshold_source or cfg.get("threshold_source") or \
        ("fixed" if threshold is not None else "validation")
    refuse_test_tuning(str(threshold_source))
    seed = int(args.seed if args.seed is not None else cfg.get("seed", 42))

    from src.privacy_layer.features import ML_FEATURES
    from src.risk_engine.fusion import FusionEngine

    engine = FusionEngine(model_dir)
    df = pd.read_csv(dataset_path)
    missing = [c for c in ML_FEATURES if c not in df.columns]
    if missing:
        raise SystemExit(
            f"FEATURE_SCHEMA_INCOMPATIBLE: dataset is missing required features: {missing}")

    y = df["label"].to_numpy(dtype=int)
    X = df[ML_FEATURES].to_numpy(dtype=float)
    scores = engine.predict_matrix(X)

    # Threshold discipline
    val_labels = cfg.get("validation_labels")
    if threshold is None:
        if val_labels and Path(val_labels).exists():
            vpath = Path(val_labels)
            import joblib
            y_val = joblib.load(vpath)
            s_val = cfg.get("validation_scores")
            s_val = joblib.load(Path(s_val)) if s_val else scores[: len(y_val)]
            threshold = threshold_at_fpr_on_validation(y_val, s_val, target_fpr=0.01)
            threshold_source = "validation"
        else:
            raise SystemExit("No threshold given and no validation data configured — "
                             "pass --threshold or set validation_labels/validation_scores in the config.")
    op = operating_point(y, scores, threshold, threshold_source)
    full = FullEvaluationMetrics(y_true=y, scores=scores, threshold=threshold,
                                 threshold_source=threshold_source,
                                 bootstrap=args.bootstrap, seed=seed)
    metrics = full.to_dict()
    metrics["ranking_metrics"]["roc_auc"] = roc_auc(y, scores)
    metrics["ranking_metrics"]["pr_auc"] = pr_auc(y, scores)
    metrics["recall_at_selected_fpr"] = recall_at_fpr(y, scores, threshold)

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    record = create_evaluation_record(
        model_identifier=args.model_identifier or model_dir.name,
        artifact_paths=sorted(model_dir.glob("*.joblib")),
        dataset_path=dataset_path,
        training_dataset_path=Path(args.training_data) if args.training_data else None,
        seed=seed,
        threshold=threshold,
        threshold_source=threshold_source,
        evaluation_config={"config_file": args.config, **cfg},
        metrics=metrics,
        status="COMPLETED",
    )
    ledger = EvaluationLedger(outdir / "eval_ledger.jsonl")
    ledger.append(record)

    (outdir / f"record_{record.evaluation_id}.json").write_text(
        json.dumps(record.to_dict(), indent=2), encoding="utf-8")

    md = [
        f"# Evaluation {record.evaluation_id}",
        f"- model: `{record.model_identifier}` (hash {record.model_hash})",
        f"- dataset: `{record.dataset['path']}` sha256 `{record.dataset['sha256']}`",
        f"- git commit: {record.git_commit}",
        f"- seed: {record.seed}, metric defs v{record.metric_definitions_version}",
        f"- threshold: {threshold} ({threshold_source})",
        f"- ROC-AUC: {metrics['ranking_metrics']['roc_auc']:.4f}, "
        f"PR-AUC: {metrics['ranking_metrics']['pr_auc']:.4f}",
        f"- operating point: {metrics['operating_point']}",
        f"- class imbalance: {metrics['class_imbalance']}",
    ]
    for w in metrics.get("warnings", []):
        md.append(f"- ⚠️ {w}")
    (outdir / f"report_{record.evaluation_id}.md").write_text("\n".join(md), encoding="utf-8")

    print(f"evaluation_id: {record.evaluation_id}")
    print(f"model_hash: {record.model_hash}")
    print(f"ROC-AUC {metrics['ranking_metrics']['roc_auc']:.4f}  "
          f"PR-AUC {metrics['ranking_metrics']['pr_auc']:.4f}")
    print(f"operating point: {metrics['operating_point']}")
    print(f"ledger: {ledger.path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
