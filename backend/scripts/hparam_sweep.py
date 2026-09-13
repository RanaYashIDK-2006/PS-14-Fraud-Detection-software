"""Hyperparameter sweep against the HONEST generalization metrics.

Runs the leave-one-archetype-out folds (the same code path as the
production group-split eval) for several MODEL_SPECS candidates and
reports: impulse recall@1%FPR (the weakest archetype), min/mean
recall@1%FPR across archetypes, and novel-legit FPR@0.6.

NOTE: fit_fold hardcodes XGB hyperparameters (reads only `use_scaler`
from MODEL_SPECS), so XGB candidates have NO effect and are excluded.
Only RF and LR specs are patched through.

Usage:
    python backend/scripts/hparam_sweep.py            # all candidates
    python backend/scripts/hparam_sweep.py base rf_deep  # subset
"""
from __future__ import annotations

import copy
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent  # repo root
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "backend" / "src"))

import numpy as np
import pandas as pd

import train_compare as tc
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

DATA = ROOT / "data" / "transactions.csv"

# Save the original baseline before any patching.
_ORIGINAL_SPECS = copy.deepcopy(dict(tc.MODEL_SPECS))

# Candidate MODEL_SPECS overrides.  Only RF and LR are patched — XGB is
# hardcoded inside fit_fold and cannot be overridden via MODEL_SPECS.
CANDIDATES: dict[str, dict] = {
    # A: production baseline (no changes)
    "base": {},

    # B: deeper RF — more trees + stronger leaf regularization
    "rf_deep": {
        "random_forest": (RandomForestClassifier(
            n_estimators=500,
            class_weight="balanced_subsample",
            min_samples_leaf=10,
            max_features="sqrt",
            n_jobs=-1,
            random_state=42), False),
    },

    # C: very regularized RF — fights leaf memorization hard
    "rf_reg": {
        "random_forest": (RandomForestClassifier(
            n_estimators=500,
            class_weight="balanced_subsample",
            min_samples_leaf=20,
            max_features="sqrt",
            n_jobs=-1,
            random_state=42), False),
    },

    # D: strong L2 LR — less overfitting on small fraud signal
    "lr_l2": {
        "logistic_regression": (LogisticRegression(
            C=0.1,
            class_weight="balanced",
            max_iter=3000,
            random_state=42), True),
    },

    # E: weaker L1 LR — sparse feature selection
    "lr_l1": {
        "logistic_regression": (LogisticRegression(
            C=0.5,
            penalty="l1",
            solver="saga",
            class_weight="balanced",
            max_iter=3000,
            random_state=42), True),
    },

    # F: deeper RF + stronger L2 LR — regularize both
    "rf_deep_lr_l2": {
        "random_forest": (RandomForestClassifier(
            n_estimators=500,
            class_weight="balanced_subsample",
            min_samples_leaf=10,
            max_features="sqrt",
            n_jobs=-1,
            random_state=42), False),
        "logistic_regression": (LogisticRegression(
            C=0.1,
            class_weight="balanced",
            max_iter=3000,
            random_state=42), True),
    },
}


def apply_candidate(name: str) -> None:
    """Patch tc.MODEL_SPECS with the candidate's entries, then restore
    the originals after each evaluation so candidates don't leak."""
    specs = copy.deepcopy(_ORIGINAL_SPECS)
    specs.update(CANDIDATES[name])
    tc.MODEL_SPECS.clear()
    tc.MODEL_SPECS.update(specs)


def restore_baseline() -> None:
    tc.MODEL_SPECS.clear()
    tc.MODEL_SPECS.update(copy.deepcopy(_ORIGINAL_SPECS))


def evaluate(name: str, df: pd.DataFrame, seed: int = 42) -> dict:
    apply_candidate(name)
    t0 = time.perf_counter()
    gen_df, _, _ = tc.run_ood_folds(df, seed)
    dt = time.perf_counter() - t0
    table = {r["held_out"]: r for r in gen_df.to_dict(orient="records")}
    archs = ["ato", "cnp", "escalation", "impulse", "mule"]
    recalls = [table[a]["recall_at_1pct_fpr"] for a in archs if a in table]
    novel = table.get("40 novel legit accounts", {})
    restore_baseline()
    return {
        "candidate": name,
        "impulse_r1": table.get("impulse", {}).get("recall_at_1pct_fpr"),
        "min_r1": min(recalls) if recalls else None,
        "mean_r1": round(float(np.nanmean(recalls)), 3) if recalls else None,
        "novel_fpr_06": novel.get("novel_account_fpr_0.6"),
        "seconds": round(dt, 1),
    }


def main() -> int:
    wanted = sys.argv[1:] or list(CANDIDATES)
    df = pd.read_csv(DATA)
    df["ts"] = pd.to_numeric(df["ts"], errors="coerce")
    df = df.sort_values("ts").reset_index(drop=True)

    rows = []
    for name in wanted:
        if name not in CANDIDATES:
            print(f"unknown candidate: {name}")
            return 2
        print(f"\n=== candidate: {name} ===")
        rows.append(evaluate(name, df))

    print("\n=== sweep summary (honest group-split metrics) ===")
    out = pd.DataFrame(rows)
    print(out.to_string(index=False))
    out.to_csv(ROOT / "models" / "hparam_sweep_results.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
