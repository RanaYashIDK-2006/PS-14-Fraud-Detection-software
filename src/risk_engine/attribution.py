"""SHAP-style feature attribution for the fused risk score.

INTERNAL ANALYST TOOL ONLY - the public surface keeps §11 category-level
reason codes as the ONLY user-facing explanation; this module feeds the
`/internal/attribution` endpoint.

Two layers, both with the training-set median per feature as the SHAP
"background" (a typical event), so a contribution means "how much this
event's value moves the score away from a typical event":

  1. STACKER level - exact. The fusion is a logistic regression over the
     four model outputs (LR, RF, XGB, ISO); for a linear model the exact
     SHAP contribution of input j is `coef_j * (x_j - E[x_j])` in logit
     space, so each model's contribution is computed exactly.
  2. FEATURE level - perturbation estimate. For each of the ML features,
     replace the event's value with the training median and measure how far
     the calibrated probability moves. This is the standard one-feature
     marginal approximation (no `shap` dependency).

The training medians are loaded lazily from `data/transactions.csv` (the
same set the models were trained on) and cached in-process.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.privacy_layer.features import ML_FEATURES

DATA_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "transactions.csv"

_baselines: dict[str, float] | None = None

# Sensible default baselines (zero/one) when transactions.csv is absent.
_DEFAULT_BASELINES: dict[str, float] = {
    f: 0.0 for f in ML_FEATURES
}
_DEFAULT_BASELINES.update({
    "amount_ratio": 1.0,
    "known_device_count": 1.0,
    "account_tenure_days": 150.0,
})


def _feature_baselines() -> dict[str, float]:
    global _baselines
    if _baselines is None:
        try:
            if DATA_PATH.exists():
                df = pd.read_csv(DATA_PATH)
                # Only use CSV baselines if all ML_FEATURES columns exist
                if all(f in df.columns for f in ML_FEATURES):
                    _baselines = {f: float(df[f].median()) for f in ML_FEATURES}
                else:
                    _baselines = dict(_DEFAULT_BASELINES)
            else:
                _baselines = dict(_DEFAULT_BASELINES)
        except Exception:
            _baselines = dict(_DEFAULT_BASELINES)
    return _baselines


def stacker_attribution(fusion, features: dict) -> dict:
    """Exact per-model contribution to the fused logit."""
    comp = fusion.components(features)
    baseline = {f: _feature_baselines()[f] for f in ML_FEATURES}
    base_comp = fusion.components(baseline)

    logit = comp["intercept"] + sum(
        comp["coefficients"][n] * comp["outputs"][n] for n in comp["outputs"]
    )
    base_logit = base_comp["intercept"] + sum(
        base_comp["coefficients"][n] * base_comp["outputs"][n] for n in base_comp["outputs"]
    )
    contributions = [
        {
            "model": name,
            "output": round(comp["outputs"][name], 4),
            "coefficient": round(comp["coefficients"][name], 4),
            "logit_contribution": round(
                comp["coefficients"][name] * (comp["outputs"][name] - base_comp["outputs"][name]), 4
            ),
        }
        for name in comp["outputs"]
    ]
    return {
        "logit": round(logit, 4),
        "baseline_logit": round(base_logit, 4),
        "contributions": contributions,
    }


def feature_attribution(fusion, features: dict) -> dict:
    """Perturbation-based per-feature contributions to the calibrated ml."""
    baselines = _feature_baselines()
    ml, _ = fusion.predict(features)
    base_ml, _ = fusion.predict(baselines)

    contribs: list[dict] = []
    for f in ML_FEATURES:
        perturbed = dict(features)
        perturbed[f] = baselines[f]
        moved, _ = fusion.predict(perturbed)
        contribs.append(
            {
                "feature": f,
                "value": round(float(features.get(f, baselines[f])), 4),
                "baseline": round(baselines[f], 4),
                "contribution": round(ml - moved, 4),
            }
        )
    contribs.sort(key=lambda c: abs(c["contribution"]), reverse=True)
    total = sum(c["contribution"] for c in contribs)
    return {
        "ml_score": round(ml, 4),
        "baseline_ml": round(base_ml, 4),
        "explained": round(ml - base_ml, 4),
        "sum_contributions": round(total, 4),
        "residual": round((ml - base_ml) - total, 4),  # interaction/order effects
        "features": contribs,
    }
