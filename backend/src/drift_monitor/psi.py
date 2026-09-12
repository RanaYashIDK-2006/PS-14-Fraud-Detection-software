"""Population stability index (PSI) drift monitoring (architecture section 6).

PSI compares a feature's distribution in the current window against the
distribution the models were trained on ("training baseline"). Standard
industry thresholds:

  PSI < 0.10   -> ok        (no significant change)
  0.10 - 0.25  -> warn      (moderate drift - monitor)
  PSI >= 0.25  -> alert     (significant drift - retraining review)

The baseline is stored as bin edges + expected proportions per feature, so
the monitor never needs the raw training data at check time - only the
window's values (section 6: drift triggers retraining review, not raw-data
retention).
"""

from __future__ import annotations

import numpy as np

# Standard PSI thresholds.
PSI_WARN = 0.10
PSI_ALERT = 0.25

_EPS = 1e-4  # smoothing for empty bins (standard PSI practice)


def bin_edges(values: np.ndarray, n_bins: int = 10) -> np.ndarray | None:
    """Equal-frequency bin edges from baseline values; None for constant."""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if v.size == 0 or np.ptp(v) == 0.0:
        return None
    edges = np.unique(np.quantile(v, np.linspace(0.0, 100.0, n_bins + 1) / 100.0))
    return edges if len(edges) >= 2 else None


def psi_proportions(ep: np.ndarray, ap: np.ndarray) -> float:
    """PSI over two proportion vectors (same length)."""
    return float(np.sum((ap - ep) * np.log(ap / ep)))


def level(psi_value: float) -> str:
    if psi_value >= PSI_ALERT:
        return "alert"
    if psi_value >= PSI_WARN:
        return "warn"
    return "ok"


def build_baseline(df, features: list[str], n_bins: int = 10) -> dict:
    """Per-feature baseline: bin edges + expected proportions + sample size."""
    baseline: dict = {}
    for f in features:
        vals = df[f].to_numpy(dtype=float)
        edges = bin_edges(vals, n_bins)
        if edges is None:
            baseline[f] = {"constant": True, "n": int(len(vals))}
            continue
        counts, _ = np.histogram(vals, bins=edges)
        baseline[f] = {
            "edges": [float(e) for e in edges],
            "expected": [float(x) for x in counts / counts.sum()],
            "n": int(len(vals)),
        }
    return baseline


def check_feature(baseline_feature: dict, values) -> dict:
    """PSI + level for one feature's window values against its baseline."""
    if baseline_feature.get("constant"):
        return {"psi": 0.0, "level": "ok", "note": "constant in baseline"}
    edges = np.asarray(baseline_feature["edges"], dtype=float)
    vals = np.asarray(values, dtype=float)
    vals = vals[~np.isnan(vals)]
    counts, _ = np.histogram(vals, bins=edges)
    nb = len(edges) - 1
    ep = np.asarray(baseline_feature["expected"], dtype=float)
    ep = ep / ep.sum()
    ap = (counts + _EPS) / (counts.sum() + _EPS * nb)
    p = psi_proportions(ep, ap)
    return {"psi": round(float(p), 4), "level": level(p)}


def check_window(baseline: dict, df_window, features: list[str]) -> dict:
    """Per-feature PSI for the whole window against the baseline."""
    out: dict = {}
    for f in features:
        b = baseline.get(f)
        if b is None:
            out[f] = {"psi": None, "level": "n/a", "note": "missing from baseline"}
            continue
        out[f] = check_feature(b, df_window[f].to_numpy(dtype=float))
    return out
