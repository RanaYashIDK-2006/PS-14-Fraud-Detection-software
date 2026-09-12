"""PSI-based drift detector for velocity features.

Measures distribution shifts between a reference (training) distribution
and a production distribution using Population Stability Index (PSI).

PSI = sum((P_prod - P_ref) * ln(P_prod / P_ref))

Thresholds:
  PSI < 0.10:  STABLE   — no action needed
  0.10 <= PSI < 0.20: WARNING  — monitor closely, investigate
  PSI >= 0.20: CRITICAL — trigger model retraining

Architecture:
  ReferenceDistribution (built from training data, serialized to JSON)
    → DriftDetector (loads reference, accepts production batches)
      → DriftReport (per-feature PSI + aggregate PSI + alerts)

Usage:
    # Build reference from training data
    ref = ReferenceDistribution.from_training_data(X_train, feature_names)
    ref.save("models/production/altman_lgb_*/reference_dist.json")

    # Monitor production data
    detector = DriftDetector.load("reference_dist.json")
    report = detector.check(X_production_batch)

    if report.needs_retrain:
        print(report.summary())
"""
from __future__ import annotations

import json
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

warnings.filterwarnings("ignore")


# ── PSI Calculation ──

def compute_psi(reference: np.ndarray, production: np.ndarray,
                n_bins: int = 20, eps: float = 1e-4) -> float:
    """Compute PSI between two distributions.

    Args:
        reference: reference distribution values (1D array)
        production: production distribution values (1D array)
        n_bins: number of bins for discretization
        eps: minimum proportion to avoid log(0)

    Returns:
        PSI value (0 = identical, higher = more different)
    """
    # Use shared bin edges from reference
    ref_clean = reference[np.isfinite(reference)]
    prod_clean = production[np.isfinite(production)]

    if len(ref_clean) < n_bins or len(prod_clean) < n_bins:
        return 0.0

    # Create bins from reference percentiles
    bin_edges = np.percentile(ref_clean, np.linspace(0, 100, n_bins + 1))
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf
    # Make edges unique
    bin_edges = np.unique(bin_edges)
    n_bins_actual = len(bin_edges) - 1

    if n_bins_actual < 2:
        return 0.0

    # Count proportions in each bin
    ref_counts = np.histogram(ref_clean, bins=bin_edges)[0].astype(float)
    prod_counts = np.histogram(prod_clean, bins=bin_edges)[0].astype(float)

    # Normalize to proportions
    ref_props = ref_counts / (ref_counts.sum() + eps)
    prod_props = prod_counts / (prod_counts.sum() + eps)

    # Apply epsilon to avoid log(0)
    ref_props = np.maximum(ref_props, eps)
    prod_props = np.maximum(prod_props, eps)

    # PSI = sum((P_prod - P_ref) * ln(P_prod / P_ref))
    psi = float(np.sum((prod_props - ref_props) * np.log(prod_props / ref_props)))
    return psi


def compute_psi_per_feature(X_ref: np.ndarray, X_prod: np.ndarray,
                            feature_names: list[str],
                            n_bins: int = 20) -> dict[str, float]:
    """Compute PSI for each feature.

    Returns:
        Dict mapping feature_name -> PSI value
    """
    psi_values = {}
    for i, name in enumerate(feature_names):
        if i < X_ref.shape[1] and i < X_prod.shape[1]:
            psi_values[name] = round(compute_psi(X_ref[:, i], X_prod[:, i], n_bins), 6)
    return psi_values


# ── Reference Distribution ──

@dataclass
class ReferenceDistribution:
    """Reference (training) distribution for drift comparison.

    Stores per-feature statistics and optionally the raw reference data.
    """
    feature_names: list[str]
    feature_stats: dict  # name -> {mean, std, min, max, p10, p50, p90}
    n_samples: int
    n_features: int
    feature_arrays: Optional[dict] = None  # name -> np.ndarray (raw values)
    created_at: float = field(default_factory=time.time)

    @classmethod
    def from_training_data(cls, X: np.ndarray, feature_names: list[str],
                          keep_raw: bool = True) -> ReferenceDistribution:
        """Build reference distribution from training data."""
        stats = {}
        arrays = {} if keep_raw else None

        for i, name in enumerate(feature_names):
            col = X[:, i]
            finite = col[np.isfinite(col)]
            if len(finite) == 0:
                stats[name] = {"mean": 0, "std": 1, "min": 0, "max": 1,
                              "p10": 0, "p50": 0, "p90": 0}
                if arrays is not None:
                    arrays[name] = np.zeros(1)
                continue

            stats[name] = {
                "mean": round(float(np.mean(finite)), 6),
                "std": round(float(np.std(finite)), 6),
                "min": round(float(np.min(finite)), 6),
                "max": round(float(np.max(finite)), 6),
                "p10": round(float(np.percentile(finite, 10)), 6),
                "p50": round(float(np.percentile(finite, 50)), 6),
                "p90": round(float(np.percentile(finite, 90)), 6),
            }
            if arrays is not None:
                arrays[name] = col.copy()

        return cls(
            feature_names=feature_names,
            feature_stats=stats,
            n_samples=len(X),
            n_features=len(feature_names),
            feature_arrays=arrays,
        )

    def save(self, path: str | Path) -> None:
        """Save reference distribution to JSON + numpy arrays."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        # Save metadata as JSON
        data = {
            "feature_names": self.feature_names,
            "feature_stats": self.feature_stats,
            "n_samples": self.n_samples,
            "n_features": self.n_features,
            "created_at": self.created_at,
        }
        p.write_text(json.dumps(data, indent=2))
        # Save raw arrays as numpy file
        if self.feature_arrays:
            arr_path = str(p).replace(".json", ".npy")
            # Stack into 2D array: (n_features, n_samples)
            arr_list = [self.feature_arrays[name] for name in self.feature_names]
            arr_2d = np.stack(arr_list, axis=0)
            np.save(arr_path, arr_2d)

    @classmethod
    def load(cls, path: str | Path) -> ReferenceDistribution:
        """Load reference distribution from JSON + numpy arrays."""
        p = Path(path)
        data = json.loads(p.read_text())
        # Try to load raw arrays
        arr_path = Path(str(p).replace(".json", ".npy"))
        arrays = None
        if arr_path.exists():
            arr_2d = np.load(str(arr_path))
            arrays = {name: arr_2d[i] for i, name in enumerate(data["feature_names"]) if i < arr_2d.shape[0]}
        return cls(
            feature_names=data["feature_names"],
            feature_stats=data["feature_stats"],
            n_samples=data["n_samples"],
            n_features=data["n_features"],
            feature_arrays=arrays,
            created_at=data.get("created_at", 0),
        )

    def get_arrays(self) -> dict[str, np.ndarray]:
        """Get raw arrays."""
        if self.feature_arrays:
            return self.feature_arrays
        return {}


# ── Drift Report ──

@dataclass
class DriftReport:
    """Result of a drift check."""
    timestamp: float
    n_samples: int
    per_feature_psi: dict[str, float]
    aggregate_psi: float
    max_psi: float
    max_psi_feature: str
    n_stable: int
    n_warning: int
    n_critical: int
    critical_features: list[str]
    warning_features: list[str]
    needs_retrain: bool
    needs_investigation: bool

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"Drift Report ({time.strftime('%Y-%m-%d %H:%M:%S')})",
            f"  Samples: {self.n_samples:,}",
            f"  Aggregate PSI: {self.aggregate_psi:.4f}",
            f"  Max PSI: {self.max_psi:.4f} ({self.max_psi_feature})",
            f"  Stable: {self.n_stable}/{len(self.per_feature_psi)}",
            f"  Warning: {self.n_warning}/{len(self.per_feature_psi)}",
            f"  Critical: {self.n_critical}/{len(self.per_feature_psi)}",
        ]
        if self.needs_retrain:
            lines.append(f"  ALERT: RETRAINING RECOMMENDED (PSI >= 0.20)")
            lines.append(f"  Critical features: {', '.join(self.critical_features[:5])}")
        elif self.needs_investigation:
            lines.append(f"  ALERT: INVESTIGATE (PSI >= 0.10)")
            lines.append(f"  Warning features: {', '.join(self.warning_features[:5])}")
        else:
            lines.append(f"  STATUS: STABLE — no action needed")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "n_samples": self.n_samples,
            "aggregate_psi": round(self.aggregate_psi, 6),
            "max_psi": round(self.max_psi, 6),
            "max_psi_feature": self.max_psi_feature,
            "n_stable": self.n_stable,
            "n_warning": self.n_warning,
            "n_critical": self.n_critical,
            "critical_features": self.critical_features,
            "warning_features": self.warning_features,
            "needs_retrain": self.needs_retrain,
            "needs_investigation": self.needs_investigation,
            "per_feature_psi": self.per_feature_psi,
        }


# ── Drift Detector ──

class DriftDetector:
    """Monitor velocity features for distribution drift.

    Maintains a sliding window of recent production data and compares
    it against the reference distribution using PSI.
    """

    def __init__(self, reference: ReferenceDistribution,
                 warning_threshold: float = 0.10,
                 critical_threshold: float = 0.20,
                 min_samples: int = 100,
                 aggregate_threshold: float = 0.15):
        self.reference = reference
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.min_samples = min_samples
        self.aggregate_threshold = aggregate_threshold
        self._recent_batches: list[np.ndarray] = []
        self._check_history: list[dict] = []

    @classmethod
    def load(cls, ref_path: str | Path, **kwargs) -> DriftDetector:
        """Load from a saved reference distribution."""
        ref = ReferenceDistribution.load(ref_path)
        return cls(ref, **kwargs)

    def check(self, X_production: np.ndarray,
              feature_names: Optional[list[str]] = None) -> DriftReport:
        """Check a production batch for drift.

        Args:
            X_production: production data (n_samples, n_features)
            feature_names: override feature names (default: use reference names)

        Returns:
            DriftReport with per-feature PSI and alerts
        """
        if feature_names is None:
            feature_names = self.reference.feature_names

        n_samples = len(X_production)
        if n_samples < self.min_samples:
            # Not enough data for reliable PSI
            return DriftReport(
                timestamp=time.time(), n_samples=n_samples,
                per_feature_psi={}, aggregate_psi=0.0,
                max_psi=0.0, max_psi_feature="",
                n_stable=0, n_warning=0, n_critical=0,
                critical_features=[], warning_features=[],
                needs_retrain=False, needs_investigation=False,
            )

        # Get reference arrays
        ref_arrays = self.reference.get_arrays()

        # Compute per-feature PSI
        per_feature_psi = {}
        for i, name in enumerate(feature_names):
            if name in ref_arrays and i < X_production.shape[1]:
                psi = compute_psi(ref_arrays[name], X_production[:, i])
                per_feature_psi[name] = round(psi, 6)

        if not per_feature_psi:
            return DriftReport(
                timestamp=time.time(), n_samples=n_samples,
                per_feature_psi={}, aggregate_psi=0.0,
                max_psi=0.0, max_psi_feature="",
                n_stable=0, n_warning=0, n_critical=0,
                critical_features=[], warning_features=[],
                needs_retrain=False, needs_investigation=False,
            )

        # Classify features
        critical = [n for n, p in per_feature_psi.items() if p >= self.critical_threshold]
        warning = [n for n, p in per_feature_psi.items()
                  if self.warning_threshold <= p < self.critical_threshold]
        stable = [n for n, p in per_feature_psi.items() if p < self.warning_threshold]

        # Aggregate PSI: mean of all feature PSIs
        psi_values = list(per_feature_psi.values())
        aggregate_psi = float(np.mean(psi_values))
        max_psi = max(psi_values)
        max_feature = max(per_feature_psi, key=per_feature_psi.get)

        needs_retrain = aggregate_psi >= self.aggregate_threshold or len(critical) >= 3
        needs_investigation = aggregate_psi >= self.warning_threshold or len(critical) >= 1

        report = DriftReport(
            timestamp=time.time(),
            n_samples=n_samples,
            per_feature_psi=per_feature_psi,
            aggregate_psi=aggregate_psi,
            max_psi=max_psi,
            max_psi_feature=max_feature,
            n_stable=len(stable),
            n_warning=len(warning),
            n_critical=len(critical),
            critical_features=critical,
            warning_features=warning,
            needs_retrain=needs_retrain,
            needs_investigation=needs_investigation,
        )

        self._check_history.append(report.to_dict())
        return report

    def get_history(self) -> list[dict]:
        """Get history of drift checks."""
        return self._check_history

    def save_history(self, path: str | Path) -> None:
        """Save drift check history."""
        Path(path).write_text(json.dumps(self._check_history, indent=2))

    def get_trend(self) -> dict:
        """Analyze drift trend over time."""
        if len(self._check_history) < 2:
            return {"trend": "insufficient_data", "checks": len(self._check_history)}

        psi_history = [h["aggregate_psi"] for h in self._check_history]
        recent = psi_history[-5:] if len(psi_history) >= 5 else psi_history

        # Simple trend: is PSI increasing?
        if len(recent) >= 2:
            slope = (recent[-1] - recent[0]) / len(recent)
            if slope > 0.01:
                trend = "increasing"
            elif slope < -0.01:
                trend = "decreasing"
            else:
                trend = "stable"
        else:
            trend = "stable"

        return {
            "trend": trend,
            "checks": len(self._check_history),
            "latest_psi": psi_history[-1],
            "mean_psi": round(float(np.mean(psi_history)), 6),
            "max_psi": round(float(np.max(psi_history)), 6),
            "psi_history": [round(p, 6) for p in psi_history[-20:]],
        }


# ── CLI ──

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PSI drift detector")
    parser.add_argument("--build-reference", action="store_true",
                       help="Build reference from training data")
    parser.add_argument("--check", type=str,
                       help="Check a data file for drift")
    parser.add_argument("--reference", type=str,
                       help="Path to reference distribution JSON")
    parser.add_argument("--output", type=str,
                       help="Output path for drift report")
    args = parser.parse_args()

    if args.build_reference:
        print("Building reference distribution from Altman training data...")
        # Load training data (same as production pipeline)
        import pandas as pd
        from collections import defaultdict

        rng = np.random.RandomState(42)
        rows_all = []
        ci = 0
        for chunk in pd.read_csv("data/credit_card_transactions-ibm_v2.csv",
            usecols=["User","Month","Day","Time","Amount","Use Chip","MCC","Errors?",
                     "Is Fraud?","Merchant Name","Merchant City","Merchant State","Zip","Year","Card"],
            low_memory=False, chunksize=1_000_000):
            c = chunk.copy()
            c["amt"] = c["Amount"].str.replace("$","",regex=False).str.replace(",","",regex=False).astype(float)
            c["label"] = (c["Is Fraud?"]=="Yes").astype(int)
            chip_map = {"Swipe Transaction":0,"Online Transaction":1,"Chip Transaction":2}
            c["chip"] = c["Use Chip"].map(chip_map).fillna(-1)
            c["mcc_n"] = c["MCC"].astype(str).str[:4].astype(float)/10000
            c["err"] = (c["Errors?"].fillna("")!="").astype(int)
            tp = c["Time"].str.split(":",expand=True)
            c["hr"] = tp[0].astype(float); c["mn"] = tp[1].astype(float)
            c["merchant_id"] = c["Merchant Name"].astype("category").cat.codes
            c["city_id"] = c["Merchant City"].astype("category").cat.codes
            c["is_online"] = (c["Merchant City"]=="ONLINE").astype(int)
            c["day_decimal"] = c["Day"]+c["hr"]/24.0+c["mn"]/1440.0
            c["amt_log"] = np.log1p(c["amt"].clip(upper=1e9))
            c["log_amt"] = np.log1p(c["amt"])
            c["amt_x_hr"] = c["amt"]*c["hr"]
            c["amt_x_mcc"] = c["amt"]*c["mcc_n"]
            c["amt_x_chip"] = c["amt"]*c["chip"]
            c["amt_sq"] = c["amt"]**2
            c["night_tx"] = ((c["hr"]>=22)|(c["hr"]<=6)).astype(int)
            fm = c["label"]==1
            rows_all.append(pd.concat([c[fm], c[~fm].sample(frac=0.01, random_state=rng)]))
            ci += 1
            if ci >= 16: break

        df = pd.concat(rows_all, ignore_index=True)
        # Use subset of velocity features that we can compute without full history
        # For reference, use the base features that are available immediately
        feature_names = ["amt", "hr", "mn", "mcc_n", "chip", "err", "merchant_id",
                        "city_id", "is_online", "amt_log", "log_amt", "amt_x_hr",
                        "amt_x_mcc", "amt_x_chip", "amt_sq", "night_tx"]
        X = df[feature_names].values.astype(np.float32)
        X = np.nan_to_num(X)

        print(f"  Training data: {len(X):,} rows, {len(feature_names)} features")
        ref = ReferenceDistribution.from_training_data(X, feature_names)
        ref.save("models/drift_reference.json")
        print(f"  Saved: models/drift_reference.json")

    elif args.check:
        print(f"Checking drift: {args.check}")
        ref_path = args.reference or "models/drift_reference.json"
        detector = DriftDetector.load(ref_path)

        data = np.load(args.check)
        report = detector.check(data)
        print(report.summary())

        if args.output:
            Path(args.output).write_text(json.dumps(report.to_dict(), indent=2))
            print(f"\n  Report saved: {args.output}")
