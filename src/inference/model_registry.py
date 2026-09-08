"""Model registry for A/B testing — loads and manages multiple model versions.

Discovers model artifacts from models/production/ and models/artifacts/,
loads them into memory, and provides a unified interface for scoring.

Each registered model version includes:
  - The model object (XGB, LGB, CB, or ensemble)
  - A scaler for feature normalization
  - Metadata (version, features, training info)
  - A scorer callable that maps (feature_vector) → fraud_probability
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import joblib
import numpy as np


ROOT = Path(__file__).resolve().parent.parent.parent


@dataclass
class ModelVersion:
    """A single registered model version."""
    version: str
    model: object          # sklearn/xgb/lgb model
    scaler: object         # sklearn scaler
    feat_cols: list[str]   # ordered feature names
    n_features: int
    metadata: dict = field(default_factory=dict)
    loaded_at: float = field(default_factory=time.time)
    hash: str = ""

    def __post_init__(self):
        if not self.hash:
            self.hash = hashlib.sha256(
                f"{self.version}:{self.n_features}:{len(self.feat_cols)}".encode()
            ).hexdigest()[:12]


class ModelRegistry:
    """Discovers, loads, and manages multiple model versions.

    Supports two loading patterns:
    1. Auto-discovery: scans models/production/ for manifest.json files
    2. Explicit registration: register_model(version, model_path, scaler_path, ...)

    Usage:
        registry = ModelRegistry()
        registry.auto_discover()
        scorer = registry.get_scorer("altman_269k_20260829_194628")
        proba = scorer(feature_vector)
    """

    def __init__(self):
        self._models: dict[str, ModelVersion] = {}
        self._scorers: dict[str, Callable] = {}

    def auto_discover(self) -> list[str]:
        """Scan models/production/ and models/artifacts/ for loadable models."""
        versions = []

        # Pattern 1: models/production/ with manifest.json
        prod_dir = ROOT / "models" / "production"
        if prod_dir.exists():
            # Check for manifest-based models
            manifest = prod_dir / "manifest.json"
            if manifest.exists():
                v = self._load_from_manifest(prod_dir, manifest)
                if v:
                    versions.append(v)

            # Check for versioned subdirectories
            for subdir in prod_dir.iterdir():
                if subdir.is_dir() and subdir.name.startswith("altman_"):
                    v = self._load_versioned_dir(subdir)
                    if v:
                        versions.append(v)

            # Check for native ensemble (altman_native/ with xgb_native.joblib)
            native_dir = prod_dir / "altman_native"
            if native_dir.exists() and (native_dir / "xgb_native.joblib").exists():
                v = self._load_native_ensemble(native_dir)
                if v:
                    versions.append(v)

        # Pattern 2: models/artifacts/*_altman_24m.joblib
        art_dir = ROOT / "models" / "artifacts"
        if art_dir.exists():
            v = self._load_artifacts_ensemble(art_dir)
            if v:
                versions.append(v)

        # Pattern 3: Original LGB production model
        orig_lgb = prod_dir / "model.joblib"
        if orig_lgb.exists() and "v1_original" not in self._models:
            v = self._load_original_lgb(prod_dir)
            if v:
                versions.append(v)

        return versions

    def _load_from_manifest(self, prod_dir: Path, manifest_path: Path) -> Optional[str]:
        """Load models from a manifest.json in models/production/."""
        try:
            m = json.loads(manifest_path.read_text())
            version = m.get("model_version", "unknown")
            if version in self._models:
                return None

            # Load individual model artifacts
            xgb_path = prod_dir / m["artifacts"].get("xgb", "xgb_production.joblib")
            lgb_path = prod_dir / m["artifacts"].get("lgb", "lgb_production.joblib")
            cb_path = prod_dir / m["artifacts"].get("cb", "cb_production.joblib")
            scaler_path = prod_dir / m["artifacts"].get("scaler", "scaler_production.joblib")

            if not all(p.exists() for p in [xgb_path, lgb_path, cb_path, scaler_path]):
                return None

            xgb = joblib.load(xgb_path)
            lgb = joblib.load(lgb_path)
            cb = joblib.load(cb_path)
            scaler = joblib.load(scaler_path)
            feat_cols = json.loads((prod_dir / "feature_list.json").read_text())

            mv = ModelVersion(
                version=version,
                model={"xgb": xgb, "lgb": lgb, "cb": cb},
                scaler=scaler,
                feat_cols=feat_cols,
                n_features=len(feat_cols),
                metadata=m,
            )

            # Build ensemble scorer
            def ensemble_scorer(X: np.ndarray) -> np.ndarray:
                Xs = scaler.transform(X)
                px = xgb.predict_proba(Xs)[:, 1]
                pl = lgb.predict_proba(Xs)[:, 1]
                pc = cb.predict_proba(Xs)[:, 1]
                return 0.4 * px + 0.35 * pl + 0.25 * pc

            self._models[version] = mv
            self._scorers[version] = ensemble_scorer
            return version

        except Exception as e:
            print(f"[registry] Failed to load {manifest_path}: {e}")
            return None

    def _load_versioned_dir(self, subdir: Path) -> Optional[str]:
        """Load from a versioned subdirectory like altman_lgb_20260829_145406/."""
        try:
            model_path = subdir / "model.joblib"
            if not model_path.exists():
                return None

            version = subdir.name
            if version in self._models:
                return None

            model = joblib.load(model_path)
            n_features = getattr(model, "n_features_", None) or getattr(model, "n_features_in_", 0)

            mv = ModelVersion(
                version=version,
                model=model,
                scaler=None,
                feat_cols=[],
                n_features=n_features,
                metadata={"source": "versioned_dir"},
            )

            def single_scorer(X: np.ndarray) -> np.ndarray:
                return model.predict_proba(X)[:, 1]

            self._models[version] = mv
            self._scorers[version] = single_scorer
            return version

        except Exception as e:
            print(f"[registry] Failed to load {subdir}: {e}")
            return None

    def _load_artifacts_ensemble(self, art_dir: Path) -> Optional[str]:
        """Load the XGB+LGB+CB ensemble from models/artifacts/*_altman_24m.joblib."""
        try:
            xgb_path = art_dir / "xgb_altman_24m.joblib"
            lgb_path = art_dir / "lgb_altman_24m.joblib"
            cb_path = art_dir / "cb_altman_24m.joblib"
            scaler_path = art_dir / "scaler_altman_24m.joblib"

            if not all(p.exists() for p in [xgb_path, lgb_path, cb_path, scaler_path]):
                return None

            version = "altman_24m_full"
            if version in self._models:
                return None

            xgb = joblib.load(xgb_path)
            lgb = joblib.load(lgb_path)
            cb = joblib.load(cb_path)
            scaler = joblib.load(scaler_path)

            n_features = xgb.n_features_in_

            mv = ModelVersion(
                version=version,
                model={"xgb": xgb, "lgb": lgb, "cb": cb},
                scaler=scaler,
                feat_cols=[],
                n_features=n_features,
                metadata={"source": "artifacts_24m", "training_rows": 1_247_612},
            )

            def ensemble_scorer(X: np.ndarray) -> np.ndarray:
                Xs = scaler.transform(X)
                px = xgb.predict_proba(Xs)[:, 1]
                pl = lgb.predict_proba(Xs)[:, 1]
                pc = cb.predict_proba(Xs)[:, 1]
                return 0.4 * px + 0.35 * pl + 0.25 * pc

            self._models[version] = mv
            self._scorers[version] = ensemble_scorer
            return version

        except Exception as e:
            print(f"[registry] Failed to load artifacts ensemble: {e}")
            return None

    def _load_original_lgb(self, prod_dir: Path) -> Optional[str]:
        """Load the original LGB production model."""
        try:
            model_path = prod_dir / "model.joblib"
            scaler_path = prod_dir / "scaler.joblib"
            if not model_path.exists():
                return None

            version = "v1_original_lgb"
            if version in self._models:
                return None

            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path) if scaler_path.exists() else None
            n_features = getattr(model, "n_features_", 0) or getattr(model, "n_features_in_", 0)

            mv = ModelVersion(
                version=version,
                model=model,
                scaler=scaler,
                feat_cols=[],
                n_features=n_features,
                metadata={"source": "original_production"},
            )

            def single_scorer(X: np.ndarray) -> np.ndarray:
                Xs = scaler.transform(X) if scaler else X
                return model.predict_proba(Xs)[:, 1]

            self._models[version] = mv
            self._scorers[version] = single_scorer
            return version

        except Exception as e:
            print(f"[registry] Failed to load original LGB: {e}")
            return None

    def _load_native_ensemble(self, native_dir: Path) -> Optional[str]:
        """Load the Altman-native XGB+LGB+CB ensemble from altman_native/."""
        try:
            xgb_path = native_dir / "xgb_native.joblib"
            lgb_path = native_dir / "lgb_native.joblib"
            cb_path = native_dir / "cb_native.joblib"
            scaler_path = native_dir / "scaler_native.joblib"

            if not all(p.exists() for p in [xgb_path, lgb_path, cb_path, scaler_path]):
                return None

            manifest_path = native_dir / "manifest.json"
            if manifest_path.exists():
                m = json.loads(manifest_path.read_text())
                version = m.get("model_version", "altman_native")
            else:
                version = "altman_native"

            if version in self._models:
                return None

            xgb = joblib.load(xgb_path)
            lgb = joblib.load(lgb_path)
            cb = joblib.load(cb_path)
            scaler = joblib.load(scaler_path)
            feat_cols = json.loads((native_dir / "feature_list.json").read_text()) if (native_dir / "feature_list.json").exists() else []

            mv = ModelVersion(
                version=version,
                model={"xgb": xgb, "lgb": lgb, "cb": cb},
                scaler=scaler,
                feat_cols=feat_cols,
                n_features=xgb.n_features_in_,
                metadata={"source": "altman_native", "is_native": True},
            )

            def native_scorer(X: np.ndarray) -> np.ndarray:
                Xs = scaler.transform(X)
                px = xgb.predict_proba(Xs)[:, 1]
                pl = lgb.predict_proba(Xs)[:, 1]
                pc = cb.predict_proba(Xs)[:, 1]
                return 0.34 * px + 0.33 * pl + 0.33 * pc

            self._models[version] = mv
            self._scorers[version] = native_scorer
            print(f"[registry] Loaded native ensemble: {version} ({xgb.n_features_in_} features)")
            return version

        except Exception as e:
            print(f"[registry] Failed to load native ensemble: {e}")
            return None

    def register_model(self, version: str, model: object, scaler: object = None,
                       feat_cols: list[str] = None, metadata: dict = None,
                       scorer: Callable = None) -> ModelVersion:
        """Manually register a model version."""
        n_features = getattr(model, "n_features_", 0) or getattr(model, "n_features_in_", 0)
        feat_cols = feat_cols or []

        mv = ModelVersion(
            version=version,
            model=model,
            scaler=scaler,
            feat_cols=feat_cols,
            n_features=n_features,
            metadata=metadata or {},
        )

        if scorer:
            self._scorers[version] = scorer
        else:
            def default_scorer(X: np.ndarray, _m=model, _s=scaler) -> np.ndarray:
                Xs = _s.transform(X) if _s else X
                return _m.predict_proba(Xs)[:, 1]
            self._scorers[version] = default_scorer

        self._models[version] = mv
        return mv

    def get_scorer(self, version: str) -> Optional[Callable]:
        """Get the scoring function for a model version."""
        return self._scorers.get(version)

    def get_model(self, version: str) -> Optional[ModelVersion]:
        """Get metadata for a model version."""
        return self._models.get(version)

    def list_versions(self) -> list[dict]:
        """List all registered model versions."""
        result = []
        for v, mv in self._models.items():
            result.append({
                "version": mv.version,
                "n_features": mv.n_features,
                "hash": mv.hash,
                "loaded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(mv.loaded_at)),
                "metadata": {k: v for k, v in mv.metadata.items() if not isinstance(v, (list, dict)) or len(str(v)) < 200},
            })
        return result

    @property
    def versions(self) -> list[str]:
        """List all registered version strings."""
        return list(self._models.keys())

    @property
    def default_version(self) -> Optional[str]:
        """Return the most recently loaded version."""
        if not self._models:
            return None
        return max(self._models, key=lambda v: self._models[v].loaded_at)
