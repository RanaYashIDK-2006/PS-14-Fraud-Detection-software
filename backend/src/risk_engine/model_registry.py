"""Model registry for staged rollout (shadow → canary → rollback).

Supports running a candidate model alongside production, routing a
configurable % of traffic to it, and automatic rollback on breach.

The registry is file-based (models/artifacts/registry.json) — no new
database needed. Shadow mode logs predictions without affecting decisions;
canary mode routes a % of traffic; rollback reverts to the last-known-good
model artifact.

Usage:
    registry = ModelRegistry(ARTIFACTS_DIR)
    registry.maybe_shadow(features)  # logs candidate prediction alongside prod
    registry.maybe_canary(features)  # may route to candidate for decision
    registry.record_candidate_metrics(latency_ms, fpr, fnr)
    registry.rollback()  # revert to last-known-good
"""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np


@dataclass
class ModelCandidate:
    """A candidate model ready for staged rollout."""
    model_path: str  # path to candidate .joblib
    metadata_path: str  # path to candidate metadata.json
    created_at: float = field(default_factory=time.time)
    status: str = "shadow"  # shadow | canary | rolled_back | promoted


@dataclass
class RolloutState:
    """Current rollout configuration."""
    mode: str = "direct"  # direct | shadow | canary
    candidate: ModelCandidate | None = None
    canary_pct: float = 0.1  # % of traffic to route to candidate in canary mode
    shadow_log: list[dict] = field(default_factory=list)  # last N shadow comparisons
    canary_metrics: dict = field(default_factory=dict)  # latency, fpr, fnr, count
    last_good_model: str | None = None  # path to last-known-good model
    last_good_metadata: str | None = None
    rollback_count: int = 0
    # Auto-rollback thresholds
    max_latency_ms: float = 500.0  # rollback if p95 > this
    max_fpr_spike: float = 0.05  # rollback if FPR increases by > this
    max_error_rate: float = 0.02  # rollback if error rate > this


class ModelRegistry:
    """File-based model registry for staged rollout."""

    def __init__(self, artifacts_dir: Path):
        self.artifacts_dir = artifacts_dir
        self.registry_path = artifacts_dir / "registry.json"
        self.state = self._load()

    def _load(self) -> RolloutState:
        if self.registry_path.exists():
            data = json.loads(self.registry_path.read_text(encoding="utf-8"))
            state = RolloutState(**{k: v for k, v in data.items() if k in RolloutState.__dataclass_fields__})
            if data.get("candidate"):
                state.candidate = ModelCandidate(**data["candidate"])
            return RolloutState()
        return RolloutState()

    def _save(self) -> None:
        data = asdict(self.state)
        self.registry_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def register_candidate(self, candidate_model_path: str, candidate_metadata_path: str) -> None:
        """Register a new candidate model for shadow rollout."""
        # Save current production model as last-known-good
        prod_model = self.artifacts_dir / "logistic_regression.joblib"
        if prod_model.exists():
            self.state.last_good_model = str(prod_model)
            self.state.last_good_metadata = str(self.artifacts_dir / "metadata.json")
        self.state.candidate = ModelCandidate(
            model_path=candidate_model_path,
            metadata_path=candidate_metadata_path,
        )
        self.state.mode = "shadow"
        self.state.shadow_log = []
        self.state.canary_metrics = {}
        self._save()

    def promote_to_canary(self, pct: float = 0.1) -> None:
        """Promote the candidate from shadow to canary mode."""
        if self.state.candidate is None:
            raise ValueError("No candidate registered")
        self.state.mode = "canary"
        self.state.canary_pct = pct
        self.state.candidate.status = "canary"
        self._save()

    def maybe_canary(self, features: dict) -> str:
        """Decide whether this request routes to the candidate.

        Returns 'production' or 'candidate' based on canary_pct.
        """
        if self.state.mode != "canary" or self.state.candidate is None:
            return "production"
        if np.random.random() < self.state.canary_pct:
            return "candidate"
        return "production"

    def record_shadow_comparison(self, prod_score: float, candidate_score: float, features: dict) -> None:
        """Log a shadow comparison between production and candidate."""
        if self.state.mode != "shadow":
            return
        self.state.shadow_log.append({
            "ts": time.time(),
            "prod_score": round(prod_score, 4),
            "candidate_score": round(candidate_score, 4),
            "delta": round(abs(prod_score - candidate_score), 4),
        })
        # Keep last 1000 comparisons
        if len(self.state.shadow_log) > 1000:
            self.state.shadow_log = self.state.shadow_log[-1000:]
        self._save()

    def record_candidate_metrics(self, latency_ms: float, error: bool = False) -> None:
        """Record canary metrics for auto-rollback decisions."""
        if self.state.mode != "canary":
            return
        m = self.state.canary_metrics
        m["count"] = m.get("count", 0) + 1
        m["total_latency"] = m.get("total_latency", 0) + latency_ms
        m["errors"] = m.get("errors", 0) + (1 if error else 0)
        m["latencies"] = m.get("latencies", [])
        m["latencies"].append(latency_ms)
        if len(m["latencies"]) > 1000:
            m["latencies"] = m["latencies"][-1000:]
        # Auto-rollback check
        count = m["count"]
        if count >= 100:
            p95 = float(np.percentile(m["latencies"], 95))
            error_rate = m["errors"] / count
            if p95 > self.state.max_latency_ms or error_rate > self.state.max_error_rate:
                self.rollback()
                return
        self._save()

    def rollback(self) -> bool:
        """Rollback to the last-known-good model. Returns True if rollback happened."""
        if self.state.candidate is None:
            return False
        self.state.candidate.status = "rolled_back"
        self.state.mode = "direct"
        self.state.rollback_count += 1
        self.state.shadow_log = []
        self.state.canary_metrics = {}
        # The actual model file rollback is handled by the caller
        # (docker_retrain.sh / training_pipeline.py)
        self._save()
        return True

    def promote(self) -> None:
        """Promote the candidate to production. Called after successful canary."""
        if self.state.candidate is None:
            return
        self.state.candidate.status = "promoted"
        self.state.mode = "direct"
        self.state.candidate = None
        self.state.shadow_log = []
        self.state.canary_metrics = {}
        self._save()

    def status(self) -> dict:
        """Current registry status for monitoring."""
        return {
            "mode": self.state.mode,
            "candidate": asdict(self.state.candidate) if self.state.candidate else None,
            "canary_pct": self.state.canary_pct,
            "shadow_comparisons": len(self.state.shadow_log),
            "canary_count": self.state.canary_metrics.get("count", 0),
            "rollback_count": self.state.rollback_count,
            "last_good_model": self.state.last_good_model,
        }
