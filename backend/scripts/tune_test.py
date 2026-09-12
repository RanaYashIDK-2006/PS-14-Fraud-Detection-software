#!/usr/bin/env python3
"""Cache + sweep tests for scripts/tune_operating_point.py.

Checks the on-disk per-event score cache (v3: ML fusion score, critical
flag, and the fired-rule boolean mask) without paying the full fusion
cost: a stubbed fusion/rules engine drives `load_or_compute`, and we
assert
  1. first run computes and writes the cache,
  2. second run hits the cache and returns identical arrays,
  3. changing the model artifacts (mtime) changes the key and forces a
     recompute,
  4. the split boundaries are part of the key,
  5. the global-scale sweep math is consistent with the per-rule math at
     multiplier 1 (both derive from the fired-rule mask),
  6. the vectorized cost agrees with the reference band math, and
  7. the constrained per-rule search respects its grid bounds, never
     worsens the validation cost, is deterministic, and breaks ties
     toward the higher multiplier (fraud coverage).

Run from the project root:
  python scripts/tune_test.py
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent  # repo root.parent  # repo root
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "backend" / "scripts"))

import numpy as np
import pandas as pd

import tune_operating_point as tune  # noqa: E402

failures: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        failures.append(name)


class FakeFusion:
    def predict(self, features: dict) -> tuple:
        # deterministic, feature-dependent value (matches FusionEngine signature)
        val = 0.01 * sum(features.values()) % 1.0
        return val, {"model_variance": 0.0, "model_disagreement": 0.0, "individual_outputs": {}}


class FakeRules:
    severity_scale = 0.45

    def evaluate(self, features: dict) -> dict:
        fired = []
        if features["amount_ratio"] > 0.5:
            fired.append("RULE_A")
        if features["is_weekend"] > 0.5:
            fired.append("RULE_B")
        return {
            "score": 0.0,  # unused by the tuner (it recomputes from masks)
            "fired_rules": fired,
            "reason_codes": fired,
            "critical": features["is_weekend"] > 0.75,
        }


RULES_YAML = """\
severity_scale: 0.45
rules:
  - id: RULE_A
    severity: 0.6
  - id: RULE_B
    severity: 0.4
"""


def make_df(n: int = 40, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cols = tune.ML_FEATURES
    data = {c: rng.random(n) for c in cols}
    data["label"] = rng.integers(0, 2, n)
    data["ts"] = pd.date_range("2025-01-01", periods=n, freq="h")
    return pd.DataFrame(data)


def main() -> int:
    print("== tuner score-cache + per-rule sweep tests ==")
    tmp = Path(tempfile.mkdtemp(prefix="ps14-tune-"))
    artifacts = tmp / "artifacts"
    cache = tmp / "cache"
    artifacts.mkdir()
    (artifacts / "stacker.joblib").write_bytes(b"fake-model-v1")
    rules_file = tmp / "rules.yaml"
    rules_file.write_text(RULES_YAML)

    # point the module's identity inputs at the temp copies
    orig = (tune.ARTIFACTS_DIR, tune.RULES_PATH, tune.CACHE_DIR)
    tune.ARTIFACTS_DIR, tune.RULES_PATH, tune.CACHE_DIR = artifacts, rules_file, cache

    df = make_df(n=40)
    fusion, rules = FakeFusion(), FakeRules()
    vi, ti = 28, 34  # scored tail = rows 28: (12 rows: 6 val + 6 test)

    try:
        # ---- 1. cold run: computes + writes --------------------------------
        ml1, fired1, crit1, sev1, key1, hit1 = tune.load_or_compute(
            df, vi, ti, fusion, rules, use_cache=True
        )
        check("cold run computes (cache miss)", hit1 is False, f"key={key1}")
        files = list(cache.glob("scores_*.npz"))
        check("cold run writes a cache file", len(files) == 1, str(files))
        check("cache arrays cover the scored tail only",
              len(ml1) == len(crit1) == len(df) - vi, f"len={len(ml1)}")
        check("fired-rule mask has a column per rule",
              fired1.shape == (len(df) - vi, 2), f"shape={fired1.shape}")
        check("cache carries the yaml severities",
              np.allclose(sev1, [0.6, 0.4]), f"sev={sev1}")

        # ---- 2. warm run: cache hit, identical values ----------------------
        ml2, fired2, crit2, sev2, key2, hit2 = tune.load_or_compute(
            df, vi, ti, fusion, rules, use_cache=True
        )
        check("warm run hits the cache", hit2 is True and key2 == key1, f"key={key2}")
        check("warm run returns identical arrays",
              np.array_equal(ml1, ml2) and np.array_equal(fired1, fired2)
              and np.array_equal(crit1, crit2) and np.array_equal(sev1, sev2))
        check("critical flags captured",
              set(crit1) <= {False, True} and bool(crit1.any()),
              f"n_critical={int(crit1.sum())}")
        check("fired-rule mask uses both columns",
              bool(fired1[:, 0].any()) and bool(fired1[:, 1].any()),
              f"RULE_A={int(fired1[:, 0].sum())} RULE_B={int(fired1[:, 1].sum())}")

        # ---- 3. rules-state semantics --------------------------------------
        # severity_scale is a pure multiplier the sweep varies: changing it
        # must NOT invalidate the cached signals (the training pipeline writes
        # the tuned scale into rules.yaml on every run).
        rules_file.write_text(RULES_YAML.replace("severity_scale: 0.45", "severity_scale: 0.60"))
        _, _, _, _, key3, hit3 = tune.load_or_compute(df, vi, ti, fusion, rules, use_cache=True)
        check("severity_scale change keeps the cache", hit3 is True and key3 == key1, f"key={key3}")
        # a per-rule severity change DOES invalidate (fired masks/weights move)
        rules_file.write_text(RULES_YAML.replace("severity: 0.6", "severity: 0.7"))
        _, _, _, _, key4, hit4 = tune.load_or_compute(df, vi, ti, fusion, rules, use_cache=True)
        check("rule severity change invalidates the cache", hit4 is False and key4 != key1, f"key={key4}")

        # ---- 4. artifact change invalidates --------------------------------
        (artifacts / "stacker.joblib").write_bytes(b"fake-model-v2")  # new mtime
        _, _, _, _, key5, hit5 = tune.load_or_compute(df, vi, ti, fusion, rules, use_cache=True)
        check("artifact change invalidates the cache", hit5 is False and key5 != key1, f"key={key5}")

        # ---- 5. split-boundary change invalidates --------------------------
        _, _, _, _, key6, hit6 = tune.load_or_compute(df, vi, ti + 1, fusion, rules, use_cache=True)
        check("split change invalidates the cache", hit6 is False and key6 != key1, f"key={key6}")

        # ---- 6. global sweep math == per-rule math at mult 1 ---------------
        ml, fired, crit, sev = ml1, fired1, crit1, sev1
        g = tune.evaluate_global(ml, fired, sev, crit, 0.55)
        p = tune.evaluate_rules(ml, fired, sev, np.ones(len(sev)), 0.55, crit)
        check("global sweep math == per-rule math at mult 1",
              np.array_equal(g, p), f"n_diff={int((g != p).sum())}")
        # the rule score itself must match the engine's cap-after-scale
        # semantics (RulesEngine.evaluate: min(1, sum(sev) * scale))
        rule_masked = tune.rule_score(fired, sev, np.ones(len(sev)), 0.55)
        rule_literal = np.minimum(1.0, (fired @ sev) * 0.55)
        check("rule score caps after scaling (engine semantics)",
              np.allclose(rule_masked, rule_literal) and np.all(rule_masked <= 0.55 + 1e-12))

        # ---- 7. vectorized cost == reference band logic ---------------------
        s = tune.evaluate_rules(ml, fired, sev, np.ones(len(sev)), FakeRules.severity_scale, crit)
        label = df["label"].to_numpy()[vi:]
        fast_cost = tune.total_cost(s, label, 20.0)

        slow = 0.0
        for i in range(len(s)):
            if label[i] == 1:
                if s[i] <= 30:
                    slow += 20.0
                elif s[i] <= 70:
                    slow += 0.25 * 20.0
            else:
                if s[i] > 70:
                    slow += 1.0
                elif s[i] > 30:
                    slow += 0.15
            # legit low-band events cost nothing (that is the whole point)
        check("vectorized total_cost == reference",
              abs(fast_cost - slow) < 1e-9, f"{fast_cost} vs {slow}")

        # ---- 8. constrained per-rule search --------------------------------
        grid = np.round(np.arange(0.5, 1.51, 0.1), 2)
        base_cost = tune.total_cost(s, label, 20.0)
        mult, cost, history = tune.search_per_rule(
            ml, fired, sev, crit, label, 20.0, FakeRules.severity_scale, grid
        )
        in_grid = all(any(abs(m - gv) < 1e-9 for gv in grid) for m in mult)
        check("search multipliers within grid bounds", in_grid, f"mult={np.round(mult, 2)}")
        check("search never worsens val cost", cost <= base_cost + 1e-9,
              f"{cost:.2f} vs base {base_cost:.2f}")
        check("search history is monotonic non-increasing",
              all(history[i + 1]["val_cost"] <= history[i]["val_cost"] + 1e-9
                  for i in range(len(history) - 1)))
        mult_b, cost_b, _ = tune.search_per_rule(
            ml, fired, sev, crit, label, 20.0, FakeRules.severity_scale, grid
        )
        check("search deterministic", np.array_equal(mult, mult_b) and abs(cost - cost_b) < 1e-9)

        # tie-break: all multipliers give identical cost (rule stays in the
        # low band at every candidate) -> the highest multiplier must win
        ml_t = np.full(4, 0.2)
        fired_t = np.ones((4, 1), dtype=bool)
        sev_t = np.array([0.6])
        crit_t = np.zeros(4, dtype=bool)
        labels_t = np.array([0, 1, 1, 0])
        mult_t, _, _ = tune.search_per_rule(
            ml_t, fired_t, sev_t, crit_t, labels_t, 20.0, 0.45, grid
        )
        check("ties keep the higher multiplier", abs(mult_t[0] - 1.5) < 1e-9,
              f"mult={mult_t[0]}")

        # ---- 9. neutral-multiplier pruning ---------------------------------
        # Rule A fires on the fraud row: raising it moves the fraud from the
        # low band (cost 20) to med (cost 5) - a strict improvement, kept.
        # Rule B fires only on legit rows that stay in the low band at every
        # candidate - pure tie-drift to 1.5, must be reverted to 1.0.
        ml_p = np.full(3, 0.2)
        fired_p = np.array([[True, False], [False, True], [False, True]])
        sev_p = np.array([0.6, 0.4])
        crit_p = np.zeros(3, dtype=bool)
        labels_p = np.array([1, 0, 0])
        mult_raw, _, _ = tune.search_per_rule(
            ml_p, fired_p, sev_p, crit_p, labels_p, 20.0, 0.45, grid
        )
        check("search drifts the neutral rule to the bound",
              abs(mult_raw[1] - 1.5) < 1e-9 and mult_raw[0] > 1.0,
              f"mult={np.round(mult_raw, 2)}")
        mult_pruned, n_rev = tune.prune_neutral_multipliers(
            mult_raw.copy(), ml_p, fired_p, sev_p, crit_p, labels_p, 20.0, 0.45
        )
        check("prune reverts the neutral rule to 1.0",
              abs(mult_pruned[1] - 1.0) < 1e-9 and abs(mult_pruned[0] - mult_raw[0]) < 1e-9,
              f"mult={np.round(mult_pruned, 2)} reverted={n_rev}")
        cost_pruned = tune.total_cost(
            tune.evaluate_rules(ml_p, fired_p, sev_p, mult_pruned, 0.45, crit_p), labels_p, 20.0
        )
        cost_raw = tune.total_cost(
            tune.evaluate_rules(ml_p, fired_p, sev_p, mult_raw, 0.45, crit_p), labels_p, 20.0
        )
        check("pruning never worsens the cost", cost_pruned <= cost_raw + 1e-9,
              f"{cost_pruned:.2f} vs {cost_raw:.2f}")

        # ---- 11. --no-cache forces recompute -------------------------------
        _, _, _, _, _, hit_nc = tune.load_or_compute(df, vi, ti, fusion, rules, use_cache=False)
        check("no-cache forces recompute", hit_nc is False)
    finally:
        tune.ARTIFACTS_DIR, tune.RULES_PATH, tune.CACHE_DIR = orig

    print("\n" + ("ALL CHECKS PASSED" if not failures else f"{len(failures)} CHECK(S) FAILED"))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
