#!/usr/bin/env python3
"""Operating-point tuning for the Risk Engine (section 6).

Two modes:

  GLOBAL SCALE (default) - scans the rule `severity_scale` on the
  VALIDATION split and picks the value that minimizes a business-aligned
  cost: a missed fraud is weighted far above a challenged legitimate user.
  Backward-compatible: the exact sweep table is unchanged.

  PER-RULE (--per-rule) - generalizes the one-multiplier sweep to a
  constrained search over individual rule severities: greedy coordinate
  descent over per-rule multipliers (default 0.5..1.5 in 0.1 steps, i.e.
  each rule's yaml severity can be rebalanced roughly in half to +50%)
  minimizing the same validation cost, with ties keeping the higher
  multiplier (preserves fraud coverage). The chosen multipliers map back to
  concrete yaml severities in the report.

Costs (parameterized by --cost-ratio c = C_FN / C_FP):
  fraud allowed outright        : c      (missed fraud - the expensive error)
  fraud stepped up              : 0.25 c (OTP/biometric likely blocks it)
  legit challenged (verify)     : 1.0    (false positive - user trust cost)
  legit stepped up              : 0.15   (friction only)

Per-event signals are scale-invariant, so they are computed ONCE and cached
on disk (`models/cache/`): the ML fusion score, the critical-rule flag, and
the fired-rule mask (which rule fired for each event - what a per-rule
sweep needs). The cache is keyed on the cache-format version, the dataset
content, the model artifacts, rules.yaml, and the val/test split, so a
re-run (any cost ratio, either mode) reuses the scores and finishes in
seconds; retraining, regenerating the data, or editing the rules
invalidates it. The sweep itself is vectorized numpy.

Run: python scripts/tune_operating_point.py [--cost-ratio 20] [--per-rule] [--no-cache]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# sklearn emits config-propagation warnings on repeated threaded predict calls
warnings.filterwarnings("ignore", message="sklearn.utils.parallel.delayed")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.privacy_layer.features import ML_FEATURES  # noqa: E402
from src.risk_engine.fusion import FusionEngine  # noqa: E402
from src.risk_engine.rules_engine import RulesEngine  # noqa: E402

STEP_FRAUD = 0.25  # step-up mitigates most fraud
STEP_LEGIT = 0.15  # step-up is friction, not a block

ARTIFACTS_DIR = ROOT / "models" / "artifacts"
RULES_PATH = ROOT / "src" / "risk_engine" / "rules.yaml"
CACHE_DIR = ROOT / "models" / "cache"

# Bump when the cache layout changes (old .npz files are ignored).
CACHE_FORMAT_VERSION = 3


def total_cost(score: np.ndarray, label: np.ndarray, cost_ratio: float) -> float:
    """Vectorized business cost over scored events (0-100 scores)."""
    low = score <= 30
    med = (score > 30) & (score <= 70)
    high = score > 70
    fraud = label == 1
    legit = label == 0
    return (
        cost_ratio * int((fraud & low).sum())
        + STEP_FRAUD * cost_ratio * int((fraud & med).sum())
        + STEP_LEGIT * int((legit & med).sum())
        + 1.0 * int((legit & high).sum())
    )


# --------------------------------------------------------------------------
# Rules configuration (ids + yaml severities + severity_scale)
# --------------------------------------------------------------------------

def _rules_config() -> tuple[list[str], np.ndarray, float]:
    cfg = yaml.safe_load(RULES_PATH.read_text(encoding="utf-8"))
    ids = [r["id"] for r in cfg["rules"]]
    sev = np.array([float(r["severity"]) for r in cfg["rules"]])
    scale = float(cfg.get("severity_scale", 1.0))
    return ids, sev, scale


# --------------------------------------------------------------------------
# Cache
# --------------------------------------------------------------------------

def _content_hash(df: pd.DataFrame) -> str:
    """Dataset identity: the feature matrix + labels + timeline order."""
    payload = df[ML_FEATURES + ["label", "ts"]].to_csv(index=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _rules_state() -> str:
    """Rules identity for the cache key: per-rule severities, levels,
    conditions, and `severity_floor` - everything the per-event signals
    depend on. `severity_scale` is deliberately EXCLUDED: it is a pure
    multiplier the sweep varies, so changing it (e.g. the training pipeline
    writing a tuned value into rules.yaml) must not invalidate the cached
    signals - otherwise every yaml write would force a ~3-minute recompute."""
    cfg = yaml.safe_load(RULES_PATH.read_text(encoding="utf-8"))
    body = {
        "severity_floor": cfg.get("severity_floor"),
        "rules": [
            {k: r.get(k) for k in ("id", "severity", "level", "condition")}
            for r in cfg["rules"]
        ],
    }
    return hashlib.sha256(
        json.dumps(body, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def _artifacts_state() -> str:
    """Model-artifact identity: every file's name + size + mtime, plus the
    rules state (per-rule severities are what a per-rule sweep adjusts)."""
    h = hashlib.sha256()
    for p in sorted(ARTIFACTS_DIR.iterdir()):
        st = p.stat()
        h.update(p.name.encode("utf-8"))
        h.update(str(st.st_size).encode("utf-8"))
        h.update(str(st.st_mtime_ns).encode("utf-8"))
    h.update(_rules_state().encode("utf-8"))
    return h.hexdigest()


def _cache_key(df: pd.DataFrame, vi: int, ti: int) -> str:
    return hashlib.sha256(
        (f"v{CACHE_FORMAT_VERSION}|" + _content_hash(df) + f"|{vi}|{ti}|" + _artifacts_state()).encode("utf-8")
    ).hexdigest()[:24]


def compute_tail(df: pd.DataFrame, vi: int, fusion: FusionEngine, rules: RulesEngine):
    """Per-event signals for the scored tail (rows [vi:], i.e. val + test):
    ML fusion score, critical flag, and the fired-rule boolean mask. This is
    the expensive part - run once, cache it. The head rows (0:vi) are never
    scored, keeping the first compute at the same cost as before the cache.
    """
    ids, sev, _ = _rules_config()
    idx = {rid: i for i, rid in enumerate(ids)}
    tail = df.iloc[vi:]
    n_rows = len(tail)
    ml = np.zeros(n_rows)
    fired = np.zeros((n_rows, len(ids)), dtype=bool)
    critical = np.zeros(n_rows, dtype=bool)
    for j, (_, r) in enumerate(tail.iterrows()):
        feats = {f: float(r[f]) for f in ML_FEATURES}
        ml[j] = fusion.predict(feats)[0]
        ev = rules.evaluate(feats)
        for rid in ev["fired_rules"]:
            fired[j, idx[rid]] = True
        critical[j] = bool(ev["critical"])
    return ml, fired, critical, sev


def load_or_compute(df: pd.DataFrame, vi: int, ti: int, fusion: FusionEngine, rules: RulesEngine, use_cache: bool):
    """Return (ml, fired, critical, sev, key, cache_hit) aligned to df.iloc[vi:]."""
    key = _cache_key(df, vi, ti)
    path = CACHE_DIR / f"scores_{key}.npz"
    want = len(df) - vi
    if use_cache and path.exists():
        z = np.load(path)
        if len(z["ml"]) == want and z["fired"].shape == (want, z["sev"].size):
            return z["ml"], z["fired"], z["critical"], z["sev"], key, True
    ml, fired, critical, sev = compute_tail(df, vi, fusion, rules)
    if use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.savez(path, ml=ml, fired=fired, critical=critical, sev=sev)
    return ml, fired, critical, sev, key, False


# --------------------------------------------------------------------------
# Scoring at a candidate operating point
# --------------------------------------------------------------------------

def rule_score(fired: np.ndarray, sev: np.ndarray, mult: np.ndarray, scale: float) -> np.ndarray:
    """Rule score at per-rule multipliers `mult` (1.0 = the yaml severities),
    computed exactly like the engine: min(1, sum(sev_i * mult_i) * scale).
    The cap applies AFTER scaling (RulesEngine.evaluate), so multi-rule
    events keep their full combined weight."""
    return np.minimum(1.0, (fired @ (sev * mult)) * scale)


def evaluate_rules(ml: np.ndarray, fired: np.ndarray, sev: np.ndarray, mult: np.ndarray,
                   scale: float, critical: np.ndarray) -> np.ndarray:
    """0-100 score for a per-rule multiplier vector (same as the Risk
    Engine: 100 * max(ml, rule), critical rules floor at 80)."""
    rule = rule_score(fired, sev, mult, scale)
    score = np.round(100 * np.minimum(1.0, np.maximum(ml, rule))).astype(int)
    score = np.where(critical, np.maximum(score, 80), score)
    return score


def evaluate_global(ml: np.ndarray, fired: np.ndarray, sev: np.ndarray, critical: np.ndarray, scale: float) -> np.ndarray:
    """Global severity_scale candidate, scored exactly as the engine would
    with that scale: rule = min(1, sum(sev) * scale)."""
    rule = np.minimum(1.0, (fired @ sev) * scale)
    score = np.round(100 * np.minimum(1.0, np.maximum(ml, rule))).astype(int)
    score = np.where(critical, np.maximum(score, 80), score)
    return score


# --------------------------------------------------------------------------
# Constrained per-rule search
# --------------------------------------------------------------------------

def search_per_rule(ml: np.ndarray, fired: np.ndarray, sev: np.ndarray, critical: np.ndarray,
                    labels: np.ndarray, cost_ratio: float, scale: float,
                    grid: np.ndarray, max_passes: int = 8) -> tuple[np.ndarray, float, list[dict]]:
    """Greedy coordinate descent over per-rule multipliers, bounded to
    `grid`. Objective: validation cost; ties keep the higher multiplier
    (preserves fraud coverage). Returns (multipliers, best_cost, history)."""
    rng = np.random.default_rng(0)  # deterministic rule order randomization
    order = rng.permutation(len(sev))  # visit rules in a fixed random order
    mult = np.ones(len(sev))
    best_cost = total_cost(evaluate_rules(ml, fired, sev, mult, scale, critical), labels, cost_ratio)
    history = [{"pass": 0, "mult": mult.copy(), "val_cost": best_cost}]
    for _ in range(max_passes):
        improved = False
        for i in order:
            cur = mult[i]
            best_m, best_c = cur, best_cost
            for m in grid:
                if abs(m - cur) < 1e-12:
                    continue
                cand = mult.copy()
                cand[i] = m
                c = total_cost(evaluate_rules(ml, fired, sev, cand, scale, critical), labels, cost_ratio)
                if c < best_c - 1e-9 or (abs(c - best_c) <= 1e-9 and m > best_m):
                    best_m, best_c = m, c
            if best_m != cur:
                mult[i] = best_m
                best_cost = best_c
                improved = True
        history.append({"pass": len(history), "mult": mult.copy(), "val_cost": best_cost})
        if not improved:
            break
    return mult, best_cost, history


def prune_neutral_multipliers(mult: np.ndarray, ml: np.ndarray, fired: np.ndarray,
                              sev: np.ndarray, critical: np.ndarray, labels: np.ndarray,
                              cost_ratio: float, scale: float) -> tuple[np.ndarray, int]:
    """Revert per-rule multipliers that do not STRICTLY reduce the validation
    cost. The greedy search accepts cost ties toward the higher multiplier
    (fraud coverage), so a rule that never moves an event across a band
    boundary drifts to the grid's upper bound without contributing anything.
    This pass holds every other rule at its chosen value and reverts any
    multiplier whose removal does not worsen the cost, iterating to a fixed
    point (reverting one rule can make another neutral). Returns the pruned
    multipliers and how many were reverted."""
    reverted = 0
    while True:
        any_change = False
        for i in range(len(mult)):
            if abs(mult[i] - 1.0) < 1e-12:
                continue
            cur = total_cost(evaluate_rules(ml, fired, sev, mult, scale, critical), labels, cost_ratio)
            cand = mult.copy()
            cand[i] = 1.0
            c = total_cost(evaluate_rules(ml, fired, sev, cand, scale, critical), labels, cost_ratio)
            if c <= cur + 1e-9:  # removing it does not worsen -> pure tie-drift
                mult = cand
                reverted += 1
                any_change = True
        if not any_change:
            break
    return mult, reverted


# --------------------------------------------------------------------------
# Generalization reporting (group-split OOD numbers)
# --------------------------------------------------------------------------

def generalization_md() -> str:
    """Append the leave-one-archetype-out generalization table produced by
    train_compare (models/artifacts/generalization_by_archetype.csv) to a
    tuning report, so the operating-point report carries the honest OOD
    numbers next to the leaky time-split ones."""
    path = ARTIFACTS_DIR / "generalization_by_archetype.csv"
    if not path.exists():
        return ""
    g = pd.read_csv(path)
    return (
        "\n## Out-of-distribution generalization (group-split eval)\n\n"
        "From `src/train_compare.py` leave-one-archetype-out retraining: the"
        " ensemble is retrained with each fraud archetype EXCLUDED, so the"
        " recall below is on fraud patterns the model has never seen - the"
        " honest counterpart to the test-split table above. `novel legit`"
        " rows are the cold-start FPR on accounts held out entirely.\n\n"
        "| held-out group | n fraud | recall (F1) | recall@1%FPR | OOD legit FPR | avg ml (fraud) |\n"
        "|---|---|---|---|---|---|\n"
        + "\n".join(
            f"| {r['held_out']} | {r['n_fraud']} | {r['recall_f1']} | "
            f"{r['recall_at_1pct_fpr']} | {r['ood_legit_fpr']} | {r['avg_ml_fraud']} |"
            for r in g.to_dict(orient="records")
        )
        + "\n"
    )


def archetype_recall_md(score: np.ndarray, labels: np.ndarray,
                        arch: np.ndarray | None, scale: float) -> str:
    """Per-archetype fraud recall at ONE operating point (0-100 scores)."""
    if arch is None:
        return ""
    rows = []
    for a in sorted(set(arch)):
        m = (arch == a) & (labels == 1)
        n = int(m.sum())
        if n == 0:
            continue
        caught = int((score[m] > 70).sum())
        rows.append(f"| {a} | {n} | {caught} | {caught / n:.1%} |")
    if not rows:
        return ""
    return (
        f"\n## Fraud recall by archetype at scale {scale:.2f} (test split)\n\n"
        "| archetype | n fraud (test) | caught (verify) | recall |\n|---|---|---|---|\n"
        + "\n".join(rows)
        + "\n\n> The test split is optimistic - every archetype also appears in"
        " training (same-pattern leakage). For the honest generalization"
        " numbers see the group-split table below.\n"
    )


# --------------------------------------------------------------------------
# Global severity_scale sweep (importable by the training pipeline)
# --------------------------------------------------------------------------

def sweep_global_scale(ml_val: np.ndarray, fired_val: np.ndarray, crit_val: np.ndarray,
                       sev: np.ndarray, lab_val: np.ndarray,
                       ml_test: np.ndarray, fired_test: np.ndarray, crit_test: np.ndarray,
                       lab_test: np.ndarray, cost_ratio: float,
                       report_path: Path | None = None,
                       arch_test: np.ndarray | None = None) -> tuple[float, str]:
    """Run the global `severity_scale` sweep on the scored tail splits and
    return (best_scale, report_md). Prints the sweep table as it goes
    (identical output to the CLI). The markdown report is written to
    `report_path` when given. Selection rule: eliminate legit challenges
    (legit_high == 0) and pick the HIGHEST such scale; if none exists, pick
    the highest scale with the fewest challenges (fraud coverage preserved)."""
    n_fraud_test = int(lab_test.sum())
    n_legit_test = int((lab_test == 0).sum())
    print("scale  val_cost  test_cost  legit_high  fraud_high(recall)  fraud_low(missed)")
    results = []
    for scale in np.round(np.arange(0.45, 1.01, 0.05), 2):
        s_val = evaluate_global(ml_val, fired_val, sev, crit_val, scale)
        s_test = evaluate_global(ml_test, fired_test, sev, crit_test, scale)
        v_cost = total_cost(s_val, lab_val, cost_ratio)
        t_cost = total_cost(s_test, lab_test, cost_ratio)
        legit_high = int(((s_test > 70) & (lab_test == 0)).sum())
        fraud_high = int(((s_test > 70) & (lab_test == 1)).sum())
        fraud_low = int(((s_test <= 30) & (lab_test == 1)).sum())
        results.append((scale, v_cost, t_cost, legit_high, fraud_high, fraud_low))
        print(f"{scale:.2f}  {v_cost:8.2f}  {t_cost:8.2f}  {legit_high:5d}  {fraud_high:6d} ({fraud_high / max(n_fraud_test, 1):6.1%})  {fraud_low:6d}")

    zero_challenge = [r for r in results if r[3] == 0]
    if zero_challenge:
        best_scale = max(zero_challenge, key=lambda r: r[0])[0]
    else:
        fewest = min(results, key=lambda r: r[3])[3]
        best_scale = max((r for r in results if r[3] == fewest), key=lambda r: r[0])[0]

    def stats(scale: float) -> dict:
        s = evaluate_global(ml_test, fired_test, sev, crit_test, scale)
        return {
            "legit_high": int(((s > 70) & (lab_test == 0)).sum()),
            "fraud_high": int(((s > 70) & (lab_test == 1)).sum()),
            "fraud_low": int(((s <= 30) & (lab_test == 1)).sum()),
            "n_fraud": n_fraud_test, "n_legit": n_legit_test,
            "cost": total_cost(s, lab_test, cost_ratio),
        }

    before, after = stats(1.0), stats(best_scale)
    md = f"""# Risk Engine operating-point tuning (section 6)

Cost-weighted objective with C_FN / C_FP = {cost_ratio:.1f}
(step-up: fraud 0.25x, legit 0.15x). Scanned `severity_scale` on the
**validation** split. Selection rule: eliminate legit challenges
(legit_high == 0), then pick the HIGHEST scale among those (closest to the
original calibration, so fraud coverage is maximally preserved). Metrics
below are on the held-out **test** split
({n_fraud_test} fraud / {n_legit_test} legit).

| metric | scale 1.0 (before) | scale {best_scale:.2f} (chosen) |
|---|---|---|
| legit events challenged (verify) | {before['legit_high']} ({before['legit_high'] / before['n_legit']:.2%}) | {after['legit_high']} ({after['legit_high'] / after['n_legit']:.2%}) |
| fraud events caught (verify) | {before['fraud_high']} ({before['fraud_high'] / before['n_fraud']:.1%}) | {after['fraud_high']} ({after['fraud_high'] / after['n_fraud']:.1%}) |
| fraud allowed outright (low) | {before['fraud_low']} ({before['fraud_low'] / before['n_fraud']:.1%}) | {after['fraud_low']} ({after['fraud_low'] / after['n_fraud']:.1%}) |
| total cost | {before['cost']:.2f} | {after['cost']:.2f} |

**Recommended `severity_scale`: {best_scale:.2f}** (set in
`src/risk_engine/rules.yaml`; critical rules still floor at
`severity_floor`, so account-takeover signals are unaffected).

Validation cost table (test counts out of {n_fraud_test} fraud / {n_legit_test} legit):

| scale | val cost | test cost | legit high | fraud high (recall) | fraud low (missed) |
|---|---|---|---|---|---|
{chr(10).join(f"| {s:.2f} | {vc:.2f} | {tc:.2f} | {lh} | {fh} ({fh / n_fraud_test:.1%}) | {fl} |" for s, vc, tc, lh, fh, fl in results)}
"""
    chosen = evaluate_global(ml_test, fired_test, sev, crit_test, best_scale)
    md += archetype_recall_md(chosen, lab_test, arch_test, best_scale)
    md += generalization_md()
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(md, encoding="utf-8")
        print(f"Report written to {report_path}")
    return best_scale, md


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cost-ratio", type=float, default=20.0)
    ap.add_argument("--data", default="data/transactions.csv")
    ap.add_argument("--out", default=None, help="report path (defaults by mode)")
    ap.add_argument("--per-rule", action="store_true", help="run the constrained per-rule severity search")
    ap.add_argument("--no-cache", action="store_true", help="recompute per-event scores, ignore the cache")
    ap.add_argument("--min-mult", type=float, default=0.5, help="per-rule multiplier lower bound (--per-rule)")
    ap.add_argument("--max-mult", type=float, default=1.5, help="per-rule multiplier upper bound (--per-rule)")
    ap.add_argument("--mult-step", type=float, default=0.1, help="per-rule multiplier step (--per-rule)")
    args = ap.parse_args()

    df = pd.read_csv(ROOT / args.data, parse_dates=["ts"]).sort_values("ts").reset_index(drop=True)
    n = len(df)
    vi, ti = int(n * 0.70), int(n * 0.85)
    val = df.iloc[vi:ti].reset_index(drop=True)
    test = df.iloc[ti:].reset_index(drop=True)

    fusion = FusionEngine(ARTIFACTS_DIR)
    rules = RulesEngine.from_yaml(RULES_PATH)
    _, sev, base_scale = _rules_config()

    print(f"cost ratio C_FN/C_FP = {args.cost_ratio:.1f}  (val={len(val)}, test={len(test)})"
          + ("  [per-rule mode]" if args.per_rule else ""))

    # Per-event signals are scale-invariant: compute once (cached on disk).
    t0 = time.time()
    ml_tail, fired_tail, crit_tail, sev_tail, key, hit = load_or_compute(
        df, vi, ti, fusion, rules, use_cache=not args.no_cache
    )
    elapsed = time.time() - t0
    print(f"per-event scores: {'cache HIT' if hit else 'computed'} ({elapsed:.1f}s)  key={key}")
    n_val = ti - vi
    ml_val, fired_val, crit_val = ml_tail[:n_val], fired_tail[:n_val], crit_tail[:n_val]
    ml_test, fired_test, crit_test = ml_tail[n_val:], fired_tail[n_val:], crit_tail[n_val:]
    lab_val = val["label"].to_numpy()
    lab_test = test["label"].to_numpy()

    n_fraud_test = int(lab_test.sum())
    n_legit_test = int((lab_test == 0).sum())

    if not args.per_rule:
        best_scale, _ = sweep_global_scale(
            ml_val, fired_val, crit_val, sev, lab_val,
            ml_test, fired_test, crit_test, lab_test,
            args.cost_ratio, ROOT / (args.out or "models/tuning_report.md"),
            arch_test=test["archetype"].to_numpy() if "archetype" in test.columns else None,
        )
        print(f"\nRecommended severity_scale: {best_scale:.2f}")
        return 0

    # ---- constrained per-rule severity search ------------------------------
    grid = np.round(np.arange(args.min_mult, args.max_mult + args.mult_step / 2, args.mult_step), 2)
    ids, _, _ = _rules_config()
    print(f"\nper-rule multiplier grid: {grid.tolist()}  (bounds [{args.min_mult}, {args.max_mult}])")
    print(f"baseline val cost (mult = 1): {total_cost(evaluate_rules(ml_val, fired_val, sev, np.ones(len(sev)), base_scale, crit_val), lab_val, args.cost_ratio):.2f}")

    mult, val_cost, history = search_per_rule(
        ml_val, fired_val, sev, crit_val, lab_val, args.cost_ratio, base_scale, grid
    )
    mult, n_reverted = prune_neutral_multipliers(
        mult, ml_val, fired_val, sev, crit_val, lab_val, args.cost_ratio, base_scale
    )
    new_sev = sev * mult

    print("\nrule rebalancing (greedy coordinate descent, ties keep higher multiplier):")
    print(f"{'rule':<26} {'base sev':>8} {'mult':>5} {'new sev':>8}")
    for i, rid in enumerate(ids):
        flag = "  <-- changed" if abs(mult[i] - 1.0) > 1e-9 else ""
        print(f"{rid:<26} {sev[i]:8.3f} {mult[i]:5.2f} {new_sev[i]:8.3f}{flag}")
    print(f"\nval cost: {history[0]['val_cost']:.2f} -> {val_cost:.2f} after {len(history) - 1} pass(es)")
    if n_reverted:
        print(f"pruned {n_reverted} neutral multiplier(s) (tie-drift only, no cost effect)")

    def pstats(m: np.ndarray) -> dict:
        s = evaluate_rules(ml_test, fired_test, sev, m, base_scale, crit_test)
        return {
            "legit_high": int(((s > 70) & (lab_test == 0)).sum()),
            "fraud_high": int(((s > 70) & (lab_test == 1)).sum()),
            "fraud_low": int(((s <= 30) & (lab_test == 1)).sum()),
            "n_fraud": n_fraud_test, "n_legit": n_legit_test,
            "cost": total_cost(s, lab_test, args.cost_ratio),
        }

    before, after = pstats(np.ones(len(sev))), pstats(mult)
    rows = "\n".join(
        f"| {ids[i]} | {sev[i]:.3f} | {mult[i]:.2f} | {new_sev[i]:.3f} |"
        for i in range(len(sev))
    )
    md = f"""# Per-rule severity tuning (section 6)

Constrained search over individual rule severities, objective: validation
cost with C_FN / C_FP = {args.cost_ratio:.1f}. Greedy coordinate descent
over per-rule multipliers bounded to [{args.min_mult}, {args.max_mult}]
(step {args.mult_step}); ties keep the higher multiplier (preserves fraud
coverage), then multipliers that do not strictly reduce the validation
cost are reverted ({n_reverted} neutral change(s) pruned) so only
cost-proven rebalances are recommended. `severity_scale` stays at
{base_scale:.2f}. Metrics below are on the held-out **test** split
({n_fraud_test} fraud / {n_legit_test} legit).

| rule | base severity | multiplier | new severity |
|---|---|---|---|
{rows}

| metric | before (mult = 1) | after (rebalanced) |
|---|---|---|
| legit events challenged (verify) | {before['legit_high']} ({before['legit_high'] / before['n_legit']:.2%}) | {after['legit_high']} ({after['legit_high'] / after['n_legit']:.2%}) |
| fraud events caught (verify) | {before['fraud_high']} ({before['fraud_high'] / before['n_fraud']:.1%}) | {after['fraud_high']} ({after['fraud_high'] / after['n_fraud']:.1%}) |
| fraud allowed outright (low) | {before['fraud_low']} ({before['fraud_low'] / before['n_fraud']:.1%}) | {after['fraud_low']} ({after['fraud_low'] / after['n_fraud']:.1%}) |
| total cost | {before['cost']:.2f} | {after['cost']:.2f} |

Validation cost trace (greedy passes):

| pass | val cost |
|---|---|
{chr(10).join(f"| {h['pass']} | {h['val_cost']:.2f} |" for h in history)}

**Recommended per-rule severities** (multiplier x base; set directly in
`src/risk_engine/rules.yaml`, `severity_scale` unchanged at {base_scale:.2f}):
{', '.join(f'{ids[i]} {new_sev[i]:.3f}' for i in range(len(sev)))}
"""
    md += generalization_md()
    out = ROOT / (args.out or "models/tuning_report_per_rule.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md, encoding="utf-8")
    print(f"\nReport written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
