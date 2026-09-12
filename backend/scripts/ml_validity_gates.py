#!/usr/bin/env python3
"""PS-14 ML-VALIDITY GATES - causal features, label latency, final-test quarantine.

Adversarial verification of the corrected feature pipeline + protocol, written
as its own harness (independent of scripts/ml_validity_rebuild.py which only
consumes the pre-featurized dataset CSV):

  A. FEATURE CAUSALITY (future-perturbation test, per feature)
     For every historical feature the test recomputes the feature vector of a
     transaction at time t from (i) the account history up to t only and
     (ii) the same history PLUS perturbed FUTURE events (fraud bursts, amount
     shocks, label flips, extra events). Causal features must be BIT-IDENTICAL
     under every perturbation (max |diff| == 0). The legacy derive_features()
     is run on the same accounts for disclosure: it shifts when a future event
     is added (account-wide median denominator).

  B. LABEL-AVAILABILITY LEAKAGE
     Static + dynamic check that no MODEL feature value is derived from any
     fraud label (the label column is copied through to the output row as the
     OUTCOME only). Because no feature reads a label, label-latency cannot
     leak into features; this is asserted, not assumed.

  C. FINAL-TEST QUARANTINE (recorded-artifact + source assertions)
     - scaler is fitted on TRAIN rows only (source assertion)
     - early stopping eval_set references VALIDATION, never FINAL TEST
     - threshold is selected on VALIDATION scores only (recomputed from the
       recorded predictions), then applied unchanged to the final test
     - the recorded protocol's locked threshold equals the validation one

Run:  ./.venv/Scripts/python.exe scripts/ml_validity_gates.py
Exit 0 = all gates PASS. Writes reports/ml_validity_causality_audit.json.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent  # repo root.parent  # repo root
sys.path.insert(0, str(ROOT / "backend"))

PROTOCOL = ROOT / "reports" / "ml_validity_protocol.json"
PRED = ROOT / "reports" / "ml_validity_predictions.npz"
AUDIT_OUT = ROOT / "reports" / "ml_validity_causality_audit.json"

# The 21 modeling features, classified by CODE INSPECTION of
# derive_features_causal: does the computation consume any prior account
# event (rows < i) or only the current event + fixed references?
HISTORICAL = {
    # expanding/rolling over prior events (prior-only medians, prior counts,
    # inter-arrival gaps, running device count, tenure)
    "amount_ratio", "txn_freq_last_24h", "days_since_last_similar_txn",
    "gradual_escalation_score", "known_device_count", "account_tenure_days",
    "amount_zscore", "velocity_deviation", "txn_regularity",
}
INSTANT = {
    # current-event value or fixed reference only - no prior rows consumed
    "hour_of_day", "is_weekend", "txn_time_unusual", "new_device_flag",
    "unusual_location_flag", "unusual_recipient_flag", "failed_auth_count_24h",
    "shared_device_accounts", "shared_recipient_accounts", "mule_ring_score",
    "hour_deviation", "recipient_novelty",
}
ALL_FEATURES = sorted(HISTORICAL | INSTANT)
TOL = 0.0  # bit-identical required (same prefix ops => identical floats)


def make_account(rng: np.random.Generator, archetype: str, n: int,
                 start: pd.Timestamp) -> list[dict]:
    """Raw events in the derive_features schema; ts strictly increasing."""
    ts = [start + pd.Timedelta(hours=int(h)) for h in np.cumsum(rng.integers(20, 80, n))]
    amounts = np.round(rng.lognormal(4.2, 0.35, n), 2)  # ~$15-250 typical
    hours = rng.integers(8, 22, n)
    new_dev = rng.random(n) < 0.04
    un_loc = rng.random(n) < 0.02
    un_rcp = rng.random(n) < 0.02
    auth = rng.choice([0, 1, 2], n, p=[0.90, 0.08, 0.02])
    lab = np.zeros(n, dtype=int)
    shared_d = np.zeros(n, dtype=int)
    shared_r = np.zeros(n, dtype=int)
    if archetype == "escalation":        # boiling-frog ramp on an old account
        ramp = np.linspace(1.0, 7.0, n)
        amounts = np.round(80.0 * ramp, 2)
    elif archetype == "ato":             # burst at the end: new device/loc
        b = int(n * 0.7)
        new_dev[b:] = True
        un_loc[b:] = True
        auth[b:] = rng.integers(1, 4, n - b)
        lab[b:] = 1
        amounts[b:] = np.round(rng.lognormal(6.0, 0.4, n - b), 2)
    elif archetype == "mule":            # ring context on every event
        shared_d[:] = rng.integers(2, 5)
        shared_r[:] = rng.integers(2, 5)
    elif archetype == "impulse":         # single big fraud near the end
        i = n - 1
        amounts[i] = round(2000.0, 2)
        new_dev[i] = True
        hours[i] = 3
        lab[i] = 1
    return [
        {
            "fraud_id": f"acct-{archetype}",
            "ts": ts[i], "amount": float(amounts[i]), "hour": int(hours[i]),
            "new_device": bool(new_dev[i]), "unusual_location": bool(un_loc[i]),
            "unusual_recipient": bool(un_rcp[i]), "failed_auth": int(auth[i]),
            "archetype": archetype, "label": int(lab[i]),
            "shared_device_accounts": int(shared_d[i]),
            "shared_recipient_accounts": int(shared_r[i]),
            "recipient": "R-001",
        }
        for i in range(n)
    ]


def future_suffix(rng: np.random.Generator, last_ts: pd.Timestamp,
                  kind: str) -> list[dict]:
    """Future events STRICTLY after last_ts; kind selects the perturbation."""
    n = 12
    ts = [last_ts + pd.Timedelta(hours=int(h))
          for h in np.cumsum(rng.integers(2, 30, n))]
    if kind == "fraud_burst":
        amounts = np.round(rng.lognormal(6.2, 0.5, n), 2)   # 3-8x typical
        lab = np.ones(n, dtype=int)
        new_dev = np.ones(n, dtype=bool)
        hours = rng.integers(0, 6, n)
    elif kind == "amount_shock":
        amounts = np.round(rng.lognormal(8.5, 0.6, n), 2)   # huge
        lab = np.zeros(n, dtype=int)
        new_dev = np.zeros(n, dtype=bool)
        hours = rng.integers(8, 22, n)
    elif kind == "label_flip":           # same events, labels inverted
        amounts = np.round(rng.lognormal(4.2, 0.35, n), 2)
        lab = np.ones(n, dtype=int)
        new_dev = np.zeros(n, dtype=bool)
        hours = rng.integers(8, 22, n)
    else:                                # extra volume, mixed
        amounts = np.round(rng.lognormal(4.4, 0.5, n), 2)
        lab = rng.random(n) < 0.5
        new_dev = rng.random(n) < 0.3
        hours = rng.integers(0, 24, n)
    un_loc = np.zeros(n, dtype=bool)
    return [
        {
            "fraud_id": "acct-future",
            "ts": ts[i], "amount": float(amounts[i]), "hour": int(hours[i]),
            "new_device": bool(new_dev[i]), "unusual_location": bool(un_loc[i]),
            "unusual_recipient": bool(rng.random() < 0.3),
            "failed_auth": int(rng.choice([0, 1], 1)[0]),
            "archetype": "future", "label": int(lab[i]),
            "shared_device_accounts": 0, "shared_recipient_accounts": 0,
            "recipient": "R-002",
        }
        for i in range(n)
    ]


def feats_at(fn, events, account_start, base_devices, graph, idx: int) -> dict:
    rows = fn(list(events), account_start, base_devices, graph)
    return {f: float(rows[idx][f]) for f in ALL_FEATURES}


def run_causality_audit() -> tuple[dict, list[str]]:
    from src.generate_synthetic_data import derive_features, derive_features_causal

    fails: list[str] = []
    results: dict = {"perturbations": [], "per_feature": {}, "verdict": ""}
    rng_master = np.random.default_rng(2026)
    scenarios = []
    for arch in ("legit", "escalation", "ato", "mule", "impulse"):
        for seed in range(2):
            rng = np.random.default_rng(seed + 1)
            scenarios.append((arch, make_account(
                rng, arch, 40, pd.Timestamp("2025-06-01", tz="UTC"))))
    kinds = ("fraud_burst", "amount_shock", "label_flip", "extra_volume")
    per_feat: dict[str, dict] = {f: {"max_causal_diff": 0.0,
                                     "max_legacy_diff": 0.0,
                                     "perturbed_scenarios": 0}
                                 for f in ALL_FEATURES}
    n_scen = 0
    for arch, events in scenarios:
        n_scen += 1
        i = len(events) // 2          # target transaction: strictly in the middle
        base_devices = 2
        start = events[0]["ts"] - pd.Timedelta(days=400)  # established account
        last_ts = events[i]["ts"]
        # PRECONDITION: perturbed events really are in the FUTURE of target i.
        assert all(e["ts"] > last_ts for e in future_suffix(
            rng_master, last_ts, "fraud_burst")), "suffix not strictly future"
        for kind in kinds:
            rng = np.random.default_rng(1000 + n_scen * 7)
            fut = future_suffix(rng, last_ts, kind)
            prefix_events = events[: i + 1]
            # baseline (no future) for the target row
            base_c = feats_at(derive_features_causal, prefix_events, start,
                              base_devices, None, i)
            # perturbed (future events appended); target row is index i again
            full_c = feats_at(derive_features_causal, prefix_events + fut,
                              start, base_devices, None, i)
            base_l = feats_at(derive_features, prefix_events, start,
                              base_devices, None, i)
            full_l = feats_at(derive_features, prefix_events + fut,
                              start, base_devices, None, i)
            row = {"scenario": f"{arch}_seed{seed}", "perturbation": kind,
                   "target_index": i, "future_rows": len(fut)}
            for f in ALL_FEATURES:
                dc = abs(base_c[f] - full_c[f])
                dl = abs(base_l[f] - full_l[f])
                per_feat[f]["max_causal_diff"] = max(
                    per_feat[f]["max_causal_diff"], dc)
                per_feat[f]["max_legacy_diff"] = max(
                    per_feat[f]["max_legacy_diff"], dl)
                if dc > TOL:
                    row[f] = {"causal_diff": float(dc), "legacy_diff": float(dl)}
            if any(abs(base_c[f] - full_c[f]) > TOL for f in ALL_FEATURES):
                results["perturbations"].append(row)
                fails.append(f"causality: scenario {arch} seed {seed} "
                             f"perturbation {kind} changed an EARLIER feature")
    for f in ALL_FEATURES:
        per_feat[f]["perturbed_scenarios"] = int(n_scen * len(kinds))
    results["per_feature"] = per_feat
    results["scenarios"] = n_scen
    results["perturbation_kinds"] = list(kinds)
    results["method"] = ("recompute target-row features from history up to t, "
                         "append strictly-future perturbed events, recompute; "
                         "causal max|diff| must be 0.0 (bit-identical)")
    return results, fails


def label_and_quarantine_checks() -> tuple[dict, list[str]]:
    fails: list[str] = []
    checks: dict = {}

    # ---- B. no model feature is label-derived ----------------------------
    src = (ROOT / "backend" / "src" / "generate_synthetic_data.py").read_text(
        encoding="utf-8")
    # In derive_features_causal the label is only copied into the output row
    # (as the OUTCOME); assert no model-feature line reads ev["label"].
    body = src[src.find("def derive_features_causal"):]
    body = body[: body.find("\n\n\ndef generate")] if "\n\n\ndef generate" in body else body
    feature_lines = [ln for ln in body.splitlines()
                     if any(f in ln for f in ALL_FEATURES)
                     and "label" in ln and "int(ev[\"label\"])" not in ln]
    checks["B_label_derived_feature_lines"] = feature_lines
    checks["B_none"] = len(feature_lines) == 0
    if feature_lines:
        fails.append("label-availability: a feature line references the label")
    checks["B_note"] = ("synthetic labels are assigned at event time, and NO "
                        "model feature consumes any label, so label-latency "
                        "cannot influence feature values; the label is the "
                        "OUTCOME only (used for validation metrics/threshold "
                        "selection, which is legitimate)")

    # ---- C. final-test quarantine: recorded artifact + source -------------
    if not PROTOCOL.exists() or not PRED.exists():
        fails.append("quarantine: protocol/predictions artifacts missing - run "
                     "scripts/ml_validity_rebuild.py first")
        return checks, fails
    proto = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    z = np.load(PRED)
    va_s, va_y = z["va_score"], z["va_y"].astype(int)
    te_s, te_y = z["te_score"], z["te_y"].astype(int)
    locked_recorded = float(z["locked_threshold"][0])

    # Independent threshold search on VALIDATION only. Definition matches the
    # rebuild's LOCKED policy: the most permissive threshold that still meets
    # validation FPR <= 1% (max recall within the cap). Implemented here as a
    # unique-score scan from low to high: FPR falls as the threshold rises, so
    # the FIRST threshold that satisfies the cap is the locked one.
    n_neg = int((va_y == 0).sum())
    neg_sorted = np.sort(va_s[va_y == 0])
    locked = None
    for thr in np.unique(va_s):
        fp = n_neg - int(np.searchsorted(neg_sorted, thr, side="left"))
        if fp / n_neg <= 0.01:
            locked = float(thr)
            break
    if locked is None:
        locked = float(va_s.min())
    checks["C_val_locked_threshold"] = locked_recorded
    checks["C_independent_val_threshold"] = locked
    # Validation FPR at the locked threshold must actually satisfy the cap.
    fp_at = int(((va_s >= locked_recorded) & (va_y == 0)).sum())
    checks["C_locked_fpr_on_validation"] = fp_at / n_neg
    checks["C_match"] = abs(locked - locked_recorded) <= 1e-9 and \
        fp_at / n_neg <= 0.01 + 1e-12
    if not checks["C_match"]:
        fails.append("quarantine: recorded locked threshold != independent "
                     "validation-only threshold (or exceeds 1% FPR on validation)")
    ft = proto.get("C_protocol", {}).get("final_test", {})
    checks["C_test_metrics_at_locked_consistent"] = bool(
        "metrics_at_locked" in ft)
    # The recorded protocol must NOT contain any test-fit threshold in use.
    checks["C_no_test_threshold_used"] = (
        "disclosure_test_picked_threshold_not_used" in ft)

    rebuild_src = (ROOT / "backend" / "scripts" / "ml_validity_rebuild.py").read_text(
        encoding="utf-8")
    # scaler: every StandardScaler().fit call consumes a TRAIN-only slice
    # (Xtr for the main model, Xgtr for each ablation group); never val/test.
    scaler_ok = (
        "StandardScaler().fit(Xtr)" in rebuild_src
        and "StandardScaler().fit(Xgtr)" in rebuild_src
        and "StandardScaler().fit(Xva" not in rebuild_src
        and "StandardScaler().fit(Xte" not in rebuild_src
        and "StandardScaler().fit(Xgva" not in rebuild_src
        and "StandardScaler().fit(Xgte" not in rebuild_src)
    checks["C_scaler_fit_train_only"] = scaler_ok
    # early stopping: EVERY eval_set line references the VALIDATION slice
    # (Xva_s for the main model, sc.transform(Xgva) for ablations); the final
    # test never appears in an eval_set.
    es_lines = [ln for ln in rebuild_src.splitlines() if "eval_set=" in ln]
    checks["C_early_stop_on_validation"] = (
        len(es_lines) >= 2
        and all(("Xva_s" in ln or "Xgva" in ln) for ln in es_lines))
    # threshold: locked from VALIDATION; the final test is evaluated with the
    # LOCKED threshold. The one threshold_for_fpr(te_score,...) call is the
    # recorded disclosure (never used to score).
    checks["C_threshold_from_validation_only"] = (
        "threshold_for_fpr(va_score, yva, max_fpr=0.01)" in rebuild_src
        and "metrics_at(te_score, yte, locked)" in rebuild_src
        and "disclosure_test_picked_threshold_not_used" in rebuild_src)
    for k in ("C_scaler_fit_train_only", "C_early_stop_on_validation",
              "C_threshold_from_validation_only", "C_no_test_threshold_used"):
        if not checks[k]:
            fails.append(f"quarantine source gate failed: {k}")
    return checks, fails


def main() -> int:
    print("PS-14 ML-VALIDITY GATES", flush=True)
    audit, fails = run_causality_audit()
    audit["built_at"] = datetime.now(timezone.utc).isoformat()
    audit["tolerance"] = TOL
    audit["verdict"] = "FAIL" if fails else "PASS"
    quarantine, qfails = label_and_quarantine_checks()
    audit["label_and_quarantine"] = quarantine
    fails += qfails
    AUDIT_OUT.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_OUT.write_text(json.dumps(audit, indent=2), encoding="utf-8")

    causal_bad = [f for f in sorted(audit["per_feature"])
                  if audit["per_feature"][f]["max_causal_diff"] > TOL]
    print(f"  scenarios x perturbations: {audit['scenarios']} x "
          f"{len(audit['perturbation_kinds'])}")
    for f in sorted(audit["per_feature"]):
        d = audit["per_feature"][f]
        tag = "HIST" if f in HISTORICAL else "instant"
        print(f"    {f:28s} {tag:8s} causal max|d|={d['max_causal_diff']:.2e} "
              f"(legacy disclosure {d['max_legacy_diff']:.4f})")
    if causal_bad:
        print("  CAUSALITY FAIL: features sensitive to future rows: "
              f"{causal_bad}")
    for k, v in quarantine.items():
        print(f"  {k} = {v}")
    print(f"VERDICT: {'PASS' if not fails else 'FAIL'} "
          f"({len(fails)} gate failure(s))")
    if fails:
        for f in fails:
            print(f"  - {f}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
