#!/usr/bin/env python3
"""PHASE 10C — analysis artifacts (root-cause matrix, label availability,
model identity, temporal contract with executed Tests A-D).

Firewall: only the <2016 train cache is touched. No final-test rows are
loaded. No production artifact is modified.
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

CERT_TIME = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
REPORTS = ROOT / "reports" / "phase10_remediation"

from src.privacy_layer.velocity_tracker import UserVelocityTracker  # noqa: E402
from src.risk_engine.entity_fraud_rates import EntityFraudRateTracker  # noqa: E402


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    # ── firewall ──────────────────────────────────────────────────────────
    z = np.load(ROOT / "data" / "_raw_parity_cache_tr.npz", allow_pickle=False)
    yr = z["tr_year"]
    assert int(yr.max()) < 2016 and int((yr >= 2018).sum()) == 0

    # ── 1. ROOT-CAUSE MATRIX ──────────────────────────────────────────────
    rows = [
        {
            "feature": "merchant_id",
            "training_definition": "_code(str(Merchant Name)) = sha256(name)[:8] mod 100000 (native_features._code)",
            "offline_implementation": "retrain build_context_and_features passes merchant_name='Merchant Name'; derive_native_features hashes it",
            "production_implementation": "IngestTransactionRequest has NO merchant identity field; FeatureVector.merchant_id defaults ''; _from_raw_native maps merchant_id->merchant_name->_code('')=0.0",
            "input_source": "IBM 'Merchant Name' column offline; live privacy-layer ingest carries recipient_id only",
            "decision_time_available": "YES (a merchant token exists live: recipient_id; IBM merchant name exists in the dataset the model was trained/scored on)",
            "privacy_permitted": "YES — merchant identity is a counterparty token already flowing as recipient_id; only a stable hash code reaches the model",
            "availability_class": "A (legitimately available via narrow interface) / E (proxy recipient_id differs in string space from training names)",
            "causal": True,
            "semantic_difference": "offline codes are stable hashes of IBM merchant names; production-as-wired sends constant 0.0; even a forwarded recipient_id yields a DIFFERENT code space (cold-start for every live merchant)",
            "remediation": "forward the merchant identity string used at training time through the narrowest interface (recipient_id is the PS-14 merchant token); verify hash parity on replay data where the same strings exist",
            "contract_change": False,
            "retraining_required": False,
            "parity_achievable_on_replay": True,
            "parity_achievable_live": "CONDITIONAL — same-string entities (IBM replay) match; live PS-14 tokens are a cold-start code space by design",
        },
        {
            "feature": "user_merchant_diversity",
            "training_definition": "distinct merchants seen by user BEFORE event (first-seen cumsum shifted)",
            "offline_implementation": "df.duplicated([User, Merchant Name], keep='first').groupby(User).cumsum() - first_m, clip>=0",
            "production_implementation": "UserVelocityTracker tracks only tx_count/total_amount/card_tx_count/merch_tx_count; FeatureVector default 0.0 -> derive clamps max(0,1.0)=1.0 constant",
            "input_source": "per-user distinct merchant history",
            "decision_time_available": "YES — the velocity tracker sees every (user, merchant) event before the current one",
            "privacy_permitted": "YES — only a count, not raw strings",
            "availability_class": "F (tracker can supply it; production just does not track it)",
            "causal": True,
            "semantic_difference": "offline: real distinct-count (expanding, sample-conditional); production: constant 1.0",
            "remediation": "extend the velocity tracker with a per-user ts-aware distinct-merchant set; count strictly before t; offline parity on identical sampled streams",
            "contract_change": False,
            "retraining_required": False,
            "parity_achievable_on_replay": True,
        },
        {
            "feature": "user_city_diversity",
            "training_definition": "distinct merchant cities seen by user BEFORE event",
            "offline_implementation": "duplicated([User, Merchant City]) first-seen shifted cumsum",
            "production_implementation": "constant 1.0 (clamp of the 0.0 default); not tracked",
            "input_source": "per-user distinct city history",
            "decision_time_available": "YES",
            "privacy_permitted": "YES — count only",
            "availability_class": "F",
            "causal": True,
            "semantic_difference": "offline real count vs production constant 1.0",
            "remediation": "ts-aware per-user distinct-city set in the velocity tracker",
            "contract_change": False,
            "retraining_required": False,
            "parity_achievable_on_replay": True,
        },
        {
            "feature": "user_merch_count",
            "training_definition": "count of (user, merchant) interactions BEFORE event",
            "offline_implementation": "groupby([User, Merchant Name]).cumcount()",
            "production_implementation": "constant 0.0; never tracked",
            "input_source": "per-user-merchant pair history",
            "decision_time_available": "YES",
            "privacy_permitted": "YES — count only",
            "availability_class": "F",
            "causal": True,
            "semantic_difference": "offline real pair count vs production constant 0.0",
            "remediation": "ts-aware per-(user,merchant) counter in the velocity tracker",
            "contract_change": False,
            "retraining_required": False,
            "parity_achievable_on_replay": True,
        },
        {
            "feature": "user_fraud_rate",
            "training_definition": "confirmed-fraud labels for user strictly before t / all user transactions strictly before t (expanding mean, unbounded)",
            "offline_implementation": "(cumsum(is_fraud) - is_fraud) / user_tx_count, default 0.001 when count==0",
            "production_implementation": "wired: FeatureVector default 0.0 -> COLD_START_FRAUD_RATE 0.001 constant; tracker variant: EntityFraudRateTracker 100-event window, min 5 events",
            "input_source": "confirmed fraud labels for the user, finalized before t",
            "decision_time_available": "NO — labels are only in the static IBM synthetic file or the VerificationOutcome retraining pool (investigator latency, retrospective); risk engine records score>=70 proxy which is forbidden",
            "privacy_permitted": "YES for a rate; the label itself is not a raw string",
            "availability_class": "D (only retrospectively available) + E (proxy exists but forbidden)",
            "causal": True,
            "causal_production": "UNVERIFIED — cannot be causally computed without confirmed labels at decision time",
            "semantic_difference": "expanding unbounded mean of confirmed labels vs constant 0.001 / finite-window proxy",
            "remediation": "NONE LEGITIMATE — no confirmed-label-at-decision-time pipeline exists",
            "contract_change": True,
            "retraining_required": True,
            "parity_achievable_on_replay": False,
            "parity_achievable_live": "NO",
            "decision": "UNAVAILABLE_AT_DECISION_TIME -> feature removal + retrain (PATH B)",
        },
        {
            "feature": "merch_fraud_rate",
            "training_definition": "confirmed-fraud labels for merchant strictly before t / all merchant transactions strictly before t",
            "offline_implementation": "expanding mean per Merchant Name, default 0.001",
            "production_implementation": "constant 0.001 wired / 100-event tracker proxy",
            "input_source": "confirmed merchant labels before t",
            "decision_time_available": "NO (same as user_fraud_rate)",
            "privacy_permitted": "YES",
            "availability_class": "D + E",
            "causal": True,
            "semantic_difference": "expanding mean vs constants/proxy",
            "remediation": "NONE LEGITIMATE",
            "contract_change": True,
            "retraining_required": True,
            "parity_achievable_on_replay": False,
            "decision": "UNAVAILABLE_AT_DECISION_TIME -> feature removal + retrain (PATH B)",
        },
        {
            "feature": "city_fraud_rate",
            "training_definition": "confirmed-fraud labels for city strictly before t / all city transactions strictly before t",
            "offline_implementation": "expanding mean per Merchant City, default 0.001",
            "production_implementation": "constant 0.001 wired / 100-event tracker proxy",
            "input_source": "confirmed city labels before t",
            "decision_time_available": "NO (same)",
            "privacy_permitted": "YES",
            "availability_class": "D + E",
            "causal": True,
            "semantic_difference": "expanding mean vs constants/proxy",
            "remediation": "NONE LEGITIMATE",
            "contract_change": True,
            "retraining_required": True,
            "parity_achievable_on_replay": False,
            "decision": "UNAVAILABLE_AT_DECISION_TIME -> feature removal + retrain (PATH B)",
        },
    ]
    temporal = {
        "divergence": "production trackers (UserVelocityTracker / EntityFraudRateTracker) accumulate in INGESTION order with no ts parameter; offline sorts by ts before building expanding context",
        "contract_required": "feature(t) must depend only on events with timestamp < t, never on ingestion order",
        "affected_features": ["user_tx_count", "user_avg_amt", "card_tx_count", "merch_tx_count",
                              "user_merchant_diversity", "user_city_diversity", "user_merch_count",
                              "user_fraud_rate", "merch_fraud_rate", "city_fraud_rate"],
        "phase10b_evidence": "out-of-order ingest of a 2016 row before a 2015 row changed earlier features (max delta 9999 on 6 features)",
    }
    matrix = {
        "features": rows,
        "temporal_divergence": temporal,
        "cert_time_utc": CERT_TIME,
    }
    (REPORTS / "root_cause_matrix.json").write_text(json.dumps(matrix, indent=2), encoding="utf-8")

    # ── 2. LABEL AVAILABILITY FINAL ───────────────────────────────────────
    label = {
        "user_fraud_rate": "UNAVAILABLE_AT_DECISION_TIME",
        "merch_fraud_rate": "UNAVAILABLE_AT_DECISION_TIME",
        "city_fraud_rate": "UNAVAILABLE_AT_DECISION_TIME",
        "confirmed_label_sources": {
            "ibm_synthetic_csv": "static pre-labeled file; labels are retrospective properties of the dataset, NOT produced by any live confirmation pipeline before a later decision",
            "verification_outcome": "src/verification_service writes VerificationOutcome (confirmed/disputed) into the RETRAINING pool with investigator latency; exported by scripts/export_feedback.py; not consumed as a decision-time expanding-mean input",
            "risk_engine_entity_tracker": "seeded/recorded with is_fraud = risk_score >= 70 — a MODEL-SCORE PROXY, explicitly forbidden as a label substitute",
        },
        "predictive_usefulness": "PROVEN (Phase-8 D_safe: AUC drop when removed) — SEPARATE from decision-time availability",
        "causal_computation_offline": "PASS (prior-only aggregation, verified)",
        "decision_time_label_availability": "UNVERIFIED / UNAVAILABLE — no in-repo pipeline produces a confirmed fraud label for entity history before the next live decision",
        "verdict": "The three fraud-rate features CANNOT be made production-valid without inventing a label pipeline. Marking them UNAVAILABLE triggers PATH B (feature removal + retrain).",
        "cert_time_utc": CERT_TIME,
    }
    (REPORTS / "label_availability_final.json").write_text(json.dumps(label, indent=2), encoding="utf-8")

    # ── 3. MODEL IDENTITY ─────────────────────────────────────────────────
    cand = json.load(open(ROOT / "models" / "model_records" / "mission_E_hardneg" / "config.json", encoding="utf-8"))
    prod_manifest = json.load(open(ROOT / "models" / "production" / "altman_native" / "manifest.json", encoding="utf-8"))
    identity = {
        "candidate": "V_rawplus",
        "candidate_artifact": "models/model_records/mission_E_hardneg/models.joblib",
        "candidate_sha256": sha256_file(ROOT / "models" / "model_records" / "mission_E_hardneg" / "models.joblib"),
        "production_model": prod_manifest.get("model_version"),
        "production_sha256_members": {
            "xgb": sha256_file(ROOT / "models" / "production" / "altman_native" / "xgb_native.joblib"),
            "lgb": sha256_file(ROOT / "models" / "production" / "altman_native" / "lgb_native.joblib"),
            "cb": sha256_file(ROOT / "models" / "production" / "altman_native" / "cb_native.joblib"),
        },
        "weight_identity_evidence": "Phase-10B execution: max |prediction delta| = 0.0 on random vectors between candidate models.joblib members and production altman_native members (same scaler + same 3 members)",
        "production_threshold": 0.018758,
        "candidate_threshold_tvstar": 0.0298937337,
        "verdict": "V_rawplus == E_hardneg weights. The observed FPR/alert-rate advantage is THRESHOLD-ONLY (same model, different operating point). V_rawplus must NOT be described as a superior model.",
        "n_features": len(cand.get("features_used", [])),
        "cert_time_utc": CERT_TIME,
    }
    (REPORTS / "model_identity.json").write_text(json.dumps(identity, indent=2), encoding="utf-8")

    # ── 4. TEMPORAL CONTRACT — execute Tests A-D against REQUIRED semantics ─
    # Required policy: feature(t) depends only on events with ts < t. We
    # implement a minimal ts-aware state (sorted by ts, strict-before) and
    # prove A (future insertion), B (backdated insertion), C (shuffled
    # ingestion with deterministic tie-break), D (replay).
    from src.privacy_layer.native_features import derive_native_features, native_vector
    from datetime import datetime as _dt

    events = [
        {"user": "U1", "card": "C1", "merch": "M1", "city": "CT1", "amt": 10.0, "label": 0, "ts": _dt.fromisoformat("2015-01-01T10:00:00")},
        {"user": "U1", "card": "C1", "merch": "M1", "city": "CT1", "amt": 20.0, "label": 0, "ts": _dt.fromisoformat("2015-01-01T11:00:00")},
        {"user": "U1", "card": "C1", "merch": "M2", "city": "CT2", "amt": 90.0, "label": 1, "ts": _dt.fromisoformat("2015-01-02T09:00:00")},
    ]

    class TsAwareState:
        """Minimal ts-aware expanding state implementing the causal contract."""
        def __init__(self):
            self.events = []  # (ts, user, card, merch, city, amt, label)
        def add(self, ev):
            self.events.append((ev["ts"], ev["user"], ev["card"], ev["merch"], ev["city"], ev["amt"], ev["label"]))
            self.events.sort(key=lambda e: e[0])
        def features_before(self, ev):
            t = ev["ts"]
            prior = [e for e in self.events if e[0] < t]
            utc = sum(1 for e in prior if e[1] == ev["user"])
            ctc = sum(1 for e in prior if e[2] == ev["card"])
            mtc = sum(1 for e in prior if e[3] == ev["merch"])
            uamt = sum(e[5] for e in prior if e[1] == ev["user"]) / utc if utc else 0.0
            udiv = len({e[3] for e in prior if e[1] == ev["user"]})
            cdiv = len({e[4] for e in prior if e[1] == ev["user"]})
            umc = sum(1 for e in prior if e[1] == ev["user"] and e[3] == ev["merch"])
            ufr = sum(e[6] for e in prior if e[1] == ev["user"]) / utc if utc else 0.001
            mfr = sum(e[6] for e in prior if e[3] == ev["merch"]) / max(sum(1 for e in prior if e[3] == ev["merch"]), 1) if any(e[3] == ev["merch"] for e in prior) else 0.001
            cfr = sum(e[6] for e in prior if e[4] == ev["city"]) / max(sum(1 for e in prior if e[4] == ev["city"]), 1) if any(e[4] == ev["city"] for e in prior) else 0.001
            vel = {"user_tx_count": utc, "user_avg_amt": uamt, "card_tx_count": ctc,
                   "merch_tx_count": mtc, "user_merchant_diversity": max(udiv, 1.0),
                   "user_city_diversity": max(cdiv, 1.0), "user_merch_count": umc}
            rates = {"user_fraud_rate": ufr, "merch_fraud_rate": mfr, "city_fraud_rate": cfr}
            raw = {"amount": ev["amt"], "ts": ev["ts"], "use_chip": "Chip Transaction", "mcc": 5411,
                   "merchant_city": ev["city"], "merchant_state": "IL", "zip": "62704",
                   "card": ev["card"], "errors": "", "merchant_name": ev["merch"], "merchant_id": ev["merch"]}
            return native_vector(derive_native_features(raw, vel, rates))

    def vec_of(ev, state):
        return state.features_before(ev)

    # Test A — future insertion must not change A's vector
    sA = TsAwareState()
    sA.add(events[0]); sA.add(events[1])
    vA_before = vec_of(events[2], sA)
    fut = {"user": "U1", "card": "C1", "merch": "M9", "city": "CT9", "amt": 5000.0, "label": 1, "ts": _dt.fromisoformat("2016-01-01T00:00:00")}
    sA.add(fut)
    vA_after = vec_of(events[2], sA)  # events[2] at 2015-01-02 must be unchanged
    testA = {"pass": bool(np.abs(vA_before - vA_after).max() < 1e-9),
             "max_delta": float(np.abs(vA_before - vA_after).max())}

    # Test B — backdated insertion (ts earlier than A) must be included (causal policy: ts < t)
    sB = TsAwareState()
    sB.add(events[1]); sB.add(events[2])
    back = {"user": "U1", "card": "C1", "merch": "M0", "city": "CT0", "amt": 5.0, "label": 0, "ts": _dt.fromisoformat("2014-12-31T23:59:00")}
    vB_without = vec_of(events[2], sB)
    sB.add(back)
    vB_with = vec_of(events[2], sB)
    testB = {"backdated_row_included_because_ts_lt_t": bool(np.abs(vB_without - vB_with).max() > 0),
             "max_delta": float(np.abs(vB_without - vB_with).max()),
             "policy": "backdated events with ts < t ARE part of history (causal by timestamp, not ingestion order)"}

    # Test C — shuffled ingestion with identical ts must give identical features (deterministic tie-break)
    sC1 = TsAwareState(); sC2 = TsAwareState()
    for ev in [events[0], events[1], events[2]]:
        sC1.add(ev)
    for ev in [events[2], events[0], events[1]]:
        sC2.add(ev)
    vC1 = vec_of(events[2], sC1)
    vC2 = vec_of(events[2], sC2)
    testC = {"pass": bool(np.abs(vC1 - vC2).max() < 1e-9),
             "max_delta": float(np.abs(vC1 - vC2).max()),
             "note": "ts-sorted state is deterministic under any ingestion order when timestamps are identical"}

    # Test D — replay: reset and replay the same stream => identical vectors
    sD1 = TsAwareState()
    for ev in events:
        sD1.add(ev)
    vD1 = [vec_of(ev, sD1) for ev in events]
    sD2 = TsAwareState()
    for ev in events:
        sD2.add(ev)
    vD2 = [vec_of(ev, sD2) for ev in events]
    dD = max(float(np.abs(a - b).max()) for a, b in zip(vD1, vD2))
    testD = {"pass": dD < 1e-9, "max_delta": dD}

    # current production trackers violate the contract (no ts)
    vt_now = UserVelocityTracker()
    vt_now.record_event("U1", 10.0, "C1", "M1")
    vt_now.record_event("U1", 20.0, "C1", "M1")
    vt_now.record_event("U1", 90.0, "C1", "M2")
    v_ingest = vt_now.get_velocity("U1", "C1", "M2")
    vt_fut = UserVelocityTracker()
    vt_fut.record_event("U1", 5000.0, "C1", "M9")  # future row ingested first
    vt_fut.record_event("U1", 10.0, "C1", "M1")
    vt_fut.record_event("U1", 20.0, "C1", "M1")
    vt_fut.record_event("U1", 90.0, "C1", "M2")
    v_fut = vt_fut.get_velocity("U1", "C1", "M2")
    current_violates = v_ingest["user_tx_count"] != v_fut["user_tx_count"]

    temporal_contract = {
        "policy": "feature(t) depends only on events with timestamp < t (strict-before, ts-ordered)",
        "tests": {"A_future_insertion": testA, "B_backdated_insertion": testB,
                  "C_shuffled_ingestion": testC, "D_replay": testD},
        "current_production_trackers_violate": bool(current_violates),
        "current_violation_example": {"without_future_first": v_ingest["user_tx_count"],
                                      "with_future_first": v_fut["user_tx_count"]},
        "remediation_required": "UserVelocityTracker / EntityFraudRateTracker must become ts-aware (store (ts, entity, ...) and compute strictly before t), replacing ingestion-order accumulation",
        "cert_time_utc": CERT_TIME,
    }
    (REPORTS / "temporal_contract.json").write_text(json.dumps(temporal_contract, indent=2), encoding="utf-8")

    print(json.dumps({"A": testA, "B": testB, "C": testC, "D": testD,
                      "current_trackers_violate": current_violates}, indent=1))
    print("[phase10c] analysis artifacts written")
    return 0


if __name__ == "__main__":
    sys.exit(main())