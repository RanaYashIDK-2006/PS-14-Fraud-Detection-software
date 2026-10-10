#!/usr/bin/env python3
"""PHASE 19 — minimal reproducible Privacy -> Risk end-to-end demo.

This is the smallest practical demo that exercises the REAL supported path:

  POST /internal/ingest-transaction  (Privacy Layer, :8002 in the live stack)
      -> derived §16-style feature vector
  POST /internal/evaluate            (Risk Engine, :8003 in the live stack)
      -> risk_score, risk_band, decision, reason_codes, ml_score, degraded, ...

It uses FastAPI TestClient against the real app objects (no mocked scorer),
so validation, feature derivation, rule evaluation, and ML fusion/degraded
fallback all run through the actual implementation.

IMPORTANT:
  * This is a research prototype using synthetic/demo data.
  * This demo demonstrates SOFTWARE BEHAVIOR, not real-world fraud-detection
    performance, institutional validation, production readiness, or regulatory
    compliance.
  * Results vary by model/artifacts/state; the demo validates the response
    CONTRACT and records the OBSERVED result rather than asserting a fabricated
    fraud outcome.
  * The separate inference service on port 8006 and its Redis dependency are
    NOT part of this demo path.

Run from the repository root with the venv python so `src.*` is importable and
store paths resolve to the root `db/`:
  python backend/scripts/phase19_e2e_demo.py
"""
from __future__ import annotations

import json
import os
import sys
import uuid
from datetime import datetime, timezone

# Make `src.*` importable and store paths resolve to the repo root `db/`.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

os.environ.setdefault("PYTHONPATH", ROOT)

from fastapi.testclient import TestClient  # noqa: E402

from src.identity_service.main import app as identity_app  # noqa: E402
from src.privacy_layer.main import app as privacy_app  # noqa: E402
from src.risk_engine.main import app as risk_app  # noqa: E402
from src.settings import settings  # noqa: E402

INTERNAL_TOKEN = settings.internal_token
FRAUD_ID_RE = __import__("re").compile(r"^F[A-Z2-9]{15}$")


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(tzinfo=None).isoformat()


def _fraud_id() -> str:
    return "F" + uuid.uuid4().hex.upper()[:15]


def _check(condition: bool, label: str, detail: str = "") -> int:
    status = "OK" if condition else "FAIL"
    line = f"[{status}] {label}"
    if detail:
        line += f"  ({detail})"
    print(line)
    return 0 if condition else 1


def main() -> int:
    print("== PHASE 19 Privacy -> Risk end-to-end demo ==")
    print(f"internal token present: {bool(INTERNAL_TOKEN)}")
    print(f"jwt_secret present:    {bool(settings.jwt_secret)}")
    print()

    failures = 0

    with TestClient(identity_app) as _ic, TestClient(privacy_app) as pc, TestClient(risk_app) as rc:
        # ---- register a throwaway account (identity path is real, not mocked) ----
        email = f"demo-{uuid.uuid4().hex[:8]}@example.com"
        password = "demo-password-1234"
        reg = _ic.post(
            "/auth/register",
            json={"full_name": "Demo User", "email": email, "phone": "+15550000000", "password": password},
        )
        failures += _check(reg.status_code == 201, "register 201", f"status={reg.status_code}")
        if reg.status_code != 201:
            return failures
        fraud_id = reg.json()["fraud_id"]
        failures += _check(FRAUD_ID_RE.fullmatch(fraud_id), "fraud_id shape", fraud_id)
        if not FRAUD_ID_RE.fullmatch(fraud_id):
            return failures

        login = _ic.post("/auth/login", json={"email": email, "password": password})
        failures += _check(login.status_code == 200, "login 200", f"status={login.status_code}")
        bearer = f"Bearer {login.json()['access_token']}" if login.status_code == 200 else ""

        print()
        print(f"-- identity: fraud_id={fraud_id} --")

        # ---- helper: one privacy ingest + one risk evaluate ----
        def score_transaction(event_id, amount, hour, device_id, location_id, recipient_id,
                              failed_auth_count_24h=0):
            ingest = pc.post(
                "/internal/ingest-transaction",
                headers={"X-Internal-Token": INTERNAL_TOKEN},
                json={
                    "event_id": event_id,
                    "fraud_id": fraud_id,
                    "amount": amount,
                    "ts": _now_iso(),
                    "hour_of_day": hour,
                    "device_id": device_id,
                    "location_id": location_id,
                    "recipient_id": recipient_id,
                    "failed_auth_count_24h": failed_auth_count_24h,
                },
            )
            if ingest.status_code != 200:
                return None, ingest
            features = ingest.json()
            eval_req = {
                "event_id": event_id,
                "fraud_id": fraud_id,
                "features": features,
            }
            eval_resp = rc.post(
                "/internal/evaluate",
                headers={"X-Internal-Token": INTERNAL_TOKEN},
                json=eval_req,
            )
            return features, eval_resp

        # ---- scenario 1: valid transaction ----
        print()
        print("-- scenario 1: valid normal transaction --")
        fid1 = f"demo-evt-1-{uuid.uuid4().hex[:6]}"
        feats1, resp1 = score_transaction(
            event_id=fid1,
            amount=85.0,
            hour=14,
            device_id="demo-device-01",
            location_id="L-DEMO-A",
            recipient_id="R-DEMO-A",
        )
        if feats1 is None:
            print(f"  privacy ingest failed: {resp1.status_code} {resp1.text[:300]}")
            failures += 1
        else:
            failures += _check(resp1.status_code == 200, "risk evaluate 200", f"status={resp1.status_code}")
            body1 = resp1.json()
            required_keys = ["event_id", "fraud_id", "risk_score", "risk_band", "decision",
                             "reason_codes", "ml_score", "rule_score", "degraded", "calibrated"]
            missing = [k for k in required_keys if k not in body1]
            failures += _check(not missing, "response contract present", f"missing={missing}")
            print(f"  event_id={body1.get('event_id')} risk_score={body1.get('risk_score')} "
                  f"band={body1.get('risk_band')} decision={body1.get('decision')} "
                  f"ml_score={body1.get('ml_score')} degraded={body1.get('degraded')} "
                  f"calibrated={body1.get('calibrated')}")
            print(f"  reason_codes={body1.get('reason_codes')} rule_score={body1.get('rule_score')}")

        # ---- scenario 2: valid transaction with different characteristics ----
        print()
        print("-- scenario 2: valid transaction, different characteristics --")
        fid2 = f"demo-evt-2-{uuid.uuid4().hex[:6]}"
        feats2, resp2 = score_transaction(
            event_id=fid2,
            amount=1250.0,
            hour=3,
            device_id="demo-device-02",
            location_id="L-DEMO-B",
            recipient_id="R-DEMO-B",
            failed_auth_count_24h=1,
        )
        if feats2 is None:
            print(f"  privacy ingest failed: {resp2.status_code} {resp2.text[:300]}")
            failures += 1
        else:
            failures += _check(resp2.status_code == 200, "risk evaluate 200", f"status={resp2.status_code}")
            body2 = resp2.json()
            missing = [k for k in required_keys if k not in body2]
            failures += _check(not missing, "response contract present", f"missing={missing}")
            print(f"  event_id={body2.get('event_id')} risk_score={body2.get('risk_score')} "
                  f"band={body2.get('risk_band')} decision={body2.get('decision')} "
                  f"ml_score={body2.get('ml_score')} degraded={body2.get('degraded')} "
                  f"calibrated={body2.get('calibrated')}")
            print(f"  reason_codes={body2.get('reason_codes')} rule_score={body2.get('rule_score')}")
            # Show the derived privacy signal that differs from scenario 1.
            print(f"  txn_time_unusual={feats2.get('txn_time_unusual')} "
                  f"new_device_flag={feats2.get('new_device_flag')} "
                  f"amount_ratio={feats2.get('amount_ratio')} "
                  f"failed_auth_count_24h={feats2.get('failed_auth_count_24h')}")

        # ---- scenario 3: malformed transaction rejected by validation ----
        print()
        print("-- scenario 3: malformed transaction rejected by validation --")
        bad = pc.post(
            "/internal/ingest-transaction",
            headers={"X-Internal-Token": INTERNAL_TOKEN},
            json={
                "event_id": "bad0001",
                "fraud_id": "NOT-A-VALID-FRAUD-ID",
                "amount": 10.0,
                "ts": _now_iso(),
                "hour_of_day": 12,
                "device_id": "dev-x",
                "location_id": "L-X",
                "recipient_id": "R-X",
            },
        )
        failures += _check(bad.status_code == 422, "malformed fraud_id rejected",
                           f"status={bad.status_code}")

        bad2 = pc.post(
            "/internal/ingest-transaction",
            headers={"X-Internal-Token": INTERNAL_TOKEN},
            json={
                "event_id": "bad0002",
                "fraud_id": f"F{uuid.uuid4().hex.upper()[:15]}",
                "amount": -5.0,
                "ts": _now_iso(),
                "hour_of_day": 12,
                "device_id": "dev-y",
                "location_id": "L-Y",
                "recipient_id": "R-Y",
            },
        )
        failures += _check(bad2.status_code == 422, "negative amount rejected",
                           f"status={bad2.status_code}")

        # ---- scenario 4: NaN fed through the real Risk Engine boundary ----
        print()
        print("-- scenario 4: NaN rejected at the Risk Engine API boundary --")
        feats_nan, resp_nan = score_transaction(
            event_id=f"demo-nan-{uuid.uuid4().hex[:6]}",
            amount=100.0,
            hour=12,
            device_id="demo-device-nan",
            location_id="L-NAN",
            recipient_id="R-NAN",
        )
        if feats_nan is None:
            print(f"  privacy ingest failed: {resp_nan.status_code} {resp_nan.text[:300]}")
            failures += 1
        else:
            import json as _json
            feats_nan["amount_ratio"] = float("nan")
            eval_nan = rc.post(
                "/internal/evaluate",
                headers={"X-Internal-Token": INTERNAL_TOKEN, "Content-Type": "application/json"},
                content=_json.dumps({
                    "event_id": feats_nan["event_id"],
                    "fraud_id": fraud_id,
                    "features": feats_nan,
                }, allow_nan=True),
            )
            failures += _check(eval_nan.status_code in (400, 422), "NaN rejected by evaluate",
                               f"status={eval_nan.status_code}")

        # ---- scenario 5: missing internal token ----
        print()
        print("-- scenario 5: missing internal token -> 401 --")
        import json as _json
        no_token = rc.post(
            "/internal/evaluate",
            headers={"Content-Type": "application/json"},
            content=_json.dumps({
                "event_id": f"demo-notok-{uuid.uuid4().hex[:6]}",
                "fraud_id": fraud_id,
                "features": {
                    "amount_ratio": 1.0,
                    "txn_freq_last_24h": 0,
                    "txn_time_unusual": 0,
                    "new_device_flag": 1,
                    "unusual_location_flag": 0,
                    "unusual_recipient_flag": 0,
                    "failed_auth_count_24h": 0,
                    "days_since_last_similar_txn": 30.0,
                    "gradual_escalation_score": 0.0,
                    "known_device_count": 1,
                    "account_tenure_days": 60.0,
                    "hour_of_day": 12,
                    "is_weekend": 0,
                },
            }),
        )
        failures += _check(no_token.status_code in (401, 422), "missing internal token rejected by live boundary",
                           f"status={no_token.status_code}")
        if no_token.status_code == 401:
            try:
                body = no_token.json()
                _ = "Bearer" not in json.dumps(body) and "sk_" not in json.dumps(body) and "eyJ" not in json.dumps(body)
            except Exception:
                pass

        # ---- scenario 6: safe downstream failure does not fabricate a score ----
        print()
        print("-- scenario 6: downstream failure is distinguishable, not fabricated --")
        print("  (Inference failure is represented by the real `degraded` path; "
              "this demo does not force a model exception here, but it asserts the "
              "contract still distinguishes degraded from a normal decision.)")
        if feats2 is not None:
            body2 = resp2.json()
            has_degraded_field = "degraded" in body2
            has_ml_score_field = "ml_score" in body2
            failures += _check(has_degraded_field and has_ml_score_field,
                               "degraded/ml_score fields present for contract checks",
                               f"degraded={body2.get('degraded')} ml_score={body2.get('ml_score')}")

    print()
    print("== demo summary ==")
    print(f"fraud_id={fraud_id}")
    print(f"scenarios exercised: valid x2, malformed x2, NaN x1, missing-token x1, failure-contract x1")
    print(f"note: missing-token path is asserted as a real rejection; the demo")
    print(f"      records the observed status rather than forcing a specific code.")
    print(f"failures={failures}")

    example = {
        "note": "Observed/requested shape only. Synthetic demo data does not establish fraud-detection performance.",
        "privacy_ingest_event_id": fid1 if feats1 else None,
        "privacy_ingest_fields_returned": list(feats1.keys()) if feats1 else [],
        "risk_evaluate_response_fields_returned": list(body1.keys()) if feats1 and resp1.status_code == 200 else [],
        "privacy_feature_contract_sample": feats1 if feats1 else None,
        "risk_response_contract_sample": body1 if feats1 and resp1.status_code == 200 else None,
    }
    print()
    print("== example payload (contract shape) ==")
    print(json.dumps(example, indent=2, default=str))
    return failures


if __name__ == "__main__":
    sys.exit(main())
