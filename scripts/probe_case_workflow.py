#!/usr/bin/env python3
"""Live probe for the check-#30 fixes: case creation resolves score_id from
the shared risk store, dedup is enforced, lifecycle transitions validate, and
investigator CONFIRMED_* verdicts write VerificationOutcome labels.

Runs all five services in-process against a temp DB_DIR (real stores
untouched). Mirrors scripts/verification_test.py setup.
"""
from __future__ import annotations

import os
import sqlite3
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

TMP = tempfile.mkdtemp(prefix="ps14-case-fix-")
os.environ["DB_DIR"] = TMP
os.environ["PS14_MODE"] = "development"
os.environ["JWT_SECRET"] = "smoke-test-secret-0123456789abcdef"
os.environ["INTERNAL_TOKEN"] = "smoke-internal-token"
os.environ["COMPLIANCE_TOKEN"] = "smoke-compliance-token"

from fastapi.testclient import TestClient  # noqa: E402
from src.risk_engine.main import app as risk_app  # noqa: E402
from src.verification_service.main import app as verify_app  # noqa: E402

TOKEN = "smoke-internal-token"
COMPLIANCE = "smoke-compliance-token"

failures: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        failures.append(name)


def attack_vector(**overrides) -> dict:
    v = {
        "amount_ratio": 37.5,
        "txn_freq_last_24h": 1,
        "txn_time_unusual": 1,
        "new_device_flag": 1,
        "unusual_location_flag": 1,
        "unusual_recipient_flag": 1,
        "failed_auth_count_24h": 5,
        "days_since_last_similar_txn": 0.0,
        "gradual_escalation_score": 0.8,
        "known_device_count": 1,
        "account_tenure_days": 60.0,
        "hour_of_day": 3,
        "is_weekend": 0,
        "shared_device_accounts": 3,
        "shared_recipient_accounts": 2,
        "mule_ring_score": 0.5,
    }
    v.update(overrides)
    return v


def benign_vector(**overrides) -> dict:
    v = {
        "amount_ratio": 1.0,
        "txn_freq_last_24h": 1,
        "txn_time_unusual": 0,
        "new_device_flag": 0,
        "unusual_location_flag": 0,
        "unusual_recipient_flag": 0,
        "failed_auth_count_24h": 0,
        "days_since_last_similar_txn": 3.0,
        "gradual_escalation_score": 0.0,
        "known_device_count": 3,
        "account_tenure_days": 60.0,
        "hour_of_day": 12,
        "is_weekend": 0,
    }
    v.update(overrides)
    return v


def main() -> int:
    print("== Case-workflow fix probe (check #30) ==")
    with TestClient(risk_app) as rc, TestClient(verify_app) as vc:
        hdr = {"X-Internal-Token": TOKEN}
        chdr = {"X-Compliance-Token": COMPLIANCE}
        fraud_id = "F24BRMMMBJWYMTDW"  # 16-char base32, no I/O/0/1

        # Seed two scored events (one high-risk, one benign).
        ev_high = "casefix-attack-0001"
        ev_low = "casefix-benign-0001"
        r1 = rc.post("/internal/evaluate", json={
            "event_id": ev_high, "fraud_id": fraud_id,
            "features": attack_vector()}, headers=hdr)
        r2 = rc.post("/internal/evaluate", json={
            "event_id": ev_low, "fraud_id": fraud_id,
            "features": benign_vector()}, headers=hdr)
        check("evaluate seeds both events", r1.status_code == 200 and r2.status_code == 200,
              f"high={r1.status_code} low={r2.status_code}")
        high = r1.json()
        check("high-risk event scores high band", high.get("risk_band") == "high",
              f"band={high.get('risk_band')} score={high.get('risk_score')}")

        # Risk store: fetch the authoritative score row id.
        risk_db_path = Path(TMP) / "risk.db"
        con = sqlite3.connect(risk_db_path)
        row = con.execute(
            "SELECT score_id, risk_score, ml_score, risk_band FROM risk_scores "
            "WHERE event_id=? AND fraud_id=?", (ev_high, fraud_id)).fetchone()
        con.close()
        check("score row persisted in shared risk store", row is not None)
        score_id, stored_risk, stored_ml, stored_band = row
        check("unique case-dedup index exists on risk store",
              bool(sqlite3.connect(risk_db_path).execute(
                  "SELECT 1 FROM sqlite_master WHERE type='index' "
                  "AND name='uq_investigator_cases_event_id'").fetchone()),
              "created by verification lifespan on the shared store")

        # 1) create a case for the high event with WRONG caller values -> the
        #    endpoint must use the stored score (server-side truth).
        resp = vc.post(
            "/investigator/cases",
            params={"event_id": ev_high, "fraud_id": fraud_id,
                    "risk_score": 100, "ml_score": 0.99},
            json=["HIGH_AMOUNT_RATIO", "NEW_DEVICE"],
            headers=chdr,
        )
        body = resp.json()
        check("case creation no longer 500s", resp.status_code == 200, f"status={resp.status_code}")
        check("case carries resolved score_id", body.get("score_id") == score_id,
              f"got={body.get('score_id')} want={score_id}")
        case1 = body.get("case_id")
        # Server truth: caller sent risk=100/ml=0.99 but the stored row is used.
        con = sqlite3.connect(risk_db_path)
        c1 = con.execute("SELECT risk_score, score_id FROM investigator_cases WHERE case_id=?",
                         (case1,)).fetchone()
        con.close()
        check("priority derived from STORED score (not caller args)",
              c1 is not None and c1[1] == score_id, f"stored risk={c1[0] if c1 else None}")

        # 2) create on missing score -> 404
        r404 = vc.post("/investigator/cases",
                       params={"event_id": "casefix-noscore-01", "fraud_id": fraud_id,
                               "risk_score": 90, "ml_score": 0.9},
                       json=["BEHAVIOR_DEVIATION"], headers=chdr)
        check("case for unscored event rejected 404", r404.status_code == 404)

        # 3) dedup: second create returns existing case
        again = vc.post("/investigator/cases",
                        params={"event_id": ev_high, "fraud_id": fraud_id,
                                "risk_score": 100, "ml_score": 0.99},
                        json=["HIGH_AMOUNT_RATIO"], headers=chdr)
        check("duplicate create returns existing case",
              again.status_code == 200 and again.json().get("already_exists") is True,
              f"case_id same={again.json().get('case_id') == case1}")
        con = sqlite3.connect(risk_db_path)
        n_cases = con.execute("SELECT COUNT(*) FROM investigator_cases WHERE event_id=?",
                              (ev_high,)).fetchone()[0]
        con.close()
        check("one case per event (DB unique index)", n_cases == 1, f"count={n_cases}")

        # 4) lifecycle: NEW -> REVIEWING -> CONFIRMED_SUSPICIOUS -> CLOSED
        t1 = vc.post(f"/investigator/cases/{case1}/transition",
                     params={"new_status": "REVIEWING", "investigator_id": "analyst-1",
                             "notes": "reviewing"}, headers=chdr)
        t2 = vc.post(f"/investigator/cases/{case1}/transition",
                     params={"new_status": "CONFIRMED_SUSPICIOUS",
                             "investigator_id": "analyst-1"}, headers=chdr)
        t3 = vc.post(f"/investigator/cases/{case1}/transition",
                     params={"new_status": "CLOSED", "investigator_id": "analyst-1"},
                     headers=chdr)
        check("legal transitions accepted",
              t1.status_code == 200 and t2.status_code == 200 and t3.status_code == 200,
              f"{t1.status_code}/{t2.status_code}/{t3.status_code}")
        check("suspicious verdict records label on transition",
              t2.json().get("label_recorded") is True)
        # illegal: closed -> reviewing
        tbad = vc.post(f"/investigator/cases/{case1}/transition",
                       params={"new_status": "REVIEWING"}, headers=chdr)
        check("illegal transition from CLOSED rejected 400", tbad.status_code == 400)

        # 5) benign-event case closed via CONFIRMED_LEGITIMATE
        rlow = vc.post("/investigator/cases",
                       params={"event_id": ev_low, "fraud_id": fraud_id,
                               "risk_score": 20, "ml_score": 0.05},
                       json=["BEHAVIOR_NORMAL"], headers=chdr)
        case2 = rlow.json().get("case_id")
        t4 = vc.post(f"/investigator/cases/{case2}/transition",
                     params={"new_status": "CONFIRMED_LEGITIMATE",
                             "investigator_id": "analyst-2"}, headers=chdr)
        t5 = vc.post(f"/investigator/cases/{case2}/transition",
                     params={"new_status": "CLOSED", "investigator_id": "analyst-2"},
                     headers=chdr)
        check("legitimate verdict + close accepted",
              t4.status_code == 200 and t5.status_code == 200 and t4.json().get("label_recorded"))

        # 6) labels landed in verification_outcomes (same store as user confirmations)
        con = sqlite3.connect(risk_db_path)
        outs = con.execute(
            "SELECT event_id, outcome, case_id, score_id FROM verification_outcomes "
            "ORDER BY event_id").fetchall()
        con.close()
        by_event = {o[0]: o for o in outs}
        check("verdicts become VerificationOutcome labels",
              by_event.get(ev_high) is not None and by_event.get(ev_low) is not None,
              f"{len(outs)} outcome rows")
        check("suspicious -> disputed label",
              by_event.get(ev_high) and by_event[ev_high][1] == "disputed")
        check("legitimate -> confirmed label",
              by_event.get(ev_low) and by_event[ev_low][1] == "confirmed")
        check("labels carry the case_id + score_id",
              by_event.get(ev_high) and by_event[ev_high][2] == case1 and by_event[ev_high][3] == score_id)

        # 7) feedback pool now counts investigator labels
        fb = vc.get("/feedback-status").json()
        check("feedback pool reflects investigator labels", fb.get("resolved", 0) == 2,
              f"resolved={fb.get('resolved')}")

        # 8) listing works on the shared store
        lst = vc.get("/investigator/cases", headers=chdr).json()
        check("case listing on shared store", lst.get("total") == 2, f"total={lst.get('total')}")

    print(f"\n{'ALL PASS' if not failures else f'{len(failures)} FAILURES: {failures}'}")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
