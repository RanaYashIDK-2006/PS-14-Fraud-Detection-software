#!/usr/bin/env python3
"""Live HTTP walkthrough across the four running PS-14 services.

Requires: identity on :8001, privacy on :8002, risk on :8003, verification
on :8004 (all with default dev secrets). Registers a throwaway user, chains
Privacy -> Risk via the Verification Service's /demo/seed, lists alerts,
confirms one as "this was me" and disputes another as "this wasn't me".

Run from the project root:
  python scripts/live_walkthrough.py
"""

from __future__ import annotations

import sys
import uuid

import httpx

IDENTITY = "http://127.0.0.1:8001"
PRIVACY = "http://127.0.0.1:8002"
RISK = "http://127.0.0.1:8003"
VERIFY = "http://127.0.0.1:8004"

email = f"live-{uuid.uuid4().hex[:8]}@example.com"
password = "hunter2-live-password"


def main() -> int:
    ok = True
    with httpx.Client(timeout=10.0) as c:
        # health
        for name, url in [("identity", IDENTITY), ("privacy", PRIVACY), ("risk", RISK), ("verification", VERIFY)]:
            r = c.get(f"{url}/health")
            print(f"[{name}] health {r.status_code} {r.json()}")
            ok &= r.status_code == 200

        # register + login
        r = c.post(f"{IDENTITY}/auth/register", json={
            "full_name": "Live Walker", "email": email, "phone": "+919876543219", "password": password})
        r.raise_for_status()
        fraud_id = r.json()["fraud_id"]
        print(f"[identity] registered user, fraud_id={fraud_id}")

        r = c.post(f"{IDENTITY}/auth/login", json={"email": email, "password": password})
        r.raise_for_status()
        token = r.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        # demo seed: verification chains privacy -> risk over HTTP. The seed
        # WARMS THE ACCOUNT UP FIRST - six normal events + an in-DB-2 aging
        # step to ~60 days - so the attack is scored against a mature
        # history, not a brand-new one (no cold-start false positives).
        # Allowed (low) warm-up events are committed to the behavioral
        # baseline; the blocked attack event is NOT.
        r = c.post(f"{VERIFY}/demo/seed", headers=headers)
        r.raise_for_status()
        seed = r.json()
        print(f"[verification] demo seed: {seed['seeded']} events, {seed['high_risk']} high-risk -> {len(seed['alerts'])} alert(s), {seed['baseline_committed']} baseline-committed, age_days={seed['age_days']}")
        ok &= seed["high_risk"] >= 1
        ok &= seed["baseline_committed"] >= 1  # allowed events entered the baseline; the blocked attack did not
        # The warm-up guarantee: the account is aged to 60 days BEFORE the
        # attack, so the ONLY high-risk event is the attack itself - the
        # warm-up history must not surface as false-positive alerts.
        ok &= seed["age_days"] == 60.0
        ok &= len(seed["alerts"]) == 1
        attack1 = seed["alerts"][0]["event_id"]

        # ---- anti-pollution re-verification --------------------------------
        # A normal transaction AFTER the blocked attack must derive against
        # the UN-polluted baseline: amount_ratio ~1 (pre-fix the 4500 event
        # moved the EMA median to ~560, so normal events scored ratio 0.32).
        # Uses the account's OWN device (the one the seed warmed up with) and
        # a unique event id, so repeated runs against a persistent DB work.
        it = {"X-Internal-Token": "ps14-dev-internal-token-change-me"}
        normal = {
            "event_id": f"post-attack-{uuid.uuid4().hex[:8]}", "fraud_id": fraud_id, "amount": 120.0,
            "ts": "2025-07-01T12:00:00", "hour_of_day": 12,
            "device_id": seed["device_id"], "location_id": "L-DEMO-01",
            "recipient_id": "R-DEMO-01", "failed_auth_count_24h": 0,
        }
        r = c.post(f"{PRIVACY}/internal/ingest-transaction", json=normal, headers=it)
        r.raise_for_status()
        nfeat = r.json()
        print(f"[privacy] post-attack normal event derived: amount_ratio={nfeat['amount_ratio']} "
              f"new_device={nfeat['new_device_flag']} (expect ratio ~1.0)")
        ok &= 0.9 < nfeat["amount_ratio"] < 1.1
        r = c.post(f"{RISK}/internal/evaluate", json={
            "event_id": normal["event_id"], "fraud_id": fraud_id, "features": nfeat}, headers=it)
        r.raise_for_status()
        neval = r.json()
        print(f"[risk] post-attack normal event: {neval['decision']} (score {neval['risk_score']}, "
              f"reasons {neval['reason_codes']})")
        # The KEY anti-pollution property: the blocked attack did NOT move the
        # baseline, so this event's ratio is ~1.0 (pre-fix it was 0.32) and it
        # is never escalated to VERIFY. A step-up is still possible when the
        # recent-window escalation feature sees the blocked spike - a
        # documented residual (features include all attempts; only the stored
        # baseline is gated).
        ok &= neval["risk_band"] != "high"
        r = c.post(f"{PRIVACY}/internal/commit-baseline", json={
            "event_id": normal["event_id"], "fraud_id": fraud_id}, headers=it)
        r.raise_for_status()
        print(f"[privacy] committed allowed post-attack event: {r.json()['committed']}")

        # a second attack event to dispute (so confirm AND dispute both run)
        attack2 = {
            "event_id": f"wt-attack-{uuid.uuid4().hex[:8]}", "fraud_id": fraud_id, "amount": 3800.0,
            "ts": "2025-07-01T02:00:00", "hour_of_day": 2,
            "device_id": "wt-attacker-2", "location_id": "L-WT-2",
            "recipient_id": "R-WT-2", "failed_auth_count_24h": 6,
        }
        r = c.post(f"{PRIVACY}/internal/ingest-transaction", json=attack2, headers=it)
        r.raise_for_status()
        a2feat = r.json()
        r = c.post(f"{RISK}/internal/evaluate", json={
            "event_id": attack2["event_id"], "fraud_id": fraud_id, "features": a2feat}, headers=it)
        r.raise_for_status()
        print(f"[risk] second attack event: {r.json()['decision']} (score {r.json()['risk_score']})")
        ok &= r.json()["risk_band"] == "high"

        # alerts
        r = c.get(f"{VERIFY}/alerts", headers=headers)
        r.raise_for_status()
        alerts = r.json()["alerts"]
        print(f"[verification] {len(alerts)} open alert(s):")
        for a in alerts:
            print(f"  - {a['event_id']}  score={a['risk_score']} band={a['risk_band']} reasons={a['reason_codes']}")
        ok &= len(alerts) >= 2

        # confirm + dispute
        first = next(a for a in alerts if a["event_id"] == attack1)
        r = c.post(f"{VERIFY}/alerts/{first['event_id']}/confirm", headers=headers, json={"outcome": "this_was_me"})
        r.raise_for_status()
        cbody = r.json()
        print(f"[verification] confirmed {first['event_id']}: {cbody['message']} (baseline_updated={cbody['baseline_updated']})")
        ok &= cbody["baseline_updated"] is True

        second = next(a for a in alerts if a["event_id"] == attack2["event_id"])
        r = c.post(f"{VERIFY}/alerts/{second['event_id']}/confirm", headers=headers, json={"outcome": "this_wasnt_me"})
        r.raise_for_status()
        body = r.json()
        print(f"[verification] disputed {second['event_id']}: {body['message']} (case {body['case_id']}, "
              f"baseline_updated={body['baseline_updated']})")
        print(f"[verification] recovery steps: {body['recovery_steps'][:2]} ...")
        ok &= body["baseline_updated"] is False

        # ---- baseline-discipline DB assertions ------------------------------
        import sqlite3
        con = sqlite3.connect("db/features.db")
        committed = {eid: flag for eid, flag in con.execute(
            "SELECT event_id, baseline_committed FROM transaction_features")}
        con.close()
        print(f"[privacy] baseline_committed flags: post-attack={committed.get(normal['event_id'])}, "
              f"{attack1}={committed.get(attack1)}, wt-attack-0002={committed.get(attack2['event_id'])}")
        ok &= committed.get(normal["event_id"]) == 1       # allowed -> committed
        ok &= committed.get(attack1) == 1                   # confirmed -> committed
        ok &= committed.get(attack2["event_id"]) == 0      # disputed -> never committed

        r = c.get(f"{VERIFY}/alerts", headers=headers)
        print(f"[verification] open alerts after resolution: {len(r.json()['alerts'])}")
        ok &= len(r.json()["alerts"]) == 0

    print("\nLIVE WALKTHROUGH " + ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
