#!/usr/bin/env python3
"""Smoke test for the PS-14 hash-chained audit trail (DB-4).

Checks that risk scoring and verification outcomes append chained events,
that /audit/integrity recomputes the chain correctly, that tampering is
detected (including a realistic attack: dropping the trigger, rewriting a
payload, and watching the integrity check fail), and that the store is
append-only at the storage layer.

Run from the project root:
  python scripts/audit_test.py
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

TMP = tempfile.mkdtemp(prefix="ps14-audit-")
os.environ["DB_DIR"] = TMP
os.environ["JWT_SECRET"] = "smoke-test-secret-0123456789abcdef"
os.environ["INTERNAL_TOKEN"] = "smoke-internal-token"
os.environ["COMPLIANCE_TOKEN"] = "smoke-compliance-token"

from fastapi.testclient import TestClient  # noqa: E402

from src.audit_service.export import verify_export_chain, verify_export_signature  # noqa: E402
from src.audit_service.main import app as audit_app  # noqa: E402
from src.identity_service.main import app as identity_app  # noqa: E402
from src.identity_service.security import create_access_token  # noqa: E402
from src.privacy_layer.main import app as privacy_app  # noqa: E402
from src.risk_engine.main import app as risk_app  # noqa: E402
from src.settings import get_settings  # noqa: E402
try:
    from src.settings import load_dotenv_and_patch
    load_dotenv_and_patch()
except ImportError:
    pass
from src.verification_service.main import app as verify_app  # noqa: E402

TOKEN = "smoke-internal-token"
FRAUD_ID = "F24BRMMMBJWYMTDW"
failures: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        failures.append(name)


def clean_vector(**overrides) -> dict:
    base = {
        "amount_ratio": 1.0, "txn_freq_last_24h": 1, "txn_time_unusual": 0,
        "new_device_flag": 0, "unusual_location_flag": 0, "unusual_recipient_flag": 0,
        "failed_auth_count_24h": 0, "days_since_last_similar_txn": 2.5,
        "gradual_escalation_score": 0.0, "known_device_count": 3,
        "account_tenure_days": 120.0, "hour_of_day": 12, "is_weekend": 0,
    }
    base.update(overrides)
    return base


def attack_vector() -> dict:
    return clean_vector(amount_ratio=3.5, txn_time_unusual=1, new_device_flag=1,
                        unusual_location_flag=1, unusual_recipient_flag=1,
                        failed_auth_count_24h=4, hour_of_day=3)


def main() -> int:
    print("== Audit trail smoke test (DB-4) ==")
    with TestClient(risk_app) as rc, TestClient(verify_app) as vc, TestClient(audit_app) as ac, \
         TestClient(privacy_app) as pc, TestClient(identity_app) as ic:

        # ---- populate the chain --------------------------------------------
        r = rc.post("/internal/evaluate", headers={"X-Internal-Token": TOKEN},
                    json={"event_id": "audit-evil-0001", "fraud_id": FRAUD_ID, "features": attack_vector()})
        assert r.status_code == 200 and r.json()["risk_band"] == "high", r.text
        r = rc.post("/internal/evaluate", headers={"X-Internal-Token": TOKEN},
                    json={"event_id": "audit-clean-01", "fraud_id": FRAUD_ID, "features": clean_vector()})
        assert r.status_code == 200 and r.json()["risk_band"] == "low", r.text

        auth = {"Authorization": f"Bearer {create_access_token(FRAUD_ID)}"}
        r = vc.post("/alerts/audit-evil-0001/confirm", headers=auth, json={"outcome": "this_wasnt_me"})
        assert r.status_code == 200, r.text
        check("chain: risk + verification events appended", True)

        # ---- chain linkage + compliance viewer -----------------------------
        r = ac.get(f"/audit/events?fraud_id={FRAUD_ID}", headers={"X-Internal-Token": TOKEN,
                                                                  "X-Audit-Actor": "compliance-officer-1"})
        assert r.status_code == 200, r.text
        events = r.json()["events"]
        types = [e["event_type"] for e in events]
        check("events logged", sorted(types) == ["score_generated", "score_generated", "verification_resolved"], str(types))
        # events are newest-first; reverse to chain order
        ordered = list(reversed(events))
        linked = all(
            ordered[i]["prev_hash"] == ordered[i - 1]["entry_hash"] for i in range(1, len(ordered))
        )
        check("hash chain links (prev == previous entry)", linked)
        check("payloads pseudonymous", all("email" not in str(e["payload"]).lower() and "name" not in str(e["payload"]).lower()
                                           for e in events), str(events[0]["payload"]))

        r = ac.get("/audit/events")  # no token
        check("audit viewer requires role (401)", r.status_code == 401)

        r = ac.get("/audit/integrity", headers={"X-Internal-Token": TOKEN})
        assert r.status_code == 200, r.text
        check("integrity ok", r.json()["ok"] is True and r.json()["n_entries"] == 3, str(r.json()))

        r = ac.get("/audit/integrity", headers={"X-Internal-Token": TOKEN})
        check("integrity stable across reads", r.json()["ok"] is True)

        # ---- privacy + identity audit coverage -----------------------------
        # A Privacy Layer ingest must append a hash-chained event; a
        # break-glass fraud-id resolution must too (PII stays in DB-1).
        r = pc.post("/internal/ingest-transaction", headers={"X-Internal-Token": TOKEN},
                    json={"event_id": "audit-ingest-01", "fraud_id": FRAUD_ID, "amount": 120.0,
                          "ts": "2025-08-01T12:00:00", "hour_of_day": 12, "device_id": "audit-dev",
                          "location_id": "L-A1", "recipient_id": "R-A1", "failed_auth_count_24h": 0})
        assert r.status_code == 200, r.text
        check("privacy: ingest returns derived vector", r.json().get("amount_ratio") is not None)

        r = ic.post("/auth/register", json={"full_name": "Audit Tester", "email": "audit@example.com",
                                            "phone": "+919876543222", "password": "hunter2-secure-password"})
        assert r.status_code == 201, r.text
        resolved_fid = r.json()["fraud_id"]
        r = ic.post("/internal/resolve-fraud-id", headers={"X-Internal-Token": TOKEN},                     json={"fraud_id": resolved_fid, "reason": "regulatory request (audit test)", "case_id": "AUDIT-TEST-001"})
        assert r.status_code == 200, r.text
        check("identity: break-glass returns masked identity",
              "email_masked" in r.json() and "audit@" not in r.json()["email_masked"])

        r = ac.get(f"/audit/events?fraud_id={FRAUD_ID}", headers={"X-Internal-Token": TOKEN})
        evs = r.json()["events"]
        types_all = {e["event_type"] for e in evs}
        check("audit: feature_ingested in the chain", "feature_ingested" in types_all, str(types_all))
        ingest_ev = next(e for e in evs if e["event_type"] == "feature_ingested")
        check("audit: feature_ingested payload minimal (event id only)",
              set(ingest_ev["payload"].keys()) == {"event_id"}
              and ingest_ev["payload"]["event_id"] == "audit-ingest-01", str(ingest_ev["payload"]))

        r = ac.get(f"/audit/events?fraud_id={resolved_fid}", headers={"X-Internal-Token": TOKEN})
        revs = r.json()["events"]
        check("audit: fraud_id_resolved in the chain",
              len(revs) == 1 and revs[0]["event_type"] == "fraud_id_resolved", str([e["event_type"] for e in revs]))
        rp = revs[0]["payload"]
        check("audit: resolve payload = actor + reason, no PII",
              rp.get("actor") == "internal:compliance" and rp.get("reason") == "regulatory request (audit test)"
              and "audit@" not in str(rp) and "Tester" not in str(rp) and "919876543222" not in str(rp), str(rp))
        # The new events are linked into the chain like every other event.
        r = ac.get("/audit/integrity", headers={"X-Internal-Token": TOKEN})
        check("audit: chain still verifies with 5 events", r.json()["ok"] is True and r.json()["n_entries"] == 5,
              str(r.json()))

        # ---- compliance viewer UI (static page + passphrase-gated API) -----
        r = ac.get("/")
        # Fetch external app.js — JS patterns moved from inline to external file
        app_js = ac.get("/static/app.js")
        js_text = app_js.text if app_js.status_code == 200 else ""
        combined = r.text + js_text  # check both HTML and external JS
        check("compliance UI served", r.status_code == 200 and "Compliance Viewer" in combined
              and "hash-chained" in combined)
        check("compliance UI: compact chips + paging",
              all(x in combined for x in ("rchip", "rchip none", "tmore", "trailMoreBtn", "TRAIL_PAGE",
                                        "Show older events", "ps14-dev-compliance-token-change-me")))
        check("compliance UI: live footer badge polls integrity",
              all(x in combined for x in ("chainBadge", "refreshBadge", "Audit chain: checking",
                                        "setInterval(refreshBadge")))
        check("compliance UI: event-type filter chips",
              all(x in combined for x in ("filterRow", "EVENT_FILTERS", "Verifications",
                                        "Break-glass", "event_type=")))
        r = ac.get("/compliance/events")
        check("compliance viewer requires passphrase (401)", r.status_code == 401)
        c_hdr = {"X-Compliance-Token": "smoke-compliance-token"}
        r = ac.get("/compliance/events", headers=c_hdr)
        assert r.status_code == 200, r.text
        c_events = r.json()["events"]
        check("compliance viewer: trail served", len(c_events) == 5, str(len(c_events)))
        score_ev = next(e for e in c_events
                        if e["event_type"] == "score_generated" and e["payload"].get("reason_codes"))
        check("compliance viewer: category reason texts", "reason_texts" in score_ev["payload"],
              str(score_ev["payload"].get("reason_codes")))

        # ---- paging: offset + total on the trail endpoints ---------------
        r = ac.get("/compliance/events?limit=2&offset=0", headers=c_hdr)
        assert r.status_code == 200, r.text
        p1 = r.json()
        r = ac.get("/compliance/events?limit=2&offset=2", headers=c_hdr)
        assert r.status_code == 200, r.text
        p2 = r.json()
        r = ac.get("/compliance/events?limit=2&offset=4", headers=c_hdr)
        assert r.status_code == 200, r.text
        p3 = r.json()
        all_seqs = [e["seq"] for e in p1["events"] + p2["events"] + p3["events"]]
        check("compliance: offset pages are disjoint + cover the trail",
              len(p1["events"]) == 2 and len(p2["events"]) == 2 and len(p3["events"]) == 1
              and len({e["seq"] for e in p1["events"] + p2["events"] + p3["events"]}) == 5
              and sorted(all_seqs) == list(range(1, 6)),
              str(all_seqs))
        check("compliance: total reflects the whole chain",
              p1.get("total") == 5 and p2.get("total") == 5 and p3.get("total") == 5,
              str(p1.get("total")))
        check("compliance: offset beyond the tail returns empty",
              ac.get("/compliance/events?limit=2&offset=10", headers=c_hdr).json()["events"] == [])
        r = ac.get("/audit/events?limit=2&offset=2", headers={"X-Internal-Token": TOKEN})
        assert r.status_code == 200, r.text
        check("audit/events: internal endpoint also pages (total present)",
              r.json().get("total") == 5 and {e["seq"] for e in r.json()["events"]} == {2, 3},
              str(r.json().get("total")))

        # ---- event-type filter on the compliance trail ---------------------
        r = ac.get("/compliance/events?event_type=verification_resolved", headers=c_hdr)
        assert r.status_code == 200, r.text
        f = r.json()
        check("compliance: event_type filter returns only that type",
              f["total"] >= 1 and f["total"] == len(f["events"])
              and all(e["event_type"] == "verification_resolved" for e in f["events"]),
              str([(e["event_type"], e["seq"]) for e in f["events"]]))
        r = ac.get("/compliance/events?event_type=feature_ingested&limit=2&offset=0", headers=c_hdr)
        f2 = r.json()
        r = ac.get("/compliance/events?event_type=feature_ingested&limit=2&offset=2", headers=c_hdr)
        f3 = r.json()
        check("compliance: filtered paging (offset + filtered total)",
              f2["total"] == f3["total"] and f2["total"] == len(f2["events"]) + len(f3["events"])
              and all(e["event_type"] == "feature_ingested" for e in f2["events"] + f3["events"]),
              str(f2.get("total")))

        # ---- chain-wide summary stats (by_type + no-flag scores) ----------
        r = ac.get("/compliance/events", headers=c_hdr)
        d = r.json()
        check("compliance: by_type covers the whole chain",
              isinstance(d.get("by_type"), dict) and sum(d["by_type"].values()) == d["total"],
              str(d.get("by_type")))
        check("compliance: no_flag_scores counts flag-free scores",
              isinstance(d.get("no_flag_scores"), int) and d["no_flag_scores"] >= 0
              and d["no_flag_scores"] <= d["by_type"].get("score_generated", 0),
              str(d.get("no_flag_scores")))
        r = ac.get("/compliance/events?event_type=verification_resolved", headers=c_hdr)
        dv = r.json()
        check("compliance: no-flag count is 0 under a non-score filter",
              dv.get("no_flag_scores") == 0 and dv["by_type"] == {"verification_resolved": dv["total"]},
              str(dv.get("no_flag_scores")))
        # ---- /audit/overview: BFF aggregate for the compliance viewer -----
        r = ac.get("/audit/overview", headers={"X-Internal-Token": TOKEN})
        assert r.status_code == 200, r.text
        ov = r.json()
        check("audit/overview: one payload with all sections",
              set(ov) == {"integrity", "latest_case", "events", "count", "total", "by_type", "no_flag_scores"},
              str(sorted(ov)))
        check("audit/overview: integrity + totals consistent",
              ov["integrity"]["ok"] is True and ov["integrity"]["n_entries"] == 5
              and ov["total"] == 5 and ov["count"] == 5 and len(ov["events"]) == 5,
              str(ov["integrity"]))
        check("audit/overview: latest_case is the newest verification_resolved",
              ov["latest_case"] is not None and ov["latest_case"]["case_id"]
              and ov["latest_case"]["outcome"] in ("confirmed", "disputed")
              and ov["latest_case"]["seq"] == max(
                  e["seq"] for e in ov["events"] if e["event_type"] == "verification_resolved"),
              str(ov["latest_case"]))
        lc = ov["latest_case"]
        sc = next(e for e in ov["events"] if e["event_type"] == "score_generated"
                  and e["payload"].get("event_id") == lc["event_id"])
        check("audit/overview: latest_case joined with its score_generated",
              lc["risk_band"] == sc["payload"]["risk_band"]
              and lc["risk_score"] == sc["payload"]["risk_score"],
              str(lc))
        check("audit/overview: reason texts enriched on scores",
              any(e["event_type"] == "score_generated" and "reason_texts" in e["payload"]
                  for e in ov["events"]))
        r2 = ac.get("/audit/overview?limit=2&offset=0", headers={"X-Internal-Token": TOKEN})
        check("audit/overview: paging respected (limit/offset)",
              r2.json()["count"] == 2 and r2.json()["total"] == 5 and len(r2.json()["events"]) == 2,
              str((r2.json().get("count"), r2.json().get("total"))))
        r = ac.get("/audit/overview")  # no token
        check("audit/overview: requires role (401)", r.status_code == 401)
        r = ac.get("/compliance/integrity", headers=c_hdr)
        check("compliance viewer: integrity ok", r.json()["ok"] is True and r.json()["n_entries"] == 5,
              str(r.json()))

        # ---- compliance export (regulator-facing, signed) ------------------
        r = ac.get("/audit/export")
        check("export: requires internal token (401)", r.status_code == 401)
        r = ac.get("/audit/export", headers={"X-Internal-Token": TOKEN})
        assert r.status_code == 200, r.text
        exp = r.json()
        check("export: format + integrity ok", exp.get("format") == "ps14-audit-export-v1"
              and exp.get("integrity", {}).get("ok") is True, str(exp.get("integrity")))
        check("export: all events dumped", len(exp.get("events", [])) == 5, f"n={len(exp.get('events'))}")
        check("export: signed (hmac-sha256)", bool(exp.get("signature"))
              and exp.get("signature_algorithm") == "hmac-sha256")
        key = get_settings().export_signing_key
        check("export: signature verifies", verify_export_signature(exp, key))
        chk = verify_export_chain(exp, exp["genesis_hash"])
        check("export: chain re-verifies independently", chk["ok"] is True and chk["n_entries"] == 5, str(chk))

        # A tampered copy (as a regulator would receive it) must fail BOTH
        # checks: the signature (content changed) and the chain (hash
        # mismatch at the altered entry).
        tampered = json.loads(json.dumps(exp))
        tampered["events"][2]["payload"]["risk_score"] = 1
        check("export: tampered copy fails signature", not verify_export_signature(tampered, key))
        cv = verify_export_chain(tampered, tampered["genesis_hash"])
        check("export: tampered copy fails chain", cv["ok"] is False and cv.get("first_bad_seq") == 3, str(cv))

        # ---- append-only guard (storage layer) -----------------------------
        con = sqlite3.connect(Path(TMP) / "audit.db")
        try:
            con.execute("UPDATE audit_events SET payload_summary = 'hacked' WHERE seq = 1")
            con.commit()
            blocked = False
        except sqlite3.DatabaseError as e:
            blocked = "append-only" in str(e)
            con.rollback()
        con.close()
        check("append-only: UPDATE blocked by trigger", blocked)

        # ---- tamper detection ----------------------------------------------
        con = sqlite3.connect(Path(TMP) / "audit.db")
        con.execute("DROP TRIGGER IF EXISTS audit_events_no_update")
        con.execute("UPDATE audit_events SET payload_summary = 'hacked' WHERE seq = 1")
        con.commit()
        con.close()

        r = ac.get("/audit/integrity", headers={"X-Internal-Token": TOKEN})
        body = r.json()
        check("tamper detected", body["ok"] is False and body.get("first_bad_seq") == 1,
              str(body))
        r = ac.get("/compliance/integrity", headers={"X-Compliance-Token": "smoke-compliance-token"})
        cbody = r.json()
        check("compliance viewer also reports tamper", cbody["ok"] is False and cbody.get("first_bad_seq") == 1,
              str(cbody))
        r2 = ac.get(f"/audit/events?fraud_id={FRAUD_ID}", headers={"X-Internal-Token": TOKEN})
        tampered_payload = r2.json()["events"][-1]["payload"]  # seq 1 is oldest -> last in newest-first
        check("tampered payload flagged for compliance", tampered_payload.get("corrupted_payload") == "hacked", str(tampered_payload))

        # The export of a BROKEN chain: still authentic (the service signed
        # exactly what it has) but the integrity report must flag the break,
        # and the regulator-side chain re-verification must reject it.
        r = ac.get("/audit/export", headers={"X-Internal-Token": TOKEN})
        exp2 = r.json()
        check("export: broken chain flagged", exp2["integrity"]["ok"] is False
              and exp2["integrity"].get("first_bad_seq") == 1, str(exp2["integrity"]))
        check("export: broken chain still authentic", verify_export_signature(exp2, key))
        cv2 = verify_export_chain(exp2, exp2["genesis_hash"])
        check("export: broken chain rejected by regulator check",
              cv2["ok"] is False and cv2.get("first_bad_seq") == 1, str(cv2))

        # ---- access logging + DB separation --------------------------------
        con = sqlite3.connect(Path(TMP) / "audit.db")
        n_access = con.execute("SELECT COUNT(*) FROM audit_access_log").fetchone()[0]
        tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        con.close()
        check("audit reads logged", n_access >= 4, f"n={n_access}")

        con = sqlite3.connect(Path(TMP) / "risk.db")
        risk_tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        con.close()
        check("DB-4 has only audit tables", tables <= {"audit_events", "audit_access_log", "sqlite_sequence"}, str(tables))
        check("audit_events not in DB-3", "audit_events" not in risk_tables)

    print("\n" + ("ALL CHECKS PASSED" if not failures else f"{len(failures)} CHECK(S) FAILED"))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
