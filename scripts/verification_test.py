#!/usr/bin/env python3
"""Smoke test for the PS-14 Verification flow (all four services in-process).

Chain: Identity (register/login) -> Privacy (ingest) -> Risk (evaluate) ->
Verification (alerts/confirm), then DB-3 assertions for verification_outcomes
and the served UI.

Run from the project root:
  python scripts/verification_test.py
"""

from __future__ import annotations

import os
import sqlite3
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

TMP = tempfile.mkdtemp(prefix="ps14-verify-")
os.environ["DB_DIR"] = TMP
os.environ["JWT_SECRET"] = "smoke-test-secret-0123456789abcdef"
os.environ["INTERNAL_TOKEN"] = "smoke-internal-token"
os.environ["COMPLIANCE_TOKEN"] = "smoke-compliance-token"

from fastapi.testclient import TestClient  # noqa: E402

from src.identity_service.main import app as identity_app  # noqa: E402
from src.privacy_layer.main import app as privacy_app  # noqa: E402
from src.risk_engine.main import app as risk_app  # noqa: E402
from src.verification_service.main import app as verify_app  # noqa: E402
from src.audit_service.main import app as audit_app  # noqa: E402
import src.verification_service.main as verify_module  # noqa: E402
from src.settings import get_settings  # noqa: E402
try:
    from src.settings import load_dotenv_and_patch
    load_dotenv_and_patch()
except ImportError:
    pass

COMPLIANCE = "smoke-compliance-token"

TOKEN = "smoke-internal-token"
EMAIL = "verify@example.com"
PASSWORD = "hunter2-secure-password"
failures: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        failures.append(name)


def main() -> int:
    print("== Verification flow smoke test (4 services) ==")
    with TestClient(identity_app) as ic, TestClient(privacy_app) as pc, \
         TestClient(risk_app) as rc, TestClient(verify_app) as vc, \
         TestClient(audit_app) as ac:

        # ---- Identity ------------------------------------------------------
        r = ic.post("/auth/register", json={"full_name": "Verify Tester", "email": EMAIL, "phone": "+919876543212", "password": PASSWORD})
        assert r.status_code == 201, r.text
        fraud_id = r.json()["fraud_id"]
        r = ic.post("/auth/login", json={"email": EMAIL, "password": PASSWORD})
        assert r.status_code == 200, r.text
        token = r.json()["access_token"]
        auth = {"Authorization": f"Bearer {token}"}
        check("identity: login ok", bool(token))

        # ---- Privacy ingest (baseline + attack event) ----------------------
        now = datetime.now(timezone.utc).isoformat()
        base = {"fraud_id": fraud_id, "ts": now, "hour_of_day": 12,
                "device_id": "verify-dev", "location_id": "L-V1", "recipient_id": "R-V1",
                "failed_auth_count_24h": 0}
        feature_map = {}
        for i in range(3):
            ev = {**base, "event_id": f"verify-base-000{i}", "amount": 120.0}
            r = pc.post("/internal/ingest-transaction", headers={"X-Internal-Token": TOKEN}, json=ev)
            assert r.status_code == 200, r.text
        attack = {**base, "event_id": "verify-attack-01", "amount": 4500.0, "hour_of_day": 3,
                  "device_id": "verify-attacker", "location_id": "L-X", "recipient_id": "R-X",
                  "failed_auth_count_24h": 5}
        r = pc.post("/internal/ingest-transaction", headers={"X-Internal-Token": TOKEN}, json=attack)
        assert r.status_code == 200, r.text
        attack_features = r.json()
        # A genuinely clean feature vector (not derived from the attack event)
        clean = {"amount_ratio": 1.0, "txn_freq_last_24h": 1, "txn_time_unusual": 0,
                 "new_device_flag": 0, "unusual_location_flag": 0, "unusual_recipient_flag": 0,
                 "failed_auth_count_24h": 0, "days_since_last_similar_txn": 2.5,
                 "gradual_escalation_score": 0.0, "known_device_count": 3,
                 "account_tenure_days": 120.0, "hour_of_day": 12, "is_weekend": 0}
        check("privacy: baseline + attack ingested", True)

        # ---- Risk evaluate -------------------------------------------------
        r = rc.post("/internal/evaluate", headers={"X-Internal-Token": TOKEN},
                    json={"event_id": "verify-attack-01", "fraud_id": fraud_id, "features": attack_features})
        assert r.status_code == 200, r.text
        attack_eval = r.json()
        check("risk: attack -> high/verify", attack_eval["risk_band"] == "high", f"score={attack_eval['risk_score']}")

        r = rc.post("/internal/evaluate", headers={"X-Internal-Token": TOKEN},
                    json={"event_id": "verify-legit-01", "fraud_id": fraud_id, "features": clean})
        assert r.status_code == 200, r.text
        legit_eval = r.json()
        check("risk: legit -> low/allow", legit_eval["risk_band"] == "low", f"score={legit_eval['risk_score']}")

        # ---- Verification: alerts ------------------------------------------
        r = vc.get("/alerts", headers=auth)
        assert r.status_code == 200, r.text
        alerts = r.json()["alerts"]
        ids = [a["event_id"] for a in alerts]
        check("alerts: contains attack, not legit", "verify-attack-01" in ids and "verify-legit-01" not in ids, str(ids))
        attack_alert = next(a for a in alerts if a["event_id"] == "verify-attack-01")
        check("alerts: category-level reasons", set(attack_alert["reason_codes"]) <= {
            "AMOUNT_UNUSUAL", "NEW_DEVICE", "UNUSUAL_TIME", "LOCATION_UNUSUAL", "RECIPIENT_UNUSUAL",
            "AUTH_ANOMALY", "FREQUENCY_ABNORMAL", "BEHAVIOR_DEVIATION",
            "DAILY_SPEND_EXCEEDED", "ACCOUNT_DAILY_LIMIT", "DEVICE_DAILY_LIMIT",
            "DOMAIN_SHIFT", "UNCERTAINTY_ESCALATION", "UNCERTAINTY_INVESTIGATION",
            "MULE_RING"}, str(attack_alert["reason_codes"]))
        check("alerts: human-readable texts", any("device" in t.lower() for t in attack_alert["reason_texts"]), str(attack_alert["reason_texts"]))

        r = vc.get("/alerts")  # no token
        check("alerts: no token 401", r.status_code == 401)

        r = vc.get(f"/alerts/verify-attack-01/reason", headers=auth)
        check("reason endpoint 200", r.status_code == 200 and r.json()["event_id"] == "verify-attack-01")

        # ---- Verification: confirm (this was me) ---------------------------
        r = vc.post("/alerts/verify-attack-01/confirm", headers=auth, json={"outcome": "this_was_me"})
        assert r.status_code == 200, r.text
        check("confirm this_was_me", r.json()["outcome"] == "confirmed" and r.json()["message"], r.text[:120])
        # The commit to the Privacy Layer is best-effort over HTTP: with the
        # in-process TestClient no server listens on settings.privacy_url, so
        # confirm must still succeed and report baseline_updated=False
        # (graceful degradation). The live walkthrough verifies the positive
        # path over real HTTP.
        check("confirm reports baseline_updated (graceful)",
              isinstance(r.json().get("baseline_updated"), bool) and r.json()["baseline_updated"] is False,
              r.text[:120])

        r = vc.post("/alerts/verify-attack-01/confirm", headers=auth, json={"outcome": "this_was_me"})
        check("duplicate confirm 409", r.status_code == 409)

        r = vc.get("/alerts", headers=auth)
        check("resolved alert gone from list", "verify-attack-01" not in [a["event_id"] for a in r.json()["alerts"]])

        # ---- Verification: dispute (this wasn't me) ------------------------
        attack2 = {**base, "event_id": "verify-attack-02", "amount": 3800.0, "hour_of_day": 2,
                   "device_id": "verify-attacker-2", "location_id": "L-Y", "recipient_id": "R-Y",
                   "failed_auth_count_24h": 6}
        r = pc.post("/internal/ingest-transaction", headers={"X-Internal-Token": TOKEN}, json=attack2)
        assert r.status_code == 200, r.text
        f2 = r.json()
        r = rc.post("/internal/evaluate", headers={"X-Internal-Token": TOKEN},
                    json={"event_id": "verify-attack-02", "fraud_id": fraud_id, "features": f2})
        assert r.status_code == 200 and r.json()["risk_band"] == "high", r.text

        r = vc.post("/alerts/verify-attack-02/confirm", headers=auth, json={"outcome": "this_wasnt_me"})
        assert r.status_code == 200, r.text
        d = r.json()
        check("dispute this_wasnt_me", d["outcome"] == "disputed" and d.get("case_id", "").startswith("PS14-"), r.text[:120])
        check("recovery steps returned", len(d.get("recovery_steps", [])) >= 4, str(d.get("recovery_steps")))
        # A disputed event must NEVER enter the baseline - no commit call.
        check("dispute never updates the baseline", d.get("baseline_updated") is False, r.text[:120])
        con = sqlite3.connect(Path(TMP) / "features.db")
        n_committed = con.execute(
            "SELECT COUNT(*) FROM transaction_features WHERE baseline_committed = 1"
        ).fetchone()[0]
        committed_ids = [r[0] for r in con.execute(
            "SELECT event_id FROM transaction_features WHERE baseline_committed = 1")]
        con.close()
        # Only the account's birth event is committed (by init). The
        # confirmed event's commit could not reach a server (TestClient), and
        # the disputed event never attempts one - so nothing else may be
        # committed.
        check("only the birth event committed (nothing else pollutes)",
              n_committed == 1 and committed_ids == ["verify-base-0000"],
              f"n={n_committed} ids={committed_ids}")

        # ---- Post-verification: case history + feedback status ------------
        r = vc.get("/cases", headers=auth)
        assert r.status_code == 200, r.text
        cases = r.json()["cases"]
        # Both outcomes resolve within the same second, so resolved_at ties
        # and DESC order is undefined - assert the set, not the order.
        check("cases: 2 resolved, both outcomes present",
              len(cases) == 2 and {c["outcome"] for c in cases} == {"confirmed", "disputed"},
              str([c["outcome"] for c in cases]))
        check("cases: joined with risk score + case id",
              all(c["case_id"].startswith("PS14-") and isinstance(c["risk_score"], int)
                  and c["risk_band"] == "high" and c["event_id"] for c in cases),
              str(cases))
        check("cases: reason codes/texts join through to history",
              all(isinstance(c["reason_codes"], list) and isinstance(c["reason_texts"], list)
                  and len(c["reason_texts"]) == len(c["reason_codes"])
                  and all(isinstance(t, str) and t for t in c["reason_texts"]) for c in cases),
              str([(c["case_id"], c["reason_texts"]) for c in cases]))
        # Fetch HTML + external JS for markup checks
        _r = vc.get("/")
        _js = vc.get("/static/app.js")
        _combined = _r.text + (_js.text if _js.status_code == 200 else "")
        check("cases: UI renders summary chips + expandable reason detail",
              "confirmed" in _combined
              and "rchip-wrap" in _combined and "Why this was flagged" in _combined
              and "caret" in _combined, "markup")
        r = vc.get("/cases")  # no token
        check("cases: no token 401", r.status_code == 401)
        r = vc.get("/feedback-status")
        fs = r.json()
        check("feedback-status: 1 confirmed + 1 disputed",
              fs["confirmed"] == 1 and fs["disputed"] == 1 and fs["resolved"] == 2, str(fs))
        check("feedback-status: retrain threshold surfaced", fs["retrain_disputed_threshold"] >= 1, str(fs))

        # ---- UI served -----------------------------------------------------
        r = vc.get("/")
        # Fetch external app.js — JS patterns moved from inline to external file
        app_js = vc.get("/static/app.js")
        js_text = app_js.text if app_js.status_code == 200 else ""
        combined = r.text + js_text  # check both HTML and external JS
        check("UI served", r.status_code == 200 and "PS-14" in combined
              and "This was me" in combined)
        check("UI has compliance view", "complianceView" in combined and "Compliance" in combined)
        check("UI has live chain-integrity footer badge",
              "chainBadge" in combined and "chain-status" in combined)
        check("UI surfaces chain status on the alerts screen",
              "alertsChainChip" in combined and "refreshCompliance" in combined)
        check("UI uses the single /overview fetch (BFF)",
              "loadOverview" in combined and "renderCases" in combined
              and "/overview" in combined and "loadAlerts" not in combined)
        check("UI has gentle /overview auto-refresh (30s silent tick)",
              "OVERVIEW_REFRESH_MS" in combined and "loadOverview(true)" in combined
              and "overviewGen" in combined and "setInterval" in combined, "markup")
        # ---- OpenAPI-generated API client (api.js) ------------------------
        # (uses its own response vars - must not clobber `r`, the served-page
        # response the following UI checks read)
        aj = vc.get("/static/api.js")
        check("UI: OpenAPI client generator served", aj.status_code == 200 and "createClient" in aj.text, "api.js")
        check("UI: calls go through the generated client, no hand-written wrappers",
              "api.js" in combined and "apiReady" in combined
              and "api.identity.login" in combined and "api.identity.register" in combined
              and "api.verify.overview" in combined and "api.verify.confirmAlert" in combined
              and "api.verify.demoSeed" in combined and "api.verify.chainStatus" in combined
              and "api.verify.complianceEvents" in combined
              and "function api(" not in combined and "api(VERIFY_URL" not in combined
              and "api(IDENTITY_URL" not in combined, "markup")
        # Schema operationIds are clean -> generated helper names are clean.
        spec_res = vc.get("/openapi.json")
        check("UI: operationIds are stable/readable in the schema",
              spec_res.status_code == 200
              and {o.get("operationId") for m in spec_res.json()["paths"].values() for o in m.values()}
                  >= {"overview", "confirm_alert", "compliance_events", "chain_status"},
              "operationIds")
        check("UI compliance view has chain-status + latest-case cards",
              all(x in combined for x in ("chainStatusCard", "chainStatusLine",
                                        "latestCaseCard", "latestCaseBody",
                                        "renderLatestCase", "updateChainStatus")))
        check("UI compliance view: single aggregate fetch (BFF)",
              "complianceOverview" in combined and "renderLatestCase(ov.latest_case)" in combined
              and "complianceIntegrity()" not in combined, "markup")
        check("UI score cards show a placeholder when no reasons",
              "no flags — nothing unusual" in combined and "rchip none" in combined)
        check("UI has post-verification case history",
              "historyBtn" in combined and "caseHistory" in combined and "renderCases" in combined)
        check("UI has account-history view (abnormalities page)",
              "historyView" in combined and "loadHistory" in combined and "api.verify.history" in combined
              and "Account history" in combined and "historyBackBtn" in combined, "markup")

        # ---- Compliance: pseudonymous decision trail -----------------------
        # Route the verification service's audit proxy to the in-process
        # Audit Service (normally http://127.0.0.1:8005).
        def fake_audit_get(path: str, actor: str = "compliance-ui"):
            resp = ac.get(path, headers={"X-Internal-Token": TOKEN, "X-Audit-Actor": actor})
            if resp.headers.get("content-type", "").startswith("application/json"):
                return resp.status_code, resp.json()
            return resp.status_code, {"detail": resp.text}

        verify_module._audit_get = fake_audit_get

        # ---- /overview: one round trip for the alerts screen (BFF) --------
        # (after the audit-proxy patch so its chain section reads the
        # in-process Audit Service like the chain-status check below)
        r = vc.get("/overview", headers=auth)
        check("overview: 200 with all four sections", r.status_code == 200
              and set(r.json()) == {"fraud_id", "alerts", "cases", "feedback", "chain"}, r.text[:200])
        ov = r.json()
        al = vc.get("/alerts", headers=auth).json()
        cs = vc.get("/cases", headers=auth).json()
        fs = vc.get("/feedback-status").json()
        check("overview: alerts match /alerts", ov["alerts"] == al["alerts"])
        check("overview: cases match /cases (reasons included)",
              ov["cases"] == cs["cases"] and all(c["reason_texts"] for c in ov["cases"]))
        check("overview: feedback matches /feedback-status", ov["feedback"] == fs)
        check("overview: chain integrity surfaced",
              ov["chain"].get("ok") is True and isinstance(ov["chain"].get("n_entries"), int), str(ov["chain"]))
        r = vc.get("/overview")  # no token
        check("overview: no token 401", r.status_code == 401)

        # ---- /history: account history + abnormality analysis -------------
        r = vc.get("/history", headers=auth)
        assert r.status_code == 200, r.text
        h = r.json()
        # The history includes clean/unresolved events too (e.g. the legit
        # score with no reasons and no outcome) - that IS the point of a
        # full account history.
        check("history: all scored events for the account (reason texts joined)",
              h["fraud_id"] == fraud_id and len(h["events"]) >= 3
              and all(e["risk_band"] in ("high", "medium", "low")
                      and isinstance(e["reason_texts"], list) for e in h["events"]),
              str(len(h["events"])))
        check("history: outcomes joined (resolved + unresolved events present)",
              {"confirmed", "disputed"} <= {e["outcome"] for e in h["events"]}
              and None in [e["outcome"] for e in h["events"]],
              str([e["outcome"] for e in h["events"]]))
        check("history: stats + abnormality flags",
              h["stats"]["total"] == len(h["events"]) and h["stats"]["disputed"] >= 1
              and isinstance(h["stats"]["flags"], list) and len(h["stats"]["flags"]) >= 1
              and isinstance(h["stats"]["repeated_reasons"], list), str(h["stats"]))
        check("history: repeated reasons carry human-readable text",
              all(r["code"] and r["count"] >= 2 and r["text"] for r in h["stats"]["repeated_reasons"]),
              str(h["stats"].get("repeated_reasons")))
        r = vc.get("/history")  # no token
        check("history: no token 401", r.status_code == 401)

        # ---- /overview server-side cache (TTL + write invalidation) -------
        # The chain section of /overview proxies to the in-process Audit
        # Service, which logs every read - so audit_access_log row growth is
        # an observable proxy for whether the cache served a read.
        def audit_access_rows() -> int:
            with sqlite3.connect(Path(TMP) / "audit.db") as c:
                return c.execute("SELECT COUNT(*) FROM audit_access_log").fetchone()[0]

        verify_module._OVERVIEW_CACHE.clear()
        verify_module.OVERVIEW_TTL_SECONDS = 3600  # long TTL: hits must not re-fetch
        rows0 = audit_access_rows()
        ov1 = vc.get("/overview", headers=auth).json()
        ov2 = vc.get("/overview", headers=auth).json()
        check("overview cache: repeated reads served from cache",
              audit_access_rows() == rows0 + 1 and ov1 == ov2,
              f"audit rows +{audit_access_rows() - rows0}")
        verify_module.OVERVIEW_TTL_SECONDS = 0
        verify_module._OVERVIEW_CACHE.clear()  # TTL changes don't retro-expire stored entries
        vc.get("/overview", headers=auth)  # miss; stored with expires=now
        vc.get("/overview", headers=auth)  # already expired -> miss again
        check("overview cache: TTL expiry re-fetches",
              audit_access_rows() == rows0 + 3, f"audit rows +{audit_access_rows() - rows0}")

        # A write must bypass the cache: seed a third attack, resolve it, and
        # the next /overview must show the new case + the grown chain.
        verify_module.OVERVIEW_TTL_SECONDS = 3600
        verify_module._OVERVIEW_CACHE.clear()
        attack3 = {**base, "event_id": "verify-attack-03", "amount": 5200.0, "hour_of_day": 4,
                   "device_id": "verify-attacker-3", "location_id": "L-Z", "recipient_id": "R-Z",
                   "failed_auth_count_24h": 7}
        r = pc.post("/internal/ingest-transaction", headers={"X-Internal-Token": TOKEN}, json=attack3)
        assert r.status_code == 200, r.text
        f3 = r.json()
        r = rc.post("/internal/evaluate", headers={"X-Internal-Token": TOKEN},
                    json={"event_id": "verify-attack-03", "fraud_id": fraud_id, "features": f3})
        assert r.status_code == 200 and r.json()["risk_band"] == "high", r.text
        cached_before = vc.get("/overview", headers=auth).json()
        n_cases_before = len(cached_before["cases"])
        r = vc.post("/alerts/verify-attack-03/confirm", headers=auth, json={"outcome": "this_was_me"})
        assert r.status_code == 200, r.text
        fresh = vc.get("/overview", headers=auth).json()
        check("overview cache: confirm invalidates (fresh cases + chain)",
              len(fresh["cases"]) == n_cases_before + 1
              and fresh["chain"]["n_entries"] > cached_before["chain"]["n_entries"],
              f"cases {n_cases_before}->{len(fresh['cases'])}, "
              f"chain {cached_before['chain']['n_entries']}->{fresh['chain']['n_entries']}")

        # ---- live chain-integrity badge (footer) ---------------------------
        r = vc.get("/chain-status")
        assert r.status_code == 200, r.text
        cs = r.json()
        check("chain-status: live badge data",
              cs.get("ok") is True and cs.get("n_entries", 0) >= 1, str(cs))
        check("chain-status: genesis hash", bool(cs.get("genesis_hash")), str(cs.get("genesis_hash", ""))[:12])

        r = vc.get("/compliance/events")
        check("compliance: no token 401", r.status_code == 401)
        r = vc.get("/compliance/events", headers={"X-Compliance-Token": "wrong"})
        check("compliance: wrong token 401", r.status_code == 401)

        c_hdr = {"X-Compliance-Token": COMPLIANCE}
        r = vc.get("/compliance/integrity", headers=c_hdr)
        assert r.status_code == 200, r.text
        integ = r.json()
        check("compliance: integrity ok", integ.get("ok") is True and integ.get("n_entries", 0) >= 1, str(integ))
        check("compliance: genesis hash", bool(integ.get("genesis_hash")), str(integ.get("genesis_hash", ""))[:12])

        r = vc.get("/compliance/events", headers=c_hdr)
        assert r.status_code == 200, r.text
        evs = r.json()["events"]
        types = {e["event_type"] for e in evs}
        check("compliance: trail has scores + outcomes",
              {"score_generated", "verification_resolved"} <= types, str(types))
        score_ev = next(e for e in evs if e["event_type"] == "score_generated")
        sp = score_ev["payload"]
        check("compliance: score payload", sp.get("risk_score") is not None and bool(sp.get("reason_codes")), str(sp))
        check("compliance: category-level reason texts", "reason_texts" in sp, str(sp.get("reason_codes")))
        res_ev = next(e for e in evs if e["event_type"] == "verification_resolved")
        rp = res_ev["payload"]
        check("compliance: outcome + case id",
              rp.get("outcome") in ("confirmed", "disputed") and rp.get("case_id", "").startswith("PS14-"), str(rp))
        check("compliance: every entry hash-chained",
              all(e.get("entry_hash") and e.get("prev_hash") for e in evs))

        # ---- /compliance/overview: BFF aggregate proxy ---------------------
        r = vc.get("/compliance/overview", headers=c_hdr)
        assert r.status_code == 200, r.text
        cov = r.json()
        check("compliance/overview: one payload with all sections",
              set(cov) == {"integrity", "latest_case", "events", "count", "total", "by_type", "no_flag_scores"}
              and cov["integrity"]["ok"] is True
              and cov["total"] == cov["count"] == len(cov["events"]), str(sorted(cov)))
        check("compliance/overview: latest case joined with its score",
              cov["latest_case"] and cov["latest_case"]["case_id"].startswith("PS14-")
              and cov["latest_case"]["risk_band"] in ("high", "medium", "low")
              and isinstance(cov["latest_case"]["risk_score"], int), str(cov["latest_case"]))
        r = vc.get("/compliance/overview")  # no token
        check("compliance/overview: no token 401", r.status_code == 401)

        con = sqlite3.connect(Path(TMP) / "audit.db")
        n_log = con.execute("SELECT COUNT(*) FROM audit_access_log").fetchone()[0]
        con.close()
        check("compliance: reads logged (audit of the audit)", n_log >= 2, f"n={n_log}")

        # ---- DB-3: verification_outcomes -----------------------------------
        con = sqlite3.connect(Path(TMP) / "risk.db")
        tables = {row[0] for row in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        n_outcomes = con.execute("SELECT COUNT(*) FROM verification_outcomes").fetchone()[0]
        outcomes = con.execute("SELECT outcome, COUNT(*) FROM verification_outcomes GROUP BY outcome").fetchall()
        con.close()
        check("DB-3 has verification_outcomes", "verification_outcomes" in tables)
        # The cache test above confirmed a third attack, so the pool is now 3.
        check("DB-3 outcomes persisted", n_outcomes == 3, f"n={n_outcomes}")
        check("DB-3 outcome mix", dict(outcomes) == {"confirmed": 2, "disputed": 1}, str(outcomes))

        # ---- Feedback loop: outcomes -> labeled training rows --------------
        sys.path.insert(0, str(ROOT / "scripts"))
        from export_feedback import export_real  # noqa: E402

        pool = export_real(get_settings())
        check("feedback: pool size = outcomes", len(pool) == 3, f"n={len(pool)}")
        lbl = dict(zip(pool["event_id"], pool["label"]))
        check("feedback: confirmed -> legit label", lbl.get("verify-attack-01") == 0, str(lbl))
        check("feedback: disputed -> fraud label", lbl.get("verify-attack-02") == 1, str(lbl))
        check("feedback: archetypes", set(pool["archetype"]) == {"feedback_legit", "feedback_fraud"}, str(set(pool["archetype"])))
        check("feedback: feature vector present", {"amount_ratio", "gradual_escalation_score", "known_device_count"} <= set(pool.columns), str(list(pool.columns)))

    print("\n" + ("ALL CHECKS PASSED" if not failures else f"{len(failures)} CHECK(S) FAILED"))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
