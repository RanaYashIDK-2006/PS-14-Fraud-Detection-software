#!/usr/bin/env python3
"""Phase 20 — demo usability test (hermetic, no live services).

Validates the Phase 20 landing-page demo surface on the front service:

  GET  /demo/readiness               honest liveness/readiness, no false "ready"
  POST /demo/synthetic-transaction   returns the REAL backend response verbatim

Checks:
  1. the demo section and its script actually ship on the landing page, and the
     old hardcoded "all systems operational" claim is gone;
  2. readiness never reports a service (or the demo path) as ready when its
     /health did not answer, and reports `null` readiness when a service does
     not publish readiness rather than inventing "ready";
  3. invalid demo input is rejected by validation (422) instead of reaching the
     internal endpoints;
  4. the success path echoes the Privacy/Risk responses exactly (no invented
     score, band, decision or reason code) with a server-minted unique event id;
  5. an unreachable service yields an understandable 503, an upstream 5xx
     yields a generic 502 with no upstream body leakage, and nothing ever
     returns a stack trace or the internal token;
  6. the per-client demo rate limit trips.

No live stack is required: downstream HTTP is stubbed with httpx.MockTransport
and the readiness cache is seeded directly. Run from the repo root:

  PYTHONPATH=backend python backend/scripts/phase20_demo_usability_test.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

# Hermetic stores BEFORE src.settings is imported (front service writes an
# admin/session store on import).
_TMP = tempfile.mkdtemp(prefix="ps14_phase20_")
os.environ["DB_DIR"] = _TMP
os.environ["PS14_MODE"] = "development"
os.environ.setdefault("INTERNAL_TOKEN", "phase20-demo-test-token")
os.environ.setdefault("JWT_SECRET", "phase20-demo-test-jwt-secret-value-32b")

import httpx  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import src.front_service.main as front  # noqa: E402
from src.settings import settings  # noqa: E402

passed = failed = 0
errors: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    global passed, failed
    if cond:
        passed += 1
        print(f"  [PASS] {name}")
    else:
        failed += 1
        errors.append(f"{name}{' — ' + detail if detail else ''}")
        print(f"  [FAIL] {name}{' — ' + detail if detail else ''}")


def seed_status(entries: dict[str, dict]) -> None:
    front._STATUS_CACHE["data"] = entries
    front._STATUS_CACHE["ts"] = datetime.now(timezone.utc)


def alive_entry(health: dict | None = None) -> dict:
    return {"status": "ok", "latency_ms": 1.0, "uptime_s": 10.0,
            "health": health if health is not None else {}}


def down_entry() -> dict:
    return {"status": "down"}


class StubDownstream:
    """Swap the httpx client used by the demo endpoint for a stub transport."""

    def __init__(self, handler):
        self._real = None
        self._handler = handler

    def __enter__(self):
        self._real = front.httpx.AsyncClient

        def factory(*args, **kwargs):
            return self._real(transport=httpx.MockTransport(self._handler), timeout=5.0)

        front.httpx.AsyncClient = factory
        return self

    def __exit__(self, *exc):
        front.httpx.AsyncClient = self._real
        return False


STUB_FEATURES = {
    "event_id": "ignored-by-stub", "fraud_id": "FSTUBSTUBSTUBSTU", "amount_ratio": 1.5,
    "txn_amount_bucket": "typical", "txn_freq_last_24h": 2, "txn_time_unusual": 0,
    "new_device_flag": 0, "unusual_location_flag": 0, "unusual_recipient_flag": 0,
    "failed_auth_count_24h": 0, "days_since_last_simular_txn": 1.0,
    "gradual_escalation_score": 0.0, "known_device_count": 3, "account_tenure_days": 60.0,
    "hour_of_day": 14, "is_weekend": 0, "label": None,
}
STUB_DECISION = {
    "event_id": "ignored-by-stub", "fraud_id": "FSTUBSTUBSTUBSTU",
    "risk_score": 12, "risk_band": "low", "decision": "allow", "reason_codes": [],
    "ml_score": 0.01, "rule_score": 0.0, "fired_rules": [], "degraded": False,
    "calibrated": True, "odds": 0.0101,
    "limits": {"pass": True, "triggers": [], "hard": False},
    "feature_version": "v1", "rule_version": "stub",
}


def ok_handler(request: httpx.Request) -> httpx.Response:
    body = json.loads(request.content.decode("utf-8"))
    if request.url.path.endswith("/internal/ingest-transaction"):
        out = dict(STUB_FEATURES)
        out["event_id"] = body.get("event_id")
        out["fraud_id"] = body.get("fraud_id")
        return httpx.Response(200, json=out)
    if request.url.path.endswith("/internal/evaluate"):
        out = dict(STUB_DECISION)
        out["event_id"] = body.get("event_id")
        out["fraud_id"] = body.get("fraud_id")
        return httpx.Response(200, json=out)
    return httpx.Response(404, json={"detail": "unknown stub path"})


def main() -> int:
    global passed, failed
    client = TestClient(front.app)  # no context manager -> lifespan (and the
    # background regression-suite runner) never starts; this suite stays hermetic.

    print("== Phase 20 demo usability test ==")
    print(f"internal token present: {bool(settings.internal_token)}")
    print()

    # ── 1. the demo surface actually ships on the landing page ──────────
    print("-- 1) landing page ships the demo surface --")
    page = client.get("/")
    check("landing page 200", page.status_code == 200, f"status={page.status_code}")
    html = page.text
    check("demo section present", 'id="demo"' in html)
    check("demo script referenced", "/static/demo-app.js" in html)
    check("hardcoded 'All systems operational' removed",
          "All systems operational" not in html)
    check("demo labeled synthetic", "synthetic" in html.lower())
    js = client.get("/static/demo-app.js")
    check("demo-app.js served", js.status_code == 200, f"status={js.status_code}")
    check("demo-app.js calls the demo API",
          "/demo/synthetic-transaction" in js.text and "/demo/readiness" in js.text)
    print()

    # ── 2. readiness honesty ────────────────────────────────────────────
    print("-- 2) readiness is honest --")
    seed_status({
        "identity": alive_entry({"status": "ok"}),
        "privacy": alive_entry({"status": "ok", "service": "privacy-layer"}),
        "risk": alive_entry({"status": "ok", "liveness": "alive", "readiness": "ready",
                             "model_readiness": "loaded", "runtime_state": "READY"}),
        "verify": alive_entry({"status": "ok"}),
        "audit": alive_entry({"status": "ok"}),
    })
    r = client.get("/demo/readiness")
    check("readiness 200", r.status_code == 200, f"status={r.status_code}")
    body = r.json()
    check("demo path ready when all alive", body.get("demo_path_ready") is True)
    check("risk model_readiness surfaced", body["services"]["risk"]["model_readiness"] == "loaded")
    check("privacy readiness null when not published",
          body["services"]["privacy"]["readiness"] is None)
    check("readiness never claims unpublished readiness is ready",
          all(v["readiness"] in (None, "ready") for v in body["services"].values()))

    seed_status({name: down_entry() for name in front.SERVICES})
    r = client.get("/demo/readiness")
    body = r.json()
    check("demo path NOT ready when services down", body.get("demo_path_ready") is False)
    check("services reported unreachable", all(
        v["liveness"] == "unreachable" for v in body["services"].values()))
    check("down services carry a detail, not a fake status",
          all(v["detail"] for v in body["services"].values()))

    seed_status({
        "identity": alive_entry(), "privacy": alive_entry(),
        "risk": alive_entry({"status": "ok", "liveness": "alive", "readiness": "ready",
                             "model_readiness": "loading"}),
        "verify": alive_entry(), "audit": alive_entry(),
    })
    body = client.get("/demo/readiness").json()
    check("model still loading -> demo path not ready", body["demo_path_ready"] is False)
    print()

    # ── 3. invalid input never reaches the internal endpoints ───────────
    print("-- 3) invalid demo input rejected --")
    seen_paths: list[str] = []

    def spy_handler(request: httpx.Request) -> httpx.Response:
        seen_paths.append(request.url.path)
        return httpx.Response(500, text="SHOULD-NOT-BE-CALLED")

    with StubDownstream(spy_handler):
        bad_amount = client.post("/demo/synthetic-transaction", json={"amount": -5})
        check("negative amount 422", bad_amount.status_code == 422,
              f"status={bad_amount.status_code}")
        bad_id = client.post("/demo/synthetic-transaction",
                             json={"amount": 10, "fraud_id": "not-a-valid-id"})
        check("invalid fraud_id 422", bad_id.status_code == 422,
              f"status={bad_id.status_code}")
        bad_hour = client.post("/demo/synthetic-transaction", json={"amount": 10, "hour_of_day": 99})
        check("hour out of range 422", bad_hour.status_code == 422,
              f"status={bad_hour.status_code}")
        bad_device = client.post("/demo/synthetic-transaction",
                                 json={"amount": 10, "device_id": "x" * 200})
        check("oversized device token 422", bad_device.status_code == 422,
              f"status={bad_device.status_code}")
    check("invalid input never hit a downstream endpoint", seen_paths == [],
          f"paths={seen_paths}")
    print()

    # ── 4. success path echoes the backend response exactly ─────────────
    print("-- 4) success path returns the real backend response --")
    with StubDownstream(ok_handler):
        r1 = client.post("/demo/synthetic-transaction", json={"amount": 85.0, "hour_of_day": 14})
        check("demo run 200", r1.status_code == 200, f"status={r1.status_code} body={r1.text[:200]}")
        body1 = r1.json()
        check("response labelled synthetic", body1.get("synthetic") is True)
        check("server minted a unique event id",
              str(body1.get("event_id", "")).startswith("DEMO-"), str(body1.get("event_id")))
        expected_decision = dict(STUB_DECISION,
                                 event_id=body1.get("event_id"), fraud_id=body1.get("fraud_id"))
        check("decision echoed verbatim", body1.get("decision") == expected_decision,
              json.dumps(body1.get("decision"))[:160])
        check("ingest echoed verbatim",
              body1.get("ingest", {}).get("event_id") == body1.get("event_id")
              and body1.get("ingest", {}).get("fraud_id") == body1.get("fraud_id"))
        check("no invented fields on the envelope",
              set(body1.keys()) == {"synthetic", "event_id", "fraud_id", "ingest", "decision", "note"})

        r2 = client.post("/demo/synthetic-transaction", json={"amount": 85.0})
        body2 = r2.json()
        check("second run gets a different event id",
              body2.get("event_id") != body1.get("event_id"),
              f"{body1.get('event_id')} vs {body2.get('event_id')}")

        supplied = client.post("/demo/synthetic-transaction",
                               json={"amount": 42.0,
                                     "fraud_id": "FABCDEF23456789X"})
        check("caller-supplied fraud_id honoured",
              supplied.json().get("fraud_id") == "FABCDEF23456789X",
              str(supplied.json().get("fraud_id")))
    print()

    # ── 5. failures are honest and leak nothing ─────────────────────────
    print("-- 5) failure handling --")
    def boom_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/internal/ingest-transaction"):
            return httpx.Response(200, json=dict(STUB_FEATURES))
        return httpx.Response(500, text="SECRET-LEAK-MARKER traceback.py line 1")

    with StubDownstream(boom_handler):
        r = client.post("/demo/synthetic-transaction", json={"amount": 10.0})
        check("upstream 5xx -> 502", r.status_code == 502, f"status={r.status_code}")
        check("502 detail is generic", "SECRET-LEAK-MARKER" not in r.text
              and "traceback" not in r.text.lower())
        check("502 says no decision was produced", "No risk decision" in r.text)

    def reject_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(422, json={"detail": ["upstream validation text"]})

    with StubDownstream(reject_handler):
        r = client.post("/demo/synthetic-transaction", json={"amount": 10.0})
        check("upstream 4xx surfaces the validation status", r.status_code == 422,
              f"status={r.status_code}")

    # Unreachable services: point the demo at a closed port.
    saved_services = dict(front.SERVICES)
    try:
        front.SERVICES["privacy"] = "http://127.0.0.1:59991"
        r = client.post("/demo/synthetic-transaction", json={"amount": 10.0})
        check("unreachable privacy -> 503", r.status_code == 503, f"status={r.status_code}")
        check("503 names the service and the fix", "Privacy Layer" in r.text
              and "readiness" in r.text)
        check("503 has no traceback", "Traceback" not in r.text)
    finally:
        front.SERVICES.clear()
        front.SERVICES.update(saved_services)

    # Risk unreachable after a successful ingest: the stub transport raises a
    # connection error exactly as a dead service would.
    def privacy_ok_risk_down(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/internal/ingest-transaction"):
            return httpx.Response(200, json=dict(STUB_FEATURES))
        raise httpx.ConnectError("connection refused", request=request)

    with StubDownstream(privacy_ok_risk_down):
        r = client.post("/demo/synthetic-transaction", json={"amount": 10.0})
    check("unreachable risk -> 503", r.status_code == 503, f"status={r.status_code} body={r.text[:120]}")
    check("risk 503 says the txn was not scored", "NOT scored" in r.text)

    for resp in (page, js, r1, r2, supplied):
        check("internal token never leaked", settings.internal_token not in resp.text)
        check("no stack trace in response", "Traceback (most recent call last)" not in resp.text)
    print()

    # ── 6. rate limit ───────────────────────────────────────────────────
    print("-- 6) demo rate limit --")
    front._demo_hits.clear()
    with StubDownstream(ok_handler):
        statuses = [client.post("/demo/synthetic-transaction", json={"amount": 1.0}).status_code
                    for _ in range(front.DEMO_RATE_MAX + 2)]
    check("requests under the limit succeed",
          all(s == 200 for s in statuses[:front.DEMO_RATE_MAX]),
          str(statuses[:front.DEMO_RATE_MAX]))
    check("requests over the limit are 429",
          all(s == 429 for s in statuses[front.DEMO_RATE_MAX:]),
          str(statuses[front.DEMO_RATE_MAX:]))
    check("429 detail is understandable",
          "rate limit" in client.post("/demo/synthetic-transaction",
                                      json={"amount": 1.0}).text.lower())
    front._demo_hits.clear()
    print()

    print("== result ==")
    print(f"checks passed={passed} failed={failed}")
    print("NOTE: synthetic/demo path test — software behavior and response "
          "contract only, not real-world fraud-detection performance.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
