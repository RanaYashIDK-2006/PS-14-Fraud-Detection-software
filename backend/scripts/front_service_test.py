#!/usr/bin/env python3
"""Smoke test for the front page service (`src/front_service`).

Checks the landing page renders, the service health endpoint answers, and
`/status` degrades gracefully both ways:
  * ONLINE (when the dev stack is up): per-service ok + latency + uptime,
    and the audit chain state on the audit entry;
  * OFFLINE (forced, always): the aggregate is pointed at an unreachable
    port, so every service must report "down" with all five keys present
    and no audit-chain key — the browser badge renders "unavailable" and
    the page never 500s (it polls /status every 8s).

The offline half is hermetic (monkeypatched URLs), so this suite never
stops or starts the real stack. For the live stop-the-stack version see
`scripts/front_offline_check.py` (opt-in, restores everything on exit).

Run from the project root:
  python scripts/front_service_test.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent  # repo root
sys.path.insert(0, str(ROOT / "backend"))

ADMIN_PASS = "test-admin-pass-123"
DB_TMP = tempfile.mkdtemp(prefix="ps14-front-")
os.environ["DB_DIR"] = DB_TMP
os.environ["ADMIN_USER"] = "admin"
os.environ["ADMIN_PASS"] = ADMIN_PASS

from fastapi.testclient import TestClient  # noqa: E402

from src.front_service.main import app  # noqa: E402
from src.front_service import main as fm  # noqa: E402
from src.middleware.totp import TOTPAuthenticator  # noqa: E402

ADMIN_FILE = Path(DB_TMP) / "admin.json"

failures: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        failures.append(name)


def main() -> int:
    print("== Front page service smoke test ==")
    with TestClient(app) as c:
        r = c.get("/")
        check("landing page 200", r.status_code == 200, f"status={r.status_code}")
        check("landing page is the PS-14 front page",
              "<title>PS-14" in r.text and "Privacy-First" in r.text)
        # The landing page deliberately does NOT link /monitor-page: monitor
        # and fraud-report are admin-only (commit 18ea5af — reachable via the
        # secret triple-click + passphrase flow). Assert the intentional
        # absence instead of a stale link expectation.
        check("landing page does not expose the admin monitor link", "/monitor-page" not in r.text,
              "monitor link leaked to public page")

        h = c.get("/health")
        check("health ok", h.status_code == 200 and h.json().get("service") == "front-page",
              str(h.json()))

        # Offline: the aggregate must still answer with all five keys (as
        # "down"/error) instead of crashing — the browser polls it every 8s.
        s = c.get("/status")
        check("status aggregate 200", s.status_code == 200, f"status={s.status_code}")
        body = s.json()
        check("status covers all five services",
              set(body) == {"identity", "privacy", "risk", "verify", "audit"},
              str(sorted(body)))
        # Works both online (all ok) and offline (all down) — the browser
        # only needs a valid per-service status, never an error. Each entry is
        # a dict: {status, latency_ms, uptime_s, [chain] on audit}.
        valid = {"ok", "down"} | {f"http {c}" for c in range(100, 600)}
        check("status returns valid per-service states",
              all(isinstance(v, dict) and v.get("status") in valid for v in body.values()),
              str(body))
        # The online latency/chain assertions need the dev stack actually up
        # (they hit real /health endpoints). Without a stack every service
        # reports down/error and latency is legitimately absent - skip the
        # online half instead of failing, and note it.
        stack_up = any(v.get("status") == "ok" for v in body.values())
        if stack_up:
            check("status reports latency per service",
                  all(isinstance(v.get("latency_ms"), (int, float)) for v in body.values()),
                  str({k: v.get("latency_ms") for k, v in body.items()}))
            check("audit chain state in status",
                  isinstance(body["audit"].get("chain"), dict)
                  and body["audit"]["chain"].get("ok") is True
                  and isinstance(body["audit"]["chain"].get("n_entries"), int),
                  str(body["audit"].get("chain")))
        else:
            print("  [BLOCKED] online latency/audit-chain checks - no live stack "
                  "(re-run with the dev stack up)")

        # ---- live batch results (/batch): placeholder until a report exists,
        # full summary once scripts/batch_cases.py has written one -----------
        print("\n-- batch results endpoint --")
        from src.front_service.main import BATCH_REPORT
        r = c.get("/batch")
        body = r.json()
        check("batch endpoint 200 without a report", r.status_code == 200, f"status={r.status_code}")
        check("batch placeholder when no report", body.get("ran_at") is None, str(body))

        BATCH_REPORT.parent.mkdir(parents=True, exist_ok=True)
        BATCH_REPORT.write_text(json.dumps({
            "ran_at": "2026-01-02T03:04:05+00:00",
            "seed": 7, "total_cases": 50, "named": 5, "perturbed": 45,
            "distribution": {"allow": 3, "step_up": 10, "verify": 37},
            "chain": {"before": 100, "after": 193, "grew": 93},
            "integrity_ok": True,
            "scenarios": [{"name": "s1", "score": 29, "band": "low",
                           "decision": "allow", "reasons": ["NEW_DEVICE"]}],
        }, indent=2), encoding="utf-8")
        r = c.get("/batch")
        body = r.json()
        check("batch report served when present", r.status_code == 200 and body.get("ran_at"),
              str(body.get("ran_at")))
        check("batch report carries distribution + chain",
              body.get("distribution") == {"allow": 3, "step_up": 10, "verify": 37}
              and body.get("chain") == {"before": 100, "after": 193, "grew": 93},
              str({k: body.get(k) for k in ("distribution", "chain")}))
        BATCH_REPORT.unlink(missing_ok=True)
        r = c.get("/batch")
        check("batch placeholder returns after report removed",
              r.json().get("ran_at") is None, str(r.json()))

        # ---- offline degradation (forced): the browser polls /status every
        # 8s, so a dead backend stack must never 500 or drop keys. Point the
        # aggregate at a refused port and re-check the shape.
        print("\n-- offline /status degradation (forced) --")
        import src.front_service.main as fm
        orig_services = dict(fm.SERVICES)
        fm.SERVICES = {k: "http://127.0.0.1:9" for k in orig_services}  # port 9: refused
        # Clear the status cache so the offline test isn't served stale data
        fm._STATUS_CACHE.clear()
        fm._CHAIN_CACHE.clear()
        try:
            s = c.get("/status")
            check("offline /status still 200", s.status_code == 200, f"status={s.status_code}")
            off = s.json()
            check("offline /status keeps all five keys",
                  set(off) == {"identity", "privacy", "risk", "verify", "audit"},
                  str(sorted(off)))
            check("offline /status reports every service down",
                  all(v.get("status") == "down" for v in off.values()),
                  str({k: v.get("status") for k, v in off.items()}))
            check("offline audit entry carries no chain (badge -> unavailable)",
                  off.get("audit", {}).get("chain", {}).get("ok") is not True,
                  str(off.get("audit")))
            # The page only needs status strings, never a crash; latency is
            # absent (no response to time) but the entry must stay a dict.
            check("offline entries are plain down dicts",
                  all(isinstance(v, dict) for v in off.values()),
                  str(off))
        finally:
            fm.SERVICES = orig_services

        # ---- admin area: passphrase-gated, minimal + encrypted at rest -----
        print("\n-- admin login / essentials --")
        r = c.get("/admin/essential")
        check("essentials without a session -> 401", r.status_code == 401, f"status={r.status_code}")

        r = c.post("/admin/login", json={"username": "admin", "passphrase": "wrong-pass"})
        check("wrong passphrase -> 401", r.status_code == 401, f"status={r.status_code}")

        r = c.post("/admin/login", json={"username": "admin", "passphrase": ADMIN_PASS})
        check("correct passphrase -> 200 + token", r.status_code == 200 and "token" in r.json(),
              f"status={r.status_code}")
        login = r.json()
        ess = login.get("essentials", {})
        check("essentials carry the service map + credentials",
              "services" in ess and "credentials" in ess and "break_glass" in ess,
              str(sorted(ess.keys())))
        check("credentials hold the three gates",
              set(ess["credentials"]) == {"jwt_secret", "internal_token", "compliance_token"},
              str(sorted(ess["credentials"])))

        auth = {"Authorization": "Bearer " + login["token"]}
        r = c.get("/admin/essential", headers=auth)
        check("essentials with session -> 200", r.status_code == 200 and "essentials" in r.json(),
              f"status={r.status_code}")

        r = c.post("/admin/rotate", headers=auth,
                   json={"current_passphrase": "wrong", "new_passphrase": "new-pass-456"})
        check("rotate with wrong current -> 401", r.status_code == 401, f"status={r.status_code}")

        new_pass = "new-admin-pass-456"
        r = c.post("/admin/rotate", headers=auth,
                   json={"current_passphrase": ADMIN_PASS, "new_passphrase": new_pass})
        check("rotate ok", r.status_code == 200, f"status={r.status_code} {r.text[:120]}")
        r = c.post("/admin/login", json={"username": "admin", "passphrase": ADMIN_PASS})
        check("old passphrase rejected after rotate", r.status_code == 401, f"status={r.status_code}")
        r = c.post("/admin/login", json={"username": "admin", "passphrase": new_pass})
        check("new passphrase works", r.status_code == 200, f"status={r.status_code}")

        # ---- minimal + encrypted at rest --------------------------------
        record = json.loads(ADMIN_FILE.read_text(encoding="utf-8"))
        check("admin record has only essential fields",
              set(record) == {"username", "pass_salt", "pass_hash", "wrap",
                              "essential", "created_at", "last_login", "login_count"},
              str(sorted(record)))
        raw = ADMIN_FILE.read_text(encoding="utf-8")
        check("passphrase not stored in plaintext", ADMIN_PASS not in raw and new_pass not in raw)
        check("essential payload is ciphertext, not JSON",
              "\"jwt_secret\"" not in raw and "\"services\"" not in raw)
        check("login count tracked", record.get("login_count", 0) >= 1, str(record.get("login_count")))

        # ---- TOTP second factor (opt-in from Access Control) -------------
        # Placed after the 8-field record check above: enabling TOTP adds the
        # ninth field ("totp"), asserted explicitly at the end.
        print("\n-- admin TOTP second factor --")
        fm._admin_login_failures.clear()  # isolate from earlier negative logins

        def _bad_code() -> str:
            """A code guaranteed to differ from the current window's code."""
            v = good.generate_code()
            return ("0" if v[0] != "0" else "1") + v[1:]

        # RFC 6238 test vectors (secret from the RFC appendices, SHA1, T=59
        # -> step 1: 94287082 at 8 digits; the 6-digit form is its mod 1e6,
        # identical to RFC 4226 HOTP counter=1).
        rfc_secret = b"12345678901234567890"
        check("RFC-6238 vector T=59 (8 digits)",
              TOTPAuthenticator(rfc_secret, digits=8).generate_code(59) == "94287082",
              TOTPAuthenticator(rfc_secret, digits=8).generate_code(59))
        check("6-digit default derives from the same truncation",
              TOTPAuthenticator(rfc_secret).generate_code(59) == "287082",
              TOTPAuthenticator(rfc_secret).generate_code(59))

        cur_pass = new_pass  # the passphrase was rotated above

        # The rotation above revoked every session — re-login for a fresh one
        # (TOTP is still off here, so no code is needed yet).
        r = c.post("/admin/login", json={"username": "admin", "passphrase": cur_pass})
        check("fresh session for the TOTP section",
              r.status_code == 200 and "token" in r.json(), f"status={r.status_code}")
        auth = {"Authorization": "Bearer " + r.json().get("token", "")}

        r = c.get("/admin/totp/status", headers=auth)
        check("totp status initially disabled",
              r.status_code == 200 and r.json().get("enabled") is False, r.text[:80])

        r = c.get("/admin/totp/setup", headers=auth)
        d = r.json()
        secret_b32 = d.get("secret", "")
        check("setup returns a pending base32 secret",
              r.status_code == 200 and len(secret_b32) == 32 and d.get("enabled") is False,
              f"status={r.status_code} len={len(secret_b32)}")

        raw_file = ADMIN_FILE.read_text(encoding="utf-8")
        record = json.loads(raw_file)
        check("totp secret encrypted at rest (base32 absent from file)",
              record.get("totp", {}).get("encrypted") is True and secret_b32 not in raw_file)

        good = TOTPAuthenticator.from_base32(secret_b32)

        r = c.post("/admin/totp/verify", headers=auth, json={"totp_code": _bad_code()})
        check("verify with wrong code -> 401 and still disabled",
              r.status_code == 401 and not fm.store.is_totp_enabled(), f"status={r.status_code}")

        r = c.post("/admin/login", json={"username": "admin", "passphrase": cur_pass})
        check("pending (unverified) secret does not gate login",
              r.status_code == 200, f"status={r.status_code}")

        r = c.post("/admin/totp/verify", headers=auth, json={"totp_code": good.generate_code()})
        check("verify with correct code -> enabled",
              r.status_code == 200 and fm.store.is_totp_enabled(),
              f"status={r.status_code} {r.text[:80]}")

        r = c.post("/admin/login", json={"username": "admin", "passphrase": cur_pass})
        check("login without code after enable -> 401 required",
              r.status_code == 401 and "TOTP code required" in r.text, r.text[:80])

        fm._admin_login_failures.clear()
        r = c.post("/admin/login", json={"username": "admin", "passphrase": cur_pass,
                                         "totp_code": _bad_code()})
        check("login with wrong code -> 401 invalid",
              r.status_code == 401 and "Invalid TOTP code" in r.text, r.text[:80])
        check("invalid code counts toward the brute-force guard",
              sum(len(v) for v in fm._admin_login_failures.values()) >= 1,
              str(fm._admin_login_failures))
        fm._admin_login_failures.clear()

        r = c.post("/admin/login", json={"username": "admin", "passphrase": cur_pass,
                                         "totp_code": good.generate_code()})
        check("login with valid code -> 200 + token",
              r.status_code == 200 and "token" in r.json(), f"status={r.status_code}")

        # One-time use: a code for the next window is accepted once, then a
        # replay of the same time-step is rejected (no clock waits needed).
        next_step_code = good.generate_code(time.time() + 30)
        r = c.post("/admin/login", json={"username": "admin", "passphrase": cur_pass,
                                         "totp_code": next_step_code})
        check("login with next-window code -> 200", r.status_code == 200, f"status={r.status_code}")
        fm._admin_login_failures.clear()
        r = c.post("/admin/login", json={"username": "admin", "passphrase": cur_pass,
                                         "totp_code": next_step_code})
        check("replayed code -> rejected as already used",
              r.status_code == 401 and "already used" in r.text, r.text[:90])
        fm._admin_login_failures.clear()

        # Bypass attempts must fail closed (extra fields are ignored by the
        # model — none of them substitutes for the missing code).
        r = c.post("/admin/login", json={"username": "admin", "passphrase": cur_pass,
                                         "force": True, "override": True,
                                         "admin_override": True, "skip_validation": True,
                                         "allow_unverified": True, "bypass": True})
        check("bypass fields without a code -> 401 (no override honored)",
              r.status_code == 401 and "required" in r.text,
              f"status={r.status_code} {r.text[:80]}")

        # Disable round-trip: authenticated + code-gated, secret retained
        r = c.post("/admin/totp/disable", headers=auth, json={"totp_code": good.generate_code()})
        check("disable with code -> 200", r.status_code == 200,
              f"status={r.status_code} {r.text[:80]}")
        check("disable keeps the secret for re-enable",
              fm.store.get_totp_secret() is not None and not fm.store.is_totp_enabled())

        r = c.post("/admin/login", json={"username": "admin", "passphrase": cur_pass})
        check("login without code works again after disable",
              r.status_code == 200, f"status={r.status_code}")
        fm._admin_login_failures.clear()

        r = c.get("/admin/totp/setup", headers=auth)
        check("re-setup after disable returns the same stored secret",
              r.status_code == 200 and r.json().get("secret") == secret_b32,
              f"status={r.status_code}")

        record = json.loads(ADMIN_FILE.read_text(encoding="utf-8"))
        check("admin record gained only the totp field",
              set(record) == {"username", "pass_salt", "pass_hash", "wrap", "essential",
                              "created_at", "last_login", "login_count", "totp"},
              str(sorted(record)))

    print("\n" + ("ALL CHECKS PASSED" if not failures else f"{len(failures)} CHECK(S) FAILED"))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
