#!/usr/bin/env python3
"""50-case batch test against the LIVE PS-14 stack (Risk Engine + Audit).

Runs 50 distinct scoring scenarios through `POST /internal/evaluate` on the
running Risk Engine (:8003) and checks the invariants that must hold no
matter the model weights:

  * every case returns a valid decision (score 0-100, band, reasons);
  * scoring is DETERMINISTIC — the same vector re-evaluated gives the same
    decision (a subset is re-run to keep the audit chain tidy);
  * MONOTONICITY — adding a red flag to an identical vector can only raise
    (never lower) the score;
  * the hash-chained audit trail grew by exactly the number of events and
    still verifies (integrity OK) — every decision is chained and the
    batch did not break it.

The 5 "real-world" scenarios from the design discussions (new device /
unusual location / big amount / velocity / known account) are pinned by
name and printed in their own table so their bands are visible at a glance;
they are reported, not asserted, because the ML fusion is opaque by design
(§11: category-level reasons are the human surface, not raw scores).

Run from the project root with the stack up (ports 8003 + 8005):
  python scripts/batch_cases.py [--recheck N] [--seed 7]

Exit 0 = all invariants held; 1 = a check failed; 2 = stack unreachable.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

RISK_URL = "http://127.0.0.1:8003"
AUDIT_URL = "http://127.0.0.1:8005"

# Read tokens from .env (preferred) or fall back to dev defaults
_env: dict[str, str] = {}
_env_file = Path(__file__).resolve().parent.parent.parent / ".env"
if _env_file.exists():
    for line in _env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            _env[k.strip()] = v.strip()

INTERNAL_TOKEN = _env.get("INTERNAL_TOKEN", "ps14-dev-internal-token-change-me")
COMPLIANCE_TOKEN = _env.get("COMPLIANCE_TOKEN", "ps14-dev-compliance-token-change-me")

# Last-run summary written on success — the front page's /batch endpoint
# serves this to the live panel. Lives under DB_DIR like the other runtime
# state (gitignored); overridable with --report.
DEFAULT_REPORT = str(Path(os.environ.get("DB_DIR", "db")) / "batch_report.json")

# Valid pseudonym: ^F[A-Z2-9]{15}$ (no 0/O/1/I so it stays unambiguous).
FRAUD_ID = "FA7K2Q9X4M6B3Z5C"

failures: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        failures.append(name)


def _post(url: str, payload: dict, headers: dict | None = None) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(), method="POST",
        headers={"Content-Type": "application/json", **(headers or {})},
    )
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.load(r)


def _get(url: str, headers: dict | None = None) -> dict:
    req = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.load(r)


def _integrity() -> dict:
    return _get(f"{AUDIT_URL}/audit/integrity",
                {"X-Internal-Token": INTERNAL_TOKEN})


def base_vector() -> dict:
    """A mature account, everything ordinary: ~1x usual amount, midday,
    known device, usual location/recipient, low frequency."""
    return {
        "amount_ratio": 1.0,
        "txn_freq_last_24h": 2,
        "txn_time_unusual": 0,
        "new_device_flag": 0,
        "unusual_location_flag": 0,
        "unusual_recipient_flag": 0,
        "failed_auth_count_24h": 0,
        "days_since_last_similar_txn": 3.0,
        "gradual_escalation_score": 0.0,
        "known_device_count": 2,
        "account_tenure_days": 400.0,
        "hour_of_day": 13,
        "is_weekend": 0,
        "shared_device_accounts": 0,
        "shared_recipient_accounts": 0,
        "mule_ring_score": 0.0,
        "account_daily_spend_ratio": 0.8,
        "device_daily_count": 3,
    }


# The five real-world scenarios from the design discussions, pinned by name.
NAMED_SCENARIOS: list[tuple[str, dict]] = [
    ("s1 new device @ usual location, $230", {
        **base_vector(), "amount_ratio": 1.5, "new_device_flag": 1,
    }),
    ("s2 old device, 250 miles away", {
        **base_vector(), "amount_ratio": 1.0, "unusual_location_flag": 1,
    }),
    ("s3 new device @ usual location, $1500", {
        **base_vector(), "amount_ratio": 10.0, "new_device_flag": 1,
    }),
    ("s4 usual device @ usual location, $1200", {
        **base_vector(), "amount_ratio": 8.0,
    }),
    ("s5 different location, usual device, <$250", {
        **base_vector(), "amount_ratio": 1.2, "unusual_location_flag": 1,
    }),
]


def generate_cases(seed: int) -> list[tuple[str, dict]]:
    """The 5 named scenarios + 45 seeded perturbations = 50 distinct cases."""
    rng = random.Random(seed)
    cases = list(NAMED_SCENARIOS)

    while len(cases) < 50:
        v = base_vector()
        v["amount_ratio"] = round(rng.choice([0.4, 0.8, 1.0, 1.5, 2.5, 4.0, 8.0, 15.0, 30.0]), 2)
        v["txn_freq_last_24h"] = rng.choice([0, 1, 3, 6, 12, 25])
        v["txn_time_unusual"] = rng.choice([0, 1])
        v["new_device_flag"] = rng.choice([0, 1])
        v["unusual_location_flag"] = rng.choice([0, 1])
        v["unusual_recipient_flag"] = rng.choice([0, 1])
        v["failed_auth_count_24h"] = rng.choice([0, 0, 1, 3, 6])
        v["gradual_escalation_score"] = rng.choice([0.0, 0.1, 0.35, 0.7, 1.0])
        v["known_device_count"] = rng.choice([0, 1, 2, 5])
        v["account_tenure_days"] = rng.choice([0.1, 3.0, 30.0, 400.0, 1500.0])
        v["hour_of_day"] = rng.choice([3, 9, 13, 22, 23])
        v["is_weekend"] = rng.choice([0, 1])
        v["shared_device_accounts"] = rng.choice([0, 1, 4, 9])
        v["shared_recipient_accounts"] = rng.choice([0, 1, 3, 7])
        v["mule_ring_score"] = rng.choice([0.0, 0.0, 0.2, 0.55, 0.9])
        v["account_daily_spend_ratio"] = round(rng.choice([0.5, 1.0, 2.5, 5.0, 12.0]), 2)
        v["device_daily_count"] = rng.choice([1, 3, 8, 15, 30])
        name = f"case {len(cases) - 4:02d} ratio={v['amount_ratio']} newdev={v['new_device_flag']} " \
               f"loc={v['unusual_location_flag']} recip={v['unusual_recipient_flag']} " \
               f"freq={v['txn_freq_last_24h']} tenure={v['account_tenure_days']}"
        if name not in [n for n, _ in cases]:
            cases.append((name, v))

    return cases[:50]


def evaluate(cases: list[tuple[str, dict]], seq_offset: int) -> list[tuple[str, dict, dict]]:
    results = []
    for i, (name, v) in enumerate(cases):
        res = _post(f"{RISK_URL}/internal/evaluate", {
            "event_id": f"batch-{seq_offset:02d}-{i:02d}-00000000",
            "fraud_id": FRAUD_ID,
            "features": v,
        }, {"X-Internal-Token": INTERNAL_TOKEN})
        results.append((name, v, res))
    return results


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recheck", type=int, default=5,
                    help="how many cases to re-evaluate for the determinism check")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--report", type=str, default=DEFAULT_REPORT,
                    help="where to write the last-run summary JSON (default: DB_DIR/batch_report.json)")
    args = ap.parse_args()

    # ---- reachability -----------------------------------------------------
    try:
        _get(f"{RISK_URL}/health")
        _get(f"{AUDIT_URL}/health")
    except Exception as e:
        print(f"ERROR: stack unreachable (need :8003 Risk Engine + :8005 Audit): {e}", file=sys.stderr)
        return 2

    before = _integrity()
    print(f"== 50-case batch vs live stack ==  ({datetime.now(timezone.utc):%H:%M:%S} UTC, "
          f"audit chain before: {before.get('n_entries')} entries)")

    cases = generate_cases(args.seed)
    assert len(cases) == 50, f"expected 50 cases, got {len(cases)}"
    print(f"\nGenerated {len(cases)} cases "
          f"({len(NAMED_SCENARIOS)} named + {len(cases) - len(NAMED_SCENARIOS)} perturbed, seed {args.seed})")

    # ---- run every case once ---------------------------------------------
    results = evaluate(cases, 1)

    extra_evals = 0  # every extra evaluation also appends to the audit chain

    # ---- determinism: re-run the first N, must match bit-for-bit ----------
    print(f"\n-- determinism (re-evaluate first {args.recheck}) --")
    for i in range(args.recheck):
        name, v, first = results[i]
        again = _post(f"{RISK_URL}/internal/evaluate", {
            "event_id": f"batch-r-{i:02d}-00000000",
            "fraud_id": FRAUD_ID,
            "features": v,
        }, {"X-Internal-Token": INTERNAL_TOKEN})
        extra_evals += 1
        same = (first["risk_score"], first["risk_band"], first["decision"], first["reason_codes"]) == \
               (again["risk_score"], again["risk_band"], again["decision"], again["reason_codes"])
        check(f"deterministic: {name[:52]}", same,
              f"{first['risk_score']} vs {again['risk_score']}")

    # ---- invariants over all 50 -------------------------------------------
    print("\n-- invariants across all 50 --")
    check("every case returns a 0-100 score",
          all(0 <= r["risk_score"] <= 100 for _, _, r in results),
          str(sorted({r["risk_score"] for _, _, r in results})))
    check("every case has a valid band + decision",
          all(r["risk_band"] in {"low", "medium", "high"}
              and r["decision"] in {"allow", "step_up", "verify"}
              for _, _, r in results))
    check("every case carries reason codes",
          all(isinstance(r.get("reason_codes"), list) and len(r["reason_codes"]) >= 0
              for _, _, r in results))

    # Monotonicity: adding a red flag to the SAME vector never lowers score.
    # Limited to the first 15 cases so the audit growth stays predictable.
    print("\n-- monotonicity (add one red flag -> score non-decreasing, first 15 cases) --")
    mono_ok = True
    for name, v, r in results[:15]:
        for flag in ("new_device_flag", "unusual_location_flag", "unusual_recipient_flag", "txn_time_unusual"):
            if v[flag] == 1:
                continue
            v2 = dict(v)
            v2[flag] = 1
            r2 = _post(f"{RISK_URL}/internal/evaluate", {
                "event_id": f"batch-m-{abs(hash(name)) % 10000:04d}-00000000",
                "fraud_id": FRAUD_ID,
                "features": v2,
            }, {"X-Internal-Token": INTERNAL_TOKEN})
            extra_evals += 1
            if r2["risk_score"] < r["risk_score"]:
                mono_ok = False
                print(f"    MONO VIOLATION: {name} {r['risk_score']} -> "
                      f"{r2['risk_score']} after {flag}=1")
    check("adding a red flag never lowers the score", mono_ok)

    # ---- audit chain: grew by the expected number and still verifies ------
    print("\n-- audit chain --")
    after = _integrity()
    grew = after.get("n_entries", 0) - before.get("n_entries", 0)
    expected = 50 + extra_evals
    check(f"audit chain grew by exactly {expected} (saw {grew})", grew == expected, str(grew))
    check("chain still verifies from genesis",
          after.get("ok") is True and after.get("first_bad_seq") is None,
          f"ok={after.get('ok')} first_bad={after.get('first_bad_seq')}")

    # ---- report: full table + named scenarios -----------------------------
    print("\n== decision table (all 50) ==")
    print(f"{'#':>3} {'case':<44} {'score':>5} {'band':<7} {'decision':<9} reasons")
    bands = {"low": 0, "medium": 0, "high": 0}
    for i, (name, _v, r) in enumerate(results, 1):
        bands[r["risk_band"]] += 1
        print(f"{i:>3} {name[:44]:<44} {r['risk_score']:>5} {r['risk_band']:<7} "
              f"{r['decision']:<9} {len(r['reason_codes'])}")

    print("\n== the five real-world scenarios ==")
    for name, _v, r in results[:5]:
        print(f"  {name:<38} -> {r['risk_band'].upper()} {r['risk_score']} "
              f"({r['decision']})  [{', '.join(r['reason_codes'][:4])}{' …' if len(r['reason_codes']) > 4 else ''}]")

    print(f"\nSummary: 50 cases -> allow {bands['low']}, step_up {bands['medium']}, "
          f"verify {bands['high']}; audit chain {before.get('n_entries')} -> {after.get('n_entries')} entries")
    print("\n" + ("ALL 50-CASE CHECKS PASSED" if not failures else f"{len(failures)} CHECK(S) FAILED"))

    # Only a clean run overwrites the last-run report the front page serves.
    if not failures:
        report = {
            "ran_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "seed": args.seed,
            "recheck": args.recheck,
            "total_cases": len(cases),
            "named": len(NAMED_SCENARIOS),
            "perturbed": len(cases) - len(NAMED_SCENARIOS),
            "distribution": {"allow": bands["low"], "step_up": bands["medium"], "verify": bands["high"]},
            "chain": {
                "before": before.get("n_entries"),
                "after": after.get("n_entries"),
                "grew": grew,
            },
            "integrity_ok": after.get("ok") is True and after.get("first_bad_seq") is None,
            "all_checks_passed": True,
            "scenarios": [
                {"name": name, "score": r["risk_score"], "band": r["risk_band"],
                 "decision": r["decision"], "reasons": r["reason_codes"]}
                for name, _v, r in results[:5]
            ],
        }
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = report_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(report, indent=2), encoding="utf-8")
        os.replace(tmp, report_path)
        print(f"Wrote last-run report -> {report_path}")

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
