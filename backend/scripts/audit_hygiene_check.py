#!/usr/bin/env python3
"""Audit-chain hygiene invariant (Phase 110).

Enforced by scripts/security_ci_gate.py (step 1) and the CI Security Scan
job.  Watches the SHARED production audit chain (db/audit.db) for test-
fixture leakage and chain corruption.  Zero parameters, deterministic,
no bypass.

Historical evidence (Phase 109): the five 2026-09-19 forks
{731, 735, 740, 745, 750} are PRESERVED rows.  They are allowed ONLY
when they match the frozen AuditForkFinding evidence exactly
(evidence-bound quarantine in phase109_audit_fork_repair.evaluate_chain);
they are never declared valid, and any missing/found extra/changed fork
row fails.  verify_chain stays strict — nothing here weakens it.

Fixture families below were written DIRECTLY into DB-4 by test code
(phase76 / phase77 / phase79 — all now bound to private temp DBs by their
suites).  Their shared-chain counts are frozen at the 2026-09-24
snapshot: a chain containing ANY fixture row must match the snapshot
exactly; a chain containing none (fresh, correctly isolated environment)
passes as pristine.  Legitimate application event types (decisions,
OUTCOME_*, admin_*, runtime_*) are intentionally NOT frozen — they grow
with real usage.

Exit 0 = hygiene clean (or no shared chain present — nothing to pollute).
Exit 1 = fixture leakage / fork drift / corruption detected.
"""
from __future__ import annotations

import hashlib  # noqa: F401  (kept for symmetry with forensic tooling)
import re
import sqlite3
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

BACKEND = Path(__file__).resolve().parent.parent
REPO_ROOT = BACKEND.parent
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from src.monitoring.phase109_audit_fork_repair import (  # noqa: E402
    AFFECTED_SEQUENCES,
    evaluate_chain,
)

LIVE_AUDIT_DB = REPO_ROOT / "db" / "audit.db"

# Frozen 2026-09-24 snapshot of every DIRECT test-fixture event type in
# the shared chain (phase77 concurrency family, phase76 lifecycle family,
# phase79 security family).  These must never change now that all three
# suites write to private temp DBs.
FIXTURE_FROZEN_COUNTS: dict[str, int] = {
    "rapid_test": 4800,
    "concurrent_test": 3840,
    "perf_test": 2400,
    "perf_conc": 1200,
    "queue_test": 720,
    "deadlock_test": 690,
    "lock_test": 480,
    "shutdown_test": 480,
    "dup_test": 48,
    "exception_test": 48,
    "seq_test": 48,
    "empty_test": 24,
    "large_test": 24,
    "restart_test": 24,
    "single_test": 24,
    "lifecycle_test_1": 25,
    "lifecycle_test_2": 25,
    "lifecycle_test_3": 25,
    "test_drain_event": 25,
    "crash_survival_test": 24,
    "security_test": 540,
    "legacy_race": 0,       # Phase-109 harness signature type: never shared
    "phase109_repro": 0,    # Phase-109 repaired-writer probes: never shared
}
FROZEN_CHAIN_FRAUD_COUNT = 75  # fraud_id LIKE 'CHAIN-%' (== lifecycle rows)
FIXTURE_TIMESTAMP_CUTOVER = "2026-09-24 08:00:00"

_TESTISH = re.compile(r"(_test|^test_)")  # catch-all for FUTURE fixture types


def _load_rows(db_path: Path) -> list[dict[str, Any]] | None:
    if not db_path.exists():
        return None
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in con.execute(
            "SELECT * FROM audit_events ORDER BY seq"
        ).fetchall()]
    finally:
        con.close()


def evaluate_audit_hygiene(rows: Iterable[Mapping[str, Any]]) -> list[str]:
    """Deterministic hygiene verdict over already-loaded chain rows.

    Takes rows only — no path, no flags, nothing a caller can tune.
    """
    row_list = list(rows)
    violations: list[str] = []

    observed = Counter(str(r.get("event_type")) for r in row_list)
    chain_fraud = sum(
        1 for r in row_list if str(r.get("fraud_id", "")).startswith("CHAIN-")
    )

    # Unknown fixture-like types (future leaks) are always a violation.
    for t in sorted(observed):
        if t not in FIXTURE_FROZEN_COUNTS and _TESTISH.search(t):
            violations.append(f"unexpected fixture-like event_type {t} x{observed[t]}")

    fixture_total = sum(observed.get(t, 0) for t in FIXTURE_FROZEN_COUNTS)
    pristine = fixture_total == 0

    if not pristine:
        # This chain carries historical fixture rows -> it must match the
        # frozen snapshot EXACTLY (growth = new leak, loss = tampering).
        for t in sorted(FIXTURE_FROZEN_COUNTS):
            expected = FIXTURE_FROZEN_COUNTS[t]
            actual = observed.get(t, 0)
            if actual != expected:
                violations.append(
                    f"fixture count drift: {t} = {actual} (frozen {expected})"
                )
        if chain_fraud != FROZEN_CHAIN_FRAUD_COUNT:
            violations.append(
                f"CHAIN-* fixture rows = {chain_fraud} "
                f"(frozen {FROZEN_CHAIN_FRAUD_COUNT})"
            )
        for r in row_list:
            t = str(r.get("event_type"))
            if t in FIXTURE_FROZEN_COUNTS and \
                    str(r.get("created_at", "")) > FIXTURE_TIMESTAMP_CUTOVER:
                violations.append(
                    f"fixture timestamp after cutover: seq {r.get('seq')} "
                    f"{t} at {r.get('created_at')}"
                )
                break
    elif chain_fraud != 0:
        violations.append(f"CHAIN-* fixture rows on pristine chain: {chain_fraud}")

    # Duplicate sequence / duplicate event id
    seqs = [r.get("seq") for r in row_list]
    if len(seqs) != len(set(seqs)):
        violations.append("duplicate audit sequence detected")
    eids = [r.get("event_id") for r in row_list]
    if len(eids) != len(set(eids)):
        violations.append("duplicate audit event_id detected")

    # Chain integrity under the Phase-109 evidence model: every row
    # strictly checked; breaks allowed only when they are EXACTLY the five
    # frozen evidence-matched historical forks — or the chain is clean
    # (fresh/isolated environment).  A vanished fork row, an extra fork,
    # or a doctored quarantined row all fail.
    res = evaluate_chain(row_list)
    if not res["ok"]:
        violations.append(f"chain not trustworthy: {res['reason']} (first_bad="
                          f"{res['first_bad_seq']})")
    else:
        breaks = list(res["quarantined_breaks"] or [])
        if breaks and set(breaks) != set(AFFECTED_SEQUENCES):
            violations.append(
                f"historical fork set drifted: {breaks} != {list(AFFECTED_SEQUENCES)}"
            )
        if breaks and res["strict_ok"]:
            violations.append("fork rows declared strictly valid (must stay False)")
        if not breaks and not res["strict_ok"]:
            violations.append("clean chain reported strict_ok=False")

    return violations


def check_shared_chain_hygiene() -> tuple[str, list[str]]:
    """Load the shared chain (read-only) and evaluate.  Returns
    (status, violations).  status is one of:
    NO_SHARED_CHAIN (fresh environment, nothing to pollute) / CLEAN / DIRTY."""
    rows = _load_rows(LIVE_AUDIT_DB)
    if rows is None:
        return "NO_SHARED_CHAIN", []
    return ("DIRTY" if (v := evaluate_audit_hygiene(rows)) else "CLEAN"), v


def main() -> int:
    status, violations = check_shared_chain_hygiene()
    print("=" * 70)
    print("PS-14 AUDIT-CHAIN HYGIENE INVARIANT (Phase 110)")
    print("=" * 70)
    print(f"  shared chain: {LIVE_AUDIT_DB}")
    print(f"  frozen fixture types: {len(FIXTURE_FROZEN_COUNTS)}, "
          f"cutover {FIXTURE_TIMESTAMP_CUTOVER}")
    print(f"  historical forks allowed (evidence-bound): {list(AFFECTED_SEQUENCES)}")
    if status == "NO_SHARED_CHAIN":
        print("  -> no shared audit chain present (fresh/isolated environment):")
        print("     nothing to pollute, nothing to verify: PASS")
        return 0
    if status == "CLEAN":
        print("  -> CLEAN: fixture snapshot exact, forks evidence-matched, "
              "no duplicates, no drift")
        return 0
    print(f"  -> DIRTY: {len(violations)} violation(s)")
    for viol in violations:
        print(f"     VIOLATION: {viol}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
