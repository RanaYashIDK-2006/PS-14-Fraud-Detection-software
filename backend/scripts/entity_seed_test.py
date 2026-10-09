#!/usr/bin/env python3
"""Phase 11 regression test for _seed_entity_tracker (entity-rate warm start).

Original defect: the seed query referenced ``m.TransactionFeature`` on
``src.risk_engine.models``, where that model never lives (it is a
privacy-layer/DB-2 model), so startup seeding always raised AttributeError
and was silently skipped — the tracker never warm-started.

This test (all in an isolated temp DB_DIR):

  1. REGRESSION: seeding must succeed and seed exactly the events that
     have both a DB-3 label and a DB-2 feature row with at least one
     non-NULL entity id (pre-fix this raises AttributeError).
  2. BOUNDARIES: empty DB, unmatched events, partial/NULL/all-empty ids,
     chunked DB-2 pairing (>500 events), repeated invocation, and
     determinism across fresh trackers.
  3. DECISION IMPACT: the live /internal/evaluate request schema carries
     no entity ids, so seeded vs cleared tracker state must produce
     identical decision payloads (compared field-by-field, excluding
     event_id).

Run from the project root:
  python backend/scripts/entity_seed_test.py
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent  # repo root
sys.path.insert(0, str(ROOT / "backend"))

TMP = tempfile.mkdtemp(prefix="ps14-entity-seed-")
os.environ["DB_DIR"] = TMP
os.environ["PS14_MODE"] = "development"
os.environ["JWT_SECRET"] = "entity-seed-test-secret-0123456789"
os.environ["INTERNAL_TOKEN"] = "entity-seed-test-token"

from src.risk_engine import models as rm  # noqa: E402
from src.risk_engine.db import SessionLocal  # noqa: E402
from src.risk_engine.db import engine as risk_engine  # noqa: E402
from src.privacy_layer import models as pm  # noqa: E402
from src.privacy_layer.db import SessionLocal as FeatureSessionLocal  # noqa: E402
from src.privacy_layer.db import engine as privacy_engine  # noqa: E402
from src.risk_engine.entity_fraud_rates import (  # noqa: E402
    EntityFraudRateTracker,
    get_tracker,
)
from src.risk_engine.main import _seed_entity_tracker  # noqa: E402

failures: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        failures.append(name)


# ── deterministic fixtures ─────────────────────────────────────────────────
# Base set: 40 labeled events; feature rows for events 0..29; event 0 has
# all-NULL entity ids; events 7/14/21/28 have NULL device; event 9/18/27
# have NULL recipient; events 30..39 have no feature row at all.
# Chunk set: 700 further events, all matched, dense ids (forces >1 chunk).
BASE_RISK = [
    (f"p11ev{i:04d}", f"F24BRMMMTEST{i:03d}", 85 if i % 5 == 0 else 10)
    for i in range(40)
]
BASE_FEAT = []
for i in range(30):
    dev = None if (i == 0 or i % 7 == 0) else f"bdev{i % 4:02d}"
    rcp = None if (i == 0 or i % 9 == 0) else f"brcp{i % 3:02d}"
    cty = None if i == 0 else f"bcty{i % 2:02d}"
    BASE_FEAT.append((f"p11ev{i:04d}", f"F24BRMMMTEST{i:03d}", dev, rcp, cty))

CHUNK_RISK = [
    (f"p11chk{i:04d}", f"F24BRMMMCHK{i:03d}", 85 if i % 10 == 0 else 10)
    for i in range(700)
]
CHUNK_FEAT = [
    (f"p11chk{i:04d}", f"F24BRMMMCHK{i:03d}",
     f"cdev{i % 4:02d}", f"crcp{i % 3:02d}", f"ccty{i % 2:02d}")
    for i in range(700)
]

RISK_ROWS = BASE_RISK + CHUNK_RISK
FEAT_ROWS = BASE_FEAT + CHUNK_FEAT
RISK_BY_EVENT = {ev: (fid, score) for ev, fid, score in RISK_ROWS}

# Expected seeding result, derived independently of the implementation:
# label row + feature row + at least one non-NULL id, in the seed query's
# ORDER BY event_id DESC order.
SEED_ORDER = sorted(RISK_BY_EVENT, reverse=True)
EXPECTED = []  # (event_id, user, merchant, city, is_fraud)
for ev in SEED_ORDER:
    feat = next((f for f in FEAT_ROWS if f[0] == ev), None)
    if feat is None:
        continue
    _, _fid, dev, rcp, cty = feat
    dev, rcp, cty = dev or "", rcp or "", cty or ""
    if not (dev or rcp or cty):
        continue
    EXPECTED.append((ev, dev, rcp, cty, RISK_BY_EVENT[ev][1] >= 70))
EXP_SEEDED = len(EXPECTED)
EXP_FRAUD = sum(1 for e in EXPECTED if e[4])

print("== Phase 11 entity-tracker seeding regression ==")
print(f"  fixture: risk_rows={len(RISK_ROWS)} feat_rows={len(FEAT_ROWS)} "
      f"expected_seeded={EXP_SEEDED} expected_fraud={EXP_FRAUD}")

# ── populate isolated DB-3 + DB-2 ─────────────────────────────────────────
rm.Base.metadata.create_all(risk_engine)
pm.Base.metadata.create_all(privacy_engine)

s = SessionLocal()
for ev, fid, score in RISK_ROWS:
    s.add(rm.RiskScore(fraud_id=fid, risk_score=score, event_id=ev,
                       risk_band="high" if score >= 70 else "low",
                       reason_codes="[]", model_version="p11-test",
                       ml_score=0.0, rule_score=0.0))
s.commit()
s.close()

fs = FeatureSessionLocal()
for ev, fid, dev, rcp, cty in FEAT_ROWS:
    fs.add(pm.TransactionFeature(event_id=ev, fraud_id=fid, amount_ratio=1.0,
                                 txn_amount_bucket="mid",
                                 days_since_last_similar_txn=3.0,
                                 hour_of_day=12, device_hash=dev,
                                 recipient_id=rcp, location_id=cty))
fs.commit()
fs.close()

# ── 1. regression: seeding succeeds and is exact ──────────────────────────
print("\n-- regression (pre-fix AttributeError) --")
t1 = EntityFraudRateTracker()
try:
    _seed_entity_tracker(t1)
    check("seeding does not raise (pre-fix: AttributeError)", True)
except Exception as exc:  # noqa: BLE001
    check("seeding does not raise (pre-fix: AttributeError)", False,
          f"{type(exc).__name__}: {exc}")

g1 = t1.get_global_stats()
check("seeded exactly the joinable, non-empty events",
      g1.get("total_events") == EXP_SEEDED,
      f"got {g1.get('total_events')} want {EXP_SEEDED}")
check("fraud labels preserved (risk_score >= 70)",
      g1.get("total_fraud") == EXP_FRAUD,
      f"got {g1.get('total_fraud')} want {EXP_FRAUD}")

# unmatched / all-NULL events must NOT contribute (740 labeled rows exist,
# only EXP_SEEDED may be recorded — no phantom counting).
check("unmatched and all-NULL events contribute nothing",
      g1.get("total_events") < len(RISK_ROWS),
      f"{g1.get('total_events')} < {len(RISK_ROWS)} labeled rows")

# ── 2. boundaries ─────────────────────────────────────────────────────────
print("\n-- boundaries --")
# chunked pairing: the fixture spans >500 events, so more than one DB-2
# chunk executed; a truncated/unchunked query would under-seed.
check("chunked DB-2 pairing covers all events (>500, 2+ chunks)",
      EXP_SEEDED > 500 and g1.get("total_events") == EXP_SEEDED,
      f"expected {EXP_SEEDED}")

# per-entity rates vs an independent replay of the seed order (windows keep
# the last 100 inserted events per entity; min 5 events before a rate).
def replay_rate(entity: str, bucket: int) -> float:
    events = []  # last-100 window of (is_fraud) for this entity
    for ev, dev, rcp, cty, is_fraud in EXPECTED:
        eid = (dev, rcp, cty)[bucket]
        if eid == entity:
            events.append(is_fraud)
    events = events[-100:]
    if len(events) < 5:
        return 0.001
    return sum(events) / len(events)


for bucket, getter in enumerate(
        (lambda u, m, c: t1.get_rates(user_id=u)["user_fraud_rate"],
         lambda u, m, c: t1.get_rates(merchant_id=m)["merch_fraud_rate"],
         lambda u, m, c: t1.get_rates(city_id=c)["city_fraud_rate"])):
    entity = sorted({(e[1], e[2], e[3])[bucket] for e in EXPECTED
                     if (e[1], e[2], e[3])[bucket]})[0]
    want = replay_rate(entity, bucket)
    got = getter(entity, entity, entity)
    # tracker rounds stored rates to 6 decimals (see get_global_stats)
    check(f"seeded rate matches reference replay ({entity})",
          abs(got - want) < 1e-6, f"got {got} want {want}")

# determinism: two fresh trackers from the same DBs are identical.
t2 = EntityFraudRateTracker()
_seed_entity_tracker(t2)
g2 = t2.get_global_stats()
check("deterministic across fresh trackers (global stats)",
      g1 == g2, f"{g1} vs {g2}")
sample = sorted({(e[1], e[2], e[3]) for e in EXPECTED})[3]
check("deterministic across fresh trackers (entity rates)",
      t1.get_rates(sample) == t2.get_rates(sample),
      str(t1.get_rates(sample)))

# repeated invocation accumulates (documents the contract: the lifespan
# invokes seeding once per process; there is no dedup guard by design).
_seed_entity_tracker(t1)
g1b = t1.get_global_stats()
check("repeated invocation accumulates (2x) — documented contract",
      g1b.get("total_events") == 2 * EXP_SEEDED,
      f"got {g1b.get('total_events')} want {2 * EXP_SEEDED}")

# ── 3. serving call-site + decision impact ────────────────────────────────
print("\n-- serving path (TestClient lifespan seeds the singleton) --")
from fastapi.testclient import TestClient  # noqa: E402
from src.risk_engine.main import app as risk_app  # noqa: E402

VECTOR = {
    "amount_ratio": 0.95, "txn_freq_last_24h": 1, "txn_time_unusual": 0,
    "new_device_flag": 0, "unusual_location_flag": 0,
    "unusual_recipient_flag": 0, "failed_auth_count_24h": 0,
    "days_since_last_similar_txn": 3.0, "gradual_escalation_score": 0.0,
    "known_device_count": 3, "account_tenure_days": 90.0, "hour_of_day": 12,
    "is_weekend": 0, "shared_device_accounts": 0, "shared_recipient_accounts": 0,
    "mule_ring_score": 0.0, "hour_deviation": 0.5, "amount_zscore": 0.0,
    "velocity_deviation": 0.0, "recipient_novelty": 0.0,
    "txn_regularity": 0.0, "account_daily_spend_ratio": 0.2,
    "device_daily_count": 1,
}


def evaluate(client, event_id: str, extra: dict | None = None) -> dict:
    body = {"event_id": event_id, "fraud_id": "F24BRMMMBJWYMTDW",
            "features": VECTOR}
    if extra:
        body.update(extra)
    r = client.post("/internal/evaluate", json=body,
                    headers={"X-Internal-Token": os.environ["INTERNAL_TOKEN"]})
    assert r.status_code == 200, f"evaluate failed: {r.status_code} {r.text}"
    return r.json()


with TestClient(risk_app) as client:
    singleton = get_tracker()
    gls = singleton.get_global_stats()
    check("lifespan seeds the singleton tracker (call-site wiring)",
          gls.get("total_events") == EXP_SEEDED,
          f"got {gls.get('total_events')} want {EXP_SEEDED}")

    seeded_payload = evaluate(client, "p11seeded-0001")
    singleton.clear()
    cleared_payload = evaluate(client, "p11cleared-0001")
    diff = {k for k in seeded_payload
            if k != "event_id" and seeded_payload.get(k) != cleared_payload.get(k)}
    check("decision payload identical: seeded vs cleared tracker (no ids in schema)",
          not diff, f"differing fields: {sorted(diff)}")

    # even attempting to smuggle ids through the endpoint cannot change the
    # decision: EvaluateRequest has no id fields (pydantic drops extras).
    smuggled_seeded = evaluate(client, "p11smuggle-seed",
                               {"user_id": "bdev01", "merchant_id": "brcp01",
                                "city_id": "bcty01"})
    singleton.clear()
    smuggled_cleared = evaluate(client, "p11smuggle-clear",
                                {"user_id": "bdev01", "merchant_id": "brcp01",
                                 "city_id": "bcty01"})
    diff2 = {k for k in smuggled_seeded
             if k != "event_id"
             and smuggled_seeded.get(k) != smuggled_cleared.get(k)}
    check("decision payload identical even with extra id fields supplied",
          not diff2, f"differing fields: {sorted(diff2)}")

# ── 4. empty-state branch (last: DB is emptied) ───────────────────────────
print("\n-- empty DB branch --")
s = SessionLocal()
s.query(rm.RiskScore).delete()
s.commit()
s.close()
t3 = EntityFraudRateTracker()
try:
    _seed_entity_tracker(t3)
    check("empty DB: seeding is a clean no-op",
          t3.get_global_stats().get("total_events") == 0,
          str(t3.get_global_stats()))
except Exception as exc:  # noqa: BLE001
    check("empty DB: seeding is a clean no-op", False, str(exc))

print("\n" + "=" * 60)
if failures:
    print(f"FAILED: {len(failures)}")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
print("== Phase 11 entity-tracker seeding: 0 failed check(s) ==")
print("ALL CHECKS PASSED")
sys.exit(0)
