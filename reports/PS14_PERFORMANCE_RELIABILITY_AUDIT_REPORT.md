# PS-14 — CHECK #26: END-TO-END PERFORMANCE, LATENCY & RELIABILITY AUDIT

**Date:** 2026-09-03
**Method:** Live load tests against the actually-deployed stack (identity :8001, privacy :8002, risk :8003 on an isolated `db/perf_audit`, single uvicorn process per service, SQLite per-store, Altman lean-15 engine `altman_lean_15feat_20260830_200346`). Measured path: `ingest-transaction (feature retrieval/calculation + DB-2 write) → /internal/evaluate (mapper → model → decision → DB-3 alert row → queued DB-4 audit append)`. Harness: `scripts/perf_reliability_audit.py`; raw data: `reports/perf_reliability_audit.json`, `db/perf_audit/*.log`.
**Scope caveat:** these are loopback numbers for the SQLite configuration actually deployed here. The Postgres/Supabase path (`.env` `DATABASE_URL`) was **not running → not measured** (UNVERIFIED).

---

## 1. Model inference (in-process, same artifacts the risk engine loaded)

| Metric | Value |
|---|---|
| Single-row predict (mapper + ensemble + calibrator) | **1.93 ms/row** (~519/s serial) |
| Batch `predict_many` (10,000 rows) | **0.014 ms/row** (~71,000 rows/s) |

Inference is *not* the bottleneck — the ~16 ms HTTP floor is framework + SQLite + decision overhead.

## 2. Results table (HTTP, loopback)

| Test | Load | p50 | p95 | p99 | Throughput | Errors | Result |
|---|---|---:|---:|---:|---:|---:|---|
| ingest-only | 2 workers | 15 ms | 16 ms | 32 ms | — | 10/150 | PASS |
| evaluate-only | 2 workers | 16 ms | 16 ms | 31 ms | — | 9/150 | FAIL (errors at 2w) |
| e2e chain | 2 workers | 16 ms | 31 ms | 32 ms | — | 1/120 | PASS |
| evaluate ramp | 1 | 0 ms | 16 ms | 16 ms | **139.7/s** | 0/600 | PASS |
| evaluate ramp | 4 | 31 ms | 32 ms | 47 ms | 104.5/s | 141/600 (24%) | **FAIL** |
| evaluate ramp | 8 | 62 ms | 78 ms | 125 ms | 93.0/s | 186/600 (31%) | **FAIL** |
| evaluate ramp | 16 | 110 ms | 187 ms | 203 ms | 88.7/s | 197/600 (33%) | **FAIL** |
| evaluate ramp | 32 | 313 ms | 422 ms | 438 ms | 100.8/s | 20/900 (2%) | FAIL |
| evaluate ramp | 64 | 656 ms | 750 ms | 750 ms | 93.3/s | 21/900 (2%) | FAIL |
| evaluate ramp | 128 | 1359 ms | 1469 ms | 1484 ms | 90.2/s | 28/900 (3%) | FAIL |
| e2e ramp | 1 | 16 ms | 31 ms | 32 ms | 67.1/s | 0/300 | PASS |
| e2e ramp | 4 | 47 ms | 63 ms | 78 ms | 85.0/s | 20/300 (7%) | FAIL |
| e2e ramp | 16 | 172 ms | 218 ms | 250 ms | 58.8/s | 108/300 (36%) | **FAIL** |
| **sustained** | 16 w, 45 s | 187 ms | 250 ms | 297 ms | 58.5/s goodput | **1348/3980 (33.9%)** | **FAIL** |
| **spike** | 128 w, 8 s | 1625 ms | 1718 ms | 1766 ms | 68.0/s goodput | 216/760 (28%) | FAIL |

## 3. THE critical reliability failure (proven live, not theorized)

**Concurrent scoring intermittently 500s on the alert-row write.** Under ≥4 concurrent `/internal/evaluate` requests the risk engine returns HTTP 500 with:

```
sqlite3.InterfaceError: bad parameter or other API misuse
[SQL: INSERT INTO risk_scores (…ml_score, rule_score, degraded)
      VALUES (?, …) RETURNING scored_at]
```

and its sibling `Single-row INSERT … did not produce a new primary key result`. Sustained 16-worker load: **33.9% error rate** (1,348 of 3,980). Root cause: concurrent SQLAlchemy sessions hitting SQLite on the same store — the `RETURNING scored_at` insert is not safe under the current pooling/connection config. Consequences: those transactions get **no score row and no alert** (DB-3 is the alert store); the client sees a loud 500, so nothing is silently approved, but **alerts are silently dropped at the failure rate measured above**. The `except` in `evaluate` only handles duplicate-`event_id` races, not this error.

## 4. Reliability checklist

| Requirement | Result | Evidence |
|---|---|---|
| Timeouts — slow dependency | **NOT TESTED** | No timeout hooks exist to force; DB ops have no explicit timeout, so a hung SQLite could stall a worker thread indefinitely |
| Retries / idempotency | **PASS (core)** | Sequential replay of an event → identical score; 10-way concurrent duplicate → all 200, **1 risk row, 1 audit event** (no duplicate alerts). Caveat: under the §3 bug the race path can also 500 |
| Backpressure / queue failure | **PARTIAL** | At 128 workers requests queue (p50 1.36 s) with no queue-limit signal; audit writer queue falls back to sync writes |
| DB-4 (audit) unavailable | **PASS** | Exclusive SQLite lock on audit.db during scoring: 15/15 responses OK, p95 156 ms, audit events drained afterwards. Append is queued/async — **AGENTS.md's "sync critical path" note is outdated** |
| Risk service down | **PASS** | Evaluate → explicit connection error (no silent approve/reject). Fail-safe must live at the caller. Ingest continues while risk is down — a caller that skips evaluate after an outage would silently drop scoring (architecture note, not measured as a bug) |
| NaN feature value | **FAIL (partial)** | Accepted (no 422 finite-value guard); routed by the model to a finite low score (risk_score 8/allow) — a NaN in a critical feature silently *lowers* risk. One NaN probe during load returned 500 (the §3 bug). Recommend explicit finite validation |
| Missing feature (corrupt vector) | **PASS** | 422 validation, fail-loud |
| Corrupt/missing model artifact | PASS | Fail-closed at startup (governance, check #18) |
| Memory / leaks | **PARTIAL** | Risk listener RSS ~321 MB after sustained load + restart (single sample); model loads once at startup. No leak curve measured over hours |
| Unbounded growth | **FAIL** | risk.db +4.0 MB, audit.db 12.3 MB (14,463 events), features.db 2.4 MB in ~4 min of load → ~1 GB/day/risk.db at 60 tx/s with no retention (check #25). Uvicorn access logs grow per request with no rotation configured |
| Audit/DB-3 consistency | **FAIL** | Under load: `score_generated` audit events (9,208) exceed risk_scores rows (8,722) by 5.6% — divergence to investigate under the §3 bug |

## 5. PRODUCTION CAPACITY VERDICT

- **Maximum tested throughput:** ~140 tx/s (single worker, evaluate-only); goodput never exceeded ~100/s at any concurrency.
- **Maximum sustainable throughput:** **~58 tx/s** end-to-end (and even that carries a 34% error rate at 16 workers — *no concurrency level ≥4 workers is reliable*).
- **p95/p99 latency:** 16/31 ms at 1 worker; 250/297 ms (e2e, 16 w) — but those p95s are meaningless when a third of requests 500.
- **Failure point:** **≥4 concurrent requests** (SQLite INSERT `RETURNING` concurrency bug, §3). Latency also degrades steeply: p50 16 ms → 1.36 s at 128 workers.
- **Safe operating capacity:** ≤2 concurrent requests, ≤ ~70–140 tx/s on a single risk process — i.e., the engine is effectively **serialized**.
- **Required scaling margin:** >10× for any realistic production load — but **no amount of horizontal scaling fixes the per-process concurrency bug**; the architecture fix (serialized writes, WAL/retry, or Postgres) is a prerequisite.
- **Remaining reliability risks:** (1) the INSERT concurrency bug drops alerts at measured rates; (2) NaN features silently lower risk; (3) unbounded DB/log growth with no retention; (4) audit↔DB-3 divergence under load; (5) no timeout guards on DB operations; (6) Postgres path completely unmeasured.

**Bottom line:** the model itself is fast (2 ms inference, 71K rows/s batched), and the single-threaded path is snappy (16 ms, 0 errors). But **PS-14 is NOT production-scalable in its currently-deployed SQLite configuration** — the concurrency ceiling is ~1–2 workers and alerts are dropped at the measured failure rates above that. Fix the DB write path first, then re-measure before any capacity claim.

**Files:** `reports/perf_reliability_audit.json`, `db/perf_audit/{identity,privacy,risk}.log`, `scripts/launch_perf_stack.py`, `scripts/launch_perf_audit.py`, `scripts/perf_reliability_audit.py`.
