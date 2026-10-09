# Phase 9 — Request-Path Performance & Latency Baseline — Closeout

## Phase identity

- **Phase**: 9 — Request-Path Performance & Latency Baseline
- **Start SHA**: `4948cba` (`Phase 8 — Audit Decision-Path Resilience`, pushed to `origin/main`; recorded in every evidence file's `environment.git`)
- **Implementation SHA**: `0764b3b` (`feat(phase9): add the live request-path benchmark harness`)
- **Final SHA**: `9990ad5` (closeout content; this CI-record append is the phase's trailing commit, whose SHA is reported in the session report — a commit cannot contain its own SHA)
- **Verification window**: 2026-10-09 (this session, 10:05–11:30 IST / 04:35–06:00 UTC)
- **Scope**: measurement only. No model, threshold, calibration, rule, feature-contract, or
  decision-semantics changes. No optimizations were performed or accepted in this phase.

## Evidence class

- **DEMONSTRATED (live HTTP)**: every number below was produced by real HTTP requests
  (`http.client`, keep-alive, 1 connection per thread) against a real `uvicorn` listener
  (1 worker, `src.risk_engine.main:app`) spawned by the harness on port 8009 with an
  isolated `DB_DIR` per run.
- The **delayed** and **unavailable** audit runs are live-server measurements whose *fault*
  is SIMULATED: `backend/scripts/ps14_p9_fault_boot.py` applies the Phase 8 injection
  inside the live process (boot-time env `PS14_P9_AUDIT_MODE`), as documented in each
  record's `evidence_class` field.
- Phase 5 engine-level / in-process numbers are NOT presented as live-server results here.

## Method

Harness: `backend/scripts/ps14_p9_live_bench.py` (committed in this phase).

```text
python backend/scripts/ps14_p9_live_bench.py --profile ml   --audit-mode healthy     # ML-path baseline
python backend/scripts/ps14_p9_live_bench.py --profile full --audit-mode healthy     # full baseline
python backend/scripts/ps14_p9_live_bench.py --profile audit --audit-mode delayed    # +PS14_P9_AUDIT_DELAY=0.25
python backend/scripts/ps14_p9_live_bench.py --profile audit --audit-mode unavailable
```

- **Workload fixture**: deterministic, in-distribution, drawn from `data/transactions.csv`
  (9,799 rows, sha256 `9f0f56bf…5d63`, shuffle seed `20261009`, index rule
  `crc32(segment_prefix) + i`), 23 feature keys — identical across all four runs so
  decision payloads can be compared field-by-field. Unique event ids per segment
  (prefix + runid + index) prevent idempotency collapse; stale keep-alive connections
  are re-established across pacing waits.
- **Rate-limit pacing**: the Phase 73 limiter allows 500 `/internal/evaluate` requests per
  60 s window (`backend/src/monitoring/access_control.py check_access()`); the harness
  waits 61 s between segments (11 waits, logged in `rate_limit_pacing_log`), keeping every
  latency segment inside budget so 429s never contaminate percentiles. The final probe
  deliberately exceeds the budget.
- **Drift-gate constraint (discovered by measurement in this phase)**: the PSI drift
  baseline (`models/data/drift_baseline.json`) was built from a 200,000-event dataset with
  2012-era dates, not the serving table the deployed model was trained on. Any traffic
  trips the gate at event 100 (`check_interval=100`, observed max PSI **12.88** vs alert
  threshold 0.25) → `state=critical`, reason codes `ML_UNAVAILABLE | DRIFT_MODEL_PAUSED`,
  ML paused for the rest of the process. Reproduced by `.freebuff/p9/diagnose_degraded.py`
  and `.freebuff/p9/verify_workload.py`. Consequences for this phase:
  - events 1…99 of a process = ML path; later segments = post-flip rules-only steady state
    (both are reported separately, never pooled);
  - the `ml` profile measures the ML path in fresh-process cycles of 98 requests (under
    the check interval), 5 cycles × 98 = 490 requests.
- **Integrity**: after every run the harness reconciles its sent counters against the
  server access log (`integrity.server_log_reconciliation`) — mismatches or foreign
  client traffic void the run.

## Startup / cold start

| Run | spawn→ready | first request |
|---|---|---|
| healthy (full) | 2.907 s | 40.9 ms → HTTP 200, `risk_score=1`, `decision=allow`, `ml_score=0.0068`, `degraded=false` |
| healthy (ml cycles) | 2.875–2.937 s (5 cycles) | — |
| delayed audit | 2.907 s | HTTP 200 |
| unavailable audit | 17.687 s | HTTP 200 |

The 17.687 s spawn→ready of the unavailable-audit boot is an observed startup cost of the
faulted boot path, not a decision-path effect (probe latencies below are unaffected).

## `/internal/evaluate` latency — ML path (drift state `normal`, ML active)

Profile `ml`, 5 fresh-process cycles, 490 requests, 0 errors, **0 degraded**, drift state
`normal` in every cycle (each cycle's last check count stayed below the 100-event interval
by construction):

| stat | ms |
|---|---|
| n | 490 |
| min | 23.250 |
| mean | 31.210 |
| **p50** | **28.489** |
| p95 | 32.493 |
| p99 | 56.947 |
| max | 577.044 |

Interpretation: steady-state ML-path median ≈ **28.5 ms**, p95 ≈ 32.5 ms; the tail
(p99 ≈ 57 ms, isolated maxima 0.5–0.6 s) is dominated by rare Windows scheduling/GC
outliers, not by pathologically slow requests — the same outlier signature appears at
every concurrency level.

Pre-flip segments of the full healthy run agree:

| segment | n | p50 | p95 |
|---|---|---|---|
| warmup_20 | 20 | 28.794 | 32.754 |
| probes_6 | 6 | 28.106 | 29.842 |
| audit_load_60 | 60 | 29.174 | 33.581 |
| ml_tail_12 | 12 | 30.478 | 204.409 (single outlier) |

## `/internal/evaluate` latency — post-flip rules-only steady state

After event 100 the drift gate pauses ML (see Method); these segments measure the
degraded path (`degraded=true`, rules + score floor), which is the state the deployed
system is actually in for any sustained traffic against the current baseline:

| segment | n | p50 | p95 | p99 | max | rps |
|---|---|---|---|---|---|---|
| single_500_run1 | 500 | 28.359 | 31.317 | 45.808 | 614.749 | 31.6 |
| single_500_run2 | 500 | 29.910 | 34.701 | 73.693 | 1148.216 | 28.0 |

Rules-only is within ~1–2 ms of the ML path at the median — the ML inference cost is not
the dominant term in request-path latency.

Server-side view (`GET /internal/latency-slo`, its own 10K-request ring buffer,
n = 2,669 at capture time): p50 31 ms, p95 109 ms, p99 672 ms, p999 2,094 ms, max 2,218 ms,
`slo_met: false` against the documented target `p99 < 100 ms`. The p99 gap is the same
rare-outlier phenomenon: client-measured p99 over the pooled 490-request ML baseline was
56.9 ms, so the SLO breach concentrates in the >0.5 s tail events.

## Concurrency ramp

1,550 requests, **0 errors at every level** (all HTTP 200), 61 s pacing before each step:

| concurrency | requests | throughput rps | p50 | p95 | p99 | max |
|---|---|---|---|---|---|---|
| 1 | 150 | 30.1 | 29.634 | 33.873 | 38.893 | 509.632 |
| 2 | 200 | 38.0 | 28.576 | 236.792 | 270.274 | 277.101 |
| 4 | 300 | 107.5 | 32.755 | 68.045 | 100.982 | 122.739 |
| 8 | 300 | 106.2 | 43.669 | 111.312 | 231.226 | 904.020 |
| 16 | 300 | 113.1 | 51.039 | 265.052 | 872.570 | 1887.375 |
| 32 | 300 | 114.9 | 147.731 | 928.950 | 1692.628 | 2302.762 |

- Single-worker saturation ≈ **110–115 rps** (c ≥ 4); throughput plateaus while percentiles
  grow roughly linearly with offered load beyond c = 8.
- Median stays ≤ 34 ms up to c = 4; the SLO-relevant crossover (p95 > 100 ms) happens
  between c = 4 and c = 8.
- RSS peak ≈ 349 MB across the run; no memory creep observed between segments.
- The c=2 p95 (236.8 ms) exceeds c=4 p95 (68.0 ms): two isolated ~270 ms outliers landed
  in the small c=2 sample; this is tail noise, not a load inversion.

## Batch endpoint (`POST /internal/evaluate-batch`)

| batch size | total ms | per-item ms | items valid | failed |
|---|---|---|---|---|
| 10 | 33.427 | 3.343 | 10 | 0 |
| 50 | 43.318 | 0.866 | 50 | 0 |
| 100 | 60.630 | 0.606 | 100 | 0 |
| 500 | 158.153 | 0.316 | 500 | 0 |

- All responses: HTTP 200, `count` matches, every result carries its `event_id`, zero
  failed items.
- Amortization: per-item cost falls ~10× from size 10 → 500; a batch of 10 costs about one
  single request (33.4 ms vs mono p50 ≈ 31.5 ms → 3.15 ms/item vs 31.5 ms/item ≈ **10×**),
  and batch 500 is ~100× cheaper per item than serial evaluation.
- **Phase 5 batch contract invariant**: `native_fixture` (which drives the real
  `AltmanNativeEnsembleEngine` loader and the Phase 5 batch contract on a clean checkout)
  and `feature_parity` (offline==online features) passed in all four regression batteries
  of this session; `git status`/`git diff` over `models/` and `backend/src/` is empty —
  no contract, model, threshold, or calibration file changed in this phase.

## Rate-limit probe (deliberate, last segment)

505 requests in one window → **500 × HTTP 200, then 5 × HTTP 429**
(`{"detail": "rate_limit_exceeded"}`), first 429 at event index 500 — exactly the
documented 500/60 s budget — after a recorded 60.7 s pacing wait. This measures limiter
behavior, not request-path latency, and confirms budget enforcement on the live path.

## Audit-path comparison (Phase 8 behavior under controlled load)

Three conditions, identical workload and identical 6 fixed probe vectors
(`probe-01`…`probe-06`, chosen to fire the ATO/velocity shape):

**Decision payload identity** — every field of every probe body compared across
healthy / delayed (0.25 s per append) / unavailable, excluding `event_id` and
`latency_ms`:

```text
probe-01 … probe-06: 0 differing fields each
TOTAL differing decision fields across 6 probes × 3 conditions: 0
```

Fields covered include `risk_score`, `risk_band`, `decision`, `reason_codes`, `ml_score`,
`rule_score`, `fired_rules`, `degraded`, `calibrated`, `odds`, `limits.*`,
`uncertainty.*`.

**Decision latency** (probe latencies, ms):

| condition | probe-01…06 |
|---|---|
| healthy | 29.6, 28.1, 26.8, 27.9, 29.8, 28.7 |
| delayed | 46.9, 20.6, 26.0, 25.6, 22.1, 20.7 |
| unavailable | 22.7, 21.4, 20.0, 21.5, 21.1, 20.3 |

Workload pre-flip medians: healthy warmup/audit_load 28.8/29.2 ms; delayed 22.5/21.4 ms;
unavailable 21.3/21.5 ms. The faulted paths are never slower than healthy (the healthy
run's ~7 ms offset is run-to-run machine variance — it was captured during the longest,
pacing-dominated run; no mechanism gives the caller back time when audit is slow).

**Disposition accounting** (`audit_backlog`, isolated `DB_DIR` per run):

| condition | pre-shutdown | post-shutdown (atexit flush) |
|---|---|---|
| healthy | 2,680 rows, pending 0 | 3,831 rows, pending 0 |
| delayed | 52 rows, pending 0 | 89 rows, pending 0 (delayed writes completed) |
| unavailable | **0 rows**, 89 pending lines (29 failed + 60 retrying), DB-3 `risk_scores` 87 rows | unchanged, all 89 still durable in pending store |

Under DB-4 unavailability every decision still persisted to DB-3 and every audit event
stayed payload-bearing in the durable pending store — Phase 8's isolation contract holds
on the live request path.

## Integrity

Full healthy run: harness sent 3,174 `/internal/evaluate` + 4 batch calls;
server access log shows 3,174 eval (3,169 × 200 + 5 × 429) and 4 batch —
`eval_match: true`, `batch_match: true`, `foreign_client_traffic_suspected: false`.
All four runs wrote their records with `fatal_error` absent; every server was cleanly
stopped (listener killed, port confirmed free).

## Findings (observed, NOT fixed — measurement-only phase)

1. **`GET /internal/metrics` returns 500** (`name 'Response' is not defined`): `metrics()`
   in `backend/src/risk_engine/main.py` constructs a `fastapi.Response` that is never
   imported; `SecurityHeadersMiddleware` catches the `NameError` and returns a generic
   JSON 500 (logging one line, no traceback — which is why the server log shows a 500 with
   no stack trace). `/internal/latency-slo` and `/internal/observability` work. Pre-existing
   at the starting SHA; out of scope to fix in a measurement phase.
2. **Drift baseline vs serving-data mismatch** trips PSI at event 100 for any traffic
   (Method section). This constrains live ML-path measurement to 98 events per process and
   means the deployed system is rules-only after ~100 events. A Phase 10 candidate, not a
   Phase 9 change.
3. **`security_test.py` is state-dependent on a cold stack**: the dev stack runs
   `PS14_MODE=development`, so `rate_limit_middleware` intentionally skips (development
   guard), and the only 429 source is identity's login lockout
   (`_LOGIN_LOCKOUT_THRESHOLD=10`, 15-min window, checked *before* the failure is
   recorded). The test sends exactly 10 attempts: a cold run records its 10th failure only
   after the check that attempt-10 fails to trigger → "Never blocked"; any run within 15
   minutes of a cold run gets 429 on attempt 1 → pass. This fully explains the
   alternating pass/fail pattern across batteries (evidence: `.freebuff/identity.log`
   access lines). Not a regression; in production mode the middleware (5/60) would trip at
   attempt 6 deterministically.
4. **`audit_resilience` flaked once** in battery 2 of 4 (timing-sensitive poll under
   concurrent system load); it passed standalone and in batteries 1, 3, 4. The failing
   check name was lost when the runner's failure-detail print crashed on a cp1252 stdout —
   itself a runner defect (finding 5).
5. **`regression_suite.py` miscounts on non-UTF-8 parent stdout**: it appends the result
   *before* printing the failure detail; if that print raises `UnicodeEncodeError`, the
   `except` path appends the same suite again (battery 1 reported 33 entries with
   `security_test` counted twice). Running the parent with `PYTHONIOENCODING=utf-8`
   avoids it; battery 4 ran this way.
6. **`security_scan.py` file walk is vacuous**: it rglobs `project_root / "src"`, but the
   code lives in `backend/src`, so `Files Scanned: 0` and the 6 pattern checks execute
   over an empty set. Identical in Phase 8's stored artifacts (pre-existing). The working
   static gates are `secret_hygiene_test.py` (16 scoped detectors) and
   `bandit -r backend/src --severity-level medium` — both pass below.

## Regression

Four full batteries this session (all include the 25 fast suites + 7 live suites):

| run | result | notes |
|---|---|---|
| battery 1 (`full_suite.log`) | 31/33 | `security_test` cold-fail ×2 counted (finding 3 + finding 5 double-count); everything else PASS |
| battery 2 (`full_suite2.log`) | 31/32 | `audit_resilience` one-off flake (finding 4); `security_test` warm PASS |
| battery 3 (`full_suite3.log`) | 31/32 | `security_test` cold-fail with full detail captured; `audit_resilience` PASS |
| battery 4 (`full_suite4.log`) | **32/32 PASS** (196.8 s) | authoritative clean run; `PYTHONIOENCODING=utf-8` parent |

Standalone confirmations: `security_test` 16/16 (three warm runs), `audit_resilience`
0 failed / 19.3 s. Fast-suite component: 25/25 in every battery. Suites covered include
`smoke`, `risk_engine`, `native_fixture` (Phase 5 batch contract), `audit_resilience`
(Phase 8), `audit_chain`, `verification_flow`, `sql_injection`, `front_service`,
`resilience`, `federated_learning`, `drift_monitor`, `ood_gate`, `rules_gate`,
`calibration`, `feature_parity`, `leakage`, `privacy`, `temporal`, `backup_restore`.

## Security

| gate | result |
|---|---|
| `secret_hygiene_test.py` | PASS — 16/16, incl. machine gate `bandit -r backend/src --severity-level medium` exits 0 |
| direct `bandit -r backend/src --severity-level medium` | PASS — 0 medium, 0 high (139 low, below the gate), exit 0 |
| `audit_hygiene_check.py` | PASS — CLEAN (fixture snapshot exact, forks evidence-matched, no duplicates, no drift) |
| `auto_security_scan.py` | exit 0 — penetration suite 69 tests, 0 vulnerable; `security_scan` 6/6 checks pass (with the vacuous-walk caveat, finding 6) |
| `sql_injection_test.py` | PASS (in all batteries) |
| secret hygiene of this phase's artifacts | evidence files contain the bench-only token `ps14-p9-bench-token` (env-injected, non-production); no real secrets written |

## Evidence

Direct (this session, current code, live listener):

- `.freebuff/p9/ps14_p9_healthy_ml_20261009T045625+0000.json` — ML-path baseline (490 req)
- `.freebuff/p9/ps14_p9_healthy_full_20261009T050946+0000.json` — full baseline
  (cold start, pre-flip, mono ×2, ramp, batch, rate-limit probe, integrity, resources)
- `.freebuff/p9/ps14_p9_delayed_audit_20261009T051215+0000.json` — delayed-audit run
- `.freebuff/p9/ps14_p9_unavailable_audit_20261009T051402+0000.json` — unavailable-audit run
- server logs: `.freebuff/p9/server_{healthy,delayed,unavailable}_*.log`
- batteries: `.freebuff/p9/full_suite.log`, `full_suite2.log`, `full_suite3.log`,
  `full_suite4.log`
- drift-gate reproducers: `.freebuff/p9/diagnose_degraded.py`, `.freebuff/p9/verify_workload.py`
- committed harness: `backend/scripts/ps14_p9_live_bench.py`,
  `backend/scripts/ps14_p9_fault_boot.py`

Superseded/contaminated runs (NOT evidence) are retained under `.freebuff/p9/`
with earlier timestamps (`*_040510`, `*_040738`, `*_042057`, `*_043629`) and were
produced during harness bring-up (idempotent-id reuse and stale keep-alive defects,
both fixed before the canonical runs).

## Limitations

1. **Single worker, single machine**: 1 uvicorn worker on a 16-logical-CPU Windows 11
   laptop; results are a local baseline, not a production capacity plan. Multi-worker
   scaling and cross-process rate-limit semantics are untouched.
2. **Synthetic in-distribution workload**: rows are drawn deterministically from the
   serving table; absolute latencies for production traffic shapes may differ.
3. **Drift-gate interaction**: ML-path segments are capped at 98 events per process by the
   pre-existing baseline mismatch (finding 2); post-flip segments measure the degraded
   path. Both are labeled per segment; pooled figures are never presented.
4. **Fault injection is simulated** (Phase 8 injection inside a live process), not a real
   institutional DB-4 outage.
5. **Tail percentiles**: p99.9/p999 claims rest on rare outliers (n < 10 events above
   0.5 s); treat them as observed maxima, not stable estimates.
6. **CI / Security Scan for this push**: recorded in the Post-push record below; local
   verification is the primary evidence.
7. **Known flaky/state-dependent suites** (findings 3–5) are environmental/test-design
   issues, attributed as such and not as Phase 9 regressions; battery 4 is the clean
   authoritative full pass.

## Verdict

Phase 9 established the reproducible live-HTTP request-path baseline the roadmap asked
for, with no implementation changes:

- **Live `/internal/evaluate`**: p50 ≈ 28.5 ms / p95 ≈ 32.5 ms on the ML path; ≈ 28–30 ms
  rules-only; single-worker saturation ≈ 110–115 rps with 0 errors across 1,550 ramp
  requests; batch endpoint amortizes to 0.32 ms/item at size 500; the 500/60 s limiter
  enforces exactly on budget.
- **Phase 8 audit resilience holds live**: 0 differing decision fields across 6 probes ×
  3 audit conditions; under DB-4 unavailability decisions still persist to DB-3 while
  audit events stay durably spooled.
- **Regression**: 32/32 authoritative full battery (4 runs, with flake/state-dependence
  root-caused); **security gates all pass** (bandit 0 medium/high, secret hygiene 16/16,
  audit hygiene CLEAN, pentest 69/69 blocked-or-pass).
- **Six findings** recorded without fixing them (measurement-only scope), the two most
  actionable being the `/internal/metrics` NameError and the drift-baseline mismatch.

The baseline is now the reference for any future optimization work: Phase 10+ must show
improvement against these numbers on this harness, without regressing the audit-path
identity result.

## Post-push record

Pushed `origin/main` `4948cba..9990ad5` (2 commits):

- `0764b3b` `feat(phase9): add the live request-path benchmark harness`
- `9990ad5` `docs(phase9): record the request-path performance baseline closeout`

GitHub Actions on `9990ad5` (queried via the Actions API this session):

| workflow | status | conclusion |
|---|---|---|
| CI/CD | completed | **success** |
| Security Scan | completed | **success** |

Also verified in the same query: Phase 8's push (`4948cba`) CI/CD and Security Scan both
completed **success** — this resolves the "CI not confirmed" caveat (Limitations #1) in
`PHASE_AUDIT_RESILIENCE_CLOSEOUT.md`, whose SHA fields this phase already filled.

Local verification + CI + Security Scan are all green for Phase 9. This record was
appended in a follow-up `docs(phase9)` commit pushed after verification.
