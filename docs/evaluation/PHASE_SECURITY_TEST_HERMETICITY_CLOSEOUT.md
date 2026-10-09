# Phase 10 — Hermetic Security-Test Fixes — Closeout

## Phase identity

- **Phase**: 10 — Hermetic Security-Test Fixes
- **Previous-phase report**: `6c24cb6` (Phase 9 closeout content SHA)
- **Actual Start SHA**: `ec17560` — verified with `git rev-parse HEAD` / `origin/main` at
  session start; three Phase 9 evidence commits (`f4cd4ca`, `dcb283d`, `ec17560`) landed
  after `6c24cb6`. The working tree was **clean** at baseline; no SHA was assumed.
- **Implementation SHA**: `c95917b` (`test(phase10): make security_test.py hermetic and environmentally honest`)
- **Final SHA**: `<pending — set in the CI record below after push>`
- **Verification window**: 2026-10-09 (this session)

## Baseline

Repository state recorded before any change:

- `git status --short` → empty (clean; Phase 9 had committed all
  evaluation-ledger/calibration records in `f4cd4ca`/`dcb283d`/`ec17560`)
- `git rev-parse HEAD` = `git rev-parse origin/main` = `ec17560`
- history: `ec17560 → dcb283d → f4cd4ca → 6c24cb6 → 9990ad5 → 0764b3b` (Phase 9 chain
  on top of Phase 8's `4948cba`)

Existing security-test commands and documented prerequisites:

| command | prerequisite (documented) | source |
|---|---|---|
| `python backend/scripts/security_test.py [--url …]` | identity service reachable for 13 of 16 checks; key-separation checks are offline | `security_test.py` docstring |
| `python backend/scripts/regression_suite.py --fast` | "no live services needed" — security_test is **not** in FAST | `regression_suite.py` docstring |
| `python backend/scripts/regression_suite.py --full` | "requires running stack"; `security_test` runs in FULL with `PYTHONIOENCODING=utf-8` | `regression_suite.py` docstring + `FULL_TESTS` |
| `python backend/scripts/sql_injection_test.py` | running stack (live-service checks) | suite membership |

Prerequisite note (recorded, unchanged): the runner sets `PS14_MODE=development` for
child processes, but the *live* services' environment comes from the stack
(`start_stack.ps1` merges `.env`, which sets `PS14_MODE=development`), so the
development-mode guard in `rate_limit_middleware` is active on this stack.

## Root cause

`DEMONSTRATED` — reproduced live this session before any edit:

`backend/scripts/security_test.py` `RateLimitTest` sent **exactly 10** failed logins and
required a 429 among them. The identity service (`backend/src/identity_service/main.py`)
enforces the only 429-producing control reachable on this stack:

- login brute-force lockout: `_LOGIN_LOCKOUT_THRESHOLD = 10`, window/duration 900 s,
  keyed by client IP;
- `_check_login_brute_force()` runs **before** `_record_login_failure()`.

Consequences:

- **Cold service** (lockout list empty — all failures aged out of the 15-minute window):
  each of the 10 attempts passes the check (list 0→9, the 10th failure recorded only
  *after* its own check) → all 401 → the test exits its loop having never seen a 429 →
  `[FAIL] Block after rate limit exceeded / Never blocked`, exit 1.
- **Warm service** (≥10 failures recorded within 15 min by any *previous* run): attempt 1
  trips the lockout → 429 → PASS, exit 0.

The middleware rate limiter (5/60 s on `/auth/login`) would fire at attempt 6, but it is
deliberately skipped outside production mode, so on this dev stack it never runs. The net
effect: **the test's result depended only on whether some earlier run had warmed the
lockout** — order-dependent, not security-dependent.

Two secondary defects, both in scope of "test isolation / reporting":

1. **Service unavailability was reported as failed security assertions**: every live
   section caught `Exception` generically and logged `False`, so a down identity service
   looked identical to a real control failure (the Phase 8 closeout had to footnote this
   as "environmental").
2. **The target was not retargetable**: `--url` existed, but ports were hardcoded as a
   `:8001` suffix, so `--url http://host:8099` produced an invalid double-port URL —
   unreachable-target scenarios could not be exercised deterministically.

### Before evidence (this session, unedited file)

| run | result | evidence |
|---|---|---|
| run 1 (cold) | **exit 1**, 15/16, "Never blocked" | `.freebuff/p10/before_run1_cold.log` |
| run 2 (immediately after) | **exit 0**, 16/16 | `.freebuff/p10/before_run2_warm.log` |

Identity access log: run 1 produced 10× `401 Unauthorized`, run 2 produced `429 Too Many
Requests` on its first attempt — the same command, opposite verdicts, differing only in
inherited server state. Phase 9's four regression batteries show the same alternation
(cold-fail in batteries 1 and 3, warm-pass in 2 and 4).

## Fix (files changed)

**Only `backend/scripts/security_test.py`.** No service code, no security control, no
assertion was touched or weakened — a 429 is still *required* for the rate-limit check to
pass.

1. **Deterministic state establishment** — `_rejection_budget()` computes
   `max(lockout_threshold, middleware_login_limit) + 1` by reading both constants from
   their source files, so the test drives enough attempts to observe a 429 from *any*
   starting state: cold → 429 at attempt 11; warm/blocked → attempt 1; production-mode
   middleware → attempt 6. A source-formatting change fails loudly (exception → exit 1)
   rather than silently shrinking the budget.
2. **Environmental classification** — `log_env()` records a third state
   (`passed=None, env=True`); `_is_unreachable()` maps `httpx.TransportError` in every
   live section to `ENVIRONMENTAL` instead of `FAIL`; an upfront readiness probe prints
   the condition banner. Non-transport exceptions still fail as assertions.
3. **Distinct exit codes** — `0` all evaluated & passed; `1` security assertion failed;
   `2` nothing failed but the service was unreachable (security NOT verified). The
   summary prints `Environmental (not evaluated): N` and refuses to print
   `ALL SECURITY TESTS PASSED` when any check went unevaluated.
4. **Retargetable identity origin** — `--url` is now a full origin
   (default `http://127.0.0.1:8001`), removing the hardcoded `:8001` suffix, enabling
   deterministic unreachable-target runs.
5. **Evidence visibility** — the 429 attempt number prints on PASS too, recording which
   starting state each run exercised.

**Deliberately not done**: isolating lockout state via a service-side reset. The lockout
is IP-keyed, in-process server state; the only way to reset it would be to add a
reset/test hook to the authentication service — itself a new security surface. Per
"smallest changes necessary" and "do not perform broad security hardening", the test
instead became idempotent from any starting state. Documented as a limitation.

**Follow-up finding (recorded, NOT fixed — Phase 9 §4 / mission §10.4)**: `GET
/internal/metrics` in `backend/src/risk_engine/main.py` raises
`name 'Response' is not defined` (missing import) and returns 500 via the
`SecurityHeadersMiddleware` generic handler. A minimal test-isolation fix does not
necessarily touch it, so it was **not** bundled into Phase 10; it remains a Phase 10
candidate follow-up for a separate change.

## Before/after reproducibility

`DEMONSTRATED` (live identity service, both directions):

| scenario | before | after |
|---|---|---|
| run twice consecutively | exit 1 then exit 0 (state-dependent) | exit 0, exit 0, exit 0 |
| cold service (fresh lockout state) | exit 1 "Never blocked" | exit 0 — `429 after 11 of 11 attempts` |
| warm/blocked service | exit 0 (attempt 1) | exit 0 — `429 after 1 of 11 attempts` |
| service unreachable | exit 1, failures indistinguishable from control failures | exit 2, banner + `ENVIRONMENTAL`, 0 failed |

## Validation results

All runs this session; commands from project root unless noted
(`cd backend && PYTHONIOENCODING=utf-8 PYTHONPATH=$PWD ../.venv/Scripts/python.exe …`).

| # | validation | command / condition | exit | result |
|---|---|---|---|---|
| 1 | isolated execution | `security_test.py` | 0 | 16/16 PASS (warm, 429 after 1 of 11) |
| 2 | consecutive repeated ×3 | three back-to-back runs | 0,0,0 | 16/16 each; log `.freebuff/p10/after_consecutive_{1,2,3}.log` |
| 3 | after authentication/lockout activity | 10 lockout-triggering POSTs (all 429) then run | 0 | 16/16, 429 after 1 of 11 — `after_lockout_burn.log` |
| 4 | cold path (fresh client state) | `--url http://10.164.84.132:8001` (different client IP → empty lockout list) | 0 | 16/16, **429 after 11 of 11**; access log 10×401 then 429 — `after_cold_via_lan.log` |
| 5 | services unavailable | `--url http://127.0.0.1:8099` (nothing listening) | **2** | banner + 4 `ENVIRONMENTAL` entries, `Passed: 3` (offline key-separation still evaluated), `Failed: 0`, "security NOT verified" — `after_dead_target.log` |
| 6 | fast regression suite | `regression_suite.py --fast` | 0 | **25/25** |
| 7 | normal regression-suite order | `regression_suite.py --full` | 0 | **32/32** (192.9 s); `security_test` PASS (0.7 s) inside the suite — `.freebuff/p10/full_battery.log` |
| 8 | security gate: secret hygiene | `secret_hygiene_test.py` (incl. bandit machine gate) | 0 | 16/16 |
| 9 | security gate: audit hygiene | `audit_hygiene_check.py` | 0 | CLEAN |
| 10 | security gate: auto scan | `auto_security_scan.py` (bandit + self-authored pentest) | 0 | 0 vulnerable; pentest suite complete |

Note on #5: unavailability was exercised by pointing at a port with no listener
(`SIMULATED` unreachable target) rather than stopping the shared development stack —
the client-side code path (connection refused → `httpx.ConnectError`) is identical, and
no running service was disturbed. Regression-suite semantics for exit 2 are unchanged
(non-zero still fails the suite runner, as it must — the run did not verify security);
only the *reporting* now distinguishes environmental from assertion failures.

## Unrelated / environmental failures

None. Every failure-free run above was executed and recorded honestly; the only
non-zero exits in this phase are the **before** reproduction (exit 1, the defect being
fixed) and the **deliberate** unreachable-target run (exit 2, correct environmental
reporting). No unrelated suite regressed: fast 25/25, full 32/32.

## Inherited working-tree artifacts preserved

At Phase 10 baseline the tree was clean — Phase 9 had already committed the accumulated
evaluation-ledger/calibration records. No inherited uncommitted records existed to
preserve; any hook-regenerated calibration record appearing during this phase is
left unstaged and untouched (only the two Phase 10 files are staged at commit time).

## Evidence classification

- Before/after behavior, cold/warm paths, suite-order and gate runs: **DEMONSTRATED**
  (real service, real HTTP, exit codes captured).
- Unreachable-target run: **SIMULATED** unavailability (no listener on the target port;
  no production service stopped).
- Budget derivation from source constants: **SELF-TESTED** (test reads its own
  service's configuration; fails loud on drift).
- CI/CD and Security Scan conclusions: recorded in *Commit / CI* below — **BLOCKED**
  if the remote API is unavailable at verification time.
- Not established here: behavior against a real institutional outage or a
  production-mode (middleware-limiter-active) deployment — **NOT ESTABLISHED** for this
  phase (the production-mode branch of the budget is derived, not exercised).

## Limitations

1. Lockout isolation is achieved by state-independent determinism, not by resetting
   server state; the server's IP-keyed lockout persists across test runs (by design —
   it is a real security control). Adding a reset hook was deliberately rejected as a
   new auth-service surface.
2. The cold path requires either an empty lockout list (≥15 min idle) or a fresh client
   IP; both were exercised (#4, and #1-3 cover warm), but a CI machine without a live
   stack gets exit 2 rather than an assertion verdict (environmental, by design).
3. `_rejection_budget()` parses source with regex; a refactor of the constant
   declarations fails the test loudly (acceptable: fail-closed) but would need a
   one-line matcher update.
4. `regression_suite.py` was not modified: it still maps any non-zero exit to FAIL.
   Distinguishing exit 2 at the runner level is possible follow-up work, deliberately
   out of scope here.
5. The `/internal/metrics` `Response` NameError remains open (recorded above as a
   follow-up finding; not fixed in this phase).

## Commit / CI

- staged: `backend/scripts/security_test.py` (commit `c95917b`),
  `docs/evaluation/PHASE_SECURITY_TEST_HERMETICITY_CLOSEOUT.md` (this commit) —
  Phase 10 files only; hook-regenerated calibration records deliberately left unstaged
- Implementation/Final SHA + CI/CD + Security Scan outcomes: see the lines below after
  push.

## Verdict

Phase 10 root-caused and removed the order dependence of `security_test.py` without
touching any security control or service code: the test now establishes the lockout
state it needs (budget `max(thresholds)+1` derived from source), passes identically in
isolated, consecutive, post-lockout, cold, warm, and suite-order execution
(before: exit 1 → exit 0 for the same command; after: 10/10 runs exit 0 across those
conditions), reports service unavailability as `ENVIRONMENTAL` with a distinct exit code
instead of fake assertion failures, and all fast/full regression and security gates are
green (25/25, 32/32, 16/16, CLEAN, scan 0).
