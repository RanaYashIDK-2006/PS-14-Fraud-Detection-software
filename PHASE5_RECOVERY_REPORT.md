# PHASE5 RECOVERY REPORT

Cloudflare **HTTP 524** (origin timeout) interrupted the Phase-5 Training-Coverage
Redesign mission during its *initial infrastructure-mapping step* (read-only file
inspection). Machine-readable record: `reports/phase5_recovery.json`.

## Cause of the 524

The HTTP request that carried the Phase-5 session timed out at the proxy layer.
The mission had not yet launched any experiment job — the only work done was
reading infrastructure files (coverage-frame schemas, merchant volume tables).

## Did the backend survive?

**Yes.** All six services were confirmed up and healthy after the restart:

| Service | Port | State |
|---|---|---|
| front | 8000 | ok |
| identity | 8001 | ok |
| privacy | 8002 | ok |
| risk | 8003 | ok — serving `altman_native_E_hardneg_cert_20260904` @ 0.018758 |
| verify | 8004 | ok |
| audit | 8005 | ok |

Each runs as the documented venv-launcher → winget-python listener pair.

## Orphan process found and terminated

A single orphaned heredoc job (`python -` launched 19:15 from bash pid 5380,
pre-524) was still alive: pid 7532 with launcher 16812. It had burned ~9,755
CPU-seconds, held an 11 MB working set, and had produced **no disk output after
20:44** — a stuck job, not a Phase-5 worker and not one of the six services.
Terminated (`taskkill //PID 7532 //F`); the launcher exited with its child.
No duplicate Phase-5 work existed, so nothing was lost or doubled.

## Stage status

| Component | Status |
|---|---|
| Data inventory | COMPLETE — all Phase-4 frames parse cleanly |
| Merchant universe | COMPLETE — `_merchant_table.npz` + `_merchant_volume.npz` |
| Coverage 5% | COMPLETE — `cov_5`, `mission_E_hardneg` |
| Coverage 10% | COMPLETE — `cov_10` |
| Coverage 20% | COMPLETE — `cov_20` |
| Coverage 40% | COMPLETE — `cov_40` |
| Fixed evaluation population | COMPLETE — `reports/coverage_fixed_population.json` |
| Establishment features | COMPLETE for 5% frame (`_cs_frame.npz`, 54-feat, leakage-audited) |
| Causal tests | COMPLETE (Phase 3, max \|Δ\| ≈ 2×10⁻⁸) |
| Label-latency audit | COMPLETE — **UNVERIFIED** (unchanged) |
| Production parity | DOCUMENTED GAP (decision-proxy labels vs ground truth) |
| Strategy C frame | **NOT STARTED** |
| Ablations A/B/C/D/E | **NOT STARTED** |
| Phase-5 training/validation/reports | **NOT STARTED** |

## Artifact integrity

All seven `.npz` frames and all five model records verified parseable with the
expected row counts; no corruption, no partial JSON, no checkpoint to recover.

## What was NOT done

No reruns of completed deterministic stages. No changes to sampling strategy,
validation population, threshold, feature definitions, model parameters, or the
final-test procedure. **LABEL LATENCY remains UNVERIFIED.**

## Resume

Phase 5 restarts from its first incomplete stage: the **Strategy-C merchant-aware
frame build** (with merchant-size caps + fraud preservation + chronological
validity), then coverage-manifest, establishment-feature extension, and the
A/B/C/D/E ablation on a fixed population. All expensive stages run **detached**
with per-stage checkpoints so a request timeout cannot kill the job.
