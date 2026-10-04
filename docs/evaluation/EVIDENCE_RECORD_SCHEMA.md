# Evidence Record Schema — Canonical Formats (Phase 105)

**Status:** active, enforced by `backend/scripts/claim_evidence_check.py` and
`backend/scripts/eval_record_test.py` (both run in CI).

There is **one canonical evidence path**:

```
execute experiment → artifact → evaluation record (append-only ledger)
        → claims-registry entry → documentation references the claim id
```

Two record types exist; neither is redundant:

1. **EvaluationRecord** — written *by executing code* (never hand-typed); binds one
   evaluation run to its data, model, git state and command.
2. **Claims-registry entry** — binds one *documented number* to an EvaluationRecord or
   a tracked artifact, and carries its evidence classification.

---

## Part 1 — EvaluationRecord (ledger schema v1.1)

**Storage:** `reports/evaluation_runs/eval_ledger.jsonl` — one JSON object per line,
**append-only**. `EvaluationLedger` never mutates or removes a line; a re-run appends a
new `evaluation_id`. Companion file per record: `record_<evaluation_id>.json`.

**Code:** `backend/scripts/eval_record.py` (`EVALUATION_RECORD_VERSION = "1.1"`).
v1.0 records remain valid legacy entries (they simply lack `command`).

### Fields

| Field | Required | Meaning | Validation rule |
|---|---|---|---|
| `evaluation_id` | yes | `eval-<UTC yyyymmddThhmmss><+0000>-<12-hex digest>`; digest covers timestamp, model hash, dataset hash, and record-content digest | unique across the ledger |
| `timestamp_utc` | yes | run time (UTC, second precision) | ISO-8601 |
| `model_identifier` | yes | human label of the evaluated system | non-empty |
| `model_hash` | yes* | aggregate SHA-256 over per-file artifact hashes (`null` only when no artifact files, e.g. research models trained in-process — record must then say so in `warnings`) | matches recomputation for persisted artifacts |
| `artifact_file_hashes` | yes | per-file SHA-256 of every model artifact | any byte change ⇒ new hash |
| `preprocessing_version` | yes | preprocessing contract version | string |
| `feature_schema_version` | yes | feature contract version (`1.0`/`v1`…) | string |
| `dataset` | yes | `{path, sha256, size_bytes, exists}` over the **raw file bytes** | `sha256` non-null for DEMONSTRATED records |
| `training_dataset` | no | same shape for training data | — |
| `seed` | yes* | RNG seed of the run; if genuinely inapplicable, set `seed` to `null` **and** put a non-empty `seed_not_applicable` reason in `evaluation_config` | one of the two must hold for DEMONSTRATED |
| `threshold` / `threshold_source` | yes | operating threshold and its provenance (`validation` \| `fixed` \| `none`); `evaluate.py` **refuses** `test`/`external`/`holdout` | no test-set tuning |
| `evaluation_config` | yes | split description, script parameters, gate results, `seed_not_applicable`, etc. | must identify the split for DEMONSTRATED |
| `git_commit` | yes | `git rev-parse HEAD` at run time | non-null for DEMONSTRATED |
| `software` | yes | python/numpy/pandas/sklearn/xgboost/scipy/platform versions | best-effort, never fabricated |
| `metric_definitions_version` | yes | metric semantics version the numbers follow | string |
| `metrics` | yes | the measured values (nested freely; **numbers come from execution only**) | numeric; registry resolves via `json_path` |
| `command` | yes for v1.1 (optional legacy) | exact invocation, e.g. `python backend/scripts/eval_ulb.py` | non-empty for DEMONSTRATED |
| `warnings` | yes | honesty notes (small samples, in-sample metrics, unpersisted models, failed gates…) | list |
| `status` | yes | `COMPLETED` \| `FAILED` — failures are preserved exactly like successes | DEMONSTRATED requires `COMPLETED` |
| `record_version` | yes | `1.0` (legacy) or `1.1` | — |

### Classification rules (how a record's evidence may be classified)

| Classification | Requirement |
|---|---|
| **DEMONSTRATED** | record `status=COMPLETED` **and** `dataset.sha256` **and** `git_commit` **and** `command` **and** (`seed` or `seed_not_applicable`) — enforced by `claim_evidence_check.py` |
| **SELF-TESTED** | ledger/artifact/suite exists; provenance may be partial (e.g. no seed/git in a historical artifact) |
| **SIMULATED** | measurement from a simulation (federated sim, synthetic institutions) — record must say so in `evaluation_config`/`warnings` |
| **NOT ESTABLISHED** | *no* usable evidence; registry entry carries `evidence.type=none` + a `reason` |
| **BLOCKED** | evidence impossible right now; registry entry carries `evidence.type=blocked` + a `blocker` |

### Artifact requirements

- Metric-bearing artifacts are **JSON written by the executing script** (e.g.
  `reports/ulb_results.json`, `reports/calibration_test/calibration_metrics.json`,
  `reports/cross_dataset/cross_dataset_report.json`); they embed `hash`/`split`/`command`
  where applicable.
- Binary model artifacts are identified by SHA-256 inside the record — the record never
  trusts a filename.
- Historical artifacts (pre-Phase-39, e.g. `misc/reports/phase24/...`) are admissible as
  `evidence.type=artifact`; if they lack dataset/split/producer metadata, the claim they
  back cannot exceed **SELF-TESTED** unless the registry supplies a complete
  `provenance` block (`dataset_sha256`, `split`, `producer`, and `seed` or
  `seed_not_applicable`).

### Example (real record, abridged — `eval-20261003T084934+0000-e58d68c2bf88`)

```json
{
  "evaluation_id": "eval-20261003T084934+0000-e58d68c2bf88",
  "timestamp_utc": "2026-10-03T08:49:34+00:00",
  "model_identifier": "ps14_ulb_research_eval",
  "model_hash": null,
  "dataset": {"path": "data/creditcard.csv",
              "sha256": "76274b691b16a6c4…", "size_bytes": 150828752, "exists": true},
  "seed": 42,
  "threshold": null, "threshold_source": "none",
  "evaluation_config": {"split": "random_stratified_80_20_test_size=0.2_random_state=42",
                        "script": "backend/scripts/eval_ulb.py", ...},
  "git_commit": "a0b600ba2edd917498d3a9551c19e10225930e4b",
  "command": "python backend/scripts/eval_ulb.py",
  "metrics": {"results": {"pattern_xgb": {"roc_auc": 0.975807, "pr_auc": 0.883465,
                                          "r1": 0.918367}, ...}},
  "warnings": ["model artifacts not persisted — reproduction is by re-running the command"],
  "status": "COMPLETED",
  "record_version": "1.1"
}
```

### Invalid record examples (structural — rejected by the checker)

- `classification: DEMONSTRATED` bound to a record with `"git_commit": null`.
- `classification: DEMONSTRATED` bound to a v1.1 record with `"command": null` (a legacy
  v1.0 record may exist, but it cannot back a DEMONSTRATED claim).
- `seed: null` with no `evaluation_config.seed_not_applicable`.
- A registry `evaluation_id` that has no line in the ledger.
- A `json_path` that does not resolve inside the record/artifact.
- `documented_value` differing from the evidence value by more than `value_tolerance`.

---

## Part 2 — Claims-registry entry

**Storage:** `docs/evaluation/claims_registry.jsonl` — one JSON object per line.

### Fields

| Field | Required | Meaning |
|---|---|---|
| `claim_id` | yes | unique, matches `C-\d{3,4}` |
| `claim` | yes | plain-language statement of the quantitative claim |
| `documented_in` | yes | repo-relative file that carries the claim; must contain a `<!-- claims:C-NNN -->` (or `claim:`) annotation for this id |
| `metric` | yes | metric name (`roc_auc`, `pr_auc`, `recall_at_1pct_fpr`, `brier`, `ece`, `fpr`, …) |
| `documented_value` | yes | the number as documented (e.g. `0.918` for "91.8%") |
| `value_tolerance` | yes | max allowed \|documented − evidence\| (choose display-rounding precision) |
| `classification` | yes | `DEMONSTRATED` \| `SELF-TESTED` \| `SIMULATED` \| `NOT ESTABLISHED` \| `BLOCKED` |
| `evidence` | yes | object, see types below |
| `provenance` | for DEMONSTRATED+artifact | `dataset_sha256`, `split`, `producer`, and `seed` or `seed_not_applicable` |
| `notes` | recommended | scope caveats, supersession notes |

### Evidence types

| `evidence.type` | Fields | Semantics |
|---|---|---|
| `ledger` | `evaluation_id`, `json_path` | resolves the value inside an append-only ledger record; the only evidence type that can be minted by running code this phase |
| `artifact` | `path`, `json_path` | resolves the value inside a tracked JSON artifact |
| `suite` | `path`, `expect` | a test-suite output file containing the expected marker (supports SELF-TESTED at most) |
| `none` | `reason` | explicit *absence* of evidence — only valid with `NOT ESTABLISHED` |
| `blocked` | `blocker` | explicit blocker — only valid with `BLOCKED` |

### Documentation annotations

- Managed regions in a surface file: `<!-- claims:managed:start … -->` …
  `<!-- claims:managed:end -->`. Every **quantitative table row** (a row with a bare
  numeric cell) inside must carry `<!-- claims:C-001 -->` or
  `<!-- claims:C-001 C-002 -->` (multi-metric rows).
- Rows are checked so the registered `documented_value` actually appears in the row —
  stale documentation (value changed in docs but not registry, or vice versa) fails.
- Annotations are also allowed outside managed regions (they then satisfy
  `documented_in` without being coverage-scanned).

### Valid example (real entry, `C-002`)

```json
{"claim_id": "C-002",
 "claim": "ULB research benchmark (pattern XGB) Recall@1%FPR 91.8%",
 "documented_in": "README.md",
 "metric": "recall_at_1pct_fpr",
 "documented_value": 0.918, "value_tolerance": 0.0005,
 "classification": "DEMONSTRATED",
 "evidence": {"type": "ledger",
              "evaluation_id": "eval-20261003T084934+0000-e58d68c2bf88",
              "json_path": "metrics.results.pattern_xgb.r1"},
 "notes": "Historical README value 91.8% reproduced exactly (0.918367)."}
```

### Valid example of an honest *absence* (real entry, `C-104`)

```json
{"claim_id": "C-104",
 "claim": "Historical headline: FPR 1.996% on ULB",
 "documented_in": "docs/PHASE_105_EVIDENCE_BASELINE.md",
 "metric": "fpr",
 "documented_value": 0.01996, "value_tolerance": 0.00005,
 "classification": "NOT ESTABLISHED",
 "evidence": {"type": "none",
              "reason": "Substring '1.996' occurs in misc/reports only as unrelated digits…"},
 "notes": "Historical value retained for traceability."}
```

### Invalid examples (each fails `claim_evidence_check.py`)

1. **Typed metric without execution** — `{"documented_value": 0.966, "classification":
   "DEMONSTRATED", "evidence": {"type": "none", "reason": "trust me"}}` →
   `evidence.type=none is only valid for NOT ESTABLISHED`.
2. **Registered claim never annotated in its document** →
   `C-0XX: README.md contains no <!-- claim:C-0XX --> annotation`.
3. **Value drift** — registry says `0.966` but the evidence (or the annotated row)
   says `0.975807` → `documented … != evidence …` / `row does not contain the
   registered value`.
4. **Unregistered quantitative row in a managed region** →
   `… lacks a <!-- claim:C-NNN --> annotation`.
5. **DEMONSTRATED with partial provenance** — artifact evidence without
   `provenance.split`, or a ledger record without `command`.

### Growth rules

- New evidence fields are added only when the architecture needs them (v1.1 added
  exactly one: `command`).
- New surfaces (files/regions) are added deliberately; unscanned prose is a
  **documented limitation**, never an implicit guarantee.
- The registry never replaces the ledger: a claim may point at a historical artifact,
  but only an executed, provenance-complete record can carry `DEMONSTRATED` via
  `ledger` evidence.
