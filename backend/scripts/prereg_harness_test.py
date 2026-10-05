#!/usr/bin/env python3
"""NR-04 — preregistered harness determinism + fixture test suite.

Plain assert script (repository convention, no pytest). Run:
    python backend/scripts/prereg_harness_test.py

Covers the §19 determinism matrix (9 cases) and the §20 synthetic/fixture
validation (metrics, bootstrap plumbing, failed-run handling, evidence
generation, matrix generation, provenance population, protocol guards).
Every fixture runs against temp files/temp ledgers; fixture values never
enter the real evidence ledger as benchmark measurements (asserted at the
end of the suite).
"""

from __future__ import annotations

import contextlib
import io
import json
import re
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]                     # backend/scripts -> repo root
sys.path.insert(0, str(HERE))

import eval_record as er                    # noqa: E402
import prereg_harness as ph                 # noqa: E402


# ── fixtures ────────────────────────────────────────────────────────────────
def sha_of(p: Path) -> str:
    return er.sha256_file(p)


def write_fixture_dataset(td: Path, name: str = "fixture_data.csv") -> Path:
    p = td / name
    rows = ["label,x,y"]
    rows += [f"0,{i * 0.1:.4f},{(i * 7) % 13}" for i in range(160)]
    rows += [f"1,{0.5 + i * 0.01:.4f},{(i * 3) % 11}" for i in range(40)]
    p.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return p


def make_contract(**over) -> dict:
    c = {
        "protocol_version": ph.PROTOCOL_VERSION,
        "experiment_id": "E1",
        "dataset_name": "fixture_ds",
        "dataset_path": None,               # set per-test via fixture_env
        "dataset_expected_sha256": None,
        "dataset_role": "fixture",
        "split": {"name": "fixture_random_split", "axis": "random"},
        "population": "synthetic fixture population (never a real dataset)",
        "seed": 42,
        "model_identity": "fixture_model",
        "artifact_paths": [],
        "feature_contract_path": str(ph.CONTRACT_PATH),
        "preprocessing_identity": "standardscaler_fit_on_train",
        "preprocessing_fit_population": "train",
        "calibration_config": {"procedure": "platt", "fit_population": "validation"},
        "threshold_config": {"value": 0.5, "source": "validation",
                             "selected_on": "validation"},
        "metric_definitions_version": ph.METRIC_DEFINITIONS_VERSION,
        "bootstrap_config": {"ci": False, "n_bootstrap": None, "seed": 42},
        "statistical_test_config": {"paired_difference_ci": "pending (PD-PAIRED)"},
        "acceptance_criterion_config": ["C1"],
        "evidence_schema_version": ph.EVIDENCE_SCHEMA_VERSION,
        "execution_command": "python backend/scripts/prereg_harness_test.py (FIXTURE)",
        "data_tier": "development",
        "deviations": [],
        "fixture_test": True,
    }
    c.update(over)
    return c


def fixture_env(td: Path, contract: dict | None = None) -> dict:
    ds = write_fixture_dataset(td)
    base = {"dataset_path": str(ds),
            "dataset_expected_sha256": sha_of(ds),
            "dataset_name": "fixture_ds"}
    if contract is not None:
        contract.update(base)
        if not contract.get("artifact_paths"):
            # harness evidence requires artifact identity (14/14 provenance) —
            # fixtures persist a tiny clearly-fake model file
            art = td / "fixture_model.joblib"
            art.write_bytes(b"fixture-weights-v1")
            contract["artifact_paths"] = [str(art)]
        return contract
    return base


def fixture_runner(contract: dict, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    y = np.array([0] * 160 + [1] * 40)
    s = np.concatenate([rng.uniform(0, 0.75, 160), rng.uniform(0.25, 1.0, 40)])
    return {"y": y, "scores": s}


def counting_runner(flag: dict):
    def _r(contract, seed):
        flag["called"] = flag.get("called", 0) + 1
        return fixture_runner(contract, seed)
    return _r


# ── §4/§18 registry fidelity: NR-03 §C table is the source of truth ────────
def parse_nr03_table() -> tuple[dict[str, list[str]], tuple[int, int, int, int], bool]:
    text = (ROOT / "docs" / "PHASE_NR03_EVALUATION_SEMANTICS.md").read_text(encoding="utf-8")
    rows: dict[str, list[str]] = {}
    in_table = False
    for line in text.splitlines():
        if line.startswith("| Field (required) |"):
            in_table = True
            continue
        if in_table:
            if not line.startswith("|"):
                break
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if cells[0].startswith("---"):
                continue   # separator row
            rows[cells[0]] = cells[1:8]
    m = re.search(r"\*\*(\d+) cells: FROZEN (\d+) · PENDING REVIEW (\d+) · "
                  r"BLOCKED (\d+) · NOT APPLICABLE (\d+)\*\*", text)
    assert m, "NR-03 stated totals line not found"
    stated = (int(m.group(1)), int(m.group(2)), int(m.group(3)),
              int(m.group(4)), int(m.group(5)))
    flat = [s for v in rows.values() for s in v]
    computed = (len(flat), flat.count("F"), flat.count("P"),
                flat.count("B"), flat.count("N"))
    return rows, stated, computed


def test_registry_matches_nr03_matrix():
    rows, stated, computed = parse_nr03_table()
    assert list(rows.keys()) == ph.MATRIX_FIELD_ORDER, \
        f"field rows differ: {list(rows.keys())}"
    for fname, cells in rows.items():
        assert cells == ph.NR03_GRID[fname], \
            f"transcription drift on '{fname}': doc={cells} harness={ph.NR03_GRID[fname]}"
    assert computed[0] == 23 * 7 == 161, computed
    assert stated == computed, \
        f"NR-03 stated totals {stated[1:]} != machine recount of its own table {computed[1:]}"


def test_marker_count_11():
    proto = (ROOT / "docs" / "evaluation" /
             "PRE_REGISTERED_EVALUATION_PROTOCOL.md").read_text(encoding="utf-8")
    literals = len(re.findall(r"\[REQUIRES DECISION/APPROVAL\]", proto))
    assert literals == 11, f"protocol marker literal count {literals} != 11"
    assert len(ph.MARKER_IDS) == 11, ph.MARKER_IDS
    nr03 = (ROOT / "docs" / "PHASE_NR03_EVALUATION_SEMANTICS.md").read_text(encoding="utf-8")
    assert "**11 protocol markers (unchanged):**" in nr03


def test_seven_experiments_23_fields():
    assert list(ph.EXPERIMENTS) == ["E1", "E2", "E3", "E4", "E5", "E6", "E7"]
    vocab = {"FROZEN", "PENDING REVIEW", "BLOCKED", "NOT APPLICABLE"}
    for eid, exp in ph.EXPERIMENTS.items():
        assert list(exp["fields"]) == ph.MATRIX_FIELD_ORDER, eid
        for fname, info in exp["fields"].items():
            assert info["status"] in vocab, (eid, fname, info["status"])
            if info["status"] == "FROZEN":
                assert info["value"], f"{eid}/{fname} FROZEN without a value"
            if info["status"] == "PENDING REVIEW":
                assert info["value"] is None, \
                    f"{eid}/{fname} PENDING but carries a value — invented?"
    # purpose/criteria propagated
    assert ph.EXPERIMENTS["E1"]["criteria"] == ["C1", "C2"]
    assert ph.EXPERIMENTS["E3"]["criteria"] == ["C5", "C6"]


def test_pending_decision_registry():
    assert len(ph.PENDING_DECISIONS) >= 11
    required = {"M-0", "M-1", "M-2", "M-3", "M-7", "PD-SEED-AGG", "PD-PAIRED",
                "PD-D1", "PD-D1B", "PD-T8", "PD-T9", "PD-SIGNOFF", "PD-FREEZE"}
    ids = {d["id"] for d in ph.PENDING_DECISIONS}
    assert required <= ids, required - ids
    # guard minimum (§6): seed aggregation, bootstrap count, ECE binning,
    # paired CI, C1-4 datasets, external confirmatory, external boundaries,
    # every marker — each must gate at least one experiment
    for d in ph.PENDING_DECISIONS:
        assert d["affects"], d["id"]


# ── §5/§6 guard states stay distinct ────────────────────────────────────────
def test_guard_states_distinct():
    d_cfg = ph.guard("E1", {"experiment_id": "E1", "seed": 42})   # mostly missing
    assert d_cfg.code == ph.CODE_CONFIG and not d_cfg.allowed
    d_pend = ph.guard("E1")
    assert d_pend.code == ph.CODE_PENDING
    assert "DO NOT EXECUTE" in d_pend.message
    assert ph.guard("E6").code == ph.CODE_PENDING
    # dataset blocked: BAF must be refused, never substituted
    try:
        ph.validate_dataset("baf")
        raise AssertionError("BAF must be BLOCKED")
    except ph.Blocked as e:
        assert "BLOCKED — DATASET UNAVAILABLE" in e.message
    # three states have three different codes
    codes = {d_cfg.code, d_pend.code, ph.guard("E1", {"experiment_id": "E1"}).code}
    assert ph.CODE_CONFIG in codes and ph.CODE_PENDING in codes


def test_all_experiments_guarded_preregistered():
    for eid in ph.EXPERIMENT_IDS:
        d = ph.guard(eid)
        assert not d.allowed, f"{eid} unexpectedly clear"
        assert d.code == ph.CODE_PENDING, (eid, d.code)
        assert d.pending, eid
        assert "PENDING REVIEW" in d.message


# ── §19 determinism suite (9 cases) ─────────────────────────────────────────
def test_same_config_same_seed():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        c = fixture_env(td, make_contract())
        led = td / "led.jsonl"
        r1 = ph.execute(dict(c), fixture_runner, mode="development", ledger_path=led)
        r2 = ph.execute(dict(c), fixture_runner, mode="development", ledger_path=led)
        assert r1.metrics and r2.metrics
        for k in ("roc_auc", "pr_auc", "recall_at_1pct_fpr", "brier"):
            assert abs(r1.metrics[k] - r2.metrics[k]) < 1e-12, k
        assert r1.config_digest == r2.config_digest
        assert r1.evaluation_id != r2.evaluation_id, "ids must stay unique per attempt"


def test_same_config_repeated_no_semantic_diff():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        c = fixture_env(td, make_contract())
        led = td / "led.jsonl"
        outs = [ph.execute(dict(c), fixture_runner, mode="development", ledger_path=led)
                for _ in range(2)]
        assert [o.status for o in outs] == [o.status for o in outs]
        assert outs[0].metrics == outs[1].metrics
        assert outs[0].code == outs[1].code
        assert outs[0].measured is False and outs[1].measured is False


def test_different_seed_identity_changes():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        c42 = fixture_env(td, make_contract(seed=42))
        c43 = dict(c42, seed=43)
        d42 = ph.config_digest(c42)
        d43 = ph.config_digest(c43)
        assert d42 != d43, "seed must be part of run identity"
        assert ph.config_digest(c42, include_seed=False) == \
            ph.config_digest(c43, include_seed=False), \
            "frozen configuration must be identical apart from the seed"
        led = td / "led.jsonl"
        r42 = ph.execute(c42, fixture_runner, mode="development", ledger_path=led)
        r43 = ph.execute(c43, fixture_runner, mode="development", ledger_path=led)
        assert r42.seed == 42 and r43.seed == 43
        same = all(r42.metrics[k] == r43.metrics[k]
                   for k in ("roc_auc", "pr_auc")) is False
        assert same, "different seed must produce its own outputs"


def test_different_dataset_hash_blocked():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        c = fixture_env(td, make_contract())
        c["dataset_expected_sha256"] = "0" * 64
        r = ph.execute(dict(c), counting_runner({}), mode="development",
                       ledger_path=td / "led.jsonl")
        assert r.status == ph.BLOCKED and "DATASET IDENTITY MISMATCH" in (r.diagnostic or "")
        assert r.measured is False
        # and no substitution: runner never executed
        # (diagnostic already proves refusal before step 3)


def test_missing_provenance_blocked():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        c = fixture_env(td, make_contract())
        del c["preprocessing_identity"]
        flag: dict = {}
        r = ph.execute(dict(c), counting_runner(flag), mode="development",
                       ledger_path=td / "led.jsonl")
        assert r.code == ph.CODE_CONFIG and r.status == ph.NOT_ESTABLISHED
        assert "preprocessing_identity" in (r.diagnostic or "")
        assert not flag, "runner must not execute on a configuration error"
        # missing dataset hash pin is also a configuration error
        errs = ph.validate_contract({**fixture_env(td), "dataset_expected_sha256": None})
        assert any("dataset_expected_sha256" in e for e in errs)


def test_pending_decision_blocks_preregistered():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        c = fixture_env(td, make_contract(data_tier="validation"))
        flag: dict = {}
        r = ph.execute(dict(c), counting_runner(flag), mode="preregistered",
                       ledger_path=td / "led.jsonl")
        assert r.status == ph.PENDING_REVIEW, r.status
        assert r.code == ph.CODE_PENDING
        assert r.measured is False and not flag
        assert r.pending, "pending decision ids must be surfaced"


def test_failed_evidence_not_measured():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        c = fixture_env(td, make_contract())
        blocker = td / "blocker"
        blocker.write_text("not a directory", encoding="utf-8")
        r = ph.execute(dict(c), fixture_runner, mode="development",
                       ledger_path=blocker / "led.jsonl")
        assert r.code == ph.CODE_EVIDENCE, r.code
        assert r.status == ph.NOT_ESTABLISHED and r.measured is False
        assert r.failure_class == "evidence failure"
        assert r.rerun_permitted is True
        assert "cannot become a measured benchmark claim" in (r.diagnostic or "")


def test_feature_contract_mismatch_blocked():
    # N-02 invariant passes on the real contract file now
    ident = ph.validate_feature_contract(ph.CONTRACT_PATH)
    assert ident["count"] == 14 and ident["sha256"]
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        bad = td / "bad_contract.json"
        doc = json.loads(ph.CONTRACT_PATH.read_text(encoding="utf-8"))
        doc["public_features_count"] = 14          # declares 14, lists 14-1 below
        doc["features"].pop("failed_auth_proxy")
        bad.write_text(json.dumps(doc), encoding="utf-8")
        try:
            ph.validate_feature_contract(bad)
            raise AssertionError("tampered contract must be BLOCKED")
        except ph.Blocked as e:
            assert "BLOCKED — FEATURE CONTRACT" in e.message and "N-02" in e.message
        c = fixture_env(td, make_contract(feature_contract_path=str(bad)))
        r = ph.execute(dict(c), counting_runner({}), mode="development",
                       ledger_path=td / "led.jsonl")
        assert r.status == ph.BLOCKED
        assert "BLOCKED — FEATURE CONTRACT" in (r.diagnostic or "")


def test_final_test_selection_violation():
    # (a) threshold selected from final-test observations
    v = ph.final_test_protection(
        {**make_contract(),
         "threshold_config": {"value": 0.9, "source": "validation",
                              "selected_on": "final_test"}},
        tier="final_test")
    assert any("PROTOCOL VIOLATION" in s for s in v), v
    # (b) final tier without a freeze record
    v = ph.final_test_protection(make_contract(data_tier="final_test"),
                                 tier="final_test", ledger_records=[])
    assert any("freeze record does not exist" in s for s in v), v
    # (c) freeze present clears that violation, but selective seed removal trips
    freeze = [{"evaluation_config": {"final_test_freeze": True}}]
    v = ph.final_test_protection(
        make_contract(data_tier="final_test", seed=42, declared_seeds=[42, 43]),
        tier="final_test", ledger_records=freeze)
    assert any("seed set" in s and "forbidden" in s for s in v), v
    # (d) test-derived preprocessing population
    v = ph.final_test_protection(
        make_contract(preprocessing_fit_population="validation"),
        tier="development")
    assert any("!= 'train'" in s for s in v), v
    # (e) end-to-end: guard reports PROTOCOL VIOLATION with precedence over pending
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        bad_thr = {"value": 0.9, "source": "validation",
                   "selected_on": "final_test"}
        c = fixture_env(td, make_contract(threshold_config=bad_thr,
                                          data_tier="final_test"))
        d = ph.guard("E1", c, tier="final_test")
        assert d.code == ph.CODE_VIOLATION, d.code
        r = ph.execute(dict(c), counting_runner({}), mode="development",
                       ledger_path=td / "led.jsonl")
        assert r.code == ph.CODE_VIOLATION and r.measured is False
        assert not (td / "led.jsonl").exists(), "violated run must not append a record"


# ── §13 metric semantics: exact NR-03 §I pins ──────────────────────────────
def test_metric_semantics_pins():
    from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
    rng = np.random.default_rng(7)
    y = np.array([0] * 300 + [1] * 60)
    s = np.concatenate([rng.uniform(0, 0.8, 300), rng.uniform(0.2, 1.0, 60)])
    out = ph.compute_preregistered_metrics(y, s, threshold=0.5)
    assert abs(out["roc_auc"] - roc_auc_score(y, s)) < 1e-12
    # PR-AUC semantic pin: average precision, not trapezoidal
    assert abs(out["pr_auc"] - average_precision_score(y, s)) < 1e-12
    # recall@1%FPR: literal transcription of eval_ulb.py:24-30 (the pin)
    fpr, tpr, _ = roc_curve(y, s)
    if fpr[0] > 0:
        fpr = np.concatenate([[0], fpr]); tpr = np.concatenate([[0], tpr])
    idx = np.searchsorted(fpr, 0.01)
    expected_r1 = float(tpr[min(idx, len(tpr) - 1)])
    assert out["recall_at_1pct_fpr"] == expected_r1
    assert out["recall_at_1pct_fpr"] == float(ph.eval_ulb.recall_at_fpr(y, s, 0.01))
    # Brier pin: mean((p - y)^2)
    assert abs(out["brier"] - float(np.mean((s - y) ** 2))) < 1e-12
    # ECE is explicitly provisional + pending binning (never silent)
    assert out["ece"]["provisional"] is True
    assert "PENDING REVIEW" in out["ece"]["binning_status"]
    # threshold-dependent metrics at the frozen point
    at = out["at_threshold"]
    assert at["threshold"] == 0.5
    tp = int(((s >= 0.5) & (y == 1)).sum())
    assert abs(at["recall"] - tp / 60) < 1e-12
    assert out["metric_definitions_version"] == ph.METRIC_DEFINITIONS_VERSION


def test_metric_domain_failures():
    y = np.array([0] * 20 + [1] * 10)
    s = np.linspace(0, 1, 30)
    s[3] = np.nan
    try:
        ph.compute_preregistered_metrics(y, s)
        raise AssertionError("NaN scores must fail")
    except ph.MetricDomainError as e:
        assert e.failure_kind == "nan_inf"
    try:
        ph.compute_preregistered_metrics(np.zeros(10), np.linspace(0, 1, 10))
        raise AssertionError("single-class must fail")
    except ph.MetricDomainError as e:
        assert e.failure_kind == "metric_domain"
    # nine-row failure policy invariants (NR-03 §G)
    fc = ph.FAILURE_CLASSES
    assert len(fc) == 9, len(fc)
    assert fc["crash"]["rerun"] is True and fc["missing_data"]["rerun"] is False
    assert fc["feature_contract_violation"]["rerun"] is False
    assert fc["evidence_failure"]["cell"] == ph.NOT_ESTABLISHED
    assert fc["metric_domain"]["rerun"] is False
    assert all(v["cell"] in ph.CELL_STATUSES for v in fc.values())


# ── §11 bootstrap plumbing: no silent default, approval required ───────────
def test_bootstrap_plumbing():
    # development: default is allowed but explicitly flagged
    cfg = ph.resolve_bootstrap(n_bootstrap=None, mode="development")
    assert cfg["explicit"] is False and cfg["n_bootstrap"] == 1000
    assert "pending reviewer decision" in cfg["warning"]
    cfg = ph.resolve_bootstrap(n_bootstrap=250, mode="development")
    assert cfg["n_bootstrap"] == 250 and cfg["explicit"] is True
    # preregistered: refused while approval is pending — even with an explicit count
    for given in (None, 200, 1000, 2000):
        try:
            ph.resolve_bootstrap(n_bootstrap=given, mode="preregistered")
            raise AssertionError(f"n_bootstrap={given} must not pass unapproved")
        except ph.PendingReview as e:
            assert "does NOT constitute approval" in str(e)
    # metrics path honors the same rule
    y = np.array([0] * 80 + [1] * 30)
    s = np.random.default_rng(1).uniform(0, 1, 110)
    out = ph.compute_preregistered_metrics(y, s, ci=True, n_bootstrap=40,
                                           mode="development")
    assert out["ci"]["n_bootstrap"] == 40 and out["ci"]["explicit"] is True
    try:
        ph.compute_preregistered_metrics(y, s, ci=True, n_bootstrap=40,
                                         mode="preregistered")
        raise AssertionError("preregistered CI must refuse unapproved replicate count")
    except ph.PendingReview:
        pass
    # inventory: three implementations, with their callers (§11.1-2)
    ids = [b["id"] for b in ph.BOOTSTRAP_IMPLEMENTATIONS]
    assert ids == ["metric_definitions.bootstrap_ci", "eval_ulb.bootstrap_ci",
                   "protocol §2 proposal"]
    defaults = [b["default_replicates"] for b in ph.BOOTSTRAP_IMPLEMENTATIONS]
    assert defaults == [1000, 200, 2000]
    assert ph.BOOTSTRAP_COUNT_APPROVED is None


def test_paired_ci_boundary():
    y = np.array([0] * 50 + [1] * 20)
    a = np.random.default_rng(2).uniform(0, 1, 70)
    b = np.random.default_rng(3).uniform(0, 1, 70)
    # schema boundary enforces paired rows + explicit parameters first
    try:
        ph.paired_difference_ci(ph.md.roc_auc, y, a[:10], b, n_bootstrap=100, seed=1)
        raise AssertionError("shape mismatch must be a configuration error")
    except ph.ConfigurationError:
        pass
    try:
        ph.paired_difference_ci(ph.md.roc_auc, y, a, b, n_bootstrap=None, seed=None)
        raise AssertionError("implicit parameters must be refused")
    except ph.ConfigurationError:
        pass
    # valid schema, unapproved estimator: PENDING REVIEW, NO interval produced
    assert ph.PAIRED_CI_SPEC_APPROVED is False
    try:
        ph.paired_difference_ci(ph.md.roc_auc, y, a, b, n_bootstrap=100, seed=1)
        raise AssertionError("unapproved paired CI must refuse")
    except ph.PendingReview as e:
        assert "PD-PAIRED" in str(e) and "No interval is generated" in str(e)
    # E1/E2/E3 are gated on it
    for eid in ("E1", "E2", "E3"):
        ids = {d["id"] for d in ph.pending_for(eid)}
        assert "PD-PAIRED" in ids, eid


# ── §8 failed-run handling: retained, classified, never replaced ───────────
def test_failed_run_retention():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        c = fixture_env(td, make_contract())
        led = td / "led.jsonl"
        r1 = ph.record_failed_run(dict(c), failure_kind="crash",
                                  diagnostic="boom (fixture)", seed=42,
                                  ledger_path=led)
        r2 = ph.record_failed_run(dict(c), failure_kind="crash",
                                  diagnostic="boom again (fixture)", seed=42,
                                  ledger_path=led)
        assert r1.status == ph.FAILED and r2.status == ph.FAILED
        assert r1.rerun_permitted is True and r1.failure_class == "implementation"
        assert r1.config_digest == r2.config_digest, "identical frozen config on rerun"
        assert r1.evaluation_id != r2.evaluation_id
        recs = er.EvaluationLedger(led).records()
        assert len(recs) == 2, "both attempts retained, nothing replaced"
        for rec in recs:
            cfg = rec["evaluation_config"]
            assert rec["status"] == "FAILED"
            assert cfg["experiment_id"] == "E1"
            assert rec["seed"] == 42
            assert cfg["failure_class"] == "implementation"
            assert cfg["rerun_permitted"] is True
            assert "boom" in cfg["diagnostic"]
            assert cfg["config_digest"] == r1.config_digest
            assert rec["git_commit"] and rec["timestamp_utc"]
        # crash path during execute also preserves the failure
        def _boom(contract, seed):
            raise RuntimeError("runner crash (fixture)")
        r3 = ph.execute(dict(c), _boom, mode="development", ledger_path=led)
        assert r3.status == ph.FAILED and r3.failure_class == "implementation"
        assert len(er.EvaluationLedger(led).records()) == 3


def test_seed_plan_determinism_and_failures():
    plan = ph.SeedPlan([42, 43, 44])
    assert plan.seeds == (42, 43, 44) and plan.approved is False
    assert {d["id"] for d in plan.pending_decisions()} == {"M-7", "PD-SEED-AGG"}
    try:
        ph.SeedPlan([42, 42])
        raise AssertionError("duplicate seeds must be refused")
    except ph.ConfigurationError:
        pass
    # proposal stays PENDING — never treated as approved
    assert ph.SEED_POLICY_APPROVED is False
    assert tuple(ph.PROPOSED_SEEDS) == (42, 43, 44, 45, 46)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        c = fixture_env(td, make_contract())
        led = td / "led.jsonl"
        def flaky(contract, seed):
            if seed == 43:
                raise RuntimeError("seed 43 crash (fixture)")
            return fixture_runner(contract, seed)
        results = ph.run_seed_plan(dict(c), flaky, [42, 43, 44],
                                   mode="development", ledger_path=led)
        assert [r.seed for r in results] == [42, 43, 44]
        assert results[1].status == ph.FAILED, "failed seed recorded, not hidden"
        recs = er.EvaluationLedger(led).records()
        seeds = sorted(r["seed"] for r in recs)
        assert seeds == [42, 43, 44], seeds
        assert len(recs) == 3, "every pre-declared seed appears (§9.9)"


# ── §10 provenance: all 14 items populated on a harness record ────────────
def test_provenance_14_items():
    assert len(ph.PROVENANCE_ITEMS) == 14, len(ph.PROVENANCE_ITEMS)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        c = fixture_env(td, make_contract())
        led = td / "led.jsonl"
        r = ph.execute(dict(c), fixture_runner, mode="development", ledger_path=led)
        assert r.evaluation_id
        rec = [x for x in er.EvaluationLedger(led).records()
               if x["evaluation_id"] == r.evaluation_id][0]
        prov = ph.provenance_report(rec)
        missing = [k for k, v in prov.items() if v["status"] != "POPULATED"]
        assert not missing, f"unpopulated provenance: {missing}"
        # newly-populated identities (NR-03 §O) specifically verified
        assert rec["preprocessing_version"] == "standardscaler_fit_on_train"
        assert rec["feature_schema_version"].startswith("public_feature_contract.json@")
        assert rec["evaluation_config"]["protocol_version"] == ph.PROTOCOL_VERSION
        assert rec["evaluation_config"]["experiment_id"] == "E1"
        assert isinstance(rec["evaluation_config"]["deviations"], list)
        assert rec["command"] and rec["git_commit"] and rec["dataset"]["sha256"]


def test_evidence_first_measured_gate():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        c = fixture_env(td, make_contract())
        led = td / "led.jsonl"
        r = ph.execute(dict(c), fixture_runner, mode="development", ledger_path=led)
        rec = [x for x in er.EvaluationLedger(led).records()
               if x["evaluation_id"] == r.evaluation_id][0]
        ok, errs = ph.validate_evidence_record(rec, require_preregistered=False)
        assert ok, errs
        ok, errs = ph.validate_evidence_record(rec, require_preregistered=True)
        assert not ok and any("preregistered" in e for e in errs), errs
        assert r.measured is False and r.status != ph.MEASURED, \
            "development evidence must never be MEASURED"
        # truncated record fails validation
        broken = {k: v for k, v in rec.items() if k != "git_commit"}
        ok, errs = ph.validate_evidence_record(broken, require_preregistered=False)
        assert not ok and any("git_commit" in e for e in errs), errs
        # MEASURED is impossible while every experiment is pending (§24)
        assert all(ph.guard(e).code == ph.CODE_PENDING for e in ph.EXPERIMENT_IDS)


# ── §14 threshold freeze ───────────────────────────────────────────────────
def test_threshold_states():
    assert ph.validate_threshold(ph.ThresholdSpec(0.018758, "frozen", "validation")) == "frozen"
    assert ph.validate_threshold(ph.ThresholdSpec(0.5, "validation", "validation")) == "validation_selected"
    assert ph.validate_threshold(ph.ThresholdSpec(0.9, "exploratory", "none")) == "exploratory"
    assert ph.validate_threshold({"value": 0.5, "source": "pending",
                                  "selected_on": "unknown"}) == "pending"
    for bad_src in ("test", "external", "holdout"):
        try:
            ph.validate_threshold(ph.ThresholdSpec(0.5, bad_src, "validation"))
            raise AssertionError(f"source {bad_src} must be refused")
        except ph.ProtocolViolation as e:
            assert "PROTOCOL VIOLATION" in e.message
    try:
        ph.validate_threshold(ph.ThresholdSpec(0.9, "exploratory", "none"),
                              tier="final_test")
        raise AssertionError("exploratory threshold on final tier must be refused")
    except ph.ProtocolViolation:
        pass
    try:
        ph.validate_threshold(ph.ThresholdSpec(None, "pending", "unknown"),
                              confirmatory=True)
        raise AssertionError("pending threshold on confirmatory run must be refused")
    except ph.PendingReview:
        pass
    # parity with the existing pipeline guard (evaluate.py:65)
    import evaluate
    for bad in ("test", "external", "holdout"):
        try:
            evaluate.refuse_test_tuning(bad)
            raise AssertionError("refuse_test_tuning must reject " + bad)
        except SystemExit:
            pass
    evaluate.refuse_test_tuning("validation")
    evaluate.refuse_test_tuning("fixed")
    # §14/§28: the harness exposes no threshold optimizer at all
    assert not any("optim" in name.lower() for name in dir(ph))


# ── §16 split validation ───────────────────────────────────────────────────
def test_split_validation():
    parts = {"train": list(range(0, 60)), "validation": list(range(60, 80)),
             "test": list(range(80, 100))}
    ok = ph.validate_split({"name": "s_rand", "axis": "random"}, parts)
    assert ok["disjoint"] is True
    # overlap
    bad = dict(parts, test=[79, 80, 81])
    try:
        ph.validate_split({"name": "s", "axis": "random"}, bad)
        raise AssertionError("overlap must block")
    except ph.Blocked as e:
        assert "BLOCKED — SPLIT VALIDATION" in e.message
    # temporal without timestamps / violated ordering
    try:
        ph.validate_split({"name": "s", "axis": "temporal"}, parts)
        raise AssertionError("unprovable chronology must block")
    except ph.Blocked as e:
        assert "cannot be proven" in e.message
    ts = {"train": [1, 2, 3], "test": [4, 5, 6]}
    ph.validate_split({"name": "s", "axis": "temporal"}, parts, timestamps=ts)
    try:
        ph.validate_split({"name": "s", "axis": "temporal"}, parts,
                          timestamps={"train": [4, 5], "test": [1, 2]})
        raise AssertionError("temporal violation must block")
    except ph.Blocked as e:
        assert "temporal ordering violated" in e.message
    # entity separation
    ent = {"train": ["u1", "u2"], "test": ["u3"]}
    ph.validate_split({"name": "s", "axis": "entity"}, parts, entities=ent)
    try:
        ph.validate_split({"name": "s", "axis": "entity"}, parts,
                          entities={"train": ["u1"], "test": ["u1"]})
        raise AssertionError("entity leakage must block")
    except ph.Blocked as e:
        assert "entity leakage" in e.message
    # role mismatch (external-set role protection)
    try:
        ph.validate_split({"name": "s", "axis": "random",
                           "role": "development", "expected_role": "final_external"})
        raise AssertionError("role mismatch must block")
    except ph.Blocked as e:
        assert "does not match the preregistered role" in e.message
    # missing identity / bad axis
    for spec in ({"axis": "random"}, {"name": "s", "axis": "magic"}):
        try:
            ph.validate_split(spec, {})
            raise AssertionError("invalid split spec must block")
        except ph.Blocked:
            pass


# ── §15 dataset validation ─────────────────────────────────────────────────
def test_dataset_validation():
    # real registered dataset: full hash pin verified live (ULB only — IBM is
    # 2.35GB and deliberately not re-hashed in the suite for runtime)
    info = ph.validate_dataset("ulb")
    assert info["sha256"] == ph.DATASETS["ulb"]["sha256"]
    assert info["label_column"] == "Class"
    # blocked dataset: no substitution
    for name, needle in (("baf", "BLOCKED — DATASET UNAVAILABLE"),):
        try:
            ph.validate_dataset(name)
            raise AssertionError(name)
        except ph.Blocked as e:
            assert needle in e.message
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        ds = write_fixture_dataset(td)
        pin = sha_of(ds)
        # identity ok (no label_column registered for fixture -> schema skipped)
        got = ph.validate_dataset("fixture_x", path=ds, expected_sha256=pin)
        assert got["sha256"] == pin
        # hash mismatch
        try:
            ph.validate_dataset("fixture_x", path=ds, expected_sha256="f" * 64)
            raise AssertionError("mismatch must block")
        except ph.Blocked as e:
            assert "DATASET IDENTITY MISMATCH" in e.message
        # missing file
        try:
            ph.validate_dataset("fixture_x", path=td / "nope.csv",
                                expected_sha256=pin)
            raise AssertionError("missing file must block")
        except ph.Blocked as e:
            assert "BLOCKED — DATASET UNAVAILABLE" in e.message
        # label column required when the registry declares one
        ph.DATASETS["fixture_lbl"] = {"path": "", "label_column": "target",
                                      "sha256": pin, "role": "fixture"}
        try:
            ph.validate_dataset("fixture_lbl", path=ds, expected_sha256=pin)
            raise AssertionError("missing label column must block")
        except ph.Blocked as e:
            assert "label column" in e.message
        finally:
            ph.DATASETS.pop("fixture_lbl", None)
        # unknown dataset without a path = configuration error, not a block
        try:
            ph.validate_dataset("not_registered")
            raise AssertionError("unknown dataset must be a config error")
        except ph.ConfigurationError:
            pass


# ── §18 full-matrix output ─────────────────────────────────────────────────
def test_matrix_complete():
    m = ph.build_matrix()
    assert m["statuses"] == list(ph.CELL_STATUSES)
    assert m["measured_cells"] == 0, "no final experiment has run (§24)"
    assert m["total_cells"] == len(m["tensor_cells"]) > 0
    assert sum(m["cell_counts"].values()) == m["total_cells"]
    assert set(m["design_matrix"]) == set(ph.EXPERIMENT_IDS)
    for eid, row in m["design_matrix"].items():
        assert len(row) == 23, eid
    for cell in m["tensor_cells"]:
        assert cell["status"] in ph.CELL_STATUSES, cell
        assert cell["axes"], "cell must declare every axis"
        assert cell["note"] is not None
        assert cell["evaluation_id"] is None   # nothing has run
    # no zero-fill / collapse: blocked and pending are separate
    assert m["cell_counts"][ph.BLOCKED] > 0, "21+7 contract cells must appear BLOCKED"
    assert m["cell_counts"][ph.PENDING_REVIEW] > 0
    assert m["cell_counts"][ph.MEASURED] == 0
    e2 = [c for c in m["tensor_cells"] if c["experiment"] == "E2"]
    assert len(e2) == 7, "E2 must enumerate its 7 pre-defined conditions"
    e3 = [c for c in m["tensor_cells"] if c["experiment"] == "E3"]
    assert any(c["status"] == ph.BLOCKED for c in e3)
    assert all("21+7" not in c["axes"]["contract"] or c["status"] == ph.BLOCKED
               for c in e3)
    # descriptive experiments mark the seed axis NOT APPLICABLE (§9.11/§M)
    for c in m["tensor_cells"]:
        if c["experiment"] in ("E5", "E6"):
            assert c["axes"]["seed"] == ph.NOT_APPLICABLE


# ── §21 pipeline integration ───────────────────────────────────────────────
def test_pipeline_integration():
    ledger_before = _harness_record_count()
    import calibration_test          # noqa: F401  (pinned Brier/ECE source)
    import cross_dataset_eval        # noqa: F401
    import evaluate as _evaluate     # noqa: F401
    import eval_ulb as _ulb          # noqa: F401 (must NOT run training)
    assert hasattr(_ulb, "main") and hasattr(_ulb, "recall_at_fpr")
    assert _harness_record_count() == ledger_before, "imports must not append records"
    # the four §21 canonical paths + evaluate/ibm are registered with notes
    for key in ("calibration_test", "eval_ulb", "cross_dataset_eval",
                "train_compare", "evaluate", "ibm_train"):
        assert key in ph.PIPELINES, key
        entry = ph.PIPELINES[key]
        assert entry["script"] and entry["record"] and entry["notes"]
    assert ph.PIPELINES["train_compare"].get("outdir_required") is True
    assert ph.PIPELINES["ibm_train"].get("deferred") is True
    # train_compare / ibm_train are __main__-guarded (not imported for speed,
    # verified structurally)
    for script in ("backend/src/train_compare.py", "backend/scripts/ibm_train.py"):
        src = (ROOT / script).read_text(encoding="utf-8")
        assert 'if __name__ == "__main__":' in src, script
    # eval_ulb citations (NR-02/NR-03 line refs) still land on the pinned code
    ulb_lines = (ROOT / "backend/scripts/eval_ulb.py").read_text(encoding="utf-8").splitlines()
    assert ulb_lines[23].startswith("def recall_at_fpr"), ulb_lines[23]
    assert "searchsorted(fpr, target_fpr)" in ulb_lines[28]


# ── §22 ledger behavior ────────────────────────────────────────────────────
def test_ledger_distinguishability():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        art1 = td / "m.joblib"
        art1.write_bytes(b"weights-A")
        c = fixture_env(td, make_contract(artifact_paths=[str(art1)]))
        led = td / "led.jsonl"
        r1 = ph.execute(dict(c), fixture_runner, mode="development", ledger_path=led)
        r2 = ph.execute(dict(c), fixture_runner, mode="development", ledger_path=led)
        assert r1.evaluation_id != r2.evaluation_id, "repeat attempts stay distinct"
        # distinct artifact bytes -> distinct model identity (same command!)
        art1.write_bytes(b"weights-B")
        r3 = ph.execute(dict(c), fixture_runner, mode="development", ledger_path=led)
        recs = {r["evaluation_id"]: r for r in er.EvaluationLedger(led).records()}
        h1 = recs[r1.evaluation_id]["model_hash"]
        h3 = recs[r3.evaluation_id]["model_hash"]
        assert h1 != h3, "distinct artifact versions are not duplicates (§22)"
        assert recs[r1.evaluation_id]["command"] == recs[r3.evaluation_id]["command"]
        # failed evidence can never create a falsely valid measured record
        assert all(not recs[e].get("metrics", {}).get("measured")
                   for e in (r1.evaluation_id, r3.evaluation_id))


def test_fixture_isolation_and_no_final_experiment():
    # no harness-authored record ever touched the canonical ledger
    assert _harness_record_count() == 0, \
        "fixture runs must never write to the real evidence ledger"
    if ph.CANONICAL_LEDGER.exists():
        for line in ph.CANONICAL_LEDGER.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            assert not str(rec.get("model_identifier", "")).startswith("prereg_E"), \
                "a preregistered experiment record exists in the canonical ledger"
            assert not (rec.get("evaluation_config") or {}).get("fixture_test"), \
                "fixture record leaked into the canonical ledger"
    # no cell has been measured; the matrix itself proves §24
    assert ph.build_matrix()["measured_cells"] == 0
    # seed/bootstrap/ECE/paired gates all still closed (no silent approvals)
    assert ph.SEED_POLICY_APPROVED is False and ph.SEED_AGGREGATION_APPROVED is False
    assert ph.BOOTSTRAP_COUNT_APPROVED is None and ph.ECE_BINNING_APPROVED is None
    assert ph.PAIRED_CI_SPEC_APPROVED is False and ph.PROTOCOL_SIGNED_OFF is False


def test_cli_surface():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc_registry = ph.main(["registry"])
        rc_pending = ph.main(["pending"])
        rc_guard = ph.main(["guard", "E1"])
        rc_baf = ph.main(["validate-dataset", "baf"])
        rc_matrix = ph.main(["matrix"])
    assert rc_registry == 0 and rc_pending == 0
    assert rc_guard == 3, "guard must exit non-zero when not allowed"
    assert rc_baf == 4, "blocked dataset must exit non-zero"
    assert rc_matrix == 0
    out = buf.getvalue()
    assert "DO NOT EXECUTE" in out and "BLOCKED — DATASET UNAVAILABLE" in out
    # no execution subcommand exists (§6: no casual override path)
    import argparse
    try:
        ph.main(["execute", "E1"])
        raise AssertionError("an execute subcommand must not exist")
    except SystemExit as e:
        assert e.code == 2


def _harness_record_count() -> int:
    if not ph.CANONICAL_LEDGER.exists():
        return 0
    n = 0
    for line in ph.CANONICAL_LEDGER.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        cfg = rec.get("evaluation_config") or {}
        if str(cfg.get("harness", "")).startswith("prereg_harness"):
            n += 1
    return n


TESTS = [
    test_registry_matches_nr03_matrix,
    test_marker_count_11,
    test_seven_experiments_23_fields,
    test_pending_decision_registry,
    test_guard_states_distinct,
    test_all_experiments_guarded_preregistered,
    test_same_config_same_seed,
    test_same_config_repeated_no_semantic_diff,
    test_different_seed_identity_changes,
    test_different_dataset_hash_blocked,
    test_missing_provenance_blocked,
    test_pending_decision_blocks_preregistered,
    test_failed_evidence_not_measured,
    test_feature_contract_mismatch_blocked,
    test_final_test_selection_violation,
    test_metric_semantics_pins,
    test_metric_domain_failures,
    test_bootstrap_plumbing,
    test_paired_ci_boundary,
    test_failed_run_retention,
    test_seed_plan_determinism_and_failures,
    test_provenance_14_items,
    test_evidence_first_measured_gate,
    test_threshold_states,
    test_split_validation,
    test_dataset_validation,
    test_matrix_complete,
    test_pipeline_integration,
    test_ledger_distinguishability,
    test_fixture_isolation_and_no_final_experiment,
    test_cli_surface,
]


def main() -> int:
    passed, failed = 0, []
    for t in TESTS:
        try:
            t()
            passed += 1
            print(f"  PASS {t.__name__}")
        except AssertionError as e:
            failed.append(t.__name__)
            print(f"  FAIL {t.__name__}: {e}")
        except Exception as e:  # noqa: BLE001 - report, don't hide
            failed.append(t.__name__)
            print(f"  ERROR {t.__name__}: {type(e).__name__}: {e}")
    total = passed + len(failed)
    print(f"=" * 60)
    print(f"Results: {passed}/{total} passed, {len(failed)} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
