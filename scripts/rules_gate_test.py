#!/usr/bin/env python3
"""Smoke test for the rules CI gate (scripts/gate_rules.py).

Runs the real gate binary against a deterministic sample of the labeled
history with a temp baseline, asserting the exit-code contract:

  * missing baseline snapshot        -> exit 2
  * candidate == baseline (no-op)    -> exit 0 without computing ML
  * FPR-increasing rule change       -> exit 1 (GATE BREACHED)
  * --accept refreshes the snapshot  -> next identical gate exits 0

The ML pass is a few seconds on the sampled rows and reuses the shared
disk cache, so the suite stays fast.

Run from the project root:
  python scripts/rules_gate_test.py
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

GATE = ROOT / "scripts" / "gate_rules.py"
SRC_RULES = ROOT / "src" / "risk_engine" / "rules.yaml"
DATA_PATH = ROOT / "data" / "transactions.csv"

failures: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        failures.append(name)


def run_gate(tmp: Path, baseline: Path, candidate: Path, extra: list[str] | None = None) -> subprocess.CompletedProcess:
    cmd = [sys.executable, str(GATE),
           "--data", str(tmp / "sample.csv"),
           "--baseline", str(baseline),
           "--rules", str(candidate)]
    if extra:
        cmd += extra
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=600)


def main() -> int:
    print("== rules CI gate smoke test ==")
    df = pd.read_csv(DATA_PATH)
    sample = df.sample(800, random_state=11).sort_values("ts").reset_index(drop=True)
    n_fraud = int(sample["label"].sum())
    print(f"  sample: {len(sample)} events, {n_fraud} fraud")

    with tempfile.TemporaryDirectory(prefix="ps14-gate-") as td:
        tmp = Path(td)
        sample.to_csv(tmp / "sample.csv", index=False)
        baseline = tmp / "baseline.yaml"
        candidate = tmp / "candidate.yaml"
        shutil.copy2(SRC_RULES, baseline)
        shutil.copy2(SRC_RULES, candidate)

        # ---- missing baseline ------------------------------------------------
        r = run_gate(tmp, tmp / "nonexistent.yaml", candidate)
        check("missing baseline -> exit 2", r.returncode == 2, f"rc={r.returncode}")

        # ---- no-op edit: identical candidate, fast path ----------------------
        r = run_gate(tmp, baseline, candidate)
        check("identical candidate -> pass (exit 0)", r.returncode == 0, f"rc={r.returncode}")
        check("identical candidate needs no ML", "no gate needed" in r.stdout, "")

        # ---- recall-dropping edit -> blocked ---------------------------------
        text = baseline.read_text(encoding="utf-8")
        # Remove all critical rules and lower severity_scale to weaken detection
        import re as _re
        # Remove lines containing 'level: critical'
        text_new = _re.sub(r'\n\s*-\s+id:\s+RULE_STRONG_AUTH_ANOMALY.*?(?=\n\s*-\s+id:|\Z)', '', text, flags=_re.DOTALL)
        text_new = _re.sub(r'\n\s*-\s+id:\s+RULE_MULTI_SIGNAL_FRAUD.*?(?=\n\s*-\s+id:|\Z)', '', text_new, flags=_re.DOTALL)
        text_new = _re.sub(r'\n\s*-\s+id:\s+RULE_MULE_RING_STRONG.*?(?=\n\s*-\s+id:|\Z)', '', text_new, flags=_re.DOTALL)
        text_new = _re.sub(r'\n\s*-\s+id:\s+RULE_RAPID_FIRE.*?(?=\n\s*-\s+id:|\Z)', '', text_new, flags=_re.DOTALL)
        text_new = _re.sub(r'(severity_scale:\s*)[0-9.]+', r'\g<1>0.10', text_new, count=1)
        candidate.write_text(text_new, encoding="utf-8")
        r = run_gate(tmp, baseline, candidate)
        check("recall-dropping edit -> exit 1 (blocked)", r.returncode == 1, f"rc={r.returncode}")
        check("verdict names the breach", "breached" in r.stdout.lower() or "recall" in r.stdout.lower(), "")

        # ---- accept refreshes the snapshot -----------------------------------
        r = run_gate(tmp, baseline, candidate, extra=["--accept"])
        check("--accept copies candidate over baseline (exit 0)", r.returncode == 0, f"rc={r.returncode}")
        check("--accept wrote the snapshot", baseline.read_text(encoding="utf-8") == candidate.read_text(encoding="utf-8"))
        r = run_gate(tmp, baseline, candidate)
        check("gate passes after accept", r.returncode == 0, f"rc={r.returncode}")

    print("\n" + ("ALL CHECKS PASSED" if not failures else f"{len(failures)} CHECK(S) FAILED"))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
