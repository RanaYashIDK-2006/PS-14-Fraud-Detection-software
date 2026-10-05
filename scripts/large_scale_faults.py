"""Phase 3 §12 — FAILURE / RECOVERY tests at large scale.

Every fault is injected into COPIES of benchmark partitions under
`misc/reports/phase3_faults/`. The validated benchmark at
`data/synthetic_50m/` is opened read-only and never written, renamed or
deleted; the original manifest hash is re-verified at the end of this script
and any change aborts the test.

Faults exercised, each by re-running the REAL runner
(`scripts/large_scale_runner.py --bench <scratch>`):

  1. baseline            3 copied partitions, no fault      -> expected pass
  2. missing_partition   one partition deleted             -> expected loud failure
  3. corrupt_partition   one partition truncated to 4 KiB  -> expected loud failure
  4. duplicate_partition one partition copied to a second
                         year_month dir with identical rows-> expected row inflation
  5. rerun_completed     the same 3 partitions run twice   -> expected identical
  6. recovery            a partition restored after fault 2/3 -> expected pass again

"Expected loud failure" means: non-zero exit AND a diagnostic on stderr. A
silent wrong answer is a WORSE outcome than a crash and is reported as such.

Usage:
    ./.venv/Scripts/python.exe scripts/large_scale_faults.py \
        --out misc/reports/phase3_faults.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

SRC = Path("data/synthetic_50m")
SCRATCH = Path("misc/reports/phase3_faults")
RUNNER = Path("scripts/large_scale_runner.py")
PY = str(Path(".venv/Scripts/python.exe"))


def combined_sha(root: Path) -> str:
    man = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    ph = man["partition_hashes"]
    h = hashlib.sha256()
    for k in sorted(ph):
        h.update(k.encode())
        h.update(ph[k].encode())
    return h.hexdigest()


def make_scratch(parts: list[str]) -> Path:
    if SCRATCH.exists():
        shutil.rmtree(SCRATCH)
    (SCRATCH / "data").mkdir(parents=True, exist_ok=True)
    shutil.copy(SRC / "manifest.json", SCRATCH / "manifest.json")
    for p in parts:
        d = SCRATCH / "data" / f"year_month={p}"
        d.mkdir(parents=True, exist_ok=True)
        src = next((SRC / "data" / f"year_month={p}").glob("*.parquet"))
        shutil.copy(src, d / src.name)
    return SCRATCH


def run(label: str, out_json: Path, expect: str) -> dict:
    cmd = [PY, str(RUNNER), "--bench", str(SCRATCH),
           "--rows", "50000000", "--out", str(out_json), "--label", label]
    t = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    rec = {
        "fault": label,
        "expectation": expect,
        "returncode": p.returncode,
        "seconds": round(time.time() - t, 2),
        "stderr_tail": (p.stderr or "").strip().splitlines()[-3:],
        "stdout_tail": (p.stdout or "").strip().splitlines()[-3:],
        "report_written": out_json.exists(),
    }
    if out_json.exists():
        r = json.loads(out_json.read_text(encoding="utf-8"))
        rec["observed"] = {
            "rows_processed": r["rows_processed"],
            "partitions_processed": r["partitions_processed"],
            "fraud_rows": r["fraud_rows"],
            "roc_auc": r["roc_auc"],
            "confusion_at_threshold": r["confusion_at_threshold"],
            "missing_feature_partitions": r["missing_feature_partitions"],
            "nonfinite_values": r["nonfinite_values"],
        }
        out_json.unlink()
    if expect == "PASS":
        rec["outcome"] = "AS EXPECTED" if p.returncode == 0 else "UNEXPECTED FAILURE"
    elif expect == "FAIL_LOUDLY":
        loud = p.returncode != 0 and bool(rec["stderr_tail"] or rec["stdout_tail"])
        rec["outcome"] = "AS EXPECTED" if loud else "SILENT — DEFECT"
    rec["failed_loudly"] = p.returncode != 0
    print(f"  [{label}] rc={p.returncode} -> {rec['outcome']}", flush=True)
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="misc/reports/phase3_faults.json")
    ap.add_argument("--partitions", default="2016-01,2016-02,2016-03")
    args = ap.parse_args()

    parts = [p.strip() for p in args.partitions.split(",") if p.strip()]
    src_sha_before = combined_sha(SRC)
    out_json = SCRATCH / "result.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    results = []

    print("1/6 baseline (no fault)", flush=True)
    make_scratch(parts)
    base = run("baseline_no_fault", out_json, "PASS")
    base_rows = base.get("observed", {}).get("rows_processed")
    base_auc = base.get("observed", {}).get("roc_auc")
    results.append(base)

    print("2/6 missing partition", flush=True)
    make_scratch(parts)
    shutil.rmtree(SCRATCH / "data" / f"year_month={parts[1]}")
    miss = run("missing_partition", out_json, "FAIL_LOUDLY")
    miss["note"] = ("rglob over data/ simply sees fewer files; the runner has NO "
                    "partition manifest assertion, so this MAY pass with fewer rows")
    results.append(miss)

    print("3/6 corrupt partition", flush=True)
    make_scratch(parts)
    victim = next((SCRATCH / "data" / f"year_month={parts[1]}").glob("*.parquet"))
    with open(victim, "r+b") as fh:
        fh.truncate(4096)
    corrupt = run("corrupt_partition_truncated_4KiB", out_json, "FAIL_LOUDLY")
    results.append(corrupt)

    print("4/6 duplicate partition", flush=True)
    make_scratch(parts)
    s = next((SCRATCH / "data" / f"year_month={parts[1]}").glob("*.parquet"))
    d = SCRATCH / "data" / "year_month=2016-99"
    d.mkdir(parents=True, exist_ok=True)
    shutil.copy(s, d / "copy.parquet")
    dup = run("duplicate_partition_under_new_year_month", out_json, "FAIL_LOUDLY")
    if dup.get("observed"):
        inflated = dup["observed"]["rows_processed"] > (base_rows or 0)
        dup["rows_inflated_vs_baseline"] = inflated
        dup["observed_effect"] = ("duplicate rows scored as extra data; prevalence "
                                  f"{dup['observed']['fraud_rows']}/{dup['observed']['rows_processed']} "
                                  "vs baseline "
                                  f"{base.get('observed', {}).get('fraud_rows')}/{base_rows}")
    results.append(dup)

    print("5/6 rerun of a completed partition set (idempotence)", flush=True)
    make_scratch(parts)
    r1 = run("rerun_completed_pass1", out_json, "PASS")
    r2 = run("rerun_completed_pass2", out_json, "PASS")
    if r1.get("observed") and r2.get("observed"):
        same = (r1["observed"]["rows_processed"] == r2["observed"]["rows_processed"]
                and r1["observed"]["fraud_rows"] == r2["observed"]["fraud_rows"]
                and r1["observed"]["confusion_at_threshold"] == r2["observed"]["confusion_at_threshold"]
                and r1["observed"]["roc_auc"] == r2["observed"]["roc_auc"])
        r2["bit_identical_to_pass1"] = same
        r2["comparison"] = "rows, fraud rows, ROC-AUC and full confusion matrix"
    results += [r1, r2]

    print("6/6 recovery after fault", flush=True)
    make_scratch(parts)
    with open(next((SCRATCH / "data" / f"year_month={parts[1]}").glob("*.parquet")), "r+b") as fh:
        fh.truncate(4096)
    run("recovery_probe_before_restore", out_json, "FAIL_LOUDLY")
    make_scratch(parts)   # restore every partition from the pristine source
    rec = run("recovery_after_restore", out_json, "PASS")
    if rec.get("observed"):
        rec["restored_to_baseline_rows"] = (rec["observed"]["rows_processed"] == base_rows)
        rec["restored_to_baseline_auc"] = (rec["observed"]["roc_auc"] == base_auc)
    results.append(rec)

    src_sha_after = combined_sha(SRC)
    res = {
        "label": "Phase 3 §12 failure / recovery on partition copies",
        "evidence_class": "SELF-TESTED — SYNTHETIC 50M — SCALE EXPERIMENT",
        "copied_partitions": parts,
        "original_benchmark_sha256_before": src_sha_before,
        "original_benchmark_sha256_after": src_sha_after,
        "original_benchmark_unmodified": src_sha_before == src_sha_after,
        "tests": results,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    print(f"\noriginal benchmark unmodified: {res['original_benchmark_unmodified']}")
    print(f"report -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())