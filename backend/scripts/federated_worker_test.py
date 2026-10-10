#!/usr/bin/env python3
"""Regression test for the federated institution worker (architecture §19).

Hermetic: launches the REAL worker process on a controlled synthetic CSV and
speaks its documented stdin/stdout protocol. No coordinator, no services, no
shared state, no network.

Regression target (F1 — found by Phase 13, fixed by Phase 15):
`federated_worker.py` had a function-local `import sys;` inside the "some
ML_FEATURES columns are missing" branch. A local import makes `sys` a LOCAL
name for all of `main()`, so whenever EVERY ML_FEATURES column was present
(the fresh-checkout / CI case) that branch was skipped and the
`for line in sys.stdin` below raised `UnboundLocalError` — every worker died
at startup and `federated_test.py` failed with
`worker t_0 failed to start (rc=1)`. The warm dev tree masked it because its
cached pool CSV lacks a production-only feature, which executed the branch.

Run from the project root:
  python backend/scripts/federated_worker_test.py
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent  # repo root
sys.path.insert(0, str(ROOT / "backend"))

from src.privacy_layer.features import ML_FEATURES  # noqa: E402

WORKER = ROOT / "backend" / "scripts" / "federated_worker.py"
N_ROWS = 40
TRAIN_RATIO = 0.7
# Production-only link-analysis columns: absent from the public/synthetic
# datasets, present in a full-coverage export (the fresh-CI-data case).
PRODUCTION_ONLY = ("shared_device_accounts", "shared_recipient_accounts", "mule_ring_score")

failures: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        failures.append(name)


def write_csv(path: Path, columns: list[str]) -> None:
    """Deterministic institution fixture: fixed values, no RNG, ts ascending."""
    with open(path, "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=["ts"] + columns + ["label"])
        wr.writeheader()
        for i in range(N_ROWS):
            row = {"ts": f"2025-08-{(i % 28) + 1:02d}T{8 + (i % 12):02d}:00:00",
                   "label": 1 if i % 5 == 0 else 0}
            for j, col in enumerate(columns):
                row[col] = (i * (j + 2) % 17) / 4.0
            wr.writerow(row)


def run_worker(csv_path: Path, n_features: int) -> subprocess.CompletedProcess:
    """One `train` request plus `shutdown`, over the documented protocol."""
    request = {"type": "train", "model": "lr", "round": 1, "epochs": 2, "lr": 0.5,
               "weights": {"w": [0.0] * n_features, "b": 0.0}}
    return subprocess.run(
        [sys.executable, str(WORKER), "--data", str(csv_path), "--name", "t_0",
         "--train-ratio", str(TRAIN_RATIO)],
        input=json.dumps(request) + "\n" + json.dumps({"type": "shutdown"}) + "\n",
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        cwd=str(ROOT), timeout=180,
    )


def main() -> int:
    print("== Federated worker startup test ==")
    with tempfile.TemporaryDirectory(prefix="ps14-fed-worker-") as td:
        tmp = Path(td)
        expected_n_train = int(N_ROWS * TRAIN_RATIO)

        # ---- 1. EVERY ML_FEATURES column present (fresh-checkout path) ------
        full = tmp / "all_features.csv"
        write_csv(full, list(ML_FEATURES))
        p = run_worker(full, len(ML_FEATURES))
        tail = p.stderr.strip().splitlines()[-1] if p.stderr.strip() else "(no stderr)"
        check("worker exits cleanly with every ML_FEATURES column present",
              p.returncode == 0, f"rc={p.returncode} stderr_tail={tail}")
        replies = [json.loads(ln) for ln in p.stdout.splitlines() if ln.strip()]
        check("worker answers a train request with exactly one weights message",
              len(replies) == 1 and replies[0].get("type") == "weights",
              str(replies)[:160])
        if len(replies) == 1:
            reply = replies[0]
            check("weights reply carries institution name + local train size",
                  reply.get("name") == "t_0" and reply.get("n_train") == expected_n_train,
                  f"name={reply.get('name')} n_train={reply.get('n_train')}")
            w = reply.get("weights", {})
            check("wire payload is weights only (LR: w + b) at full width",
                  set(w) == {"w", "b"} and len(w.get("w", [])) == len(ML_FEATURES)
                  and all(isinstance(v, float) for v in w.get("w", [])),
                  f"keys={sorted(w)} |w|={len(w.get('w', []))}")

        # ---- 2. reduced column set (production-only features absent) --------
        reduced_cols = [f for f in ML_FEATURES if f not in PRODUCTION_ONLY]
        reduced = tmp / "reduced_features.csv"
        write_csv(reduced, reduced_cols)
        p2 = run_worker(reduced, len(reduced_cols))
        check("worker exits cleanly when columns are missing",
              p2.returncode == 0, f"rc={p2.returncode}")
        info_tail = p2.stderr.strip().splitlines()[0] if p2.stderr.strip() else "(none)"
        check("worker reports the missing-column fallback on stderr",
              f"{len(ML_FEATURES) - len(reduced_cols)} features not in CSV" in p2.stderr,
              info_tail)
        replies2 = [json.loads(ln) for ln in p2.stdout.splitlines() if ln.strip()]
        widths = [len(r.get("weights", {}).get("w", [])) for r in replies2]
        check("missing-column run uses only the available features",
              len(replies2) == 1 and widths == [len(reduced_cols)], str(widths))

    print("\n" + ("ALL CHECKS PASSED" if not failures else f"{len(failures)} CHECK(S) FAILED"))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
