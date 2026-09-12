#!/usr/bin/env python3
"""Tests for the OOD recall gate (`check_ood_recall_gate` in train_compare.py).

The gate fails a retrain when a named held-out archetype's
`recall_at_1pct_fpr` drops below a floor. Recall@1%FPR is the ranking-based
metric on purpose: `recall_f1` is unreliable at low fold fraud prevalence
(the ato fold's validation is ~0.2% fraud, so its F1 threshold lands far
above held-out ato scores even when ranking is intact).

Run from the project root:
  python scripts/ood_gate_test.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent  # repo root.parent  # repo root
sys.path.insert(0, str(ROOT / "backend"))

from src.train_compare import check_ood_recall_gate  # noqa: E402

failures: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        failures.append(name)


def gen_df(**overrides) -> pd.DataFrame:
    rows = [
        {"held_out": "ato", "n_fraud": 67, "recall_f1": 0.09, "recall_at_1pct_fpr": 1.0},
        {"held_out": "cnp", "n_fraud": 30, "recall_f1": 0.93, "recall_at_1pct_fpr": 1.0},
        {"held_out": "mule", "n_fraud": 34, "recall_f1": 0.56, "recall_at_1pct_fpr": 1.0},
    ]
    for r in rows:
        r.update(overrides.get(r["held_out"], {}))
    return pd.DataFrame(rows)


def main() -> int:
    print("== OOD recall gate tests ==")

    # ---- pass: all gated archetypes above the floor ----------------------
    rows = check_ood_recall_gate(gen_df(), ["ato", "mule"], 0.60)
    check("healthy run passes", bool(rows) and all(r["passed"] for r in rows),
          str([(r["archetype"], r["recall_at_1pct_fpr"]) for r in rows]))
    check("gate rows carry the metric + floor",
          bool(rows) and all(r["recall_at_1pct_fpr"] == 1.0 and r["floor"] == 0.60 for r in rows))

    # ---- fail: ato recall drops below the floor --------------------------
    rows = check_ood_recall_gate(gen_df(ato={"recall_at_1pct_fpr": 0.4}), ["ato", "mule"], 0.60)
    check("ato below floor -> fail row", any(r["archetype"] == "ato" and not r["passed"] for r in rows),
          str([(r["archetype"], r["passed"]) for r in rows]))
    check("mule still passes", any(r["archetype"] == "mule" and r["passed"] for r in rows))

    # ---- floor 0 never fails ---------------------------------------------
    rows = check_ood_recall_gate(gen_df(ato={"recall_at_1pct_fpr": 0.01}), ["ato"], 0.0)
    check("floor 0 never fails", bool(rows) and rows[0]["passed"])

    # ---- missing archetype is skipped, not failed ------------------------
    rows = check_ood_recall_gate(gen_df(), ["ato", "escalation"], 0.60)
    check("absent archetype skipped", [r["archetype"] for r in rows] == ["ato"])

    # ---- empty gen_df (--no-group-eval) -> skipped -----------------------
    rows = check_ood_recall_gate(pd.DataFrame(), ["ato", "mule"], 0.60)
    check("no group eval -> gate skipped", rows == [])

    # ---- nan recall (no fraud rows in the fold) -> skipped ---------------
    rows = check_ood_recall_gate(gen_df(mule={"recall_at_1pct_fpr": np.nan}), ["ato", "mule"], 0.60)
    check("nan recall skipped", [r["archetype"] for r in rows] == ["ato"])

    print("\n" + ("ALL CHECKS PASSED" if not failures else f"{len(failures)} CHECK(S) FAILED"))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
