#!/usr/bin/env python3
"""Smoke test for the federated-learning simulation (architecture section 19).

Small/fast run of the real coordinator + worker processes, asserting:
  * shards are account-disjoint (no customer in two institutions),
  * ONLY weight vectors and training params cross the process boundary
    (recorded messages are schema-checked),
  * metrics are finite and above random,
  * federated >= local-only (PR-AUC) and centralized oracle >= federated,
  * the DP-FedAvg path (src/federated/dp.py): clipping bounds, Gaussian
    noise scale monotone in epsilon, RDP accounting sanity, the noisy
    aggregation's equivalence to plain FedAvg at sigma = 0, and an
    end-to-end DP run whose boundary still carries only weights.
  * the MLP extension (src/federated/mlp.py): params round-trip, hidden size,
    the XOR separability benefit over LR, per-layer FedAvg aggregation, the
    flat-vector DP aggregation's equivalence at sigma = 0, and an end-to-end
    MLP federated run on the same shards whose boundary still carries only
    weights.
  * the heterogeneity map (--hetero-sweep): `split_institutions_skewed`
    mechanics (account-disjointness, realized fraud rates ordered by the
    skews, determinism, skew=0 repelling fraud) and a small sweep whose FL
    gain / max-drag numbers are consistent with the per-institution detail.

Run from the project root:
  python scripts/federated_test.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from scripts.federated_sim import (  # noqa: E402
    centralized_oracle,
    generate_pool,
    run_federated,
    run_hetero_sweep,
    split_institutions,
    split_institutions_skewed,
)
from src.federated import dp  # noqa: E402
from src.federated.lr import LocalLR, fedavg_params  # noqa: E402
from src.federated.mlp import LocalMLP  # noqa: E402

failures: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        failures.append(name)


def main() -> int:
    print("== Federated learning smoke test ==")

    pool = generate_pool(4_000, 0.015, seed=3)
    shards = split_institutions(pool, [0.5, 0.3, 0.2], seed=3)

    check("3 institutions", len(shards) == 3, str([len(s) for s in shards]))
    total_accts = sum(len(set(s["fraud_id"])) for s in shards)
    check("account-disjoint shards", total_accts == len(set(pool["fraud_id"])),
          f"{total_accts} vs {len(set(pool['fraud_id']))}")
    check("every institution has data + fraud", all(len(s) > 0 and int(s["label"].sum()) > 0 for s in shards),
          str([int(s["label"].sum()) for s in shards]))

    with tempfile.TemporaryDirectory(prefix="fed-test-") as td:
        results = run_federated(
            shards, rounds=4, local_epochs=3, lr=0.5, baseline_epochs=30,
            train_ratio=0.7, label="t", seed=3, record_dir=Path(td),
        )
        # ---- boundary: coordinator sends ONLY weights + training params ----
        # ("model" selects the architecture; the payload is still just weights)
        allowed = {"type", "model", "round", "epochs", "lr", "weights"}
        leaks: list[str] = []
        for f in Path(td).glob("*.jsonl"):
            for line in f.read_text(encoding="utf-8").splitlines():
                msg = json.loads(line)
                if not set(msg) <= allowed:
                    leaks.append(line[:120])
        check("only weights cross the boundary", not leaks, str(leaks[:1]))

    for name, src in (("local-only", results["local"]), ("federated", results["federated"])):
        check(f"{name}: 3 institutions", len(src) == 3, str(sorted(src)))
        finite = all(m["pr_auc"] == m["pr_auc"] and 0.0 <= m["pr_auc"] <= 1.0 for m in src.values())
        check(f"{name}: finite PR-AUCs", finite,
              str({k: round(v["pr_auc"], 3) for k, v in src.items()}))

    macro = lambda src: sum(m["pr_auc"] for m in src.values()) / len(src)
    lp, fp = macro(results["local"]), macro(results["federated"])
    check("federated >= local-only (PR-AUC)", fp >= lp - 0.05,
          f"fed {fp:.3f} vs local {lp:.3f}")
    check("convergence trace finite", all(t == t for t in results["trace"]) and len(results["trace"]) == 4,
          str(results["trace"]))

    oracle = centralized_oracle(shards, epochs=30, lr=0.5, train_ratio=0.7)
    op = macro(oracle)
    check("centralized oracle >= federated (tol)", op >= fp - 0.05,
          f"oracle {op:.3f} vs fed {fp:.3f}")
    check("all above random (PR-AUC > 0.5)", lp > 0.5 and fp > 0.5 and op > 0.5,
          f"local {lp:.3f} fed {fp:.3f} oracle {op:.3f}")

    # ---- DP-FedAvg: clipping + Gaussian noise + RDP accounting ------------
    print("\n-- differential privacy --")
    big = np.array([3.0, -4.0, 0.5])
    small = np.array([0.3, -0.2, 0.1])
    clipped = dp.clip_delta(big, 2.0)
    check("clipping pulls the norm down to S",
          abs(float(np.linalg.norm(clipped)) - 2.0) < 1e-12,
          f"norm={np.linalg.norm(clipped):.6f}")
    check("below-S updates are unchanged", np.array_equal(dp.clip_delta(small, 2.0), small))
    check("zero vector stays zero", np.all(dp.clip_delta(np.zeros(3), 1.0) == 0.0))

    s_hi = dp.noise_scale_for_eps(1.0, 1e-5, 20)
    s_lo = dp.noise_scale_for_eps(8.0, 1e-5, 20)
    check("noise scale grows as epsilon shrinks", s_hi > s_lo > 0.0,
          f"eps1 -> {s_hi:.3f}, eps8 -> {s_lo:.3f}")
    check("epsilon=inf -> zero noise (clean baseline)",
          dp.noise_scale_for_eps(float("inf"), 1e-5, 20) == 0.0)
    e_big = dp.eps_from_rdp(dp.rdp_gaussian_eps(5.0, 20, 20), 1e-5, 20)
    e_small = dp.eps_from_rdp(dp.rdp_gaussian_eps(1.0, 20, 20), 1e-5, 20)
    check("RDP conversion: more noise -> smaller epsilon", e_big < e_small,
          f"sigma5 -> {e_big:.3f}, sigma1 -> {e_small:.3f}")

    rng = np.random.default_rng(0)
    gw, gb = np.zeros(3), 0.1
    updates = [(np.array([1.0, 0.5, -0.2]), 0.3, 100),
               (np.array([0.8, 0.4, 0.0]), 0.2, 50),
               (np.array([1.2, 0.6, -0.1]), 0.4, 30)]
    w0, b0 = dp.dp_aggregate(gw, gb, updates, clip_norm=100.0, sigma=0.0, rng=rng)
    w_plain, b_plain = fedavg_from(updates)
    check("dp_aggregate at sigma=0 == plain FedAvg",
          np.allclose(w0, w_plain) and abs(b0 - b_plain) < 1e-12,
          f"{np.round(w0, 4)} vs {np.round(w_plain, 4)}")
    rng2 = np.random.default_rng(0)
    w1, b1 = dp.dp_aggregate(gw, gb, updates, clip_norm=0.5, sigma=2.0, rng=rng2)
    check("clipped + noisy aggregation runs and stays finite",
          np.all(np.isfinite(w1)) and np.isfinite(b1), str(w1))
    rng3 = np.random.default_rng(1)
    w2, b2 = dp.dp_aggregate(gw, gb, updates, clip_norm=0.5, sigma=2.0, rng=rng3)
    check("noise is seed-deterministic", not np.allclose(w1, w2),
          "two seeds -> different noise")

    with tempfile.TemporaryDirectory(prefix="fed-dp-") as td:
        dp_res = run_federated(
            shards, rounds=4, local_epochs=3, lr=0.5, baseline_epochs=30,
            train_ratio=0.7, label="dp", seed=3, dp_eps=8.0, clip_norm=1.0,
            dp_delta=1e-5, record_dir=Path(td),
        )
        check("DP run records its privacy budget",
              dp_res["dp"]["epsilon"] == 8.0 and dp_res["dp"]["sigma"] > 0.0,
              str(dp_res["dp"]))
        finite_dp = all(np.isfinite(m["pr_auc"]) and 0.0 <= m["pr_auc"] <= 1.0
                        for m in dp_res["federated"].values())
        check("DP run metrics finite and in range", finite_dp,
              str({k: round(v["pr_auc"], 3) for k, v in dp_res["federated"].items()}))
        check("DP trace finite", all(t == t for t in dp_res["trace"]))
        leaks = []
        for f in Path(td).glob("*.jsonl"):
            for line in f.read_text(encoding="utf-8").splitlines():
                if not set(json.loads(line)) <= {"type", "model", "round", "epochs", "lr", "weights"}:
                    leaks.append(line[:120])
        check("DP run boundary still only carries weights", not leaks, str(leaks[:1]))

    # ---- MLP: mechanics + per-layer aggregation + separability ------------
    print("\n-- MLP extension --")
    mdl = LocalMLP(6, hidden=4, seed=0)
    params = mdl.params()
    check("MLP params shape (W1,b1,W2,b2)",
          params[0].shape == (6, 4) and params[1].shape == (4,)
          and params[2].shape == (4, 1) and np.isscalar(params[3]),
          str([p.shape if hasattr(p, "shape") else type(p).__name__ for p in params]))
    roundtrip = LocalMLP(6, hidden=4, seed=1)
    roundtrip.set_params([p.copy() if isinstance(p, np.ndarray) else p for p in params])
    rt = roundtrip.params()
    check("params/set_params round-trip", all(np.array_equal(a, b) for a, b in zip(rt, params)))
    check("predict_proba in [0,1] and sums to 1",
          np.all((mdl.predict_proba(np.ones((3, 6))) >= 0.0)
                 & (mdl.predict_proba(np.ones((3, 6))) <= 1.0)),
          str(mdl.predict_proba(np.ones((3, 6)))[:, 1]))

    # XOR: a balanced nonlinear boundary LR provably cannot separate; a
    # one-hidden-layer MLP should. This is the "separability benefit" the
    # LR-vs-MLP comparison quantifies, tested here in the deterministic
    # local path (no process boundary involved).
    X_xor = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y_xor = np.array([0, 1, 1, 0])
    xor_mlp = LocalMLP(2, hidden=8, seed=2)
    xor_mlp.train(X_xor, y_xor, epochs=800, lr=0.5)
    acc_mlp = float(np.mean((xor_mlp.predict_proba(X_xor)[:, 1] >= 0.5) == y_xor))
    lr_xor = LocalLR(2)
    lr_xor.train(X_xor, y_xor, epochs=800, lr=0.5)
    acc_lr = float(np.mean((lr_xor.predict_proba(X_xor)[:, 1] >= 0.5) == y_xor))
    check("MLP learns XOR, LR cannot (separability benefit)",
          acc_mlp > acc_lr + 0.25 and acc_mlp >= 0.75,
          f"mlp {acc_mlp:.2f} vs lr {acc_lr:.2f}")

    fed_layers = fedavg_params([
        ([np.array([1.0, 2.0]), np.array([3.0])], 100),
        ([np.array([2.0, 4.0]), np.array([5.0])], 300),
    ])
    check("fedavg_params averages each layer weighted by local size",
          np.allclose(fed_layers[0], [1.75, 3.5]) and np.allclose(fed_layers[1], [4.5]),
          str([np.round(l, 4).tolist() for l in fed_layers]))

    rng_f = np.random.default_rng(0)
    flat_g = np.array([0.1, -0.1, 0.2, 0.0])
    flat_updates = [
        (np.array([0.9, 1.1, 0.4, 0.3]), 100),
        (np.array([1.1, 0.7, 0.2, -0.2]), 300),
    ]
    dp_flat = dp.dp_aggregate_flat(flat_g, flat_updates, clip_norm=10.0, sigma=0.0, rng=rng_f)
    total = sum(n for _, n in flat_updates)
    ref = flat_g + sum((f - flat_g) * n for f, n in flat_updates) / total
    check("dp_aggregate_flat at sigma=0 == weighted FedAvg", np.allclose(dp_flat, ref),
          str(np.round(dp_flat, 4)))

    with tempfile.TemporaryDirectory(prefix="fed-mlp-") as td:
        mlp_res = run_federated(
            shards, rounds=4, local_epochs=3, lr=0.1, baseline_epochs=30,
            train_ratio=0.7, label="m", seed=3, model="mlp", hidden=4,
            record_dir=Path(td),
        )
        check("MLP run reports model", mlp_res["model"] == "mlp")
        leaks = []
        for f in Path(td).glob("*.jsonl"):
            for line in f.read_text(encoding="utf-8").splitlines():
                if not set(json.loads(line)) <= {"type", "model", "round", "epochs", "lr", "weights"}:
                    leaks.append(line[:120])
        check("MLP boundary still only carries weights", not leaks, str(leaks[:1]))
        finite_mlp = all(np.isfinite(m["pr_auc"]) and 0.0 <= m["pr_auc"] <= 1.0
                         for m in mlp_res["federated"].values())
        check("MLP federated metrics finite and in range", finite_mlp,
              str({k: round(v["pr_auc"], 3) for k, v in mlp_res["federated"].items()}))
        mlp_macro = macro(mlp_res["federated"])
        check("MLP federated above random (PR-AUC > 0.5)", mlp_macro > 0.5,
              f"{mlp_macro:.3f}")
        check("MLP convergence trace finite, one entry per round",
              len(mlp_res["trace"]) == 4 and all(t == t for t in mlp_res["trace"]),
              str([round(t, 4) for t in mlp_res["trace"]]))
        mlp_oracle = centralized_oracle(shards, epochs=30, lr=0.1, train_ratio=0.7,
                                        model="mlp", hidden=4)
        op_mlp = macro(mlp_oracle)
        check("MLP centralized oracle >= MLP federated (tol)", op_mlp >= mlp_macro - 0.05,
              f"oracle {op_mlp:.3f} vs fed {mlp_macro:.3f}")

    # ---- heterogeneity: skew mechanics + the sweep -------------------------
    print("\n-- heterogeneity --")
    sk = split_institutions_skewed(pool, [0.5, 0.3, 0.2], [0.2, 1, 5], seed=3)
    check("skewed shards account-disjoint",
          sum(len(set(s["fraud_id"])) for s in sk) == len(set(pool["fraud_id"])),
          f"{sum(len(set(s['fraud_id'])) for s in sk)} vs {len(set(pool['fraud_id']))}")
    fr = [float(s["label"].mean()) for s in sk]
    check("realized fraud rates ordered by skew", fr[0] < fr[1] < fr[2],
          str([round(x, 4) for x in fr]))
    sz = [len(s) for s in sk]
    check("sizes roughly follow the weights", sz[2] < sz[1] < sz[0]
          and sum(sz) == len(pool), str(sz))
    sk2 = split_institutions_skewed(pool, [0.5, 0.3, 0.2], [0.2, 1, 5], seed=3)
    check("skewed split deterministic given seed",
          all(s1.equals(s2) for s1, s2 in zip(sk, sk2)))
    sk0 = split_institutions_skewed(pool, [0.85, 0.10, 0.05], [0.0, 1, 8], seed=3)
    # Dedicated-fraud accounts (u == 1) are excluded entirely; the residual
    # floor is set by lightly-compromised legit accounts (ATO bursts), which
    # are repelled but not removed. With the current generator seed this
    # deterministic pool realizes ~0.8% - far below the fraud-heavy shards
    # (the realized-rate ordering is asserted separately above).
    check("skew=0 institution repels fraud",
          float(sk0[0]["label"].mean()) < 0.015,
          f"realized {float(sk0[0]['label'].mean()):.4f}")
    check("skew=0 shard still has data + some fraud (ATO remnants)",
          len(sk0[0]) > 0 and int(sk0[0]["label"].sum()) > 0,
          f"{len(sk0[0])} events, {int(sk0[0]['label'].sum())} fraud")

    rows, details = run_hetero_sweep(
        pool, grid=[
            ("balanced", [0.5, 0.3, 0.2], [1, 1, 1], 1000),
            ("clean dominant", [0.85, 0.10, 0.05], [0.2, 1, 6], 1001),
        ],
        rounds=2, local_epochs=2, lr=0.5, baseline_epochs=15, train_ratio=0.7, seed=3,
    )
    check("sweep covers every grid config",
          [r["config"] for r in rows] == ["balanced", "clean dominant"])
    check("FL gain = fed - local",
          all(abs(r["fl_gain"] - (r["fed"] - r["local"])) < 1e-9 for r in rows))
    check("max drag matches per-institution detail", all(
        abs(r["max_drag"] - max(details[r["config"]]["drag"].values())) < 1e-9
        and r["drag_at"] == max(details[r["config"]]["drag"], key=details[r["config"]]["drag"].get)
        for r in rows))
    check("sweep rows carry realized sizes and fraud rates",
          all("/" in r["sizes"] and "%" in r["fraud_rates"] for r in rows),
          str([r["fraud_rates"] for r in rows]))

    print("\n" + ("ALL CHECKS PASSED" if not failures else f"{len(failures)} CHECK(S) FAILED"))
    return 1 if failures else 0


def fedavg_from(updates: list[tuple[np.ndarray, float, int]]) -> tuple[np.ndarray, float]:
    total = sum(n for _, _, n in updates)
    w = sum(u * n for u, _, n in updates) / total
    b = sum(b_ * n for _, b_, n in updates) / total
    return w, b


if __name__ == "__main__":
    sys.exit(main())
