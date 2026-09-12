#!/usr/bin/env python3
"""Federated institution worker process (architecture section 19).

Spawned by `scripts/federated_sim.py`, one per simulated institution. Each
worker owns its disjoint local dataset; over stdin/stdout it exchanges ONLY
weight vectors and aggregate metrics with the coordinator. Raw data, feature
vectors, and labels never leave this process — the boundary is the model.

`--model lr|mlp` selects the architecture (balanced logistic regression or
a one-hidden-layer ReLU MLP); the weight payload is {"w", "b"} for LR and
{"params": [W1, b1, W2, b2]} for the MLP.

Protocol (one JSON object per line on stdin, replies on stdout):
  {"type":"train", "model":m, "round":r, "epochs":e, "lr":x, "weights":...}
      -> {"type":"weights", "name":n, "n_train":k, "weights":...}
  {"type":"evaluate", "model":m, "weights":...}
      -> {"type":"metrics", "name":n, ...eval, "n_test":k, "n_pos":j}
  {"type":"baseline", "model":m, "epochs":e, "lr":x}  (train from init, then eval)
      -> {"type":"metrics", ...}
  {"type":"shutdown"}                          -> exit
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent  # repo root.parent  # repo root
sys.path.insert(0, str(ROOT / "backend"))

from src.federated.lr import LocalLR, eval_metrics  # noqa: E402
from src.federated.mlp import LocalMLP  # noqa: E402
from src.privacy_layer.features import ML_FEATURES  # noqa: E402


def make_model(n_features: int, model: str, hidden: int) -> LocalLR | LocalMLP:
    return LocalLR(n_features) if model == "lr" else LocalMLP(n_features, hidden=hidden)


def weights_of(model: LocalLR | LocalMLP) -> dict:
    """The wire representation of a model's params (b2 stays a scalar)."""
    if isinstance(model, LocalMLP):
        return {"params": [float(p) if np.ndim(p) == 0 else p.tolist()
                           for p in model.params()]}
    return {"w": model.w.tolist(), "b": float(model.b)}


def load_weights(model: LocalLR | LocalMLP, weights: dict) -> None:
    """Set a model's params from the wire representation."""
    if isinstance(model, LocalMLP):
        model.set_params([float(p) if np.ndim(p) == 0 else np.asarray(p, dtype=float)
                          for p in weights["params"]])
    else:
        model.w = np.asarray(weights["w"], dtype=float)
        model.b = float(weights["b"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True, help="this institution's disjoint CSV")
    ap.add_argument("--name", default="inst", help="institution id")
    ap.add_argument("--train-ratio", type=float, default=0.7, help="time-based local split")
    ap.add_argument("--model", default="lr", choices=["lr", "mlp"],
                    help="architecture: balanced logistic regression or a one-hidden-layer MLP")
    ap.add_argument("--hidden", type=int, default=16, help="MLP hidden size (--model mlp)")
    args = ap.parse_args()

    # ---- load & split the LOCAL dataset (time-based, section 6) -----------
    df = pd.read_csv(args.data, parse_dates=["ts"]).sort_values("ts").reset_index(drop=True)
    n = len(df)
    split = int(n * args.train_ratio)
    train, test = df.iloc[:split], df.iloc[split:]
    # Intersect ML_FEATURES with columns actually present in the CSV
    # (production-only features like shared_device_accounts may not exist here)
    available = [f for f in ML_FEATURES if f in df.columns]
    if len(available) < len(ML_FEATURES):
        import sys; print(f"  [info] {len(ML_FEATURES) - len(available)} features not in CSV, using {len(available)} available", file=sys.stderr, flush=True)
    Xtr = train[available].to_numpy(dtype=float)
    ytr = train["label"].to_numpy(dtype=int)
    Xte = test[available].to_numpy(dtype=float)
    yte = test["label"].to_numpy(dtype=int)

    # Local standardization: each institution scales its own features with its
    # own statistics (no cross-institution statistics are shared). The shared
    # model therefore lives in each institution's standardized space.
    mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0)
    sd[sd == 0] = 1.0
    Xtr = (Xtr - mu) / sd
    Xte = (Xte - mu) / sd

    def respond(obj: dict) -> None:
        sys.stdout.write(json.dumps(obj) + "\n")
        sys.stdout.flush()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        msg = json.loads(line)
        typ = msg.get("type")
        if typ == "train":
            model = make_model(Xtr.shape[1], args.model, args.hidden)
            load_weights(model, msg["weights"])
            model.train(Xtr, ytr, epochs=int(msg["epochs"]), lr=float(msg["lr"]))
            respond({
                "type": "weights",
                "name": args.name,
                "n_train": int(len(ytr)),
                "weights": weights_of(model),
            })
        elif typ == "evaluate":
            model = make_model(Xtr.shape[1], args.model, args.hidden)
            load_weights(model, msg["weights"])
            p1 = model.predict_proba(Xte)[:, 1]
            respond({
                "type": "metrics",
                "name": args.name,
                **eval_metrics(yte, p1),
                "n_test": int(len(yte)),
                "n_pos": int(yte.sum()),
            })
        elif typ == "baseline":
            model = make_model(Xtr.shape[1], args.model, args.hidden)
            model.train(Xtr, ytr, epochs=int(msg["epochs"]), lr=float(msg["lr"]))
            p1 = model.predict_proba(Xte)[:, 1]
            respond({
                "type": "metrics",
                "name": args.name,
                **eval_metrics(yte, p1),
                "n_test": int(len(yte)),
                "n_pos": int(yte.sum()),
            })
        elif typ == "shutdown":
            break
    return 0


if __name__ == "__main__":
    sys.exit(main())
