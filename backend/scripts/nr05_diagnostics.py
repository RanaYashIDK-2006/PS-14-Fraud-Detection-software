#!/usr/bin/env python3
"""NR-05 — Exploratory method and evaluation diagnostics (EXPLORATORY ONLY).

Boundary (see docs/RESEARCH_PLAN.md §14 and
docs/evaluation/DATASET_EXPOSURE_LEDGER.md):

  * Uses ONLY previously exposed datasets: ULB, Kaggle fraudTrain/fraudTest,
    IBM v2. Nothing here makes an exposed dataset "untouched".
  * Every result produced by this script is EXPLORATORY / NON-INDEPENDENT.
    It must never be presented as confirmatory Track M evidence, as
    independent replication, or as a preregistered result.
  * No freeze placeholder is resolved here; every methodological choice made
    for convenience is recorded in the emitted JSON + the NR-05 report as an
    issue for the later statistical-review/freeze process.
  * The Research Plan stays DRAFT / NOT FROZEN / NOT APPROVED.

What it does, per dataset (Track M method-level diagnostics):

  1. Discrimination: simple baseline (LR) and representative single model
     (XGB) vs the intended ensemble (LR+RF+XGB+IF mean → Platt on the CAL
     window), same data/split/features/preprocessing/tuning budget for all.
  2. Calibration: training → calibration/validation → test; Brier + ECE on
     the test window (functions reused from calibration_test.py).
  3. Temporal / entity / segment / business (alert-volume) diagnostics.
  4. Gating: row-level contract enforcement (Phase-41 semantics rebuilt from
     the TRAINING window only) + window-level PSI gate (src/drift_monitor),
     evaluated on Class A (designed-for) and Class B (not-designed-for)
     perturbations vs an explicitly matched ungated comparator.
  5. Dependence-aware resampling (entity-clustered / temporal-block) marked
     EXPLORATORY — METHOD NOT YET FROZEN; IID shown only for contrast.

Outputs:
  reports/nr05/<experiment>.json         — machine-readable results
  reports/nr05/artifacts/<experiment>/*  — trained model artifacts (joblib)
  reports/evaluation_runs/eval_ledger.jsonl — append-only evidence records
      (existing eval_record.py infrastructure; NOT a competing schema)

Usage:
  python backend/scripts/nr05_diagnostics.py --datasets ulb,kaggle,ibm
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "backend" / "scripts"))

from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from eval_record import EvaluationLedger, create_evaluation_record
from metric_definitions import (
    bootstrap_ci,
    confusion_counts,
    prevalence_report,
    pr_auc,
    roc_auc,
    small_sample_warning,
    threshold_at_fpr_on_validation,
    recall_at_fpr,
)
from calibration_test import compute_brier_score, compute_ece, reliability_curve
from src.drift_monitor.psi import PSI_ALERT, build_baseline, check_window

# --------------------------------------------------------------------------
# Exploratory constants — EVERY one of these is an NR-05 exploratory choice
# that the confirmatory freeze must later ratify or replace. They are fixed
# a priori (before any result was inspected) and never tuned to outcomes.
# --------------------------------------------------------------------------
SEED = 42                      # single seed (§18 min-seeds is [TO BE FROZEN])
N_BOOT = 300                   # bootstrap replicates (preregistered count PENDING)
CONFIDENT_HI = 0.8             # confident ML decision: p >= 0.8 ...
CONFIDENT_LO = 0.2             # ... or p <= 0.2 (definition frozen for NR-05 only)
INJECT_FRAC = 0.10             # Class A: 10% of test rows perturbed (4 kinds × 2.5%)
CLASSB_FRAC = 0.30             # Class B: 30% of test rows perturbed per condition
NOISE_SIGMAS = (0.25, 0.5, 1.0)   # Class B1: × train-feature std (full range recorded)
AMOUNT_MULTS = (1.10, 1.25, 1.50)  # Class B2: amount multiplier (full range recorded)
HOUR_SHIFT = 3                 # Class B3: cyclic hour shift (window = all rows)
CONTRACT_QLO, CONTRACT_QHI = 0.01, 0.99
CONTRACT_PAD_IQR_MULT = 2.0    # bounds = qlo/qhi ± 2·IQR  (wide: only extremes block)
FALLBACK_Q = 0.995             # fallback rule: amount > train q99.5 → fraud
ALERT_FRACS = (0.01, 0.05)     # business: top-1% / top-5% alert volume
N_TIME_QUARTILES = 4
BOOT_MODES_NOTE = ("EXPLORATORY — METHOD NOT YET FROZEN: dependence-aware "
                   "resampling for diagnosis only; not a confirmatory CI")

OUT_DIR = ROOT / "reports" / "nr05"
ART_DIR = OUT_DIR / "artifacts"
LEDGER_PATH = ROOT / "reports" / "evaluation_runs" / "eval_ledger.jsonl"


# --------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------
def sanitize(obj):
    """JSON-safe: NaN/inf → None, numpy scalars → python."""
    if isinstance(obj, dict):
        return {str(k): sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return f if math.isfinite(f) else None
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return sanitize(obj.tolist())
    if isinstance(obj, Path):
        return str(obj)
    return obj


def emit(name: str, payload: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    p = OUT_DIR / f"{name}.json"
    p.write_text(json.dumps(sanitize(payload), indent=2, sort_keys=True),
                 encoding="utf-8")
    print(f"[nr05] wrote {p.relative_to(ROOT)}")


def save_models(exp_id: str, members: dict) -> list[Path]:
    import joblib
    d = ART_DIR / exp_id
    d.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, model in members.items():
        p = d / f"{name}.joblib"
        joblib.dump(model, p)
        paths.append(p)
    return paths


def append_record(*, exp_id: str, dataset_path: Path, artifact_paths: list,
                  metrics: dict, config: dict, seed: int,
                  threshold: float | None, threshold_source: str,
                  feature_schema_version: str, warnings: list,
                  preprocessing_version: str = "median_impute_fit_on_train") -> str:
    """Append one evidence record via the EXISTING eval_record infrastructure."""
    rec = create_evaluation_record(
        model_identifier=exp_id,
        artifact_paths=artifact_paths,
        dataset_path=dataset_path,
        seed=seed,
        threshold=threshold,
        threshold_source=threshold_source,
        preprocessing_version=preprocessing_version,
        feature_schema_version=feature_schema_version,
        evaluation_config=config,
        metrics=sanitize(metrics),
        command="python backend/scripts/nr05_diagnostics.py",
        warnings=warnings,
        status="COMPLETED",
    )
    EvaluationLedger(LEDGER_PATH).append(rec)
    side = ROOT / "reports" / "evaluation_runs" / f"record_{rec.evaluation_id}.json"
    side.write_text(json.dumps(sanitize(rec.to_dict()), indent=2, sort_keys=True),
                    encoding="utf-8")
    print(f"[nr05] evidence record {rec.evaluation_id} → ledger")
    return rec.evaluation_id


# --------------------------------------------------------------------------
# Data loaders — dataset-appropriate representations (Track M), documented
# per dataset; NOT the native 21/48-feature contract.
# --------------------------------------------------------------------------
def temporal_split(n: int, train_frac: float, cal_frac: float):
    i1 = int(n * train_frac)
    i2 = int(n * (train_frac + cal_frac))
    return slice(0, i1), slice(i1, i2), slice(i2, n)


def load_ulb() -> dict:
    """ULB: V1..V28 PCA + Amount/log(Amount). No entity id → entity-disjoint
    NOT APPLICABLE. Chronological order = Time (seconds from first txn)."""
    df = pd.read_csv(ROOT / "data" / "creditcard.csv")
    df = df.sort_values("Time").reset_index(drop=True)
    df["LogAmount"] = np.log1p(df["Amount"].astype(float))
    feats = [f"V{i}" for i in range(1, 29)] + ["Amount", "LogAmount"]
    tr, ca, te = temporal_split(len(df), 0.60, 0.20)
    return dict(
        name="ulb",
        path=ROOT / "data" / "creditcard.csv",
        feats=feats,
        X=df[feats], y=df["Class"].to_numpy(dtype=int),
        t=df["Time"].to_numpy(dtype=float),
        entity=None,
        amount=df["Amount"].to_numpy(dtype=float),
        idx=dict(train=tr, cal=ca, test=te),
        entity_test_idx=None,
        note=("ULB PCA feature representation; Time = seconds from first "
              "transaction (chronological order, no calendar semantics); "
              "entity ids absent → entity-disjoint NOT APPLICABLE"),
        label="Class",
    )


def _kaggle_frame(path: Path, cats: list[str] | None):
    df = pd.read_csv(path)
    dt = pd.to_datetime(df["trans_date_trans_time"], format="%Y-%m-%d %H:%M:%S")
    out = pd.DataFrame(index=df.index)
    amt = df["amt"].astype(float)
    out["amt"] = amt
    out["log_amt"] = np.log1p(amt)
    out["hour"] = dt.dt.hour.astype(float)
    out["dow"] = dt.dt.dayofweek.astype(float)
    out["is_weekend"] = dt.dt.dayofweek.isin([5, 6]).astype(float)
    if cats is None:
        cats = sorted(df["category"].unique().tolist())
    out["category_code"] = pd.Categorical(df["category"], categories=cats).codes.astype(float)
    out["log_city_pop"] = np.log1p(df["city_pop"].astype(float))
    lat, lon = df["lat"].astype(float), df["long"].astype(float)
    mlat, mlon = df["merch_lat"].astype(float), df["merch_long"].astype(float)
    out["dist_km"] = haversine_km(lat, lon, mlat, mlon)
    age = (dt - pd.to_datetime(df["dob"])).dt.days / 365.25
    out["age"] = age.astype(float)
    meta = pd.DataFrame({
        "unix_time": df["unix_time"].astype(float),
        "cc_num": df["cc_num"].astype(str),
        "is_fraud": df["is_fraud"].astype(int),
        "amt": amt,
        "ts": dt,
    })
    return out, meta, cats


def haversine_km(lat1, lon1, lat2, lon2) -> pd.Series:
    r = 6371.0
    p1, p2 = np.radians(lat1.to_numpy()), np.radians(lat2.to_numpy())
    dp = np.radians(lat2.to_numpy() - lat1.to_numpy())
    dl = np.radians(lon2.to_numpy() - lon1.to_numpy())
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return pd.Series(2 * r * np.arcsin(np.sqrt(a)))


def load_kaggle(rng: np.random.Generator) -> tuple[dict, dict]:
    """Kaggle fraudTrain/fraudTest: dataset-appropriate 9-feature
    representation, feature categories fitted on TRAIN only.
    Returns (discrimination_setup, entity_disjoint_setup)."""
    tr_path = ROOT / "data" / "kaggle_fraud" / "fraudTrain.csv"
    te_path = ROOT / "data" / "kaggle_fraud" / "fraudTest.csv"
    Xtr_full, mtr, cats = _kaggle_frame(tr_path, None)
    Xte, mte, _ = _kaggle_frame(te_path, cats)
    order = np.argsort(mtr["unix_time"].to_numpy(), kind="stable")
    n_tr = int(len(order) * 0.70)
    tr_i, ca_i = order[:n_tr], order[n_tr:]
    feats = Xtr_full.columns.tolist()
    disc = dict(
        name="kaggle",
        path=tr_path,
        path_test=te_path,
        feats=feats,
        X=Xtr_full, y=mtr["is_fraud"].to_numpy(int),
        t=mtr["unix_time"].to_numpy(float),
        entity=mtr["cc_num"].to_numpy(str),
        amount=mtr["amt"].to_numpy(float),
        idx=dict(train=slice(None), cal=slice(None), test=slice(None)),
        row_index=dict(train=tr_i, cal=ca_i, test=None),
        X_test=Xte, y_test=mte["is_fraud"].to_numpy(int),
        t_test=mte["unix_time"].to_numpy(float),
        entity_test=mte["cc_num"].to_numpy(str),
        amount_test=mte["amt"].to_numpy(float),
        note=("9-feature dataset-appropriate representation; category codes "
              "fit on TRAIN only; fraudTest is the natural chronological "
              "final test (already exposed — exploratory only)"),
        label="is_fraud",
        temporal_test=True,
    )
    # entity-disjoint variant: hold out 20% of cc_num from fraudTrain
    users = np.unique(mtr["cc_num"].to_numpy(str))
    held = rng.choice(users, size=max(1, int(len(users) * 0.20)), replace=False)
    held_set = set(held.tolist())
    is_held = np.array([u in held_set for u in mtr["cc_num"].to_numpy(str)])
    rest = order[~is_held[order]]
    n_rest_tr = int(len(rest) * 0.70)
    ent = dict(
        name="kaggle_entity",
        path=tr_path,
        feats=feats,
        X=Xtr_full, y=mtr["is_fraud"].to_numpy(int),
        t=mtr["unix_time"].to_numpy(float),
        entity=mtr["cc_num"].to_numpy(str),
        amount=mtr["amt"].to_numpy(float),
        row_index=dict(train=rest[:n_rest_tr], cal=rest[n_rest_tr:],
                       test=np.where(is_held)[0]),
        note=(f"entity-disjoint: {len(held)} of {len(users)} cc_num held out "
              "(20%, seed 42); train/cal on remaining users' rows; test = "
              "held-out-user rows (same time period)"),
        label="is_fraud",
        n_held_users=int(len(held)),
        n_users=int(len(users)),
    )
    return disc, ent


def load_ibm(stride: int = 37) -> dict:
    """IBM v2: stride sample over the whole 24.4M-row file (deterministic,
    no seed), 11 dataset-appropriate features, temporal 60/20/20 split,
    entity = User."""
    path = ROOT / "data" / "credit_card_transactions-ibm_v2.csv"
    usecols = ["User", "Card", "Year", "Month", "Day", "Time", "Amount",
               "Use Chip", "MCC", "Errors?", "Is Fraud?"]
    df = pd.read_csv(path, usecols=usecols,
                     skiprows=lambda i: i > 0 and i % stride != 0)
    dt = (pd.to_datetime(dict(year=df["Year"], month=df["Month"], day=df["Day"]),
                         errors="coerce")
          + pd.to_timedelta(df["Time"].astype(str).str.split(":", expand=True)[0]
                            .astype(float), unit="h")
          + pd.to_timedelta(df["Time"].astype(str).str.split(":", expand=True)[1]
                            .astype(float), unit="m"))
    amt = pd.to_numeric(df["Amount"].astype(str).str.replace("$", "", regex=False)
                        .str.replace(",", "", regex=False), errors="coerce")
    out = pd.DataFrame(index=df.index)
    out["amount"] = amt
    out["log_amount"] = np.log1p(amt)
    out["hour"] = dt.dt.hour.astype(float)
    out["dow"] = dt.dt.dayofweek.astype(float)
    out["is_weekend"] = dt.dt.dayofweek.isin([5, 6]).astype(float)
    out["day"] = df["Day"].astype(float)
    out["month"] = df["Month"].astype(float)
    out["chip_code"] = pd.Categorical(df["Use Chip"]).codes.astype(float)
    out["errors_flag"] = df["Errors?"].notna().astype(float)
    # vectorized per-user history (no python loops)
    tmp = out.assign(_ts=dt, _user=df["User"].astype(str))
    tmp = tmp.sort_values(["_user", "_ts"])
    first = tmp.groupby("_user")["_ts"].transform("min")
    tmp["tenure_days"] = (tmp["_ts"] - first).dt.total_seconds() / 86400.0
    med = tmp.groupby("_user")["amount"].transform("median")
    tmp["amount_ratio"] = (tmp["amount"] / med.clip(lower=0.01)).astype(float)
    y = (df.loc[tmp.index, "Is Fraud?"].astype(str).str.strip()
         .isin(["Yes", "yes", "1", "True"]).astype(int).to_numpy())
    user = df.loc[tmp.index, "User"].astype(str).to_numpy()
    ts_vals = tmp["_ts"].to_numpy()
    # chronological order for the temporal split
    ord2 = np.argsort(ts_vals.astype("datetime64[s]").astype(np.int64), kind="stable")
    out = tmp.drop(columns=["_ts", "_user"]).iloc[ord2].reset_index(drop=True)
    y = y[ord2]
    user = user[ord2]
    ts_vals = ts_vals[ord2]
    t = ts_vals.astype("datetime64[s]").astype(np.float64)
    feats = out.columns.tolist()
    tr, ca, te = temporal_split(len(out), 0.60, 0.20)
    return dict(
        name="ibm",
        path=path,
        feats=feats,
        X=out, y=y,
        t=t,
        entity=user,
        amount=out["amount"].to_numpy(float),
        idx=dict(train=tr, cal=ca, test=te),
        note=(f"stride-{stride} deterministic sample of the 24.4M-row file "
              f"({len(out)} rows), 11 dataset-appropriate features, "
              "vectorized per-user tenure/amount-ratio history; temporal "
              "60/20/20 split after sorting by parsed timestamp"),
        label="Is Fraud?",
        n_file_rows=24_386_899,
        stride=stride,
    )


# ---- split accessor: one convention for all datasets ---------------------
def split_arrays(ds: dict):
    """-> (train, cal, test) each a dict(X, y, t, entity, amount)."""
    def pack(X, y, t, ent, amt, rows=None):
        if rows is not None:
            return dict(X=X.iloc[rows], y=y[rows], t=t[rows],
                        entity=ent[rows], amount=amt[rows])
        return dict(X=X, y=y, t=t, entity=ent, amount=amt)

    if ds["name"] in ("ulb", "ibm"):
        i = ds["idx"]
        X, y, t = ds["X"], ds["y"], ds["t"]
        ent = (ds.get("entity") if ds.get("entity") is not None
               else np.array(["na"] * len(y)))
        amt = ds["amount"]
        return (pack(X.iloc[i["train"]], y[i["train"]], t[i["train"]],
                     ent[i["train"]], amt[i["train"]]),
                pack(X.iloc[i["cal"]], y[i["cal"]], t[i["cal"]],
                     ent[i["cal"]], amt[i["cal"]]),
                pack(X.iloc[i["test"]], y[i["test"]], t[i["test"]],
                     ent[i["test"]], amt[i["test"]]))
    if ds["name"] == "kaggle":
        ri = ds["row_index"]
        X, y, t, ent, amt = ds["X"], ds["y"], ds["t"], ds["entity"], ds["amount"]
        test = pack(ds["X_test"], ds["y_test"], ds["t_test"],
                    ds["entity_test"], ds["amount_test"])
        return (pack(X, y, t, ent, amt, rows=ri["train"]),
                pack(X, y, t, ent, amt, rows=ri["cal"]), test)
    if ds["name"] == "kaggle_entity":
        ri = ds["row_index"]
        X, y, t, ent, amt = ds["X"], ds["y"], ds["t"], ds["entity"], ds["amount"]
        return (pack(X, y, t, ent, amt, rows=ri["train"]),
                pack(X, y, t, ent, amt, rows=ri["cal"]),
                pack(X, y, t, ent, amt, rows=ri["test"]))
    raise KeyError(ds["name"])


# ---- models ---------------------------------------------------------------
def make_members(seed: int) -> dict:
    """Fixed-a-priori configs — NO hyperparameter search (NR-05 §6): every
    dataset/model family gets the identical budget."""
    return {
        "lr": make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=1000, C=1.0,
                                               random_state=seed)),
        "rf": RandomForestClassifier(n_estimators=100, min_samples_leaf=5,
                                     n_jobs=-1, random_state=seed),
        "xgb": XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                             subsample=0.8, colsample_bytree=0.8,
                             random_state=seed, n_jobs=-1,
                             eval_metric="logloss"),
        "if": IsolationForest(n_estimators=100, random_state=seed, n_jobs=-1),
    }


def train_members(tr: dict, seed: int):
    """Fit members on TRAIN with shared train-median imputation.
    Returns (members, state) where state = imputation + IF normalization."""
    Xtr = tr["X"]
    medians = Xtr.median(numeric_only=True).fillna(0.0)
    Xi = Xtr.fillna(medians).to_numpy(dtype=float)
    y = tr["y"]
    members = make_members(seed)
    t0 = time.time()
    members["lr"].fit(Xi, y)
    print(f"    lr   fit {time.time() - t0:.1f}s")
    t0 = time.time()
    members["rf"].fit(Xi, y)
    print(f"    rf   fit {time.time() - t0:.1f}s")
    t0 = time.time()
    members["xgb"].fit(Xi, y)
    print(f"    xgb  fit {time.time() - t0:.1f}s")
    t0 = time.time()
    members["if"].fit(Xi)
    # fraud direction: higher = more anomalous = more fraud-like — negated
    # decision_function, matching the repo's FusionEngine (fusion.py:
    # ``iso_score = -decision_function``), so the member is aligned with the
    # others before averaging.
    d = -members["if"].decision_function(Xi)
    print(f"    if   fit {time.time() - t0:.1f}s")
    state = dict(medians=medians.to_dict(),
                 if_min=float(np.min(d)), if_max=float(np.max(d)),
                 feats=Xtr.columns.tolist())
    return members, state


def member_scores(members: dict, state: dict, X: pd.DataFrame) -> dict:
    # Same frozen preprocessing for EVERY system (gated and ungated):
    # coerce to numeric → ±Inf replaced by train median → NaN filled with
    # train median. This is why the ungated comparator can score invalid
    # inputs instead of crashing (plan §5.1).
    Xc = X.apply(pd.to_numeric, errors="coerce")
    Xc = Xc.replace([np.inf, -np.inf], np.nan)
    Xi = Xc.fillna(pd.Series(state["medians"])).to_numpy(dtype=float)
    out = {}
    for name in ("lr", "rf", "xgb"):
        out[name] = members[name].predict_proba(Xi)[:, 1]
    d = -members["if"].decision_function(Xi)  # fraud direction (fusion.py parity)
    span = max(state["if_max"] - state["if_min"], 1e-12)
    out["if"] = np.clip((d - state["if_min"]) / span, 0.0, 1.0)
    out["ensemble_raw"] = np.mean([out[k] for k in ("lr", "rf", "xgb", "if")],
                                   axis=0)
    return out


# ---- evaluation core ------------------------------------------------------
def score_bundle(y, scores, t_thr: float, prefix: str = "") -> dict:
    """Headline metrics per metric_definitions.py semantics."""
    p = prefix
    out = {}
    out[p + "roc_auc"] = roc_auc(y, scores)
    out[p + "pr_auc"] = pr_auc(y, scores)
    out[p + "recall_at_1pct_fpr"] = recall_at_fpr(y, scores, t_thr)
    tp, fp, tn, fn = confusion_counts(y, scores, t_thr)
    out[p + "threshold"] = float(t_thr)
    out[p + "tp"], out[p + "fp"], out[p + "tn"], out[p + "fn"] = tp, fp, tn, fn
    out[p + "precision_at_thr"] = tp / (tp + fp) if (tp + fp) else None
    out[p + "recall_at_thr"] = tp / (tp + fn) if (tp + fn) else None
    # business / alert-volume metrics (ranking-based, no labels used)
    order = np.argsort(-scores, kind="stable")
    n = len(y)
    for frac in ALERT_FRACS:
        k = max(1, int(n * frac))
        caught = int(y[order[:k]].sum())
        out[p + f"recall_at_top{int(frac * 100)}pct"] = caught / y.sum() if y.sum() else None
        out[p + f"alerts_top{int(frac * 100)}pct"] = k
    out[p + "prevalence"] = prevalence_report(y)
    warns = small_sample_warning(y)
    if warns:
        out[p + "small_sample_warnings"] = warns
    return out


def dependence_note(mode: str, entity_available: bool) -> str:
    if mode == "entity":
        return (BOOT_MODES_NOTE + " | entity-clustered bootstrap (resample entity "
                "clusters with replacement)")
    if mode == "block":
        return (BOOT_MODES_NOTE + " | temporal block bootstrap (20 contiguous "
                "time blocks, blocks sampled with replacement)")
    return (BOOT_MODES_NOTE + " | IID stratified row bootstrap (baseline for "
            "contrast only — NOT the confirmatory default)")


def _resample(mode: str, y, entities, times, rng):
    if mode == "iid":
        parts = []
        for cls in (0, 1):
            c = np.where(y == cls)[0]
            if len(c):
                parts.append(rng.choice(c, size=len(c), replace=True))
        return np.concatenate(parts) if parts else np.arange(len(y))
    if mode == "entity":
        uniq, inv = np.unique(entities, return_inverse=True)
        chosen = rng.choice(len(uniq), size=len(uniq), replace=True)
        return np.concatenate([np.where(inv == c)[0] for c in chosen])
    if mode == "block":
        order = np.argsort(times, kind="stable")
        blocks = np.array_split(order, 20)
        chosen = rng.integers(0, len(blocks), size=len(blocks))
        return np.concatenate([blocks[c] for c in chosen])
    raise ValueError(mode)


def boot_compare(y, s_main, s_ref_a, s_ref_b, mode, entity, times,
                 n=N_BOOT, seed=SEED) -> dict:
    """One resampling loop → CI for main metric + paired deltas vs both refs.
    All dependence-aware intervals are EXPLORATORY — METHOD NOT YET FROZEN."""
    y = np.asarray(y, dtype=int)
    rng = np.random.default_rng(seed)
    vals, d_a, d_b = [], [], []
    for _ in range(n):
        idx = _resample(mode, y, entity, times, rng)
        yy = y[idx]
        if len(np.unique(yy)) < 2:
            continue
        m0 = roc_auc(yy, s_main[idx])
        ma = roc_auc(yy, s_ref_a[idx])
        mb = roc_auc(yy, s_ref_b[idx])
        vals.append(m0)
        d_a.append(m0 - ma)
        d_b.append(m0 - mb)
    def ci(v, point):
        if len(v) < n * 0.5:
            return {"point": point, "ci_low": None, "ci_high": None,
                    "withheld_reason": f"only {len(v)}/{n} draws had both classes",
                    "n_valid": len(v)}
        lo, hi = np.percentile(v, [2.5, 97.5])
        return {"point": point, "ci_low": float(lo), "ci_high": float(hi),
                "n_valid": len(v)}
    full = roc_auc(y, s_main)
    full_a = roc_auc(y, s_ref_a)
    full_b = roc_auc(y, s_ref_b)
    return {
        "mode": mode, "n_bootstrap": n, "seed": seed,
        "label": dependence_note(mode, entity is not None),
        "main_roc_auc": ci(vals, full),
        "paired_delta_vs_xgb": ci(d_a, full - full_a),
        "paired_delta_vs_lr": ci(d_b, full - full_b),
    }


# ---- gating ---------------------------------------------------------------
# Row-level contract: Phase-41 enforcement SEMANTICS (BLOCK on missing /
# non-finite / out-of-range / wrong type) rebuilt from the TRAINING window
# only — plan §5.0 Track M requirement (no native-contract assumptions).
def fit_contract(train_df: pd.DataFrame, feats: list, amount_col: str) -> dict:
    bounds = {}
    for f in feats:
        v = pd.to_numeric(train_df[f], errors="coerce").dropna().to_numpy(float)
        q1, q99 = np.quantile(v, [CONTRACT_QLO, CONTRACT_QHI])
        iqr = float(np.quantile(v, 0.75) - np.quantile(v, 0.25))
        pad = CONTRACT_PAD_IQR_MULT * iqr
        bounds[f] = dict(lo=float(q1 - pad), hi=float(q99 + pad),
                         q1=float(q1), q99=float(q99), iqr=iqr,
                         std=float(np.std(v)), median=float(np.median(v)))
    amt = pd.to_numeric(train_df[amount_col], errors="coerce").dropna().to_numpy(float)
    return dict(bounds=bounds, amount_col=amount_col,
                fallback_thr=float(np.quantile(amt, FALLBACK_Q)),
                medians={f: bounds[f]["median"] for f in feats})


def row_gate(dfw: pd.DataFrame, contract: dict) -> dict:
    feats = list(contract["bounds"].keys())
    blocked = np.zeros(len(dfw), dtype=bool)
    reasons = dict(wrong_type=0, missing=0, non_finite=0, out_of_range=0)
    for f in feats:
        raw = dfw[f]
        is_str = raw.map(lambda v: isinstance(v, str)).to_numpy()
        num = pd.to_numeric(raw, errors="coerce").to_numpy(
            dtype=float, na_value=np.nan)
        is_missing = np.isnan(num) & ~is_str
        is_nonf = np.isinf(num)
        b = contract["bounds"][f]
        is_oor = ~np.isnan(num) & ((num < b["lo"]) | (num > b["hi"]))
        bad = is_str | is_missing | is_nonf | is_oor
        reasons["wrong_type"] += int(is_str.sum())
        reasons["missing"] += int(is_missing.sum())
        reasons["non_finite"] += int(is_nonf.sum())
        reasons["out_of_range"] += int(is_oor.sum())
        blocked |= bad
    return dict(accept=~blocked, blocked=blocked, reasons=reasons)


A_KINDS = ("A_nan", "A_inf", "A_oor", "A_type")


def perturb(df: pd.DataFrame, kind: str, contract: dict, feats: list,
            rng: np.random.Generator) -> pd.DataFrame:
    """Deterministic given rng. Labels are NEVER modified."""
    out = df.copy()
    n = len(out)
    if kind == "clean":
        return out
    if kind in A_KINDS:
        # exactly ONE injection kind per condition; row sets are disjoint
        # quarters of a single permutation (each kind = INJECT_FRAC/4 of rows)
        perm = rng.permutation(n)
        per = int(n * INJECT_FRAC) // 4
        i = A_KINDS.index(kind)
        rows = perm[i * per:(i + 1) * per]
        cols = rng.integers(0, len(feats), size=len(rows))
        if kind == "A_type":
            for c in sorted(set(cols.tolist())):
                out[feats[c]] = out[feats[c]].astype(object)
        for r, c in zip(rows, cols):
            f = feats[c]
            b = contract["bounds"][f]
            if kind == "A_nan":
                out.at[r, f] = np.nan
            elif kind == "A_inf":
                out.at[r, f] = np.inf
            elif kind == "A_oor":
                out.at[r, f] = b["hi"] + 10.0 * abs(b["iqr"] + 1.0)
            else:
                out.at[r, f] = "not_a_number"
        return out
    if kind.startswith("B_noise_"):
        sigma = float(kind.split("B_noise_")[1])
        rows = rng.permutation(n)[:int(n * CLASSB_FRAC)]
        top3 = sorted(feats, key=lambda f: -contract["bounds"][f]["std"])[:3]
        for f in top3:
            sd = contract["bounds"][f]["std"]
            delta = rng.normal(0.0, sigma * sd, size=len(rows))
            out.loc[rows, f] = (pd.to_numeric(out.loc[rows, f], errors="coerce")
                                + delta).to_numpy()
        return out
    if kind.startswith("B_amount_"):
        mult = float(kind.split("B_amount_")[1])
        f = contract["amount_col"]
        rows = rng.permutation(n)[:int(n * CLASSB_FRAC)]
        out.loc[rows, f] = (pd.to_numeric(out.loc[rows, f], errors="coerce")
                            * mult).to_numpy()
        return out
    if kind == "B_hour":
        if "hour" in feats:
            out["hour"] = (pd.to_numeric(out["hour"], errors="coerce")
                           + HOUR_SHIFT) % 24.0
        return out
    raise ValueError(kind)


def gating_conditions(feats: list) -> list:
    conds = ["clean", *A_KINDS]
    conds += [f"B_noise_{s}" for s in NOISE_SIGMAS]
    conds += [f"B_amount_{m}" for m in AMOUNT_MULTS]
    if "hour" in feats:
        conds.append("B_hour")
    return conds


def _decisions(p: np.ndarray, t: float):
    pred = (p >= t).astype(int)
    conf = (p >= CONFIDENT_HI) | (p <= CONFIDENT_LO)
    return pred, conf


def evaluate_gating(*, members, state, platt, contract, train_df: pd.DataFrame,
                    te: dict, t_thr: float, feats: list, seed: int) -> dict:
    """Gated vs matched ungated on identical perturbed inputs.

    Plan §5.0: the two gate components are measured SEPARATELY —
    ``row_gated`` = row-level enforcement only; ``window_gated`` = row
    enforcement + window-level PSI blocking (all rows fall back when the
    window blocks). ``ungated`` = same pipeline, both gates bypassed.
    """
    rng = np.random.default_rng(seed)
    test_df = te["X"].reset_index(drop=True)
    y = te["y"]
    amount_col = contract["amount_col"]
    fallback_amt = pd.to_numeric(test_df[amount_col], errors="coerce").to_numpy(
        dtype=float, na_value=np.nan)
    fallback_rule = np.where(np.isnan(fallback_amt), False,
                             fallback_amt > contract["fallback_thr"]).astype(int)
    psi_train = train_df[feats].apply(pd.to_numeric, errors="coerce")
    psi_base = build_baseline(psi_train, feats)

    def system(accepted, pred, conf):
        blocked = ~accepted
        tp = int(np.sum((pred == 1) & (y == 1)))
        fp = int(np.sum((pred == 1) & (y == 0)))
        fn = int(np.sum((pred == 0) & (y == 1)))
        # fallback/rules decisions are always "confident" (binary rules)
        wc = (int(np.sum(accepted & conf & (pred != y)))
              + int(np.sum(blocked & (fallback_rule != y))))
        return dict(
            n_blocked=int(blocked.sum()), n_accepted=int(accepted.sum()),
            n_fallback=int(blocked.sum()),
            fallback_errors=int(np.sum(blocked & (fallback_rule != y))),
            coverage=float(accepted.mean()),
            recall=(tp / (tp + fn)) if (tp + fn) else None,
            precision=(tp / (tp + fp)) if (tp + fp) else None,
            wrong_confident=wc,
            false_reject_legit=int(np.sum(blocked & (y == 0))),
            frauds_caught_by_fallback=int(np.sum(blocked & (y == 1) &
                                                  (fallback_rule == 1))),
            accepted_correct=int(np.sum(accepted & (pred == y))))

    rows_out = {}
    for kind in gating_conditions(feats):
        p_df = perturb(test_df, kind, contract, feats, rng)
        p_cal = platt.predict(member_scores(members, state, p_df)["ensemble_raw"])
        pred_u, conf_u = _decisions(p_cal, t_thr)
        n = len(y)
        ungated = system(np.ones(n, dtype=bool), pred_u, conf_u)
        gate = row_gate(p_df, contract)
        wn = p_df[feats].apply(pd.to_numeric, errors="coerce")
        wres = check_window(psi_base, wn, feats)
        psi_vals = [r["psi"] for r in wres.values() if r.get("psi") is not None]
        psi_max = max(psi_vals) if psi_vals else 0.0
        window_blocked = bool(psi_max >= PSI_ALERT)
        accepted = gate["accept"]
        pred_row = np.where(accepted, pred_u, fallback_rule)
        row_gated = system(accepted, pred_row, conf_u)
        acc_win = np.zeros(n, dtype=bool) if window_blocked else accepted
        pred_win = np.where(acc_win, pred_u, fallback_rule)
        window_gated = system(acc_win, pred_win, conf_u)
        rows_out[kind] = dict(
            n_total=n,
            psi_max=psi_max,
            window_blocked=window_blocked,
            gate_reasons=gate["reasons"],
            ungated=ungated,
            row_gated=row_gated,
            window_gated=window_gated,
            perturbation_class="A_designed_for" if kind.startswith("A_")
                               or kind == "clean"
                               else "B_not_designed_for_primary",
        )
    return rows_out


# ---- experiment runners ---------------------------------------------------
def segment_metrics(train_amt, amt, y, scores) -> dict:
    bands = np.quantile(train_amt, [0.0, 0.25, 0.5, 0.75, 1.0])
    sid = np.searchsorted(bands[1:-1], amt, side="right")
    out = {}
    for b in range(4):
        m = sid == b
        yy, ss = y[m], scores[m]
        out[f"q{b + 1}"] = dict(
            n=int(m.sum()), positives=int(yy.sum()),
            amount_range=[float(bands[b]), float(bands[b + 1])],
            roc_auc=roc_auc(yy, ss) if m.sum() and len(np.unique(yy)) > 1 else None,
            pr_auc=pr_auc(yy, ss) if yy.sum() else None)
    return out


def time_quartile_metrics(t, y, scores) -> dict:
    qs = np.quantile(t, [0.0, 0.25, 0.5, 0.75, 1.0])
    out = {}
    for b in range(4):
        m = ((t >= qs[b]) & (t <= qs[b + 1])) if b == 3 else \
            ((t >= qs[b]) & (t < qs[b + 1]))
        yy, ss = y[m], scores[m]
        out[f"w{b + 1}"] = dict(
            n=int(m.sum()), positives=int(yy.sum()),
            roc_auc=roc_auc(yy, ss) if m.sum() and len(np.unique(yy)) > 1 else None,
            pr_auc=pr_auc(yy, ss) if yy.sum() else None)
    return out


def run_discrimination(ds: dict, exp_id: str, seed: int) -> dict:
    from src.risk_engine.calibration import PlattCalibration
    from eval_record import sha256_file
    import joblib

    print(f"\n[nr05] ===== {exp_id} =====")
    tr, ca, te = split_arrays(ds)
    print(f"  train={len(tr['y'])} (pos {int(tr['y'].sum())}) "
          f"cal={len(ca['y'])} (pos {int(ca['y'].sum())}) "
          f"test={len(te['y'])} (pos {int(te['y'].sum())})")
    members, state = train_members(tr, seed)
    paths = save_models(exp_id, members)
    art = ART_DIR / exp_id
    sp = art / "pipeline_state.joblib"
    joblib.dump(state, sp)
    paths.append(sp)
    cal_scores = member_scores(members, state, ca["X"])
    platt = PlattCalibration().fit(cal_scores["ensemble_raw"], ca["y"])
    pp = art / "platt.joblib"
    joblib.dump(platt, pp)
    paths.append(pp)
    cal_cal = platt.predict(cal_scores["ensemble_raw"])
    t_thr = threshold_at_fpr_on_validation(ca["y"], cal_cal, 0.01)

    y_te = te["y"]
    te_scores = member_scores(members, state, te["X"])
    te_cal = platt.predict(te_scores["ensemble_raw"])
    member_metrics = {n: score_bundle(y_te, te_scores[n], t_thr)
                      for n in ("lr", "rf", "xgb", "if")}
    ens_raw = score_bundle(y_te, te_scores["ensemble_raw"], t_thr)
    ens_cal = score_bundle(y_te, te_cal, t_thr)

    calib = dict(
        method="PlattCalibration (2-param sigmoid) fit on CAL window only",
        cal_n=int(len(ca["y"])), cal_positives=int(ca["y"].sum()),
        test_n=int(len(y_te)), test_positives=int(y_te.sum()),
        ensemble_raw=dict(brier=compute_brier_score(y_te, te_scores["ensemble_raw"]),
                          ece=compute_ece(y_te, te_scores["ensemble_raw"])),
        ensemble_platt=dict(brier=compute_brier_score(y_te, te_cal),
                            ece=compute_ece(y_te, te_cal)),
        xgb=dict(brier=compute_brier_score(y_te, te_scores["xgb"]),
                 ece=compute_ece(y_te, te_scores["xgb"])),
        lr=dict(brier=compute_brier_score(y_te, te_scores["lr"]),
                ece=compute_ece(y_te, te_scores["lr"])),
        reliability_platt=reliability_curve(y_te, te_cal),
        note="No historical calibration numbers reused — all values executed here.",
    )

    dep_mode = "entity" if ds.get("entity") is not None else "block"
    ent_arg = te["entity"] if dep_mode == "entity" else None
    boots = {}
    for tag, mode in (("iid", "iid"), ("dependence_aware", dep_mode)):
        boots[tag] = boot_compare(y_te, te_cal, te_scores["xgb"],
                                  te_scores["lr"], mode, ent_arg, te["t"])
    iid_ci = bootstrap_ci(roc_auc, y_te, te_cal, n_bootstrap=N_BOOT, seed=seed)

    days = float((te["t"].max() - te["t"].min()) / 86400.0) or None
    business = {}
    for frac in ALERT_FRACS:
        key = f"top{int(frac * 100)}pct"
        business[key] = dict(
            recall=ens_cal[f"recall_at_{key}"],
            alerts=ens_cal[f"alerts_{key}"],
            alerts_per_day=(ens_cal[f"alerts_{key}"] / days) if days else None,
            window_days=days)

    payload = dict(
        experiment_id=exp_id,
        exploratory=True, non_independent=True,
        dataset=dict(name=ds["name"], path=str(ds["path"].relative_to(ROOT)),
                     sha256=sha256_file(ds["path"]), note=ds["note"]),
        label=ds["label"], seed=seed,
        features=ds["feats"],
        n=dict(train=len(tr["y"]), cal=len(ca["y"]), test=len(y_te),
               train_pos=int(tr["y"].sum()), cal_pos=int(ca["y"].sum()),
               test_pos=int(y_te.sum())),
        split=("temporal (sorted by time): 60/20/20" if ds["name"] != "kaggle"
               else "fraudTrain by unix_time 70/30 train/cal; fraudTest = final test"),
        threshold=dict(value=float(t_thr), target_fpr=0.01,
                       source="CAL window only (threshold_at_fpr_on_validation)",
                       applied="test unchanged"),
        models=dict(members=["lr", "rf", "xgb", "if"],
                    ensemble="mean(lr,rf,xgb,if) → Platt on cal",
                    configs="fixed a priori — no hyperparameter search"),
        member_metrics=member_metrics,
        ensemble_raw=ens_raw, ensemble_platt=ens_cal,
        calibration=calib,
        bootstraps=boots,
        bootstrap_iid_ref=iid_ci.to_dict() if hasattr(iid_ci, "to_dict") else iid_ci,
        segments=segment_metrics(tr["amount"], te["amount"], y_te, te_cal),
        time_quartiles=time_quartile_metrics(te["t"], y_te, te_cal),
        business=business,
        prevalence_test=prevalence_report(y_te),
        classification="EXPLORATORY — previously exposed dataset; NOT confirmatory "
                       "Track M evidence; NOT independent replication",
    )
    headline = dict(roc_auc=ens_cal["roc_auc"], pr_auc=ens_cal["pr_auc"],
                    recall_at_1pct_fpr=ens_cal["recall_at_1pct_fpr"],
                    brier=calib["ensemble_platt"]["brier"],
                    ece=calib["ensemble_platt"]["ece"],
                    test_n=int(len(y_te)), test_pos=int(y_te.sum()))
    payload["headline"] = headline
    emit(exp_id, payload)
    warnings = [
        "EXPLORATORY: previously exposed dataset — not confirmatory Track M evidence",
        "NON-INDEPENDENT: prior project exposure (see DATASET_EXPOSURE_LEDGER.md)",
        "single seed 42 — §18 minimum-seed rule is [TO BE FROZEN]",
        "dependence-aware CIs are EXPLORATORY — METHOD NOT YET FROZEN",
        "baseline identity (LR vs XGB) is [TO BE FROZEN] in plan §3 — both reported",
    ] + small_sample_warning(y_te)
    append_record(
        exp_id=exp_id, dataset_path=ds["path"], artifact_paths=paths,
        metrics=dict(headline=headline,
                     ensemble_platt={k: ens_cal[k] for k in
                                     ("roc_auc", "pr_auc", "recall_at_1pct_fpr",
                                      "tp", "fp", "tn", "fn")},
                     xgb={k: member_metrics["xgb"][k] for k in
                          ("roc_auc", "pr_auc", "recall_at_1pct_fpr")},
                     lr={k: member_metrics["lr"][k] for k in
                         ("roc_auc", "pr_auc", "recall_at_1pct_fpr")},
                     calibration=dict(
                         brier=calib["ensemble_platt"]["brier"],
                         ece=calib["ensemble_platt"]["ece"],
                         cal_n=calib["cal_n"]),
                     bootstraps={k: v["main_roc_auc"] for k, v in boots.items()}),
        config=dict(script="backend/scripts/nr05_diagnostics.py",
                    split=payload["split"], features=ds["feats"],
                    exposure="PREVIOUSLY EXPLORATORY/EXPOSED",
                    exploratory=True, non_independent=True,
                    n_boot=N_BOOT, seed=seed,
                    note=ds["note"]),
        seed=seed, threshold=float(t_thr), threshold_source="validation",
        feature_schema_version=f"nr05_{ds['name']}_features_v1",
        warnings=warnings)
    return payload


def run_gating(ds: dict, disc_exp: str, seed: int) -> dict:
    import joblib
    from eval_record import sha256_file

    print(f"\n[nr05] ===== nr05_{ds['name']}_gating (model = {disc_exp}) =====")
    art = ART_DIR / disc_exp
    members = {n: joblib.load(art / f"{n}.joblib") for n in ("lr", "rf", "xgb", "if")}
    state = joblib.load(art / "pipeline_state.joblib")
    platt = joblib.load(art / "platt.joblib")
    tr, ca, te = split_arrays(ds)
    cal_cal = platt.predict(member_scores(members, state, ca["X"])["ensemble_raw"])
    t_thr = threshold_at_fpr_on_validation(ca["y"], cal_cal, 0.01)
    amount_col = {"ulb": "Amount", "kaggle": "amt", "ibm": "amount"}[ds["name"]]
    contract = fit_contract(tr["X"], ds["feats"], amount_col)
    rows = evaluate_gating(members=members, state=state, platt=platt,
                           contract=contract, train_df=tr["X"], te=te,
                           t_thr=t_thr, feats=ds["feats"], seed=seed)
    payload = dict(
        experiment_id=f"nr05_{ds['name']}_gating",
        exploratory=True, non_independent=True,
        dataset=dict(name=ds["name"], path=str(ds["path"].relative_to(ROOT)),
                     sha256=sha256_file(ds["path"]), note=ds["note"]),
        model_artifacts=str(art.relative_to(ROOT)),
        threshold=dict(value=float(t_thr), source="CAL window @1%FPR (re-derived, identical)"),
        definitions=dict(
            gated="(component 1) row-level contract enforcement — Phase-41 "
                  "semantics, TRAIN-window bounds; (component 2) window-level PSI "
                  "gate (block if any feature PSI >= 0.25). Reported SEPARATELY as "
                  "row_gated and window_gated (plan §5.0); ungated = both bypassed",
            ungated="SAME model/preprocessing/imputation/inputs/population with both "
                    "gates bypassed; invalid inputs scored via frozen median imputation "
                    "(never crashes)",
            fallback="amount > train q99.5 → FRAUD, else LEGIT (train-only rule; "
                     "rules are always 'confident')",
            confident="ML confident iff p >= 0.8 or p <= 0.2 (NR-05 exploratory "
                      "definition; plan §5.3 remains [TO BE FROZEN])",
            class_A="NaN / +Inf / clearly out-of-range / wrong-type: 10% of test rows "
                    "total (4 kinds × 2.5%), one random feature per row — SECONDARY evidence",
            class_B=f"in-range subtle shifts on {int(CLASSB_FRAC * 100)}% of rows: "
                    f"Gaussian noise σ∈{NOISE_SIGMAS}×train-std on top-3-variance "
                    f"features; amount ×{AMOUNT_MULTS}; cyclic hour +{HOUR_SHIFT}h "
                    "— PRIMARY exploratory investigation; magnitudes fixed a priori, "
                    "full range recorded (never tuned to outcomes)",
            window_block_rule="PSI >= 0.25 on ≥1 feature → entire window falls back "
                              "(psi.py standard ALERT threshold, reused unchanged)",
            labels="never modified; wrong decisions judged against original labels",
        ),
        contract=dict(
            bounds={f: dict(lo=b["lo"], hi=b["hi"]) for f, b in
                    contract["bounds"].items()},
            fallback_amount_threshold=contract["fallback_thr"],
            source="TRAIN window only (q01/q99 ± 2·IQR)"),
        conditions=rows,
        classification="EXPLORATORY — previously exposed dataset; NOT confirmatory "
                       "Track M evidence",
    )
    clean = rows.get("clean", {})
    b_rows = {k: v for k, v in rows.items()
              if v["perturbation_class"] == "B_not_designed_for_primary"}
    a_rows = {k: v for k, v in rows.items()
              if v["perturbation_class"] == "A_designed_for" and k != "clean"}

    def _cmp(subset, system):
        """wrong-confident comparison of one gated system vs ungated."""
        better = worse = equal = 0
        for v in subset.values():
            d = v[system]["wrong_confident"] - v["ungated"]["wrong_confident"]
            better += d < 0
            worse += d > 0
            equal += d == 0
        return dict(better=int(better), worse=int(worse), equal=int(equal),
                    min_coverage=min((v[system]["coverage"] for v in subset.values()),
                                     default=None))

    summary = dict(
        clean=dict(
            ungated=dict(recall=clean.get("ungated", {}).get("recall"),
                         wrong_confident=clean.get("ungated", {}).get("wrong_confident")),
            row_gated=dict(
                coverage=clean.get("row_gated", {}).get("coverage"),
                recall=clean.get("row_gated", {}).get("recall"),
                wrong_confident=clean.get("row_gated", {}).get("wrong_confident"),
                false_reject_legit=clean.get("row_gated", {}).get("false_reject_legit")),
            window_gated=dict(
                coverage=clean.get("window_gated", {}).get("coverage"),
                recall=clean.get("window_gated", {}).get("recall"),
                wrong_confident=clean.get("window_gated", {}).get("wrong_confident"),
                window_blocked=clean.get("window_blocked"))),
        classB=dict(
            conditions=len(b_rows),
            row_gate=dict(_cmp(b_rows, "row_gated"),
                          recall_deltas={
                              k: (v["row_gated"]["recall"] - v["ungated"]["recall"])
                              if v["row_gated"]["recall"] is not None and
                              v["ungated"]["recall"] is not None else None
                              for k, v in b_rows.items()}),
            window_gate=dict(_cmp(b_rows, "window_gated"),
                             windows_blocked=sum(1 for v in b_rows.values()
                                                 if v["window_blocked"]))),
        classA=dict(
            conditions=len(a_rows),
            row_gate_block_rates={k: 1.0 - v["row_gated"]["coverage"]
                                  for k, v in a_rows.items()},
            windows_blocked={k: v["window_blocked"] for k, v in a_rows.items()}),
    )
    payload["summary"] = summary
    emit(payload["experiment_id"], payload)
    warnings = [
        "EXPLORATORY: previously exposed dataset — not confirmatory Track M evidence",
        "NON-INDEPENDENT: prior project exposure",
        "gate contract/PSI fitted on TRAIN window only; Class B magnitudes fixed a priori",
        "gating criterion thresholds (plan §5.5) remain [TO BE FROZEN]",
        "row-level and window-level gate components reported separately (plan §5.0)",
    ]
    append_record(
        exp_id=payload["experiment_id"], dataset_path=ds["path"], artifact_paths=[],
        metrics=dict(summary=summary,
                     conditions={k: dict(
                         psi_max=v["psi_max"], window_blocked=v["window_blocked"],
                         ungated=dict(recall=v["ungated"]["recall"],
                                      wrong_confident=v["ungated"]["wrong_confident"]),
                         row_gated=dict(coverage=v["row_gated"]["coverage"],
                                        recall=v["row_gated"]["recall"],
                                        wrong_confident=v["row_gated"]["wrong_confident"]),
                         window_gated=dict(coverage=v["window_gated"]["coverage"],
                                           recall=v["window_gated"]["recall"],
                                           wrong_confident=v["window_gated"]["wrong_confident"]))
                                 for k, v in rows.items()}),
        config=dict(script="backend/scripts/nr05_diagnostics.py",
                    model_artifacts=str(art.relative_to(ROOT)),
                    definitions=payload["definitions"],
                    exploratory=True, non_independent=True, seed=seed),
        seed=seed, threshold=float(t_thr), threshold_source="validation",
        feature_schema_version=f"nr05_{ds['name']}_features_v1",
        warnings=warnings)
    return payload


def main() -> int:
    ap = argparse.ArgumentParser(description="NR-05 exploratory diagnostics")
    ap.add_argument("--datasets", default="ulb,kaggle,ibm",
                    help="comma subset of: ulb,kaggle,ibm")
    ap.add_argument("--skip-gating", action="store_true")
    ap.add_argument("--only-gating", action="store_true",
                    help="reuse saved discrimination artifacts; run gating only")
    args = ap.parse_args()
    want = {d.strip() for d in args.datasets.split(",") if d.strip()}
    seed = SEED
    rng = np.random.default_rng(seed)
    summary: dict = {}

    setups: dict = {}
    if "ulb" in want:
        setups["ulb"] = load_ulb()
    if "kaggle" in want:
        disc, ent = load_kaggle(rng)
        setups["kaggle"] = disc
        setups["kaggle_entity"] = ent
    if "ibm" in want:
        setups["ibm"] = load_ibm()

    for name in ("ulb", "kaggle", "ibm"):
        if name not in setups:
            continue
        if not args.only_gating:
            p = run_discrimination(setups[name], f"nr05_{name}_discrimination", seed)
            summary[f"{name}_discrimination"] = p["headline"]
        if not args.skip_gating:
            g = run_gating(setups[name], f"nr05_{name}_discrimination", seed)
            summary[f"{name}_gating"] = g["summary"]

    if "kaggle_entity" in setups and not args.only_gating:
        p = run_discrimination(setups["kaggle_entity"],
                               "nr05_kaggle_entity_disjoint", seed)
        summary["kaggle_entity_disjoint"] = p["headline"]

    emit("summary", dict(seed=seed, n_boot=N_BOOT, summary=summary,
                         classification="EXPLORATORY — NOT confirmatory Track M"))
    print("\n[nr05] DONE — all results EXPLORATORY on previously exposed datasets")
    return 0


if __name__ == "__main__":
    sys.exit(main())
