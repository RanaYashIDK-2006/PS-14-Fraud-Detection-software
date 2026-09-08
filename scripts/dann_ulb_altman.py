#!/usr/bin/env python3
"""Domain-Adversarial Neural Network (DANN) for ULB-Altman fraud transfer.

Learns domain-invariant fraud representations by jointly training:
  1. Feature extractor: maps raw features to latent space
  2. Fraud classifier: predicts fraud/legit from latent features
  3. Domain discriminator: predicts ULB vs Altman from latent features
  4. Gradient Reversal Layer: reverses domain gradients so the extractor
     learns representations that fool the domain discriminator

Result: latent features that capture fraud patterns shared between
ULB and Altman, while discarding domain-specific noise.
"""
from __future__ import annotations
import json, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
NJ = 4

torch.set_num_threads(NJ)


def recall_at_fpr(y_true, scores, target=0.01):
    fpr, tpr, _ = roc_curve(y_true, scores)
    idx = np.searchsorted(fpr, target, side="right")
    return float(tpr[idx - 1]) if idx > 0 else 0.0


# ── Shared Semantic Feature Space ──

SHARED_FEATURES = [
    "amount_log", "amount_zscore", "hour_sin", "hour_cos",
    "is_night", "amount_x_hour", "amount_sq", "high_amount", "amount_bucket",
]


def load_ulb_shared():
    """Load ULB mapped to shared feature space."""
    df = pd.read_csv(ROOT / "data" / "creditcard.csv")
    y = df["Class"].values.astype(int)
    amt = df["Amount"].values
    hour = (df["Time"].values % 86400) / 3600.0

    feats = {}
    feats["amount_log"] = np.log1p(np.clip(amt, 0, 1e6))
    feats["amount_zscore"] = (amt - amt.mean()) / max(amt.std(), 0.01)
    feats["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    feats["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    feats["is_night"] = ((hour >= 22) | (hour <= 6)).astype(float)
    feats["amount_x_hour"] = amt * hour / 24
    feats["amount_sq"] = amt ** 2
    feats["high_amount"] = (amt > 100).astype(float)
    feats["amount_bucket"] = np.nan_to_num(
        pd.cut(amt, bins=[0, 10, 50, 200, 1000, 1e9], labels=False).astype(float), nan=0.0)

    X = np.column_stack([feats[k] for k in SHARED_FEATURES])
    return np.nan_to_num(X).astype(np.float32), y, SHARED_FEATURES


def load_altman_shared():
    """Load Altman mapped to shared feature space."""
    rng = np.random.RandomState(42)
    rows_all = []
    ci = 0
    for chunk in pd.read_csv(ROOT / "data" / "credit_card_transactions-ibm_v2.csv",
        usecols=["User", "Month", "Day", "Time", "Amount", "Use Chip", "MCC", "Errors?",
                 "Is Fraud?", "Merchant Name", "Merchant City", "Merchant State", "Zip", "Year", "Card"],
        low_memory=False, chunksize=1_000_000):
        fm = chunk["Is Fraud?"] == "Yes"
        rows_all.append(pd.concat([chunk[fm], chunk[~fm].sample(frac=0.01, random_state=rng)]))
        ci += 1
        if ci >= 16:
            break
    raw = pd.concat(rows_all, ignore_index=True)
    y = (raw["Is Fraud?"] == "Yes").values.astype(int)

    tp = raw["Time"].str.split(":", expand=True)
    hr = pd.to_numeric(tp[0], errors="coerce").fillna(12).values
    amt = pd.to_numeric(raw["Amount"].str.replace("$", "", regex=False)
                        .str.replace(",", "", regex=False), errors="coerce").fillna(0).values

    feats = {}
    feats["amount_log"] = np.log1p(np.clip(amt, 0, 1e6))
    feats["amount_zscore"] = (amt - amt.mean()) / max(amt.std(), 0.01)
    feats["hour_sin"] = np.sin(2 * np.pi * hr / 24)
    feats["hour_cos"] = np.cos(2 * np.pi * hr / 24)
    feats["is_night"] = ((hr >= 22) | (hr <= 6)).astype(float)
    feats["amount_x_hour"] = amt * hr / 24
    feats["amount_sq"] = amt ** 2
    feats["high_amount"] = (amt > 100).astype(float)
    feats["amount_bucket"] = np.nan_to_num(
        pd.cut(amt, bins=[0, 10, 50, 200, 1000, 1e9], labels=False).astype(float), nan=0.0)

    X = np.column_stack([feats[k] for k in SHARED_FEATURES])
    return np.nan_to_num(X).astype(np.float32), y, SHARED_FEATURES


# ── DANN Components ──

class GradientReversalFunction(torch.autograd.Function):
    """Reverses gradient during backward pass."""
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


class GradientReversalLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, alpha=1.0):
        return GradientReversalFunction.apply(x, alpha)


class FeatureExtractor(nn.Module):
    """Shared feature encoder."""
    def __init__(self, input_dim, hidden_dim=64, latent_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, x):
        return self.net(x)


class FraudClassifier(nn.Module):
    """Predicts fraud probability from latent features."""
    def __init__(self, latent_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
        )

    def forward(self, z):
        return self.net(z).squeeze(-1)


class DomainDiscriminator(nn.Module):
    """Predicts ULB vs Altman from latent features (via GRL)."""
    def __init__(self, latent_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
        )

    def forward(self, z):
        return self.net(z).squeeze(-1)


class DANN(nn.Module):
    """Domain-Adversarial Neural Network."""
    def __init__(self, input_dim, latent_dim=32):
        super().__init__()
        self.extractor = FeatureExtractor(input_dim, latent_dim=latent_dim)
        self.fraud_clf = FraudClassifier(latent_dim)
        self.domain_disc = DomainDiscriminator(latent_dim)
        self.grl = GradientReversalLayer()

    def forward(self, x, alpha=1.0):
        z = self.extractor(x)
        fraud_logit = self.fraud_clf(z)
        z_rev = self.grl(z, alpha)
        domain_logit = self.domain_disc(z_rev)
        return fraud_logit, domain_logit

    def encode(self, x):
        with torch.no_grad():
            return self.extractor(x)

    def predict_fraud(self, x):
        with torch.no_grad():
            z = self.extractor(x)
            return torch.sigmoid(self.fraud_clf(z))


# ── Training ──

def train_dann(model, ulb_X, ulb_y, alt_X, alt_y, n_epochs=300, batch_size=512, lr=0.001):
    """Train DANN on ULB + Altman jointly."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    # Balance fraud weights
    total_fraud = int(ulb_y.sum()) + int(alt_y.sum())
    total_legit = (len(ulb_y) - int(ulb_y.sum())) + (len(alt_y) - int(alt_y.sum()))
    fraud_w = min(total_legit / max(total_fraud, 1), 20.0)
    fraud_pos_weight = torch.tensor([fraud_w])

    # Domain labels: 0=ULB, 1=Altman
    ulb_domain = np.zeros(len(ulb_y), dtype=np.float32)
    alt_domain = np.ones(len(alt_y), dtype=np.float32)

    # Create combined dataset
    X_all = np.vstack([ulb_X, alt_X])
    y_all = np.concatenate([ulb_y, alt_y]).astype(np.float32)
    d_all = np.concatenate([ulb_domain, alt_domain])

    X_t = torch.tensor(X_all, dtype=torch.float32)
    y_t = torch.tensor(y_all, dtype=torch.float32)
    d_t = torch.tensor(d_all, dtype=torch.float32)

    dataset = TensorDataset(X_t, y_t, d_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    best_loss = float("inf")
    patience = 40
    no_improve = 0
    best_state = None

    for epoch in range(n_epochs):
        model.train()
        # Linear schedule for alpha (domain adversarial strength)
        p = epoch / n_epochs
        alpha = 2.0 / (1.0 + np.exp(-10 * p)) - 1.0  # 0 → 1

        total_fraud_loss = 0
        total_domain_loss = 0
        n_batches = 0

        for X_batch, y_batch, d_batch in loader:
            fraud_logit, domain_logit = model(X_batch, alpha)

            # Fraud classification loss
            fraud_loss = F.binary_cross_entropy_with_logits(
                fraud_logit, y_batch, pos_weight=fraud_pos_weight)

            # Domain classification loss
            domain_loss = F.binary_cross_entropy_with_logits(
                domain_logit, d_batch)

            loss = fraud_loss + 0.5 * domain_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_fraud_loss += fraud_loss.item()
            total_domain_loss += domain_loss.item()
            n_batches += 1

        scheduler.step()
        avg_fraud = total_fraud_loss / n_batches
        avg_domain = total_domain_loss / n_batches
        avg_total = avg_fraud + 0.5 * avg_domain

        if avg_total < best_loss:
            best_loss = avg_total
            no_improve = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
            if no_improve >= patience:
                break

        if (epoch + 1) % 50 == 0:
            print(f"    Epoch {epoch+1:>3}: fraud={avg_fraud:.4f} domain={avg_domain:.4f} alpha={alpha:.3f}")

    if best_state:
        model.load_state_dict(best_state)
    return model


def evaluate_transfer(model, X_test, y_test, domain_name=""):
    """Evaluate fraud detection on target domain."""
    model.eval()
    with torch.no_grad():
        probs = model.predict_fraud(torch.tensor(X_test, dtype=torch.float32)).numpy()
    auc = roc_auc_score(y_test, probs)
    r1 = recall_at_fpr(y_test, probs, 0.01)
    r05 = recall_at_fpr(y_test, probs, 0.005)
    return {"auc": round(auc, 6), "r1": round(r1, 6), "r05": round(r05, 6), "domain": domain_name}


# ── Baselines ──

def train_xgb_baseline(Xtr, ytr, Xte, yte):
    """Direct XGB training baseline."""
    from xgboost import XGBClassifier
    n_pos = int(ytr.sum())
    m = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
                      subsample=0.8, colsample_bytree=0.7, gamma=2,
                      min_child_weight=5,
                      scale_pos_weight=min(n_pos / max(len(ytr) - n_pos, 1) * 50, 200),
                      random_state=42, n_jobs=NJ, eval_metric="auc")
    m.fit(Xtr, ytr, verbose=False)
    preds = m.predict_proba(Xte)[:, 1]
    return {"auc": round(roc_auc_score(yte, preds), 6),
            "r1": round(recall_at_fpr(yte, preds), 6),
            "r05": round(recall_at_fpr(yte, preds, 0.005), 6)}


def train_source_only_baseline(Xtr, ytr, Xte, yte):
    """Simple neural network trained only on source (no adversarial)."""
    input_dim = Xtr.shape[1]
    model = nn.Sequential(
        nn.Linear(input_dim, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(32, 1),
    )
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
    ytr_t = torch.tensor(ytr, dtype=torch.float32)
    Xte_t = torch.tensor(Xte, dtype=torch.float32)

    pos_w = torch.tensor([min((len(ytr) - ytr.sum()) / max(ytr.sum(), 1), 20.0)])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    for epoch in range(200):
        model.train()
        pred = model(Xtr_t).squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(pred, ytr_t, pos_weight=pos_w)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(Xte_t)).squeeze(-1).numpy()
    return {"auc": round(roc_auc_score(yte, probs), 6),
            "r1": round(recall_at_fpr(yte, probs), 6),
            "r05": round(recall_at_fpr(yte, probs, 0.005), 6)}


# ── Main ──

def main():
    print("=" * 80)
    print("  DOMAIN-ADVERSARIAL NEURAL NETWORK (DANN)")
    print("  ULB ↔ Altman fraud transfer learning")
    print("=" * 80)
    t0 = time.time()

    # Load
    print("\n[1] Loading ULB and Altman (shared feature space)...")
    ulb_X, ulb_y, feat_names = load_ulb_shared()
    alt_X, alt_y, _ = load_altman_shared()
    print(f"  ULB:    {len(ulb_y):>8,} rows, {int(ulb_y.sum()):>5,} fraud ({ulb_y.mean()*100:.3f}%)")
    print(f"  Altman: {len(alt_y):>8,} rows, {int(alt_y.sum()):>5,} fraud ({alt_y.mean()*100:.3f}%)")
    print(f"  Features: {len(feat_names)} shared: {feat_names}")

    # Standardize (fit on combined data)
    scaler = StandardScaler()
    X_all = np.vstack([ulb_X, alt_X])
    scaler.fit(X_all)
    ulb_X_s = scaler.transform(ulb_X).astype(np.float32)
    alt_X_s = scaler.transform(alt_X).astype(np.float32)

    input_dim = ulb_X_s.shape[1]

    # ── 5-fold evaluation ──
    print("\n[2] Evaluating transfer learning (5-fold)...")
    skf_ulb = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    skf_alt = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    results = {
        "dann_forward": [],  # train ULB+Altman, test Altman
        "dann_reverse": [],  # train ULB+Altman, test ULB
        "xgb_ulb_to_alt": [],
        "xgb_alt_to_ulb": [],
        "xgb_ulb_direct": [],
        "xgb_alt_direct": [],
        "nn_source_only_fwd": [],
        "nn_source_only_rev": [],
    }

    for fold, ((ulb_tr, ulb_te), (alt_tr, alt_te)) in enumerate(
        zip(skf_ulb.split(ulb_X_s, ulb_y), skf_alt.split(alt_X_s, alt_y))):
        print(f"\n  --- Fold {fold+1} ---")

        # Train DANN on ULB_train + Altman_train
        model = DANN(input_dim, latent_dim=32)
        model = train_dann_fold(model, ulb_X_s[ulb_tr], ulb_y[ulb_tr],
                                alt_X_s[alt_tr], alt_y[alt_tr])

        # Evaluate on held-out Altman (forward transfer)
        r_fwd = evaluate_transfer(model, alt_X_s[alt_te], alt_y[alt_te], "Altman_test")
        results["dann_forward"].append(r_fwd)
        print(f"    DANN forward (test Altman):  AUC={r_fwd['auc']:.4f}  R@1%={r_fwd['r1']:.4f}")

        # Evaluate on held-out ULB (reverse transfer)
        r_rev = evaluate_transfer(model, ulb_X_s[ulb_te], ulb_y[ulb_te], "ULB_test")
        results["dann_reverse"].append(r_rev)
        print(f"    DANN reverse (test ULB):     AUC={r_rev['auc']:.4f}  R@1%={r_rev['r1']:.4f}")

        # Baselines: XGB trained on ULB, tested on Altman
        r_xgb_fwd = train_xgb_baseline(ulb_X_s[ulb_tr], ulb_y[ulb_tr],
                                         alt_X_s[alt_te], alt_y[alt_te])
        results["xgb_ulb_to_alt"].append(r_xgb_fwd)
        print(f"    XGB ULB->Altman:             AUC={r_xgb_fwd['auc']:.4f}  R@1%={r_xgb_fwd['r1']:.4f}")

        # Baselines: XGB trained on Altman, tested on ULB
        r_xgb_rev = train_xgb_baseline(alt_X_s[alt_tr], alt_y[alt_tr],
                                         ulb_X_s[ulb_te], ulb_y[ulb_te])
        results["xgb_alt_to_ulb"].append(r_xgb_rev)
        print(f"    XGB Altman->ULB:             AUC={r_xgb_rev['auc']:.4f}  R@1%={r_xgb_rev['r1']:.4f}")

        # XGB in-domain baselines
        r_ulb_id = train_xgb_baseline(ulb_X_s[ulb_tr], ulb_y[ulb_tr],
                                        ulb_X_s[ulb_te], ulb_y[ulb_te])
        results["xgb_ulb_direct"].append(r_ulb_id)
        print(f"    XGB ULB in-domain:           AUC={r_ulb_id['auc']:.4f}  R@1%={r_ulb_id['r1']:.4f}")

        r_alt_id = train_xgb_baseline(alt_X_s[alt_tr], alt_y[alt_tr],
                                        alt_X_s[alt_te], alt_y[alt_te])
        results["xgb_alt_direct"].append(r_alt_id)
        print(f"    XGB Altman in-domain:        AUC={r_alt_id['auc']:.4f}  R@1%={r_alt_id['r1']:.4f}")

        # Source-only neural net (train on ULB, test on Altman)
        r_nn_fwd = train_source_only_baseline(ulb_X_s[ulb_tr], ulb_y[ulb_tr],
                                               alt_X_s[alt_te], alt_y[alt_te])
        results["nn_source_only_fwd"].append(r_nn_fwd)
        print(f"    NN source-only fwd:          AUC={r_nn_fwd['auc']:.4f}  R@1%={r_nn_fwd['r1']:.4f}")

    # ── Summary ──
    print("\n" + "=" * 80)
    print("  RESULTS SUMMARY (5-fold mean +/- std)")
    print("=" * 80)

    def mean_std(key, metric="auc"):
        vals = [r[metric] for r in results[key]]
        return np.mean(vals), np.std(vals)

    print(f"\n  {'Method':<30} {'AUC':>12} {'R@1%FPR':>12}")
    print(f"  {'-'*30} {'-'*12} {'-'*12}")

    rows = [
        ("XGB ULB in-domain", "xgb_ulb_direct"),
        ("XGB Altman in-domain", "xgb_alt_direct"),
        ("", None),
        ("XGB ULB->Altman (direct)", "xgb_ulb_to_alt"),
        ("XGB Altman->ULB (direct)", "xgb_alt_to_ulb"),
        ("NN source-only ULB->Altman", "nn_source_only_fwd"),
        ("", None),
        ("DANN ULB+Altman -> test Altman", "dann_forward"),
        ("DANN ULB+Altman -> test ULB", "dann_reverse"),
    ]

    for label, key in rows:
        if key is None:
            print()
            continue
        mu_auc, std_auc = mean_std(key, "auc")
        mu_r1, std_r1 = mean_std(key, "r1")
        print(f"  {label:<30} {mu_auc:>6.4f}+/-{std_auc:.4f} {mu_r1:>6.4f}+/-{std_r1:.4f}")

    # Transfer gain
    print("\n  Transfer gain (DANN vs direct XGB):")
    for direction, dann_key, xgb_key, test_name in [
        ("ULB->Altman", "dann_forward", "xgb_ulb_to_alt", "Altman"),
        ("Altman->ULB", "dann_reverse", "xgb_alt_to_ulb", "ULB"),
    ]:
        dann_auc = mean_std(dann_key, "auc")[0]
        xgb_auc = mean_std(xgb_key, "auc")[0]
        gain = dann_auc - xgb_auc
        arrow = "+" if gain > 0 else ""
        print(f"    {direction}: XGB {xgb_auc:.4f} -> DANN {dann_auc:.4f} ({arrow}{gain:.4f})")

    # Feature importance via gradient analysis
    print("\n  Feature importance (gradient magnitude on fraud loss):")
    model = DANN(input_dim, latent_dim=32)
    model = train_dann_fold(model, ulb_X_s, ulb_y, alt_X_s, alt_y)
    model.eval()
    X_t = torch.tensor(np.vstack([ulb_X_s, alt_X_s]), dtype=torch.float32, requires_grad=True)
    fraud_logit, _ = model(X_t)
    fraud_loss = F.binary_cross_entropy_with_logits(
        fraud_logit, torch.tensor(np.concatenate([ulb_y, alt_y]), dtype=torch.float32))
    fraud_loss.backward()
    grad_abs = X_t.grad.abs().mean(dim=0).detach().numpy()
    grad_sorted = sorted(zip(feat_names, grad_abs), key=lambda x: -x[1])
    for name, g in grad_sorted:
        print(f"    {g:.4f}  {name}")

    # Save
    out = REPORTS / "dann_ulb_altman_results.json"
    save_results = {k: v for k, v in results.items()}
    save_results["summary"] = {
        method: {"auc_mean": round(mean_std(method, "auc")[0], 6),
                 "r1_mean": round(mean_std(method, "r1")[0], 6)}
        for method in results if results[method]
    }
    out.write_text(json.dumps(save_results, indent=2))
    print(f"\n  Saved: {out}")
    print(f"  Total: {time.time()-t0:.0f}s")
    print("=" * 80)


def train_dann_fold(model, ulb_X, ulb_y, alt_X, alt_y, n_epochs=150, lr=0.001):
    """Train DANN on one fold."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    total_fraud = int(ulb_y.sum()) + int(alt_y.sum())
    total_legit = len(ulb_y) - int(ulb_y.sum()) + len(alt_y) - int(alt_y.sum())
    fraud_weight = min(total_legit / max(total_fraud, 1), 20.0)
    pos_w = torch.tensor([fraud_weight])

    ulb_domain = np.zeros(len(ulb_y), dtype=np.float32)
    alt_domain = np.ones(len(alt_y), dtype=np.float32)

    X_all = np.vstack([ulb_X, alt_X])
    y_all = np.concatenate([ulb_y, alt_y]).astype(np.float32)
    d_all = np.concatenate([ulb_domain, alt_domain])

    X_t = torch.tensor(X_all, dtype=torch.float32)
    y_t = torch.tensor(y_all, dtype=torch.float32)
    d_t = torch.tensor(d_all, dtype=torch.float32)

    dataset = TensorDataset(X_t, y_t, d_t)
    loader = DataLoader(dataset, batch_size=1024, shuffle=True, drop_last=True)

    best_loss = float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(n_epochs):
        model.train()
        p = epoch / n_epochs
        alpha = 2.0 / (1.0 + np.exp(-10 * p)) - 1.0

        total_loss = 0
        n_b = 0
        for xb, yb, db in loader:
            fraud_logit, domain_logit = model(xb, alpha)
            fl = F.binary_cross_entropy_with_logits(fraud_logit, yb, pos_weight=pos_w)
            dl = F.binary_cross_entropy_with_logits(domain_logit, db)
            loss = fl + 0.5 * dl
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_b += 1

        scheduler.step()
        avg = total_loss / n_b
        if avg < best_loss:
            best_loss = avg
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= 30:
                break

    if best_state:
        model.load_state_dict(best_state)
    return model


if __name__ == "__main__":
    main()
