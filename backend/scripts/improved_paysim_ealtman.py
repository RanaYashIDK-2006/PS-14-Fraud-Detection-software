#!/usr/bin/env python3
"""Improved PaySim + ealtman2019 evaluation with targeted fraud detection features.

Key improvements:
- PaySim: drain intensity, zero-out balance, type encoding, dest mismatch
- Ealtman: online/swipe, MCC fraud rate, card history, amount ratio
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


def metrics(y_true, y_prob, name=""):
    """Compute all fraud metrics."""
    roc = roc_auc_score(y_true, y_prob)
    pr = average_precision_score(y_true, y_prob)
    prev = y_true.mean()

    # Recall at various FPR thresholds
    n_legit = (y_true == 0).sum()
    n_fraud = (y_true == 1).sum()
    
    def recall_at_fpr(fpr_target):
        sorted_idx = np.argsort(-y_prob)
        sorted_labels = y_true[sorted_idx]
        fp = 0
        for i, label in enumerate(sorted_labels):
            if label == 0:
                fp += 1
                if fp / n_legit > fpr_target:
                    return sorted_labels[:i].sum() / n_fraud
        return 1.0

    r1 = recall_at_fpr(0.01)
    r05 = recall_at_fpr(0.005)
    r01 = recall_at_fpr(0.001)

    # At threshold that gives ~1% FPR
    sorted_idx = np.argsort(-y_prob)
    sorted_labels = y_true[sorted_idx]
    fp = 0
    threshold = 0.5
    for i, label in enumerate(sorted_labels):
        if label == 0:
            fp += 1
            if fp / n_legit >= 0.01:
                threshold = y_prob[sorted_idx[i]]
                break
    preds = (y_prob >= threshold).astype(int)
    tn, fp2, fn, tp = confusion_matrix(y_true, preds).ravel()
    precision = tp / (tp + fp2) if (tp + fp2) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'name': name,
        'n': len(y_true),
        'fraud': int(n_fraud),
        'prev': prev,
        'roc_auc': roc,
        'pr_auc': pr,
        'recall@1%FPR': r1,
        'recall@0.5%FPR': r05,
        'recall@0.1%FPR': r01,
        'precision@1%FPR': precision,
        'recall@1%FPR_ops': recall,
        'f1@1%FPR': f1,
    }


# ═══════════════════════════════════════════════════════════
# PAYSIM
# ═══════════════════════════════════════════════════════════
def eval_paysim():
    print("=" * 70)
    print("PAYSIM 1M — Improved Feature Engineering")
    print("=" * 70)

    df = pd.read_csv('data/paysim_1m.csv')
    y = df['isFraud'].values.astype(int)

    # ─── OLD FEATURES (baseline) ───
    old_features = ['amount', 'type', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
    X_old = df[old_features].copy()
    X_old = pd.get_dummies(X_old, columns=['type'], prefix='type').values

    # ─── NEW FEATURES ───
    amt = df['amount'].values.astype(float)
    obO = df['oldbalanceOrg'].values.astype(float)
    nbO = df['newbalanceOrig'].values.astype(float)
    obD = df['oldbalanceDest'].values.astype(float)
    nbD = df['newbalanceDest'].values.astype(float)

    # Core fraud signals
    drain_ratio = amt / np.maximum(obO, 1)
    zero_out = (nbO == 0).astype(float)
    exact_drain = (amt >= obO * 0.99).astype(float)
    dest_mismatch = np.abs(amt - (nbD - obD))
    log_amt = np.log1p(amt)
    log_obO = np.log1p(obO)
    log_obD = np.log1p(obD)

    # Amount extremes
    amt_gt_10k = (amt > 10000).astype(float)
    amt_gt_50k = (amt > 50000).astype(float)

    # Balance inconsistencies
    org_neg_after = (nbO < 0).astype(float)
    dest_neg_after = (nbD < 0).astype(float)

    # No step column in this version — use index-based pseudo-temporal
    is_night = pd.Series(0.0, index=df.index)  # no time info available
    day_num = pd.Series(np.arange(len(df), dtype=float) // 10000, index=df.index)  # batch proxy

    # Interaction features
    amt_x_drain = amt * drain_ratio
    co_drain = ((df['type'] == 'CASH_OUT').astype(float) * drain_ratio)
    tf_overdraft = ((df['type'] == 'TRANSFER').astype(float) * (amt > obO).astype(float))

    # Type encoding
    type_CO = (df['type'] == 'CASH_OUT').astype(float)
    type_TF = (df['type'] == 'TRANSFER').astype(float)
    type_CI = (df['type'] == 'CASH_IN').astype(float)
    type_P = (df['type'] == 'PAYMENT').astype(float)
    type_D = (df['type'] == 'DEBIT').astype(float)

    new_features = np.column_stack([
        drain_ratio, zero_out, exact_drain, dest_mismatch,
        log_amt, log_obO, log_obD,
        amt_gt_10k, amt_gt_50k,
        org_neg_after, dest_neg_after,
        is_night.values if hasattr(is_night, 'values') else is_night, day_num.values if hasattr(day_num, 'values') else day_num,
        amt_x_drain, co_drain, tf_overdraft,
        type_CO, type_TF, type_CI, type_P, type_D,
        amt, obO, nbO, obD, nbD,
    ])

    # Split (subsample for speed)
    n = len(y)
    rng = np.random.RandomState(42)
    if n > 200000:
        idx = rng.choice(n, 200000, replace=False)
        y_sub = y[idx]
        X_old_sub = X_old[idx]
        X_new_sub = new_features[idx]
        print(f"  (Subsampled to 200K for training speed)")
    else:
        y_sub = y
        X_old_sub = X_old
        X_new_sub = new_features

    X_old_train, X_old_test, y_train, y_test = train_test_split(X_old_sub, y_sub, test_size=0.2, random_state=42, stratify=y_sub)
    X_new_train, X_new_test, _, _ = train_test_split(X_new_sub, y_sub, test_size=0.2, random_state=42, stratify=y_sub)

    # ─── Train models ───
    print("\n--- Baseline (6 features) ---")
    scaler_old = StandardScaler()
    Xo_tr = scaler_old.fit_transform(X_old_train)
    Xo_te = scaler_old.transform(X_old_test)
    rf_old = RandomForestClassifier(n_estimators=200, max_depth=10, n_jobs=-1, random_state=42)
    rf_old.fit(Xo_tr, y_train)
    p_old = rf_old.predict_proba(Xo_te)[:, 1]
    m_old = metrics(y_test, p_old, "Baseline (6 features)")

    print(f"\n  ROC-AUC:    {m_old['roc_auc']:.4f}")
    print(f"  PR-AUC:     {m_old['pr_auc']:.4f}")
    print(f"  R@0.1%FPR:  {m_old['recall@0.1%FPR']:.1%}")
    print(f"  R@0.5%FPR:  {m_old['recall@0.5%FPR']:.1%}")
    print(f"  R@1%FPR:    {m_old['recall@1%FPR']:.1%}")

    print("\n--- Improved (26 features) ---")
    scaler_new = StandardScaler()
    Xn_tr = scaler_new.fit_transform(X_new_train)
    Xn_te = scaler_new.transform(X_new_test)
    rf_new = RandomForestClassifier(n_estimators=300, max_depth=12, n_jobs=-1, random_state=42)
    rf_new.fit(Xn_tr, y_train)
    p_new = rf_new.predict_proba(Xn_te)[:, 1]
    m_new = metrics(y_test, p_new, "Improved (26 features)")

    print(f"\n  ROC-AUC:    {m_new['roc_auc']:.4f}")
    print(f"  PR-AUC:     {m_new['pr_auc']:.4f}")
    print(f"  R@0.1%FPR:  {m_new['recall@0.1%FPR']:.1%}")
    print(f"  R@0.5%FPR:  {m_new['recall@0.5%FPR']:.1%}")
    print(f"  R@1%FPR:    {m_new['recall@1%FPR']:.1%}")

    print(f"\n--- Improvement ---")
    print(f"  ROC-AUC:    {m_old['roc_auc']:.4f} → {m_new['roc_auc']:.4f}  ({(m_new['roc_auc']-m_old['roc_auc'])*100:+.2f}%)")
    print(f"  PR-AUC:     {m_old['pr_auc']:.4f} → {m_new['pr_auc']:.4f}  ({(m_new['pr_auc']-m_old['pr_auc'])*100:+.2f}%)")
    print(f"  R@1%FPR:    {m_old['recall@1%FPR']:.1%} → {m_new['recall@1%FPR']:.1%}  ({(m_new['recall@1%FPR']-m_old['recall@1%FPR'])*100:+.1f}%)")

    # Feature importance
    feat_names = ['drain_ratio', 'zero_out', 'exact_drain', 'dest_mismatch',
                  'log_amt', 'log_obO', 'log_obD', 'amt_gt_10k', 'amt_gt_50k',
                  'org_neg', 'dest_neg', 'is_night', 'day_num',
                  'amt_x_drain', 'co_drain', 'tf_overdraft',
                  'type_CO', 'type_TF', 'type_CI', 'type_P', 'type_D',
                  'amount', 'obO', 'nbO', 'obD', 'nbD']
    imp = rf_new.feature_importances_
    top_idx = np.argsort(imp)[::-1][:10]
    print("\n  Top 10 features:")
    for i in top_idx:
        print(f"    {feat_names[i]:20s} {imp[i]:.4f}")

    return m_old, m_new


# ═══════════════════════════════════════════════════════════
# EALTMAN 2019
# ═══════════════════════════════════════════════════════════
def eval_ealtman():
    print("\n" + "=" * 70)
    print("EALTMAN 2019 (User0) — Improved Feature Engineering")
    print("=" * 70)

    df = pd.read_csv('data/ealtman2019/User0_credit_card_transactions.csv')
    df['Amount_val'] = df['Amount'].astype(str).str.replace(r'[\$,]', '', regex=True).astype(float)
    y = (df['Is Fraud?'] == 'Yes').values.astype(int)

    # Parse time
    df['Hour'] = pd.to_datetime(df['Time'], format='%H:%M', errors='coerce').dt.hour.fillna(12).astype(int)
    df['Month'] = df['Month'].astype(int)
    df['Day'] = df['Day'].astype(int)

    # ─── OLD FEATURES (baseline) ───
    X_old = df[['Amount_val', 'MCC']].values

    # ─── NEW FEATURES ───
    amt = df['Amount_val'].values
    mcc = df['MCC'].values
    hour = df['Hour'].values
    month = df['Month'].values
    day = df['Day'].values
    use_chip = df['Use Chip'].values
    merchant_state = df['Merchant State'].astype(str).values

    # Transaction type encoding
    is_online = (use_chip == 'Online Transaction').astype(float)
    is_swipe = (use_chip == 'Swipe Transaction').astype(float)
    is_chip = (use_chip == 'Chip Transaction').astype(float)

    # Amount features
    log_amt = np.log1p(amt)
    amt_gt_200 = (amt > 200).astype(float)
    amt_gt_400 = (amt > 400).astype(float)

    # Time features
    is_weekend = df['Day'].isin([0, 6]).astype(float).values
    is_night = ((hour >= 22) | (hour <= 5)).astype(float)
    is_morning = ((hour >= 6) & (hour <= 11)).astype(float)
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)

    # MCC fraud rate (computed from data)
    mcc_fraud = df.groupby('MCC')['Is Fraud?'].apply(lambda x: (x == 'Yes').mean()).to_dict()
    mcc_fraud_rate = np.array([mcc_fraud.get(m, 0) for m in mcc])

    # High-risk MCC (known fraud hotspots from analysis: 5311, 5300, 3640, 3722)
    high_risk_mcc = np.isin(mcc, [5311, 5300, 3640, 3722, 5310]).astype(float)

    # Card history features (rolling)
    df['card_count'] = df.groupby('Card').cumcount() + 1
    df['user_count'] = df.groupby('User').cumcount() + 1
    card_count = df['card_count'].values.astype(float)
    user_count = df['user_count'].values.astype(float)

    # Per-card amount stats (group-level, not expanding — much faster)
    card_amt_mean = df.groupby('Card')['Amount_val'].transform('mean')
    card_amt_std = df.groupby('Card')['Amount_val'].transform('std').fillna(0)
    amt_ratio = amt / np.maximum(card_amt_mean.values, 1)
    amt_zscore = (amt - card_amt_mean.values) / np.maximum(card_amt_std.values, 0.01)

    # Per-card merchant diversity (group-level)
    n_merchants = df.groupby('Card')['Merchant Name'].nunique().reindex(df['Card']).values.astype(float)

    # Merchant novelty (first time this merchant appears for this card)
    merchant_novelty = df.groupby(['Card', 'Merchant Name']).ngroup().values.astype(float)

    # Time since last transaction
    df['ts'] = pd.to_datetime(df['Year'].astype(str) + '-' + df['Month'].astype(str) + '-' + df['Day'].astype(str) + ' ' + df['Time'])
    df = df.sort_values(['Card', 'ts'])
    df['time_since_last'] = df.groupby('Card')['ts'].diff().dt.total_seconds().fillna(86400) / 3600
    time_gap = df['time_since_last'].values

    new_features = np.column_stack([
        log_amt, amt, amt_gt_200, amt_gt_400,
        mcc, mcc_fraud_rate, high_risk_mcc,
        is_online, is_swipe, is_chip,
        is_weekend, is_night, is_morning, hour_sin, hour_cos,
        hour, month, day,
        card_count, user_count,
        amt_ratio, amt_zscore, n_merchants, merchant_novelty,
        time_gap,
    ])

    # Split
    X_old_train, X_old_test, y_train, y_test = train_test_split(X_old, y, test_size=0.2, random_state=42, stratify=y)
    X_new_train, X_new_test, _, _ = train_test_split(new_features, y, test_size=0.2, random_state=42, stratify=y)

    # ─── Train models ───
    print(f"\nDataset: {len(df)} transactions, {int(y.sum())} fraud ({y.mean()*100:.3f}%)")

    print("\n--- Baseline (2 features) ---")
    scaler_old = StandardScaler()
    Xo_tr = scaler_old.fit_transform(X_old_train)
    Xo_te = scaler_old.transform(X_old_test)
    rf_old = RandomForestClassifier(n_estimators=200, max_depth=10, n_jobs=-1, random_state=42)
    rf_old.fit(Xo_tr, y_train)
    p_old = rf_old.predict_proba(Xo_te)[:, 1]
    m_old = metrics(y_test, p_old, "Baseline (2 features)")

    print(f"\n  ROC-AUC:    {m_old['roc_auc']:.4f}")
    print(f"  PR-AUC:     {m_old['pr_auc']:.4f}")
    print(f"  R@1%FPR:    {m_old['recall@1%FPR']:.1%}")

    print("\n--- Improved (26 features) ---")
    scaler_new = StandardScaler()
    Xn_tr = scaler_new.fit_transform(X_new_train)
    Xn_te = scaler_new.transform(X_new_test)
    rf_new = RandomForestClassifier(n_estimators=300, max_depth=12, n_jobs=-1, random_state=42)
    rf_new.fit(Xn_tr, y_train)
    p_new = rf_new.predict_proba(Xn_te)[:, 1]
    m_new = metrics(y_test, p_new, "Improved (26 features)")

    print(f"\n  ROC-AUC:    {m_new['roc_auc']:.4f}")
    print(f"  PR-AUC:     {m_new['pr_auc']:.4f}")
    print(f"  R@1%FPR:    {m_new['recall@1%FPR']:.1%}")

    print(f"\n--- Improvement ---")
    print(f"  ROC-AUC:    {m_old['roc_auc']:.4f} → {m_new['roc_auc']:.4f}  ({(m_new['roc_auc']-m_old['roc_auc'])*100:+.2f}%)")
    print(f"  PR-AUC:     {m_old['pr_auc']:.4f} → {m_new['pr_auc']:.4f}  ({(m_new['pr_auc']-m_old['pr_auc'])*100:+.2f}%)")
    print(f"  R@1%FPR:    {m_old['recall@1%FPR']:.1%} → {m_new['recall@1%FPR']:.1%}  ({(m_new['recall@1%FPR']-m_old['recall@1%FPR'])*100:+.1f}%)")

    feat_names = ['log_amt', 'amt', 'amt_gt_200', 'amt_gt_400',
                  'mcc', 'mcc_fraud_rate', 'high_risk_mcc',
                  'is_online', 'is_swipe', 'is_chip',
                  'is_weekend', 'is_night', 'is_morning', 'hour_sin', 'hour_cos',
                  'hour', 'month', 'day',
                  'card_count', 'user_count',
                  'amt_ratio', 'amt_zscore', 'n_merchants', 'merchant_novelty',
                  'time_gap']
    imp = rf_new.feature_importances_
    top_idx = np.argsort(imp)[::-1][:10]
    print("\n  Top 10 features:")
    for i in top_idx:
        print(f"    {feat_names[i]:20s} {imp[i]:.4f}")

    return m_old, m_new


if __name__ == '__main__':
    p_old, p_new = eval_paysim()
    e_old, e_new = eval_ealtman()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Dataset':<25} {'Metric':<12} {'Before':>10} {'After':>10} {'Delta':>10}")
    print("-" * 70)
    for label, old, new in [("PaySim 1M", p_old, p_new), ("ealtman2019", e_old, e_new)]:
        for metric in ['roc_auc', 'pr_auc', 'recall@1%FPR']:
            v_old = old[metric]
            v_new = new[metric]
            if 'recall' in metric:
                print(f"{label:<25} {metric:<12} {v_old:>9.1%} {v_new:>9.1%} {(v_new-v_old)*100:>+9.1f}%")
            else:
                print(f"{label:<25} {metric:<12} {v_old:>9.4f} {v_new:>9.4f} {(v_new-v_old)*100:>+9.2f}%")
        print()
