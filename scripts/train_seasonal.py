#!/usr/bin/env python3
"""Train XGBoost with seasonal decomposition weights.

Improves detection across all months/hours by:
1. Analyzing seasonal fraud patterns (month, hour, day-of-week)
2. Computing sample weights that upweight under-represented fraud patterns
3. Training with these weights so the model doesn't overfit to one season
4. Evaluating month-by-month to verify improvement

Run: .venv/Scripts/python.exe scripts/train_seasonal.py
"""

import sys, os, json, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
import joblib
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.risk_engine.fusion import ML_FEATURES
from src.risk_engine.seasonal import (
    full_decomposition, compute_sample_weights,
    decompose_monthly, decompose_hourly,
)
SEED = 42
N_FEATURES = len(ML_FEATURES)


def compute_features(df: pd.DataFrame) -> np.ndarray:
    """Compute all 21 ML_FEATURES with full per-card feature engineering."""
    n = len(df)
    X = np.zeros((n, N_FEATURES), dtype=np.float32)

    amt = df['amt'].values.astype(np.float32)
    hours = pd.to_datetime(df['trans_date_trans_time']).dt.hour.values
    dow = pd.to_datetime(df['trans_date_trans_time']).dt.dayofweek.values
    unix = df['unix_time'].values.astype(np.float64)

    # 1. amount_ratio
    card_median = df.groupby('cc_num')['amt'].transform('median').values
    X[:, ML_FEATURES.index('amount_ratio')] = np.where(card_median > 0, amt / card_median, 0.0)

    # 2. txn_freq_last_24h
    card_counts = df.groupby('cc_num')['cc_num'].transform('count').values
    X[:, ML_FEATURES.index('txn_freq_last_24h')] = card_counts.astype(np.float32)

    # 3. txn_time_unusual
    X[:, ML_FEATURES.index('txn_time_unusual')] = hours.astype(np.float32) / 23.0

    # 4. new_device_flag
    X[:, ML_FEATURES.index('new_device_flag')] = (df.groupby(['cc_num', 'category']).cumcount() == 0).values.astype(np.float32)

    # 5. unusual_location_flag
    lat1, lon1 = df['lat'].values, df['long'].values
    lat2, lon2 = df['merch_lat'].values, df['merch_long'].values
    dist = np.sqrt((lat1 - lat2)**2 + (lon1 - lon2)**2)
    card_dist_median = df.assign(dist=dist).groupby('cc_num')['dist'].transform('median').values
    X[:, ML_FEATURES.index('unusual_location_flag')] = (dist > card_dist_median * 2).astype(np.float32)

    # 6. unusual_recipient_flag
    X[:, ML_FEATURES.index('unusual_recipient_flag')] = (df.groupby(['cc_num', 'merchant']).cumcount() == 0).values.astype(np.float32)

    # 7. failed_auth_count_24h
    X[:, ML_FEATURES.index('failed_auth_count_24h')] = 0.0

    # 8. days_since_last_similar_txn
    df_sorted = df.sort_values(['cc_num', 'unix_time'])
    gaps = df_sorted.groupby('cc_num')['unix_time'].diff().fillna(86400).values / 86400.0
    X[:, ML_FEATURES.index('days_since_last_similar_txn')] = np.clip(gaps[:n], 0, 365).astype(np.float32)

    # 9. gradual_escalation_score
    rolling_mean = df.groupby('cc_num')['amt'].transform(lambda x: x.expanding().mean()).values
    X[:, ML_FEATURES.index('gradual_escalation_score')] = np.where(rolling_mean > 0, amt / rolling_mean, 0.0).astype(np.float32)

    # 10. known_device_count
    X[:, ML_FEATURES.index('known_device_count')] = df.groupby('cc_num')['merchant'].transform('nunique').values.astype(np.float32)

    # 11. account_tenure_days
    first_tx = df.groupby('cc_num')['unix_time'].transform('min').values
    X[:, ML_FEATURES.index('account_tenure_days')] = ((unix - first_tx) / 86400.0).astype(np.float32)

    # 12. hour_of_day
    X[:, ML_FEATURES.index('hour_of_day')] = hours.astype(np.float32)

    # 13. is_weekend
    X[:, ML_FEATURES.index('is_weekend')] = (dow >= 5).astype(np.float32)

    # 14. shared_device_accounts
    X[:, ML_FEATURES.index('shared_device_accounts')] = df.groupby('merchant')['cc_num'].transform('nunique').values.astype(np.float32)

    # 15. shared_recipient_accounts
    X[:, ML_FEATURES.index('shared_recipient_accounts')] = X[:, ML_FEATURES.index('shared_device_accounts')]

    # 16. mule_ring_score
    X[:, ML_FEATURES.index('mule_ring_score')] = df.groupby('cc_num')['city'].transform('nunique').values.astype(np.float32)

    # 17. hour_deviation
    median_hour = df.groupby('cc_num')['trans_date_trans_time'].transform(lambda x: pd.to_datetime(x).dt.hour.median()).values
    X[:, ML_FEATURES.index('hour_deviation')] = np.abs(hours - median_hour).astype(np.float32)

    # 18. amount_zscore
    grp_mean = df.groupby('cc_num')['amt'].transform('mean').values
    grp_std = df.groupby('cc_num')['amt'].transform('std').fillna(1.0).values
    X[:, ML_FEATURES.index('amount_zscore')] = np.where(grp_std > 0, (amt - grp_mean) / grp_std, 0.0).astype(np.float32)

    # 19. velocity_deviation
    X[:, ML_FEATURES.index('velocity_deviation')] = card_counts.astype(np.float32) / 24.0

    # 20. recipient_novelty
    seen = df.groupby(['cc_num', 'merchant']).cumcount().values + 1
    X[:, ML_FEATURES.index('recipient_novelty')] = (1.0 / seen).astype(np.float32)

    # 21. txn_regularity
    std_gap = df.groupby('cc_num')['unix_time'].transform(lambda x: x.diff().fillna(0).expanding().std()).values / 3600.0
    X[:, ML_FEATURES.index('txn_regularity')] = np.clip(std_gap, 0, 100).astype(np.float32)

    return X


def eval_metrics(y_true, y_prob, prefix=""):
    """Compute standard fraud detection metrics."""
    roc = roc_auc_score(y_true, y_prob)
    pr = average_precision_score(y_true, y_prob)
    fpr_arr, tpr_arr, _ = roc_curve(y_true, y_prob)
    mets = {}
    for tf, lab in [(0.0001,'r001pct'), (0.0005,'r005pct'), (0.001,'r01pct'),
                     (0.005,'r05pct'), (0.01,'r1pct')]:
        mask = fpr_arr <= tf
        mets[prefix + lab] = round(float(tpr_arr[np.where(mask)[0][-1]]), 4) if mask.any() else 0.0
    mets[prefix + 'roc_auc'] = round(float(roc), 4)
    mets[prefix + 'pr_auc'] = round(float(pr), 4)
    return mets


def main():
    np.random.seed(SEED)
    OUT = Path("data/real_data_evaluation")

    print("=" * 70)
    print("SEASONAL DECOMPOSITION + WEIGHTED TRAINING")
    print("=" * 70)

    # --- Phase 1: Load data ---
    print("\n--- Phase 1: Load data ---")
    df = pd.concat([
        pd.read_csv('data/kaggle_fraud/fraudTrain.csv'),
        pd.read_csv('data/kaggle_fraud/fraudTest.csv'),
    ], ignore_index=True)
    df = df.drop(columns=['first', 'last', 'street', 'state', 'zip', 'dob', 'job', 'trans_num', 'Unnamed: 0'], errors='ignore')
    df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
    df = df.sort_values('unix_time').reset_index(drop=True)
    print(f"Total: {len(df):,} rows, {int(df.is_fraud.sum()):,} fraud ({df.is_fraud.mean()*100:.3f}%)")

    # --- Phase 2: Seasonal decomposition ---
    print("\n--- Phase 2: Seasonal decomposition ---")
    decomp = full_decomposition(df)

    print(f"\n  Overall fraud rate: {decomp['overall_fraud_rate_pct']:.3f}%")
    print(f"  Trend: {decomp['monthly']['trend_direction']} (slope={decomp['monthly']['trend_slope']:.8f})")
    print(f"  High-risk months: {decomp['monthly']['high_risk_months']}")
    print(f"  Low-risk months: {decomp['monthly']['low_risk_months']}")
    print(f"  Peak danger hours: {decomp['hourly']['peak_danger_hours']}")
    print(f"  Night/day fraud ratio: {decomp['hourly']['night_vs_day_ratio']}x")
    print(f"  Peak day: {decomp['day_of_week']['peak_day']}")
    print(f"  Dangerous interactions: {len(decomp['dangerous_interactions'])}")

    # --- Phase 3: Compute seasonal weights ---
    print("\n--- Phase 3: Compute seasonal weights ---")
    weights = compute_sample_weights(df, method="inverse_rate")
    fraud_mask = df['is_fraud'].values.astype(bool)

    print(f"  Weight stats (fraud): mean={weights[fraud_mask].mean():.2f}, "
          f"min={weights[fraud_mask].min():.2f}, max={weights[fraud_mask].max():.2f}")
    print(f"  Weight stats (legit): mean={weights[~fraud_mask].mean():.2f}")

    # Show weight by month
    months = df['trans_date_trans_time'].dt.month.values
    print("\n  Month weights (fraud samples):")
    for m in range(1, 13):
        m_mask = fraud_mask & (months == m)
        if m_mask.sum() > 0:
            avg_w = weights[m_mask].mean()
            print(f"    Month {m:2d}: avg_weight={avg_w:.2f} ({m_mask.sum()} fraud samples)")

    # --- Phase 4: Feature engineering ---
    print("\n--- Phase 4: Feature engineering ---")
    t0 = time.time()
    X = compute_features(df)
    y = df['is_fraud'].values.astype(int)
    print(f"  Feature matrix: {X.shape}, NaN: {np.isnan(X).sum()}")
    print(f"  Time: {time.time()-t0:.1f}s")

    # Impute + scale
    imputer = SimpleImputer(strategy='median')
    X_clean = imputer.fit_transform(X)
    X_clean = np.nan_to_num(X_clean, nan=0.0, posinf=10.0, neginf=-10.0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clean)

    # --- Phase 5: Temporal split ---
    print("\n--- Phase 5: Temporal train/test split ---")
    n = len(df)
    split_idx = int(n * 0.8)
    X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    w_train = weights[:split_idx]
    months_train = months[:split_idx]
    months_test = months[split_idx:]

    print(f"  Train: {len(X_train):,} rows, {int(y_train.sum()):,} fraud")
    print(f"  Test:  {len(X_test):,} rows, {int(y_test.sum()):,} fraud")

    # --- Phase 6: Train WITHOUT seasonal weights (baseline) ---
    print("\n--- Phase 6a: Train baseline (no seasonal weights) ---")
    from xgboost import XGBClassifier

    n_pos = int(y_train.sum()); n_neg = len(y_train) - n_pos
    scale = max(n_neg / max(n_pos, 1), 1.0)

    model_base = XGBClassifier(
        n_estimators=800, max_depth=12, learning_rate=0.02,
        subsample=0.8, colsample_bytree=0.8, scale_pos_weight=scale,
        reg_lambda=3.0, reg_alpha=1.0, min_child_weight=5,
        random_state=SEED, eval_metric='auc', use_label_encoder=False,
        early_stopping_rounds=50, n_jobs=-1, verbosity=0,
    )
    t0 = time.time()
    model_base.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    print(f"  Trained in {time.time()-t0:.1f}s (best_iter={model_base.best_iteration})")

    prob_base = model_base.predict_proba(X_test)[:, 1]
    base_overall = eval_metrics(y_test, prob_base, prefix="base_")
    print(f"  Baseline ROC-AUC: {base_overall['base_roc_auc']:.4f}")
    print(f"  Baseline PR-AUC: {base_overall['base_pr_auc']:.4f}")

    # --- Phase 7: Train WITH seasonal weights ---
    print("\n--- Phase 6b: Train with seasonal weights ---")
    model_seasonal = XGBClassifier(
        n_estimators=800, max_depth=12, learning_rate=0.02,
        subsample=0.8, colsample_bytree=0.8, scale_pos_weight=scale,
        reg_lambda=3.0, reg_alpha=1.0, min_child_weight=5,
        random_state=SEED, eval_metric='auc', use_label_encoder=False,
        early_stopping_rounds=50, n_jobs=-1, verbosity=0,
    )
    t0 = time.time()
    model_seasonal.fit(X_train, y_train, sample_weight=w_train,
                       eval_set=[(X_test, y_test)], verbose=False)
    print(f"  Trained in {time.time()-t0:.1f}s (best_iter={model_seasonal.best_iteration})")

    prob_seasonal = model_seasonal.predict_proba(X_test)[:, 1]
    seasonal_overall = eval_metrics(y_test, prob_seasonal, prefix="sea_")
    print(f"  Seasonal ROC-AUC: {seasonal_overall['sea_roc_auc']:.4f}")
    print(f"  Seasonal PR-AUC: {seasonal_overall['sea_pr_auc']:.4f}")

    # --- Phase 8: Month-by-month comparison ---
    print("\n" + "=" * 70)
    print("MONTH-BY-MONTH COMPARISON")
    print("=" * 70)
    print(f"{'Month':>6} {'N':>7} {'Fraud':>6} {'Prev':>6} │ {'Base ROC':>9} {'Sea ROC':>9} {'Δ':>7} │ {'Base R@1%':>10} {'Sea R@1%':>10} {'Δ':>7}")
    print("-" * 100)

    month_names = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',
                   7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}
    monthly_results = []

    for m in range(1, 13):
        mask = months_test == m
        if mask.sum() == 0:
            continue
        y_m = y_test[mask]
        p_b = prob_base[mask]
        p_s = prob_seasonal[mask]
        if y_m.sum() == 0:
            print(f"{month_names[m]:>6} {mask.sum():>7} {0:>6} {0:>6.2f} │ {'N/A':>9} {'N/A':>9} {'':>7} │ {'N/A':>10} {'N/A':>10} {'':>7}")
            continue

        b_mets = eval_metrics(y_m, p_b, prefix="b_")
        s_mets = eval_metrics(y_m, p_s, prefix="s_")
        prev = y_m.mean() * 100

        roc_delta = s_mets['s_roc_auc'] - b_mets['b_roc_auc']
        r1_delta = s_mets['s_r1pct'] - b_mets['b_r1pct']
        print(f"{month_names[m]:>6} {mask.sum():>7} {int(y_m.sum()):>6} {prev:>6.2f} │ "
              f"{b_mets['b_roc_auc']:>9.4f} {s_mets['s_roc_auc']:>9.4f} {roc_delta:>+7.4f} │ "
              f"{b_mets['b_r1pct']:>10.4f} {s_mets['s_r1pct']:>10.4f} {r1_delta:>+7.4f}")
        monthly_results.append({
            "month": month_names[m], "n": int(mask.sum()), "fraud": int(y_m.sum()),
            "prev_pct": round(prev, 3),
            "base_roc": b_mets['b_roc_auc'], "seasonal_roc": s_mets['s_roc_auc'],
            "delta_roc": round(roc_delta, 4),
            "base_r1pct": b_mets['b_r1pct'], "seasonal_r1pct": s_mets['s_r1pct'],
            "delta_r1pct": round(r1_delta, 4),
        })

    # Summary
    base_avg_roc = np.mean([r['base_roc'] for r in monthly_results])
    sea_avg_roc = np.mean([r['seasonal_roc'] for r in monthly_results])
    base_avg_r1 = np.mean([r['base_r1pct'] for r in monthly_results])
    sea_avg_r1 = np.mean([r['seasonal_r1pct'] for r in monthly_results])

    print("-" * 100)
    print(f"{'AVG':>6} {'':>7} {'':>6} {'':>6} │ "
          f"{base_avg_roc:>9.4f} {sea_avg_roc:>9.4f} {sea_avg_roc-base_avg_roc:>+7.4f} │ "
          f"{base_avg_r1:>10.4f} {sea_avg_r1:>10.4f} {sea_avg_r1-base_avg_r1:>+7.4f}")

    # Improvement summary
    improved_months = sum(1 for r in monthly_results if r['delta_roc'] > 0)
    print(f"\n  Months improved: {improved_months}/{len(monthly_results)}")
    print(f"  Avg ROC-AUC delta: {sea_avg_roc - base_avg_roc:+.4f}")
    print(f"  Avg Recall@1%FPR delta: {sea_avg_r1 - base_avg_r1:+.4f}")

    # --- Phase 9: Save artifacts ---
    print("\n--- Phase 7: Save artifacts ---")
    # Save the better model
    if seasonal_overall['sea_roc_auc'] >= base_overall['base_roc_auc']:
        final_model = model_seasonal
        final_prob = prob_seasonal
        final_mets = seasonal_overall
        model_tag = "seasonal_weighted"
    else:
        final_model = model_base
        final_prob = prob_base
        final_mets = base_overall
        model_tag = "baseline"

    joblib.dump(final_model, 'models/artifacts/xgboost.joblib')
    joblib.dump(scaler, 'models/artifacts/scaler.joblib')
    joblib.dump(imputer, 'models/artifacts/imputer.joblib')

    # Save seasonal weights for drift detector
    seasonal_meta = {
        "method": model_tag,
        "decomposition": decomp,
        "monthly_comparison": monthly_results,
        "overall_baseline": base_overall,
        "overall_seasonal": seasonal_overall,
        "sample_weight_stats": {
            "mean_fraud_weight": round(float(weights[fraud_mask].mean()), 3),
            "max_fraud_weight": round(float(weights[fraud_mask].max()), 3),
            "method": "inverse_rate",
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "seasonal_decomposition.json").write_text(json.dumps(seasonal_meta, indent=2))

    # Save threshold config
    fpr_arr, tpr_arr, thresholds = roc_curve(y_test, final_prob)
    target_005 = fpr_arr <= 0.0005
    target_01 = fpr_arr <= 0.001
    target_1 = fpr_arr <= 0.01
    threshold_config = {
        'threshold_005fpr': float(thresholds[np.where(target_005)[0][-1]]) if target_005.any() else 0.5,
        'threshold_01fpr': float(thresholds[np.where(target_01)[0][-1]]) if target_01.any() else 0.5,
        'threshold_1fpr': float(thresholds[np.where(target_1)[0][-1]]) if target_1.any() else 0.5,
        'roc_auc': final_mets[f'{model_tag[:3]}_roc_auc'] if f'{model_tag[:3]}_roc_auc' in final_mets else final_mets.get('sea_roc_auc', final_mets.get('base_roc_auc')),
        'pr_auc': final_mets.get('sea_pr_auc', final_mets.get('base_pr_auc')),
    }
    with open('models/artifacts/threshold_config.json', 'w') as f:
        json.dump(threshold_config, f, indent=2)

    print(f"  Saved: xgboost.joblib, scaler.joblib, imputer.joblib, threshold_config.json")
    print(f"  Model: {model_tag}")
    print(f"  Seasonal report: {OUT / 'seasonal_decomposition.json'}")

    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
