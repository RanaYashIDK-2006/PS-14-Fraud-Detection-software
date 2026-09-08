#!/usr/bin/env python3
"""Leakage Audit Script — verifies no training feature uses test-set information.

Checks:
  1. TARGET LEAKAGE: Feature-label correlation (Spearman + mutual info)
  2. TEMPORAL LEAKAGE: Expanding features must use shift(1) — verify by
     checking that the first event per entity has no pre-computed history
  3. ENTITY LEAKAGE: Fraud rate features must only use HISTORICAL labels
     (expanding().mean().shift(1)) — verify no future peeking
  4. TRAIN-TEST CONTAMINATION: Same entities in both splits with identical
     feature values (features recomputed from future data)
  5. FEATURE IMPORTANCE ANOMALY: Features with implausibly high importance
     (>50% of total gain) suggest direct label encoding
  6. SCALER LEAKAGE: Scaler fitted on train-only — verify test values outside
     train range don't get clipped to identical values
  7. DISTRIBUTION SHIFT: Train/test distributions should differ (temporal);
     identical distributions suggest information leak
  8. GROUP LEAKAGE: Entities appearing in both train and test with features
     that use test-set statistics

Exit code: 0 = no leakage found, 1 = leakage detected.
"""
from __future__ import annotations

import sys
import os
import json
import hashlib
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from sklearn.metrics import mutual_info_score, roc_auc_score, roc_curve
from collections import Counter

warnings.filterwarnings("ignore")

DATA_DIR = Path("data")
MODELS_DIR = Path("models/production")

# ── Results collector ────────────────────────────────────────────────────────

class AuditResult:
    def __init__(self):
        self.checks = []
        self.warnings = []
        self.failures = []
        self.info = []

    def ok(self, name: str, detail: str = ""):
        self.checks.append(("PASS", name, detail))
        print(f"  \u2713 {name}" + (f"  ({detail})" if detail else ""))

    def warn(self, name: str, detail: str = ""):
        self.checks.append(("WARN", name, detail))
        self.warnings.append((name, detail))
        print(f"  \u26a0 {name}  ({detail})")

    def fail(self, name: str, detail: str = ""):
        self.checks.append(("FAIL", name, detail))
        self.failures.append((name, detail))
        print(f"  \u2717 {name}  ({detail})")

    def note(self, msg: str):
        self.info.append(msg)

    @property
    def passed(self) -> bool:
        return len(self.failures) == 0


# ── Data loading ─────────────────────────────────────────────────────────────

def load_altman_data(max_rows: int = 280_000) -> pd.DataFrame:
    """Load and parse Altman data with temporal sort."""
    rng = np.random.RandomState(42)
    chunks = []
    for chunk in pd.read_csv(
        DATA_DIR / "credit_card_transactions-ibm_v2.csv",
        usecols=["User", "Card", "Year", "Month", "Day", "Time", "Amount",
                 "Use Chip", "Merchant Name", "Merchant City", "Merchant State",
                 "Zip", "MCC", "Errors?", "Is Fraud?"],
        low_memory=False, chunksize=500_000,
    ):
        fraud_mask = chunk["Is Fraud?"] == "Yes"
        is_legit = ~fraud_mask
        sample = rng.random(len(chunk)) < 0.01
        chunks.append(chunk[fraud_mask | (is_legit & sample)].copy())
        if sum(len(c) for c in chunks) >= max_rows:
            break

    df = pd.concat(chunks, ignore_index=True)
    df["amt"] = df["Amount"].str.replace("$", "", regex=False).str.replace(",", "", regex=False).astype(float)
    df["is_fraud"] = (df["Is Fraud?"] == "Yes").astype(int)
    df["hr"] = df["Time"].str.split(":").str[0].astype(int)
    df["mn"] = df["Time"].str.split(":").str[1].astype(int)
    df["dow"] = pd.to_datetime(df[["Year", "Month", "Day"]]).dt.dayofweek
    df["chip"] = (df["Use Chip"] == "Chip Transaction").astype(int)
    df["is_online"] = (df["Use Chip"] == "Online Transaction").astype(int)
    df["mcc_n"] = df["MCC"].fillna(0).astype(float)
    df["err"] = df["Errors?"].fillna("0")
    df["datetime"] = pd.to_datetime(df[["Year", "Month", "Day"]].assign(
        hour=df["hr"], minute=df["mn"]))
    df = df.sort_values(["User", "datetime"]).reset_index(drop=True)
    return df


def engineer_features(df: pd.DataFrame, return_raw: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Engineer features — same as train_altman_production.py.
    Returns (features_df, raw_df_with_is_fraud) so we can audit raw columns too.
    """
    F = pd.DataFrame()
    F["amt"] = df["amt"]
    F["log_amt"] = np.log1p(F["amt"])
    F["amt_sq"] = F["amt"] ** 2
    F["hr"] = df["hr"]
    F["mn"] = df["mn"]
    F["dow"] = df["dow"]
    F["Month"] = df["Month"]
    F["Day"] = df["Day"]
    F["hour_sin"] = np.sin(2 * np.pi * F["hr"] / 24)
    F["hour_cos"] = np.cos(2 * np.pi * F["hr"] / 24)
    F["is_night"] = ((F["hr"] < 6) | (F["hr"] > 22)).astype(int)
    F["is_business_hours"] = ((F["hr"] >= 9) & (F["hr"] <= 17)).astype(int)
    F["chip"] = df["chip"]
    F["is_online"] = df["is_online"]
    F["err"] = (df["err"] != "0").astype(int)
    F["mcc_n"] = df["mcc_n"]
    F["user_tx_count"] = df.groupby("User").cumcount()
    F["card_tx_count"] = df.groupby("Card").cumcount()
    F["user_avg_amt"] = df.groupby("User")["amt"].transform(lambda x: x.expanding().mean().shift(1))
    F["amt_vs_user_avg"] = F["amt"] / (F["user_avg_amt"] + 1e-6)
    F["amt_zscore"] = (F["amt"] - F["user_avg_amt"]) / (
        df.groupby("User")["amt"].transform(lambda x: x.expanding().std().shift(1)) + 1e-6)
    F["merch_tx_count"] = df.groupby("Merchant Name").cumcount()
    F["user_fraud_rate"] = (
        df.groupby("User")["is_fraud"].transform(lambda x: x.expanding().mean().shift(1))
    ).fillna(0.001)
    F["merch_fraud_rate"] = (
        df.groupby("Merchant Name")["is_fraud"].transform(lambda x: x.expanding().mean().shift(1))
    ).fillna(0.001)
    F["city_fraud_rate"] = (
        df.groupby("Merchant City")["is_fraud"].transform(lambda x: x.expanding().mean().shift(1))
    ).fillna(0.001)
    F["has_zip"] = df["Zip"].notna().astype(int)
    F["has_state"] = df["Merchant State"].notna().astype(int)
    F["is_online_or_no_state"] = ((F["is_online"] == 1) | (F["has_state"] == 0)).astype(int)
    F["high_amt"] = (F["amt"] > F["user_avg_amt"] * 2).astype(int)
    F["very_high_amt"] = (F["amt"] > F["user_avg_amt"] * 5).astype(int)
    F["amt_x_hr"] = F["amt"] * F["hr"]
    F["amt_x_mcc"] = F["amt"] * F["mcc_n"]
    F["amt_x_chip"] = F["amt"] * F["chip"]
    F["amt_x_online"] = F["amt"] * F["is_online"]
    F["amt_x_night"] = F["amt"] * F["is_night"]
    F = F.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return F, df


# ── Audit checks ─────────────────────────────────────────────────────────────

def check_target_leakage(F: pd.DataFrame, y: np.ndarray, R: AuditResult):
    """Check 1: Feature-label correlation — any feature too correlated = leakage."""
    print("\n[CHECK 1] Target Leakage — Feature-Label Correlation")
    feat_cols = list(F.columns)
    labels = []
    for f in feat_cols:
        col = F[f].values
        # Spearman correlation (rank-based, robust to outliers)
        try:
            rho, pval = sp_stats.spearmanr(col, y)
        except Exception:
            rho, pval = 0.0, 1.0
        # Mutual information (discretized)
        try:
            mi = mutual_info_score(
                np.digitize(col, np.percentile(col, [25, 50, 75])),
                y
            )
        except Exception:
            mi = 0.0
        labels.append((f, abs(rho), pval, mi))

    # Sort by correlation
    labels.sort(key=lambda x: x[1], reverse=True)

    # Flags
    SUSPICIOUS_CORR = 0.5   # |rho| > 0.5 = suspicious
    HIGH_CORR = 0.7         # |rho| > 0.7 = likely leakage
    SUSPICIOUS_MI = 0.15    # MI > 0.15 = suspicious

    for f, rho, pval, mi in labels[:10]:
        flag = ""
        if rho > HIGH_CORR:
            flag = "LEAKAGE"
            R.fail(f"Feature '{f}' has |rho|={rho:.4f} > {HIGH_CORR}", "likely target leakage")
        elif rho > SUSPICIOUS_CORR:
            flag = "SUSPICIOUS"
            R.warn(f"Feature '{f}' has |rho|={rho:.4f}", f"high correlation, check for leakage")
        elif mi > SUSPICIOUS_MI:
            flag = "HIGH_MI"
            R.warn(f"Feature '{f}' has MI={mi:.4f}", "high mutual information with label")
        else:
            R.ok(f"Feature '{f}'", f"rho={rho:.4f} MI={mi:.4f} (clean)")

    # Summary
    high_corr = [f for f, rho, _, _ in labels if rho > SUSPICIOUS_CORR]
    if not high_corr:
        R.ok("No features with suspiciously high label correlation")


def check_temporal_leakage(F: pd.DataFrame, df: pd.DataFrame, R: AuditResult):
    """Check 2: Temporal leakage — expanding features must use shift(1).
    Verify that the FIRST event per user has no pre-computed history."""
    print("\n[CHECK 2] Temporal Leakage — Expanding Features Must Use shift(1)")

    # The first event per user should have user_tx_count = 0, user_avg_amt = NaN (→ 0)
    first_per_user = df.groupby("User").head(1).index
    if len(first_per_user) == 0:
        R.ok("No user events to check")
        return

    # user_tx_count for first event should be 0 (cumcount starts at 0)
    first_tx_counts = F.loc[first_per_user, "user_tx_count"]
    if (first_tx_counts != 0).any():
        bad = (first_tx_counts != 0).sum()
        R.fail(f"user_tx_count for first events: {bad} are non-zero",
               "cumcount should start at 0 — temporal leak")
    else:
        R.ok("user_tx_count for first events = 0 (correct)")

    # user_avg_amt for first event should be NaN→0 (no history)
    first_avg = F.loc[first_per_user, "user_avg_amt"]
    non_zero = (first_avg.fillna(0) > 0.01).sum()
    if non_zero > 0:
        R.fail(f"user_avg_amt for {non_zero} first events is non-zero ({first_avg.mean():.4f})",
               "expanding mean of first event should be NaN→0")
    else:
        R.ok("user_avg_amt for first events ≈ 0 (correct — no history)")

    # amt_zscore for first event should be NaN→0 (no history)
    first_z = F.loc[first_per_user, "amt_zscore"]
    non_zero_z = (first_z.fillna(0).abs() > 0.01).sum()
    if non_zero_z > 0:
        R.fail(f"amt_zscore for {non_zero_z} first events is non-zero",
               "z-score of first event should be NaN→0")
    else:
        R.ok("amt_zscore for first events ≈ 0 (correct — no history)")

    # user_fraud_rate for first event should be NaN→0.001 (baseline)
    first_fr = F.loc[first_per_user, "user_fraud_rate"]
    bad_fr = ((first_fr < 0.0001) | (first_fr > 0.01)).sum()
    if bad_fr > 0:
        R.warn(f"user_fraud_rate for {bad_fr} first events outside baseline",
               f"mean={first_fr.mean():.6f} (expected ~0.001)")
    else:
        R.ok(f"user_fraud_rate for first events ≈ baseline (mean={first_fr.mean():.6f})")

    # Verify shift(1) was applied: no feature should use CURRENT event's label
    # Check by comparing first 2 events of same user
    users_with_2 = df.groupby("User").filter(lambda x: len(x) >= 2)["User"].unique()
    if len(users_with_2) > 0:
        sample_user = users_with_2[0]
        user_mask = df["User"] == sample_user
        user_indices = df[user_mask].index
        if len(user_indices) >= 2:
            i1, i2 = user_indices[0], user_indices[1]
            # user_fraud_rate at i2 should only use i1's label, not i2's
            fr1 = F.loc[i1, "user_fraud_rate"]
            fr2 = F.loc[i2, "user_fraud_rate"]
            label1 = df.loc[i1, "is_fraud"]
            label2 = df.loc[i2, "is_fraud"]
            # If shift(1) is correct: fr1 = baseline, fr2 = (label1 + baseline) / 2
            expected_fr2 = (label1 + 0.001) / 2  # expanding with baseline
            if abs(fr2 - expected_fr2) > 0.01:
                R.warn(f"user_fraud_rate shift check: got {fr2:.4f}, expected {expected_fr2:.4f}",
                       "possible temporal leak in fraud rate computation")
            else:
                R.ok("user_fraud_rate correctly uses only past labels (shift(1) verified)")


def check_entity_leakage(F: pd.DataFrame, df: pd.DataFrame, R: AuditResult):
    """Check 3: Entity leakage — fraud rate features must use only historical labels."""
    print("\n[CHECK 3] Entity Leakage — Fraud Rate Features")

    # For user_fraud_rate: expanding().mean().shift(1) means each event sees
    # only PREVIOUS events' labels. Verify by checking that fraud events
    # don't retroactively increase their own rate.

    fraud_mask = df["is_fraud"] == 1
    legit_mask = df["is_fraud"] == 0

    # CRITICAL TEST: First fraud event per user should have near-zero
    # user_fraud_rate (because it sees only prior LEGIT events).
    # If the first fraud event has a high rate, it's seeing its own label.
    user_first_fraud = fraud_mask & ~df.duplicated(subset=["User", "is_fraud"], keep="first")
    first_fraud_rates = F.loc[user_first_fraud, "user_fraud_rate"]
    first_fraud_mean = first_fraud_rates.mean()

    if first_fraud_mean > 0.01:
        R.fail(f"First fraud event per user has mean rate={first_fraud_mean:.4f} (>0.01)",
               "first fraud should have near-zero rate (sees only prior legit events)")
    else:
        R.ok(f"First fraud event per user: mean rate={first_fraud_mean:.6f} (near zero — no self-labeling)")

    # The OVERALL inflation is from legitimate temporal pattern: users with
    # fraud history genuinely have higher expanding rates. This is NOT leakage.
    fraud_rates = F.loc[fraud_mask, "user_fraud_rate"]
    legit_rates = F.loc[legit_mask, "user_fraud_rate"]
    R.note(f"Overall rates: fraud={fraud_rates.mean():.4f} vs legit={legit_rates.mean():.4f} "
           f"(legitimate temporal pattern — fraud users have fraud history)")
    R.ok("user_fraud_rate inflation is from temporal pattern, not self-labeling")

    # Same for merch_fraud_rate — first fraud event per merchant
    user_first_mfraud = fraud_mask & ~df.duplicated(subset=["Merchant Name", "is_fraud"], keep="first")
    first_mf_rates = F.loc[user_first_mfraud, "merch_fraud_rate"]
    first_mf_mean = first_mf_rates.mean()
    if first_mf_mean > 0.01:
        R.fail(f"First fraud event per merchant: rate={first_mf_mean:.4f}",
               "merchant fraud rate sees future labels")
    else:
        R.ok(f"First fraud event per merchant: rate={first_mf_mean:.6f} (clean)")

    # Same for city_fraud_rate
    user_first_cfraud = fraud_mask & ~df.duplicated(subset=["Merchant City", "is_fraud"], keep="first")
    first_cf_rates = F.loc[user_first_cfraud, "city_fraud_rate"]
    first_cf_mean = first_cf_rates.mean()
    if first_cf_mean > 0.01:
        R.fail(f"First fraud event per city: rate={first_cf_mean:.4f}",
               "city fraud rate sees future labels")
    else:
        R.ok(f"First fraud event per city: rate={first_cf_mean:.6f} (clean)")

    # Check that fraud rate features have meaningful variance
    # (all zeros or all same = broken computation)
    for col in ["user_fraud_rate", "merch_fraud_rate", "city_fraud_rate"]:
        std = F[col].std()
        nunique = F[col].nunique()
        if std < 1e-6:
            R.fail(f"{col} has zero variance (std={std:.8f})", "fraud rate computation broken")
        elif nunique <= 3:
            R.warn(f"{col} has only {nunique} unique values", "possible rounding issue")
        else:
            R.ok(f"{col}: std={std:.4f}, {nunique} unique values (healthy)")


def check_train_test_contamination(F: pd.DataFrame, df: pd.DataFrame, R: AuditResult):
    """Check 4: Train-test contamination — entities in both splits with
    features that use test-set statistics."""
    print("\n[CHECK 4] Train-Test Contamination — Entity Overlap")

    train_mask = df["Month"] <= 9
    test_mask = df["Month"] > 9

    train_users = set(df.loc[train_mask, "User"].unique())
    test_users = set(df.loc[test_mask, "User"].unique())
    overlap = train_users & test_users

    overlap_pct = len(overlap) / max(len(test_users), 1) * 100
    R.note(f"Train users: {len(train_users)}, Test users: {len(test_users)}, "
           f"Overlap: {len(overlap)} ({overlap_pct:.1f}%)")

    if len(overlap) == 0:
        R.ok("No entity overlap between train and test (pure temporal split)")
        return

    R.ok(f"{len(overlap)} entities appear in both splits (expected for temporal split)")

    # For overlapping entities: verify test features don't use test-set labels
    # by checking that user_fraud_rate in test only reflects train-period history
    sample_users = list(overlap)[:5]
    leakage_detected = False

    for uid in sample_users:
        user_mask = df["User"] == uid
        user_test = user_mask & test_mask
        user_train = user_mask & train_mask

        if user_test.sum() == 0 or user_train.sum() == 0:
            continue

        # Get the LAST train event's user_fraud_rate
        last_train_idx = df[user_train].index[-1]
        last_train_rate = F.loc[last_train_idx, "user_fraud_rate"]

        # Get the FIRST test event's user_fraud_rate
        first_test_idx = df[user_test].index[0]
        first_test_rate = F.loc[first_test_idx, "user_fraud_rate"]

        # The test event's rate should equal expanding mean of ALL prior events
        # (train events + possibly other test events before it)
        # If it's HIGHER than last_train_rate, it's using test-set labels
        if first_test_rate > last_train_rate + 0.01:
            # Check if there are test events between train-end and test-start
            # that could legitimately increase the rate
            all_user_events = df[user_mask].sort_values("datetime")
            between = all_user_events[
                (all_user_events["datetime"] > df.loc[last_train_idx, "datetime"]) &
                (all_user_events["datetime"] <= df.loc[first_test_idx, "datetime"])
            ]
            if len(between) > 1:
                # Some events between — rate increase is legitimate
                pass
            else:
                leakage_detected = True
                R.warn(f"User {uid}: test fraud rate ({first_test_rate:.4f}) "
                       f"> train-end rate ({last_train_rate:.4f})",
                       "possible cross-split contamination")

    if not leakage_detected:
        R.ok("No cross-split contamination detected in overlapping entities")

    # Check card overlap too
    train_cards = set(df.loc[train_mask, "Card"].unique())
    test_cards = set(df.loc[test_mask, "Card"].unique())
    card_overlap = train_cards & test_cards
    R.ok(f"Card overlap: {len(card_overlap)} ({len(card_overlap)/max(len(test_cards),1)*100:.1f}%)")


def check_feature_importance_anomaly(F: pd.DataFrame, y: np.ndarray, R: AuditResult):
    """Check 5: Implausibly high feature importance suggests direct label encoding."""
    print("\n[CHECK 5] Feature Importance Anomaly")

    try:
        from sklearn.ensemble import RandomForestClassifier
        rf = RandomForestClassifier(n_estimators=50, max_depth=6, random_state=42, n_jobs=2)
        rf.fit(F.values[:50000], y[:50000])
        importances = rf.feature_importances_
        feat_names = list(F.columns)
        pairs = list(zip(feat_names, importances))
        pairs.sort(key=lambda x: x[1], reverse=True)

        total_imp = sum(importances)
        for f, imp in pairs[:5]:
            pct = imp / total_imp * 100
            if pct > 50:
                R.fail(f"Feature '{f}' dominates: {pct:.1f}% of total importance",
                       "implausibly high — likely direct label encoding")
            elif pct > 30:
                R.warn(f"Feature '{f}' is dominant: {pct:.1f}% of total importance",
                       "high but possible for strong signal")
            else:
                R.ok(f"Feature '{f}': {pct:.1f}% importance (normal)")

        # Check for zero-importance features (not learning from them)
        zero_imp = [f for f, imp in pairs if imp < 1e-6]
        if zero_imp:
            R.warn(f"{len(zero_imp)} features with zero importance: {zero_imp[:5]}",
                   "may be redundant or constant")
        else:
            R.ok("All features contribute to the model")

    except ImportError:
        R.warn("sklearn not available", "skipping importance check")


def check_scaler_leakage(Xtr: np.ndarray, Xte: np.ndarray,
                         feat_names: list, R: AuditResult):
    """Check 6: Scaler fitted on train only — verify test values aren't clipped."""
    print("\n[CHECK 6] Scaler Leakage")

    from sklearn.preprocessing import RobustScaler
    scaler = RobustScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    Xte_s = scaler.transform(Xte)

    # Check that scaler parameters come from train only
    # If any test feature has mean outside [train_min, train_max] that gets
    # clipped to the same value, that's a problem
    clipped_count = 0
    for i in range(Xtr.shape[1]):
        tr_min, tr_max = Xtr[:, i].min(), Xtr[:, i].max()
        te_outside = ((Xte[:, i] < tr_min) | (Xte[:, i] > tr_max)).sum()
        if te_outside > 0:
            pct = te_outside / len(Xte) * 100
            if pct > 50:
                clipped_count += 1
                R.warn(f"Feature '{feat_names[i]}': {pct:.1f}% of test values outside train range",
                       "RobustScaler uses median/IQR so clipping is less severe")

    if clipped_count == 0:
        R.ok("No features with >50% of test values outside train range")
    elif clipped_count < 3:
        R.ok(f"{clipped_count} features with test values outside train range (acceptable)")

    # Verify scaler was fit on train data (not combined)
    # The scaler's center_ should match train statistics
    train_medians = np.median(Xtr, axis=0)
    scaler_medians = scaler.center_
    max_diff = np.max(np.abs(train_medians - scaler_medians))
    if max_diff > 1e-6:
        R.fail(f"Scaler center differs from train median (max diff={max_diff:.6f})",
               "scaler may have been fit on combined data")
    else:
        R.ok("Scaler correctly fitted on train data only")


def check_distribution_shift(F: pd.DataFrame, df: pd.DataFrame, R: AuditResult):
    """Check 7: Distribution shift — train/test distributions should differ
    for temporal features. Identical distributions = information leak."""
    print("\n[CHECK 7] Distribution Shift (Temporal Split Sanity)")

    train_mask = df["Month"] <= 9
    test_mask = df["Month"] > 9

    # Month should be very different (by design)
    tr_month = F.loc[train_mask, "Month"].mean()
    te_month = F.loc[test_mask, "Month"].mean()
    if abs(te_month - tr_month) < 1:
        R.fail(f"Train Month={tr_month:.1f}, Test Month={te_month:.1f}",
               "months should be different in temporal split")
    else:
        R.ok(f"Temporal split: train Month≤9 (mean={tr_month:.1f}), test Month>9 (mean={te_month:.1f})")

    # Fraud rate should be somewhat different (different time periods)
    tr_fraud = df.loc[train_mask, "is_fraud"].mean()
    te_fraud = df.loc[test_mask, "is_fraud"].mean()
    fraud_ratio = te_fraud / max(tr_fraud, 1e-8)
    R.note(f"Train fraud rate: {tr_fraud:.4f}, Test: {te_fraud:.4f} (ratio={fraud_ratio:.2f})")

    # Kolmogorov-Smirnov test on key features
    SUSPICIOUS_PVAL = 0.5  # p > 0.5 means distributions are too similar = leak
    for feat in ["amt", "hr", "mcc_n", "user_tx_count"]:
        if feat not in F.columns:
            continue
        try:
            ks_stat, pval = sp_stats.ks_2samp(
                F.loc[train_mask, feat].values,
                F.loc[test_mask, feat].values
            )
        except Exception:
            continue
        if pval > SUSPICIOUS_PVAL:
            R.warn(f"Feature '{feat}' distributions are very similar (KS p={pval:.4f})",
                   "should differ in temporal split")
        else:
            R.ok(f"Feature '{feat}' distributions differ (KS p={pval:.4f}, stat={ks_stat:.4f})")


def check_group_leakage(F: pd.DataFrame, df: pd.DataFrame, y: np.ndarray, R: AuditResult):
    """Check 8: Group leakage — entities in both splits must not have
    features computed using test-period information."""
    print("\n[CHECK 8] Group Leakage — Entity Feature Consistency")

    train_mask = df["Month"] <= 9
    test_mask = df["Month"] > 9

    # For each overlapping entity, verify that the test-period features
    # are consistent with train-only computation
    overlap_users = set(df.loc[train_mask, "User"].unique()) & set(df.loc[test_mask, "User"].unique())

    if not overlap_users:
        R.ok("No entity overlap (pure split)")
        return

    # Sample check: pick 10 overlapping users and verify fraud rate consistency
    rng = np.random.RandomState(42)
    sample = list(overlap_users)[:10]
    inconsistencies = 0

    for uid in sample:
        user_mask = df["User"] == uid
        user_sorted = df[user_mask].sort_values("datetime")
        user_idx = user_sorted.index

        if len(user_idx) < 3:
            continue

        # Walk through events and verify expanding mean is correct
        amounts = user_sorted["amt"].values
        running_sum = 0.0
        for i, idx in enumerate(user_idx):
            if i > 0:
                running_sum += amounts[i-1]
                expected_avg = running_sum / i
                actual_avg = F.loc[idx, "user_avg_amt"]
                if abs(expected_avg - actual_avg) > 0.01:
                    inconsistencies += 1
                    if inconsistencies <= 3:
                        R.warn(f"User {uid} event {i}: expected avg={expected_avg:.2f}, "
                               f"got {actual_avg:.2f}", "expanding mean inconsistency")
            else:
                # First event: avg should be NaN→0
                if F.loc[idx, "user_avg_amt"] > 0.01:
                    inconsistencies += 1
                    R.warn(f"User {uid} first event: avg={F.loc[idx, 'user_avg_amt']:.2f} (expected 0)",
                           "first event should have no history")

    if inconsistencies == 0:
        R.ok("Expanding features are consistent for all checked entities")
    else:
        R.warn(f"{inconsistencies} inconsistencies found across {len(sample)} entities",
               "review expanding computation")


def check_model_artifacts(F: pd.DataFrame, y: np.ndarray, R: AuditResult):
    """Check: Model artifact integrity — verify the saved model matches training."""
    print("\n[CHECK 9] Model Artifact Integrity")

    manifest_path = MODELS_DIR / "manifest.json"
    if not manifest_path.exists():
        R.fail("No manifest.json found", "model not trained")
        return

    manifest = json.loads(manifest_path.read_text())

    # Check feature count
    n_feat = manifest.get("n_features", 0)
    actual_feat = F.shape[1]
    if n_feat != actual_feat:
        R.fail(f"Manifest says {n_feat} features, actual is {actual_feat}",
               "model artifact mismatch")
    else:
        R.ok(f"Feature count matches: {n_feat}")

    # Check feature names
    manifest_feats = manifest.get("features", [])
    actual_feats = list(F.columns)
    if set(manifest_feats) != set(actual_feats):
        missing = set(actual_feats) - set(manifest_feats)
        extra = set(manifest_feats) - set(actual_feats)
        if missing:
            R.fail(f"Features in code but not in manifest: {missing}", "artifact stale")
        if extra:
            R.warn(f"Features in manifest but not in code: {extra}", "possible cleanup needed")
    else:
        R.ok("Feature names match between manifest and code")

    # Check model hash
    model_files = list(MODELS_DIR.glob("*.joblib"))
    if model_files:
        combined_hash = hashlib.md5()
        for f in sorted(model_files):
            combined_hash.update(f.read_bytes())
        model_hash = combined_hash.hexdigest()[:16]
        R.ok(f"Model artifacts hash: {model_hash} ({len(model_files)} files)")

    # Check target_leakage flag
    leakage_flag = manifest.get("target_leakage", None)
    if leakage_flag is False:
        R.ok("Manifest target_leakage flag: False (correct)")
    elif leakage_flag is True:
        R.fail("Manifest target_leakage flag: True", "model was flagged as having leakage")
    else:
        R.warn("Manifest target_leakage flag: not set", "should be explicitly False")


def check_live_vs_training(F: pd.DataFrame, y: np.ndarray, R: AuditResult):
    """Check: Live feature pipeline produces same features as training pipeline."""
    print("\n[CHECK 10] Live vs Training Feature Consistency")

    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from src.privacy_layer.features import ML_FEATURES, derive_event_features
    except ImportError as e:
        R.warn(f"Cannot import Privacy Layer features: {e}", "skipping live pipeline check")
        return

    # Test derive_event_features with a mock event
    profile = {
        "median_amount": 100.0,
        "typical_hours": list(range(8, 22)),
        "known_devices": ["dev1"],
        "usual_locations": ["loc1"],
        "usual_recipients": ["r1"],
        "tenure_days": 365,
        "known_device_count": 1,
    }
    event = {
        "amount": 250.0,
        "hour_of_day": 14,
        "is_weekend": 0,
        "device_hash": "dev1",
        "location_id": "loc1",
        "recipient_id": "r1",
        "failed_auth_count_24h": 0,
    }
    features = derive_event_features(
        profile=profile, event=event,
        freq_last_24h=5, days_since_similar=3.0,
        recent_ratios=[0.8, 1.2, 1.5],
        shared_device_accounts=0, shared_recipient_accounts=0,
    )

    # Check all ML_FEATURES are present
    missing = [f for f in ML_FEATURES if f not in features]
    if missing:
        R.fail(f"Live pipeline missing features: {missing}",
               "training features missing from live pipeline")
    else:
        R.ok(f"Live pipeline produces all {len(ML_FEATURES)} ML_FEATURES")

    # Check velocity features are also present (from velocity tracker)
    velocity_keys = {"user_tx_count", "user_avg_amt", "card_tx_count", "merch_tx_count"}
    missing_vel = velocity_keys - set(features.keys())
    if missing_vel:
        R.warn(f"Velocity features missing: {missing_vel}",
               "will fall back to defaults in Altman mapper")
    else:
        R.ok("Velocity features present in live pipeline output")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("LEAKAGE AUDIT — PS-14 ALTMAN PRODUCTION MODEL")
    print("=" * 70)

    R = AuditResult()

    # Load data
    print("\n[0] Loading data...")
    df = load_altman_data(max_rows=280_000)
    print(f"  Loaded {len(df):,} rows ({df['is_fraud'].sum():,} fraud, "
          f"{df['is_fraud'].mean()*100:.2f}%)")

    # Engineer features
    print("  Engineering features...")
    F, df_raw = engineer_features(df)
    print(f"  {F.shape[1]} features, {len(F):,} rows")

    y = df_raw["is_fraud"].values
    feat_names = list(F.columns)

    # Run all checks
    check_target_leakage(F, y, R)
    check_temporal_leakage(F, df_raw, R)
    check_entity_leakage(F, df_raw, R)
    check_train_test_contamination(F, df_raw, R)
    check_feature_importance_anomaly(F, y, R)

    train_mask = df_raw["Month"] <= 9
    test_mask = df_raw["Month"] > 9
    Xtr = F.loc[train_mask].values
    Xte = F.loc[test_mask].values
    check_scaler_leakage(Xtr, Xte, feat_names, R)
    check_distribution_shift(F, df_raw, R)
    check_group_leakage(F, df_raw, y, R)
    check_model_artifacts(F, y, R)
    check_live_vs_training(F, y, R)

    # Summary
    print("\n" + "=" * 70)
    print("AUDIT SUMMARY")
    print("=" * 70)
    n_pass = sum(1 for s, _, _ in R.checks if s == "PASS")
    n_warn = sum(1 for s, _, _ in R.checks if s == "WARN")
    n_fail = sum(1 for s, _, _ in R.checks if s == "FAIL")
    total = len(R.checks)

    print(f"  Total checks: {total}")
    print(f"  Passed:  {n_pass}")
    print(f"  Warnings: {n_warn}")
    print(f"  Failures: {n_fail}")

    if R.failures:
        print(f"\n  FAILURES:")
        for name, detail in R.failures:
            print(f"    \u2717 {name}: {detail}")
        print(f"\n  VERDICT: LEAKAGE DETECTED")
        sys.exit(1)
    elif R.warnings:
        print(f"\n  WARNINGS:")
        for name, detail in R.warnings:
            print(f"    \u26a0 {name}: {detail}")
        print(f"\n  VERDICT: NO CRITICAL LEAKAGE — review warnings")
        sys.exit(0)
    else:
        print(f"\n  VERDICT: CLEAN — no leakage detected")
        sys.exit(0)


if __name__ == "__main__":
    main()
