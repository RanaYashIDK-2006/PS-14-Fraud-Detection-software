#!/usr/bin/env python3
"""Phase 2: Fast feature engineering on IBM Altman 24M (vectorized, no slow groupby.apply)."""
import numpy as np, pandas as pd, time, hashlib, os
from pathlib import Path

DATA_DIR = Path("data")

def main():
    print("=" * 60)
    print("IBM ALTMAN 24M FEATURE ENGINEERING (FAST)")
    print("=" * 60)

    t0 = time.time()
    print("Loading 24M rows...")
    df = pd.read_csv(
        DATA_DIR / "credit_card_transactions-ibm_v2.csv",
        usecols=["User","Card","Year","Month","Day","Time","Amount","Use Chip",
                  "Merchant Name","Merchant City","Merchant State","Zip","MCC","Errors?","Is Fraud?"],
        low_memory=False,
    )
    print(f"  Loaded {len(df):,} rows in {time.time()-t0:.0f}s")

    # Parse
    df["amt"] = df["Amount"].str.replace("$","",regex=False).str.replace(",","",regex=False).astype(float)
    df["is_fraud"] = (df["Is Fraud?"] == "Yes").astype(int)
    df["datetime"] = pd.to_datetime(df[["Year","Month","Day"]].assign(
        hour=df["Time"].str.split(":").str[0].astype(int),
        minute=df["Time"].str.split(":").str[1].astype(int),
    ))
    df = df.sort_values(["User","datetime"]).reset_index(drop=True)

    fraud_count = df["is_fraud"].sum()
    print(f"  Fraud: {fraud_count:,} ({fraud_count/len(df)*100:.3f}%)")
    print(f"  Users: {df['User'].nunique()}, MCCs: {df['MCC'].nunique()}")

    F = pd.DataFrame(index=df.index)
    print("  Engineering features...")

    # === AMOUNT FEATURES ===
    F["amount"] = df["amt"].values
    F["amount_log"] = np.log1p(np.abs(df["amt"].values))
    F["is_negative"] = (df["amt"].values < 0).astype(np.float32)

    # Per-user amount stats (using cumulative transforms - fast)
    g = df.groupby("User")
    F["user_amt_mean"] = g["amt"].cumsum() / (g.cumcount() + 1)  # expanding mean, fast
    F["user_amt_sq_sum"] = g["amt"].transform(lambda x: x.cumsum() ** 2)  # for variance
    user_count = g.cumcount() + 1
    F["user_amt_var"] = (F["user_amt_sq_sum"] / user_count) - F["user_amt_mean"] ** 2
    F["user_amt_std"] = np.sqrt(np.clip(F["user_amt_var"], 0, None))
    F["user_amt_max"] = g["amt"].cummax()
    F["user_amt_zscore"] = (F["amount"] - F["user_amt_mean"]) / (F["user_amt_std"] + 1e-8)
    F["user_amt_ratio"] = F["amount"] / (F["user_amt_mean"].clip(lower=1))
    F.drop(columns=["user_amt_sq_sum", "user_amt_var"], inplace=True)

    # === TEMPORAL FEATURES ===
    F["hour"] = df["datetime"].dt.hour.values.astype(np.float32)
    F["dow"] = df["datetime"].dt.dayofweek.values.astype(np.float32)
    F["is_weekend"] = (F["dow"] >= 5).astype(np.float32)
    F["is_night"] = ((F["hour"] >= 22) | (F["hour"] <= 5)).astype(np.float32)
    F["is_business"] = ((F["hour"] >= 9) & (F["hour"] <= 17)).astype(np.float32)

    # Time since last txn (fast diff)
    F["hours_since_last"] = df.groupby("User")["datetime"].diff().dt.total_seconds().fillna(0).values / 3600.0
    F["hours_since_last"] = np.clip(F["hours_since_last"], 0, 720).astype(np.float32)

    # === VELOCITY ===
    F["txn_count"] = (g.cumcount() + 1).values.astype(np.float32)

    # Approximate 24h frequency using time-based rolling
    # For speed, bin time into 24h windows and count
    F["day_bin"] = (df["datetime"].values.astype(np.int64) // (86400 * 10**9)).astype(np.float32)
    # Count txns per user per day
    F["user_day_count"] = df.groupby(["User", "day_bin"]).cumcount().values + 1
    F["user_day_count"] = F["user_day_count"].astype(np.float32)

    # === MCC FEATURES ===
    F["mcc"] = df["MCC"].values.astype(np.float32)

    # MCC fraud rate (using expanding fraud count / count)
    mcc_fraud_cumsum = g["is_fraud"].cumsum()
    F["mcc_fraud_rate"] = df.groupby("MCC")["is_fraud"].transform(
        lambda x: x.expanding().mean().shift(1).fillna(0)
    ).values.astype(np.float32)

    # === MERCHANT FEATURES ===
    # Encode merchant as category code (fast)
    F["merchant_code"] = df["Merchant Name"].astype("category").cat.codes.values.astype(np.float32)

    # Merchant fraud rate (point-in-time)
    F["merchant_fraud_rate"] = df.groupby("Merchant Name")["is_fraud"].transform(
        lambda x: x.expanding().mean().shift(1).fillna(0)
    ).values.astype(np.float32)

    # === LOCATION ===
    F["city_code"] = df["Merchant City"].astype("category").cat.codes.values.astype(np.float32)
    F["state_code"] = df["Merchant State"].fillna("UNK").astype("category").cat.codes.values.astype(np.float32)
    F["has_zip"] = df["Zip"].notna().values.astype(np.float32)

    # === CARD/CHIP ===
    F["card_num"] = df["Card"].values.astype(np.float32)
    chip_map = {"Swipe Transaction": 0, "Chip Transaction": 1, "Online Transaction": 2}
    F["chip_type"] = df["Use Chip"].map(chip_map).fillna(3).values.astype(np.float32)
    F["has_error"] = df["Errors?"].notna().values.astype(np.float32)

    # === USER BEHAVIORAL (fast: use cumulative sum for unique proxy) ===
    # Merchant novelty: ratio of unique merchants to total txns (proxy)
    F["user_merchant_diversity"] = F["user_n_unique_merchants"] = 0.0  # placeholder
    # Use merchant code frequency as novelty proxy
    F["merchant_seen_before"] = df.groupby(["User","Merchant Name"]).cumcount().values.astype(np.float32)
    F["city_seen_before"] = df.groupby(["User","Merchant City"]).cumcount().values.astype(np.float32)
    F["mcc_seen_before"] = df.groupby(["User","MCC"]).cumcount().values.astype(np.float32)

    # === ESCALATION (fast: last 5 vs prior 5 using rolling) ===
    F["escalation"] = g["amt"].transform(
        lambda x: x.rolling(10, min_periods=2).apply(
            lambda v: (np.median(v[len(v)//2:]) - np.median(v[:len(v)//2])) / (np.median(v[:len(v)//2]) + 1e-8) if len(v) >= 2 else 0,
            raw=True
        )
    ).fillna(0).clip(-5, 5).values.astype(np.float32)

    # === INTERACTION ===
    F["amt_x_freq"] = F["amount_log"] * np.log1p(F["user_day_count"])
    F["night_x_amt"] = F["is_night"] * F["amount_log"]

    # === REGULARITY ===
    F["txn_regularity"] = df.groupby("User")["datetime"].transform(
        lambda x: x.diff().dt.total_seconds().expanding().std().fillna(0)
    ).values.astype(np.float32)

    # Clean
    F = F.replace([np.inf, -np.inf], np.nan).fillna(0)
    F["label"] = df["is_fraud"].values

    # Drop day_bin helper
    F.drop(columns=["day_bin"], inplace=True)

    elapsed = time.time() - t0
    print(f"\n  Features: {F.shape[0]:,} rows x {F.shape[1]} cols")
    print(f"  Time: {elapsed:.0f}s")

    # Feature stats
    print("\n  Feature separation (fraud vs legit):")
    fraud_mask = F["label"] == 1
    for col in F.columns:
        if col in ("label",):
            continue
        fm = F.loc[fraud_mask, col].mean()
        lm = F.loc[~fraud_mask, col].mean()
        fs = F.loc[fraud_mask, col].std() + 1e-8
        sep = abs(fm - lm) / fs
        marker = " ***" if sep > 0.3 else ""
        print(f"    {col:30s}: fraud={fm:10.3f}  legit={lm:10.3f}  sep={sep:.3f}{marker}")

    # Save
    out = DATA_DIR / "altman_features_v2.csv"
    F.to_csv(out, index=False)
    print(f"\n  Saved to {out} ({os.path.getsize(out)/1e6:.0f}MB)")

if __name__ == "__main__":
    main()
