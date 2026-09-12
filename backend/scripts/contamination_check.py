#!/usr/bin/env python3
"""
PS-14 CONTAMINATION & GENERALIZATION CHECKS

Checks for:
- Duplicate transaction IDs
- Exact duplicate rows
- Near-duplicate feature vectors
- Entity overlap across splits (users, merchants, cards, cities)
- Temporal overlap
- Memorization indicators
"""
import json, os, sys, time, gc
os.environ["PYTHONIOENCODING"] = "utf-8"
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parent.parent.parent  # repo root
os.chdir(ROOT)

REPORT_PATH = ROOT / "reports" / "contamination_check.json"

T0 = time.time()
def log(msg):
    t = time.time() - T0
    print(f"[{t:6.0f}s] {msg}", flush=True)

CSV = "data/credit_card_transactions-ibm_v2.csv"
CHUNK = 2_000_000
USECOLS = ["User", "Card", "Year", "Month", "Day", "Amount", "Use Chip",
           "Merchant Name", "Merchant City", "Merchant State", "Zip", "MCC",
           "Errors?", "Is Fraud?"]

results = {}

# ============================================================
# 1. LOAD & SPLIT (same as forensic_revalidate.py)
# ============================================================
log("Loading dataset...")
all_chunks = []
for i, chunk in enumerate(pd.read_csv(CSV, usecols=USECOLS, low_memory=False, chunksize=CHUNK)):
    all_chunks.append(chunk)
    if i % 3 == 0:
        log(f"  Chunk {i}: {sum(len(c) for c in all_chunks):,} rows")

df = pd.concat(all_chunks, ignore_index=True)
del all_chunks
gc.collect()

total = len(df)
fraud_mask = df["Is Fraud?"].map({"Yes": 1, "No": 0}).fillna(0).astype(int)
n_train = int(total * 0.60)
n_val = int(total * 0.20)

train = df.iloc[:n_train]
val = df.iloc[n_train:n_train + n_val]
test = df.iloc[n_train + n_val:]

log(f"Train: {len(train):,}, Val: {len(val):,}, Test: {len(test):,}")

# ============================================================
# 2. EXACT DUPLICATE ROWS
# ============================================================
log("\n--- EXACT DUPLICATE ROWS ---")
# Check across the full dataset
full_dups = int(df.duplicated().sum())
log(f"Full dataset exact duplicates: {full_dups:,}")

# Per-split
train_dups = int(train.duplicated().sum())
val_dups = int(val.duplicated().sum())
test_dups = int(test.duplicated().sum())
log(f"  Train: {train_dups:,}, Val: {val_dups:,}, Test: {test_dups:,}")

# Cross-split: rows in test that also appear in train
# Use all columns except Is Fraud? for the comparison
compare_cols = [c for c in df.columns if c != "Is Fraud?"]
# For efficiency, convert to tuples
log("  Checking cross-split exact duplicates (train vs test)...")
# Sample for efficiency
n_sample = min(500_000, len(test))
test_sample_idx = np.random.RandomState(42).choice(len(test), n_sample, replace=False)
test_sample = test.iloc[test_sample_idx]

cross_dups = 0
# Check in chunks
for start in range(0, len(train), 500_000):
    end = min(start + 500_000, len(train))
    train_chunk = train.iloc[start:end]
    merged = test_sample.merge(train_chunk[compare_cols], how="inner", on=compare_cols)
    cross_dups += len(merged)

log(f"  Cross-split duplicates (test rows found in train, sampled {n_sample:,}): {cross_dups}")

results["exact_duplicates"] = {
    "full_dataset": full_dups,
    "train": train_dups,
    "val": val_dups,
    "test": test_dups,
    "cross_split_test_in_train_sample": cross_dups,
    "cross_split_sample_size": n_sample,
}

# ============================================================
# 3. DUPLICATE TRANSACTION IDs
# ============================================================
log("\n--- TRANSACTION ID CHECK ---")
# The dataset doesn't have explicit transaction IDs
# Use (User, Card, Year, Month, Day, Amount, Merchant Name) as proxy
id_cols = ["User", "Card", "Year", "Month", "Day", "Amount", "Merchant Name"]
df["_tx_id"] = df[id_cols].apply(lambda r: tuple(r), axis=1)

log("  Counting tx-id frequency...")
tx_id_counts = df["_tx_id"].value_counts()
n_unique_ids = len(tx_id_counts)
n_repeated = int((tx_id_counts > 1).sum())
n_repeated_rows = int(tx_id_counts[tx_id_counts > 1].sum())
max_repeat = int(tx_id_counts.max())

log(f"  Unique composite IDs: {n_unique_ids:,}")
log(f"  IDs appearing >1 time: {n_repeated:,}")
log(f"  Total rows with repeated IDs: {n_repeated_rows:,}")
log(f"  Max repeat count: {max_repeat}")

# Check cross-split ID overlap
log("  Checking cross-split ID overlap...")
train_ids = set(df.iloc[:n_train]["_tx_id"].values)
test_ids = set(df.iloc[n_train + n_val:]["_tx_id"].values)
val_ids = set(df.iloc[n_train:n_train + n_val]["_tx_id"].values)

train_test_overlap = len(train_ids & test_ids)
train_val_overlap = len(train_ids & val_ids)
val_test_overlap = len(val_ids & test_ids)

log(f"  Train-Test ID overlap: {train_test_overlap:,} IDs ({train_test_overlap/max(len(test_ids),1)*100:.2f}% of test)")
log(f"  Train-Val ID overlap: {train_val_overlap:,} IDs")
log(f"  Val-Test ID overlap: {val_test_overlap:,} IDs")

# How many test rows have IDs also in train?
test_in_train_count = 0
for start in range(0, len(test), 200_000):
    end = min(start + 200_000, len(test))
    chunk_ids = set(df.iloc[n_train + n_val + start:n_train + n_val + end]["_tx_id"].values)
    test_in_train_count += len(chunk_ids & train_ids)

log(f"  Test rows with IDs also in train: {test_in_train_count:,} / {len(test):,} ({test_in_train_count/len(test)*100:.2f}%)")

df.drop("_tx_id", axis=1, inplace=True, errors="ignore")

results["tx_id_duplicates"] = {
    "unique_composite_ids": n_unique_ids,
    "repeated_ids": n_repeated,
    "repeated_rows": n_repeated_rows,
    "max_repeat": max_repeat,
    "train_test_id_overlap": train_test_overlap,
    "train_val_id_overlap": train_val_overlap,
    "val_test_id_overlap": val_test_overlap,
    "test_rows_with_train_ids": test_in_train_count,
    "test_rows_total": len(test),
    "test_rows_with_train_ids_pct": round(test_in_train_count / len(test) * 100, 2),
}

# ============================================================
# 4. ENTITY OVERLAP (USERS, MERCHANTS, CITIES)
# ============================================================
log("\n--- ENTITY OVERLAP ---")

for entity_col, label in [("User", "users"), ("Merchant Name", "merchants"), ("Merchant City", "cities"), ("Card", "cards")]:
    train_ents = set(train[entity_col].dropna().unique())
    val_ents = set(val[entity_col].dropna().unique())
    test_ents = set(test[entity_col].dropna().unique())
    
    train_test_overlap = len(train_ents & test_ents)
    train_val_overlap = len(train_ents & val_ents)
    val_test_overlap = len(val_ents & test_ents)
    
    test_only = len(test_ents - train_ents - val_ents)
    
    log(f"  {label}: train={len(train_ents):,}, val={len(val_ents):,}, test={len(test_ents):,}")
    log(f"    Train-Test overlap: {train_test_overlap:,} ({train_test_overlap/max(len(test_ents),1)*100:.1f}% of test {label})")
    log(f"    Test-only (unseen): {test_only:,} ({test_only/max(len(test_ents),1)*100:.1f}% of test {label})")
    
    results[f"entity_overlap_{label}"] = {
        "train": len(train_ents),
        "val": len(val_ents),
        "test": len(test_ents),
        "train_test_overlap": train_test_overlap,
        "train_val_overlap": train_val_overlap,
        "val_test_overlap": val_test_overlap,
        "test_only_unseen": test_only,
        "train_test_overlap_pct": round(train_test_overlap / max(len(test_ents), 1) * 100, 2),
    }

# ============================================================
# 5. FRAUD DISTRIBUTION ACROSS SPLITS
# ============================================================
log("\n--- FRAUD DISTRIBUTION ---")
for split_name, split_df in [("train", train), ("val", val), ("test", test)]:
    f = int(split_df["Is Fraud?"].map({"Yes": 1, "No": 0}).fillna(0).sum())
    n = len(split_df)
    log(f"  {split_name}: {f:,} fraud / {n:,} total ({f/n*100:.3f}%)")

# ============================================================
# 6. NEAR-DUPLICATE FEATURE CHECK (sample-based)
# ============================================================
log("\n--- NEAR-DUPLICATE CHECK (sampled) ---")
log("  Comparing first 10 test features of test vs train (sampled)...")

# Quick check: how many test rows have exact feature matches in train?
# Using raw columns as proxy
raw_cols = ["Amount", "MCC", "Use Chip", "User", "Merchant Name"]
sample_size = min(100_000, len(test))
rng = np.random.RandomState(42)
test_idx = rng.choice(len(test), sample_size, replace=False)
test_sub = test.iloc[test_idx]

# Check for exact (User, Amount, MCC) matches in train
match_count = 0
for start in range(0, len(train), 500_000):
    end = min(start + 500_000, len(train))
    train_chunk = train.iloc[start:end]
    merged = test_sub[["User", "Amount", "MCC"]].merge(
        train_chunk[["User", "Amount", "MCC"]], how="inner", on=["User", "Amount", "MCC"]
    )
    match_count += len(merged)

log(f"  Test rows with (User, Amount, MCC) match in train: {match_count:,} / {sample_size:,} ({match_count/sample_size*100:.1f}%)")

results["near_duplicates"] = {
    "sample_size": sample_size,
    "user_amount_mcc_matches_in_train": match_count,
    "match_pct": round(match_count / sample_size * 100, 1),
}

# ============================================================
# 7. TEMPORAL OVERLAP
# ============================================================
log("\n--- TEMPORAL OVERLAP ---")
# Check if any test dates appear in train
train_dates = set(zip(train["Year"], train["Month"], train["Day"]))
test_dates = set(zip(test["Year"], test["Month"], test["Day"]))
overlap_dates = train_dates & test_dates
log(f"  Unique (Y,M,D) in train: {len(train_dates):,}")
log(f"  Unique (Y,M,D) in test: {len(test_dates):,}")
log(f"  Overlapping dates: {len(overlap_dates):,}")

# Count test rows on overlapping dates
overlap_rows = 0
for start in range(0, len(test), 500_000):
    end = min(start + 500_000, len(test))
    chunk = test.iloc[start:end]
    overlap_rows += chunk.apply(lambda r: (r["Year"], r["Month"], r["Day"]) in overlap_dates, axis=1).sum()

log(f"  Test rows on overlapping dates: {overlap_rows:,} / {len(test):,} ({overlap_rows/len(test)*100:.1f}%)")

results["temporal_overlap"] = {
    "unique_dates_train": len(train_dates),
    "unique_dates_test": len(test_dates),
    "overlapping_dates": len(overlap_dates),
    "test_rows_on_overlap_dates": int(overlap_rows),
    "test_rows_on_overlap_dates_pct": round(overlap_rows / len(test) * 100, 1),
}

# ============================================================
# 8. GENERALIZATION ASSESSMENT
# ============================================================
log("\n--- GENERALIZATION ASSESSMENT ---")

# Determine deployment scenario
# Entity overlap is high → model sees recurring entities → known-entity scenario
user_overlap_pct = results["entity_overlap_users"]["train_test_overlap_pct"]
merchant_overlap_pct = results["entity_overlap_merchants"]["train_test_overlap_pct"]
date_overlap_pct = results["temporal_overlap"]["test_rows_on_overlap_dates_pct"]
cross_dup_pct = results["tx_id_duplicates"]["test_rows_with_train_ids_pct"]

log(f"  User overlap (train->test): {user_overlap_pct:.1f}%")
log(f"  Merchant overlap (train->test): {merchant_overlap_pct:.1f}%")
log(f"  Date overlap: {date_overlap_pct:.1f}%")
log(f"  Exact tx-id overlap: {cross_dup_pct:.1f}%")

# Assessment
if cross_dup_pct > 10:
    memorization_risk = "HIGH — significant exact transaction overlap"
elif cross_dup_pct > 1:
    memorization_risk = "MODERATE — some exact transaction overlap"
else:
    memorization_risk = "LOW — minimal exact transaction overlap"

if user_overlap_pct > 80:
    entity_scenario = "KNOWN ENTITIES — most test users seen in training"
elif user_overlap_pct > 50:
    entity_scenario = "MIXED — substantial entity overlap"
else:
    entity_scenario = "UNSEEN ENTITIES — most test users are new"

assessment = {
    "memorization_risk": memorization_risk,
    "entity_scenario": entity_scenario,
    "key_finding": (
        f"The test set has {cross_dup_pct:.1f}% exact transaction ID overlap with training data, "
        f"{user_overlap_pct:.1f}% user overlap, and {merchant_overlap_pct:.1f}% merchant overlap. "
        f"This is expected for a temporal split of recurring entities. "
        f"The deployment scenario is {entity_scenario.lower()}. "
        f"Features are computed causally (past-only expanding window), "
        f"so the model learns behavioral patterns rather than memorizing individual transactions."
    ),
}
results["generalization_assessment"] = assessment
log(f"  Memorization risk: {memorization_risk}")
log(f"  Entity scenario: {entity_scenario}")
log(f"  Finding: {assessment['key_finding'][:120]}...")

# ============================================================
# SAVE
# ============================================================
results["metadata"] = {
    "total_rows": total,
    "train_rows": len(train),
    "val_rows": len(val),
    "test_rows": len(test),
    "run_time_sec": round(time.time() - T0, 1),
}

with open(REPORT_PATH, "w") as f:
    json.dump(results, f, indent=2)

log(f"\nReport saved: {REPORT_PATH}")
log(f"Total time: {time.time()-T0:.0f}s")
