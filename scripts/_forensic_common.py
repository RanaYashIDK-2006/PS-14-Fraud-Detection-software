# -*- coding: utf-8 -*-
"""
_forensic_common.py -- shared constants + VERBATIM copy of the forensic audit
pipeline's expanding-window feature function, used by data_provenance.py (check
#23) and independent_validation.py (check #24).

PROVENANCE
----------
- Source: ``scripts/forensic_revalidate.py`` lines 205-345 (the
  ``expanding_features_fixed`` function as it ran to produce the recorded
  metrics in ``reports/forensic_revalidation.json``).
- The function body below is a byte-for-byte copy of that source region
  (only imports/constants added around it).
- ``data_provenance.py`` independently verifies the copy: it AST-extracts the
  original function from ``forensic_revalidate.py`` and compares its source
  text with this module's copy (see ``verify_verbatim_copy``).

WHY REUSE (not reimplement)?
- Check #23 reproducibility requires running the *same* experiment; check #24
  independence is delivered by the independently implemented metrics, leakage
  tests, dataset statistics and threshold provenance -- not by re-deriving the
  feature function (a from-scratch reimplementation would be a *different*
  experiment and could not validate the recorded claims).
"""

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants (verbatim from forensic_revalidate.py)
# ---------------------------------------------------------------------------
CSV_ALTMAN = "data/credit_card_transactions-ibm_v2.csv"
CHUNK = 2_000_000
SEED = 42
USECOLS = ["User", "Card", "Year", "Month", "Day", "Amount", "Use Chip",
           "Merchant Name", "Merchant City", "Merchant State", "Zip", "MCC",
           "Errors?", "Is Fraud?"]

FEATURE_NAMES = [
    "log_amt", "amt_sq", "year", "month", "day", "chip", "is_online", "mcc_n",
    "has_zip", "has_state", "user_tx_count", "merch_fraud_rate", "city_fraud_rate",
    "very_high_amt", "amt_x_mcc", "amt_x_online", "merch_popularity",
    "user_fraud_rate", "amt_ratio", "amt_zscore", "amt_acceleration",
    "mfr_x_ufr", "city_popularity", "user_merch_diversity", "amt_x_chip",
]

# Feature indices that are target-derived (fraud-rate) and therefore the focus
# of the chronological-contamination quantification in independent_validation.py
TARGET_DERIVED_IDX = {11, 12, 21}  # merch_fraud_rate, city_fraud_rate, mfr_x_ufr


def new_state():
    """Fresh expanding-window state dict (same shape forensic_revalidate uses)."""
    return {
        "user_tx_count": {}, "user_fraud_count": {}, "user_total_amt": {},
        "user_amt_sq": {}, "user_last_amt": {}, "merch_tx_count": {},
        "merch_fraud_count": {}, "city_tx_count": {}, "city_fraud_count": {},
    }


def new_pop_state():
    return {"total_tx": 0, "total_fraud": 0, "total_amt": 0.0,
            "total_amt_sq": 0.0, "n_users": 0}


def expanding_features_fixed(df_chunk, state, pop_state=None):
    """Compute features using ONLY data from 'state' (running stats up to current row).
    
    COLD-START FIX: For unseen users (first transaction), features fall back to
    population-level statistics from pop_state instead of zeros/degenerate values.
    This ensures the model gets meaningful features for new users.
    
    No chunk-level statistics are used. Every feature depends only on
    strictly-past rows.
    """
    n = len(df_chunk)
    
    amt = pd.to_numeric(df_chunk["Amount"].str.replace("$", "", regex=False), errors="coerce").fillna(0).values.astype(np.float32)
    users = df_chunk["User"].astype(str).values
    merchs = df_chunk["Merchant Name"].astype(str).values
    cities = df_chunk["Merchant City"].astype(str).values
    mccs = df_chunk["MCC"].fillna(0).astype(int).values
    use_chip = df_chunk["Use Chip"].fillna("Online Transaction").values
    year = df_chunk["Year"].fillna(2019).values.astype(float)
    month = df_chunk["Month"].fillna(1).values.astype(float)
    day = df_chunk["Day"].fillna(1).values.astype(float)
    zip_val = df_chunk["Zip"].fillna(0).values
    has_zip = (np.array(zip_val, dtype=float) > 0).astype(float)
    merchant_state = df_chunk["Merchant State"].fillna("").values
    has_state = (np.array(merchant_state) != "").astype(float)
    is_fraud = df_chunk["Is Fraud?"].map({"Yes": 1, "No": 0}).fillna(0).astype(int).values
    
    log_amt = np.log1p(amt)
    amt_sq = amt ** 2
    chip = np.where(np.array(use_chip) == "Chip Transaction", 1.0,
            np.where(np.array(use_chip) == "Swipe Transaction", 0.5, 0.0))
    is_online = np.where(np.array(use_chip) == "Online Transaction", 1.0, 0.0)
    mcc_n = mccs.astype(np.float32) / 6000.0
    
    # Expanding window features - row by row
    utc = np.zeros(n, dtype=np.float32)
    ufr = np.zeros(n, dtype=np.float32)
    uavg = np.zeros(n, dtype=np.float32)
    ustd = np.zeros(n, dtype=np.float32)
    mfr = np.zeros(n, dtype=np.float32)
    cfr = np.zeros(n, dtype=np.float32)
    mtc = np.zeros(n, dtype=np.float32)
    accel = np.zeros(n, dtype=np.float32)
    ctc = np.zeros(n, dtype=np.float32)
    
    # Local state dicts for speed
    utc_s = state["user_tx_count"]
    ufc_s = state["user_fraud_count"]
    uta_s = state["user_total_amt"]
    uas_s = state["user_amt_sq"]
    ula_s = state["user_last_amt"]
    mtc_s = state["merch_tx_count"]
    mfc_s = state["merch_fraud_count"]
    ctc_s = state["city_tx_count"]
    cfc_s = state["city_fraud_count"]
    
    # Population-level fallback state for cold-start users
    if pop_state is None:
        pop_state = {"total_tx": 0, "total_fraud": 0, "total_amt": 0.0,
                      "total_amt_sq": 0.0, "n_users": 0}
    _ptx = pop_state["total_tx"]
    _pfraud = pop_state["total_fraud"]
    _pamt = pop_state["total_amt"]
    _pamtsq = pop_state["total_amt_sq"]
    _pn = pop_state["n_users"]
    # Population fallback values (safe even if _ptx == 0)
    pop_fraud_rate = _pfraud / max(_ptx, 1)
    pop_avg_amt = _pamt / max(_ptx, 1)
    pop_std_amt = max((_pamtsq / max(_ptx, 1)) - pop_avg_amt ** 2, 1e-10) ** 0.5
    pop_avg_tx_per_user = _ptx / max(_pn, 1)
    
    for i in range(n):
        u = users[i]
        m = merchs[i]
        c = cities[i]
        a = float(amt[i])
        is_new_user = u not in utc_s
        
        # Read state BEFORE this row
        utx = utc_s.get(u, 0)
        utc[i] = utx
        uf = ufc_s.get(u, 0)
        ut = uta_s.get(u, 0.0)
        us = uas_s.get(u, 0.0)
        
        if is_new_user and _ptx > 0:
            # COLD-START: unseen user, use population fallbacks
            ufr[i] = pop_fraud_rate
            ua = pop_avg_amt
            uavg[i] = ua
            ustd[i] = pop_std_amt
            accel[i] = 0.0  # no previous transaction
        else:
            # Normal: user has history
            ufr[i] = uf / max(utx, 1)
            ua = ut / max(utx, 1)
            uavg[i] = ua
            uv = max(us / max(utx, 1) - ua * ua, 1e-10)
            ustd[i] = uv ** 0.5
            prev = ula_s.get(u, a)
            accel[i] = abs(a - prev) / max(prev, 0.01) if prev > 0 else 0.0
        
        mt = mtc_s.get(m, 0)
        mtc[i] = mt
        mf = mfc_s.get(m, 0)
        mfr[i] = mf / max(mt, 1)
        
        ct = ctc_s.get(c, 0)
        ctc[i] = ct
        cf = cfc_s.get(c, 0)
        cfr[i] = cf / max(ct, 1)
        
        # UPDATE state AFTER computing features for this row
        utc_s[u] = utx + 1
        ufc_s[u] = uf + int(is_fraud[i])
        uta_s[u] = ut + a
        uas_s[u] = us + a * a
        ula_s[u] = a
        mtc_s[m] = mt + 1
        mfc_s[m] = mf + int(is_fraud[i])
        ctc_s[c] = ct + 1
        cfc_s[c] = cf + int(is_fraud[i])
        # Update population stats
        _ptx += 1
        _pfraud += int(is_fraud[i])
        _pamt += a
        _pamtsq += a * a
        if is_new_user:
            _pn += 1
    
    # Write back pop_state
    pop_state["total_tx"] = _ptx
    pop_state["total_fraud"] = _pfraud
    pop_state["total_amt"] = _pamt
    pop_state["total_amt_sq"] = _pamtsq
    pop_state["n_users"] = _pn
    
    # Derived features
    very_high_amt = (amt > 5000).astype(float)
    amt_x_mcc = amt * mcc_n
    amt_x_online = amt * is_online
    amt_x_chip = amt * chip
    mfr_x_ufr = mfr * ufr
    
    # FIX: Use expanding count directly as popularity metric
    # (no chunk-level median normalization)
    # merch_popularity = log(1 + merch_tx_count) - capped at reasonable range
    # This is strictly causal: count only increases, never depends on future
    merch_pop = np.minimum(np.log1p(mtc), 5.0)
    city_pop = np.minimum(np.log1p(ctc), 5.0)
    
    # For unseen users: amt_ratio = amt/pop_avg_amt (meaningful ratio)
    # For seen users: amt_ratio = amt/user_avg_amt (personal deviation)
    amt_ratio = amt / np.maximum(uavg, 0.01)
    amt_zscore = (amt - uavg) / np.maximum(ustd, 0.01)
    # For unseen users: diversity = 0 (first transaction, no history)
    umdiv = np.minimum(utc / np.maximum(mtc, 1), 10.0)
    
    F = np.column_stack([
        log_amt, amt_sq,
        year, month, day,
        chip, is_online, mcc_n,
        has_zip, has_state,
        utc,
        mfr, cfr,
        very_high_amt,
        amt_x_mcc, amt_x_online,
        merch_pop,
        ufr,
        amt_ratio, amt_zscore, accel,
        mfr_x_ufr,
        city_pop, umdiv, amt_x_chip,
    ]).astype(np.float32)
    
    F = np.nan_to_num(F, nan=0.0, posinf=10.0, neginf=-10.0)
    return F


def verify_verbatim_copy():
    """AST-extract `expanding_features_fixed` from forensic_revalidate.py and
    compare source text with this module's copy. Returns (ok, detail)."""
    import ast
    import inspect
    import hashlib

    src = open("scripts/forensic_revalidate.py", encoding="utf-8").read()
    tree = ast.parse(src)
    orig = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "expanding_features_fixed":
            orig = ast.get_source_segment(src, node)
            break
    if orig is None:
        return False, "function not found in forensic_revalidate.py"
    mine = inspect.getsource(expanding_features_fixed)
    # Normalize CRLF for a fair byte comparison
    if orig.replace("\r\n", "\n").strip() == mine.replace("\r\n", "\n").strip():
        h = hashlib.sha256(orig.encode("utf-8")).hexdigest()
        return True, f"byte-identical after CRLF normalization; source sha256={h[:16]}"
    return False, "TEXT MISMATCH: verbatim copy diverged from forensic_revalidate.py"