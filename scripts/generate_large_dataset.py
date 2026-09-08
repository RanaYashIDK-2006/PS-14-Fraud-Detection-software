#!/usr/bin/env python3
"""Scalable fraud dataset generator for PS-14.

Generates 1-2M synthetic transactions with 8+ distinct fraud archetypes,
realistic temporal patterns, and strict label-integrity guarantees.

Target: ~1.5M transactions, ~25k fraud events (~1.67% fraud rate)
Archetypes: ATO, CNP, Mule, Velocity, Device Compromise, Credential Abuse,
            Synthetic Identity, Gradual Escalation, Merchant Fraud

Each archetype has materially different behavioral distributions —
not just renamed versions of the same pattern.

Usage:
    python scripts/generate_large_dataset.py [--n-transactions 1500000]
                                             [--seed 42] [--output data/transactions_large.csv]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from privacy_layer.features import (
    ML_FEATURES,
    amount_bucket,
    escalation_score,
    mule_ring_score,
    hour_deviation_score,
    amount_zscore,
    velocity_deviation_score,
    recipient_novelty_score,
    txn_regularity_score,
    typical_hours_default,
)

# ── Constants ────────────────────────────────────────────────────────────

FRAUD_ID_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
FRAUD_ID_LENGTH = 16

# 24-month timeline: 2024-01-01 to 2025-12-31
GLOBAL_START = pd.Timestamp("2024-01-01 00:00:00")
GLOBAL_END = pd.Timestamp("2025-12-31 23:59:59")
TIMELINE_DAYS = (GLOBAL_END - GLOBAL_START).days  # 730

# Hour-of-day weight profiles (24 bins)
LEGIT_HOUR_WEIGHTS = np.array(
    [0, 0, 0, 0, 0, 0, 1, 3, 8, 12, 14, 12, 10, 10, 11, 12, 14, 16, 18, 15, 9, 4, 1, 0],
    dtype=float,
)
FRAUD_HOUR_WEIGHTS = np.array(
    [4, 4, 4, 4, 3, 2, 1, 1, 2, 3, 4, 5, 6, 6, 6, 6, 7, 7, 7, 7, 6, 6, 5, 5],
    dtype=float,
)
ATO_HOUR_WEIGHTS = np.array(
    [8, 8, 9, 9, 8, 7, 5, 3, 2, 2, 3, 4, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 7, 8],
    dtype=float,
)
VELOCITY_HOUR_WEIGHTS = np.array(
    [2, 2, 2, 2, 2, 2, 2, 2, 8, 12, 14, 12, 10, 10, 11, 12, 14, 16, 18, 15, 9, 4, 2, 2],
    dtype=float,
)

# Fraud archetype configuration
ARCHETYPE_CONFIG = {
    "ato": {
        "weight": 0.18,  # 18% of fraud events
        "hour_weights": ATO_HOUR_WEIGHTS,
        "amount_log_mean": 4.0,
        "amount_log_std": 0.6,
        "amount_multiplier_range": (1.5, 3.5),
        "n_txns_range": (2, 6),
        "new_device_prob": 0.85,
        "unusual_location_prob": 0.70,
        "unusual_recipient_prob": 0.75,
        "failed_auth_range": (2, 9),
        "inter_arrival_hours": (0.2, 3.0),
        "description": "Account Takeover - credential compromise, rapid burst",
    },
    "cnp": {
        "weight": 0.15,
        "hour_weights": ATO_HOUR_WEIGHTS,
        "amount_log_mean": 2.8,
        "amount_log_std": 0.4,
        "amount_multiplier_range": (0.8, 1.2),
        "n_txns_range": (6, 14),
        "new_device_prob": 0.50,
        "unusual_location_prob": 0.50,
        "unusual_recipient_prob": 0.85,
        "failed_auth_range": (0, 2),
        "inter_arrival_hours": (0.5, 6.0),
        "description": "Card-Not-Present - small rapid transactions to new merchants",
    },
    "mule": {
        "weight": 0.12,
        "hour_weights": FRAUD_HOUR_WEIGHTS,
        "amount_log_mean": 3.2,
        "amount_log_std": 0.5,
        "amount_multiplier_range": (0.8, 1.5),
        "n_txns_range": (2, 6),
        "new_device_prob": 1.0,  # always new device (shared ring device)
        "unusual_location_prob": 0.40,
        "unusual_recipient_prob": 1.0,  # always unusual (sink recipient)
        "failed_auth_range": (0, 2),
        "inter_arrival_hours": (6, 48),
        "description": "Mule Activity - shared devices/recipients, money convergence",
    },
    "velocity": {
        "weight": 0.12,
        "hour_weights": VELOCITY_HOUR_WEIGHTS,
        "amount_log_mean": 3.5,
        "amount_log_std": 0.3,
        "amount_multiplier_range": (0.5, 2.0),
        "n_txns_range": (15, 40),
        "new_device_prob": 0.10,
        "unusual_location_prob": 0.10,
        "unusual_recipient_prob": 0.30,
        "failed_auth_range": (0, 1),
        "inter_arrival_hours": (0.1, 2.0),
        "description": "Velocity Fraud - rapid-fire transactions exceeding limits",
    },
    "device_compromise": {
        "weight": 0.10,
        "hour_weights": FRAUD_HOUR_WEIGHTS,
        "amount_log_mean": 3.8,
        "amount_log_std": 0.5,
        "amount_multiplier_range": (0.8, 2.5),
        "n_txns_range": (3, 8),
        "new_device_prob": 1.0,
        "unusual_location_prob": 0.20,
        "unusual_recipient_prob": 0.30,
        "failed_auth_range": (0, 1),
        "inter_arrival_hours": (2, 24),
        "description": "Device Compromise - malware, session hijacking",
    },
    "credential_abuse": {
        "weight": 0.10,
        "hour_weights": ATO_HOUR_WEIGHTS,
        "amount_log_mean": 3.0,
        "amount_log_std": 0.6,
        "amount_multiplier_range": (0.5, 2.0),
        "n_txns_range": (10, 30),
        "new_device_prob": 0.90,
        "unusual_location_prob": 0.80,
        "unusual_recipient_prob": 0.40,
        "failed_auth_range": (5, 20),
        "inter_arrival_hours": (0.1, 1.0),
        "description": "Credential Abuse - password stuffing, brute force",
    },
    "synthetic_id": {
        "weight": 0.08,
        "hour_weights": FRAUD_HOUR_WEIGHTS,
        "amount_log_mean": 3.0,
        "amount_log_std": 0.5,
        "amount_multiplier_range": (0.5, 2.0),
        "n_txns_range": (3, 8),
        "new_device_prob": 0.80,
        "unusual_location_prob": 0.50,
        "unusual_recipient_prob": 0.70,
        "failed_auth_range": (0, 1),
        "inter_arrival_hours": (12, 168),  # weekly-ish
        "description": "Synthetic Identity - fabricated identity, thin file",
    },
    "gradual_escalation": {
        "weight": 0.08,
        "hour_weights": LEGIT_HOUR_WEIGHTS,
        "amount_log_mean": 3.5,
        "amount_log_std": 0.4,
        "amount_multiplier_range": (1.0, 1.0),  # overridden by escalation
        "n_txns_range": (8, 20),
        "new_device_prob": 0.15,
        "unusual_location_prob": 0.15,
        "unusual_recipient_prob": 0.20,
        "failed_auth_range": (0, 0),
        "inter_arrival_hours": (24, 168),
        "description": "Gradual Escalation - boiling frog, slow amount ramp",
    },
    "merchant_fraud": {
        "weight": 0.07,
        "hour_weights": FRAUD_HOUR_WEIGHTS,
        "amount_log_mean": 3.8,
        "amount_log_std": 0.5,
        "amount_multiplier_range": (1.0, 3.0),
        "n_txns_range": (4, 12),
        "new_device_prob": 0.20,
        "unusual_location_prob": 0.30,
        "unusual_recipient_prob": 0.80,
        "failed_auth_range": (0, 1),
        "inter_arrival_hours": (6, 72),
        "description": "Merchant Fraud - collusion, fake merchant patterns",
    },
}


def gen_fraud_id(rng: np.random.Generator) -> str:
    chars = rng.choice(list(FRAUD_ID_ALPHABET), size=FRAUD_ID_LENGTH - 1)
    return "F" + "".join(chars.tolist())


def draw_hour(rng: np.random.Generator, weights: np.ndarray) -> int:
    return int(rng.choice(24, p=weights / weights.sum()))


def is_weekend(ts: pd.Timestamp) -> bool:
    return ts.weekday() >= 5


# ── Account generators ──────────────────────────────────────────────────

def make_legit_events(
    rng: np.random.Generator,
    fraud_id: str,
    start_ts: pd.Timestamp,
    base_devices: int,
    n_txns_range: tuple[int, int] = (18, 48),
) -> list[dict]:
    """Generate legitimate transaction events for an account."""
    median_amount = float(rng.lognormal(4.0, 0.6))
    sigma = 0.35
    n_txns = int(rng.integers(*n_txns_range))
    events: list[dict] = []
    t_days = 0.0

    for i in range(n_txns):
        if i == 0 and rng.random() < 0.5:
            t_days += float(rng.uniform(0.001, 0.5))
        else:
            t_days += float(rng.exponential(2.2))
        ts = start_ts + pd.Timedelta(days=t_days) + pd.Timedelta(minutes=int(rng.integers(0, 60)))
        hour = draw_hour(rng, LEGIT_HOUR_WEIGHTS)
        amount = median_amount * float(rng.lognormal(0, sigma)) * (1.15 if is_weekend(ts) else 1.0)
        is_new_device = rng.random() > 0.92 or (base_devices == 0 and i == 0)

        events.append({
            "fraud_id": fraud_id,
            "ts": ts,
            "amount": amount,
            "hour": hour,
            "new_device": is_new_device,
            "unusual_location": rng.random() > 0.90,
            "unusual_recipient": rng.random() > 0.93,
            "failed_auth": int(rng.choice([0, 1, 2, 3], p=[0.96, 0.03, 0.008, 0.002])),
            "archetype": "legit",
            "label": 0,
        })
    return events


def make_fraud_events(
    rng: np.random.Generator,
    fraud_id: str,
    start_ts: pd.Timestamp,
    config: dict,
    is_escalation: bool = False,
) -> list[dict]:
    """Generate fraud transaction events based on archetype configuration."""
    median_amount = float(rng.lognormal(config["amount_log_mean"], config["amount_log_std"]))
    sigma = 0.35
    n_txns = int(rng.integers(*config["n_txns_range"]))
    events: list[dict] = []
    t_days = 0.0

    for i in range(n_txns):
        t_days += float(rng.exponential(np.mean(config["inter_arrival_hours"]) / 24.0))
        ts = start_ts + pd.Timedelta(days=t_days) + pd.Timedelta(minutes=int(rng.integers(0, 60)))

        if is_escalation:
            progress = i / max(n_txns - 1, 1)
            mult = 1.0 + 3.0 * (progress ** 0.7)
        else:
            lo, hi = config["amount_multiplier_range"]
            mult = float(rng.uniform(lo, hi))

        amount = median_amount * float(rng.lognormal(0, sigma)) * mult

        events.append({
            "fraud_id": fraud_id,
            "ts": ts,
            "amount": amount,
            "hour": draw_hour(rng, config["hour_weights"]),
            "new_device": rng.random() > (1.0 - config["new_device_prob"]),
            "unusual_location": rng.random() > (1.0 - config["unusual_location_prob"]),
            "unusual_recipient": rng.random() > (1.0 - config["unusual_recipient_prob"]),
            "failed_auth": int(rng.integers(config["failed_auth_range"][0], config["failed_auth_range"][1] + 1)),
            "archetype": config.get("_archetype_name", "fraud"),
            "label": 1,
        })
    return events


def make_mule_ring_events(
    rng: np.random.Generator,
    size: int,
    start_ts: pd.Timestamp,
) -> list[tuple[str, list[dict]]]:
    """Generate mule ring events — multiple accounts sharing device/recipient."""
    sink = f"R-SINK-{uuid.uuid4().hex[:8]}"
    out: list[tuple[str, list[dict]]] = []

    for _ in range(size):
        fid = gen_fraud_id(rng)
        median_amount = float(rng.lognormal(3.2, 0.5))
        n_txns = int(rng.integers(2, 6))
        events: list[dict] = []
        t_days = float(rng.uniform(0, 60))

        for _ in range(n_txns):
            t_days += float(rng.exponential(1.2))
            ts = start_ts + pd.Timedelta(days=t_days) + pd.Timedelta(minutes=int(rng.integers(0, 60)))
            amount = median_amount * float(rng.lognormal(0, 0.3))

            events.append({
                "fraud_id": fid,
                "ts": ts,
                "amount": amount,
                "hour": draw_hour(rng, FRAUD_HOUR_WEIGHTS),
                "new_device": True,
                "unusual_location": rng.random() > 0.6,
                "unusual_recipient": True,
                "failed_auth": int(rng.choice([0, 1, 2], p=[0.90, 0.08, 0.02])),
                "archetype": "mule",
                "label": 1,
                "shared_device_accounts": size - 1,
                "shared_recipient_accounts": size - 1,
            })
        out.append((fid, events))
    return out


def make_ato_burst_events(
    rng: np.random.Generator,
    fraud_id: str,
    start_ts: pd.Timestamp,
    burst_day: float,
) -> list[dict]:
    """ATO burst inserted into an existing legitimate account."""
    median_amount = float(rng.lognormal(4.0, 0.6))
    n = int(rng.integers(2, 6))
    events: list[dict] = []
    t_days = burst_day

    for _ in range(n):
        t_days += float(rng.uniform(0.01, 0.15))
        ts = start_ts + pd.Timedelta(days=t_days) + pd.Timedelta(minutes=int(rng.integers(0, 60)))
        amount = median_amount * float(rng.lognormal(0, 0.3)) * float(rng.uniform(1.5, 3.5))
        events.append({
            "fraud_id": fraud_id,
            "ts": ts,
            "amount": amount,
            "hour": draw_hour(rng, ATO_HOUR_WEIGHTS),
            "new_device": rng.random() > 0.15,
            "unusual_location": rng.random() > 0.30,
            "unusual_recipient": rng.random() > 0.25,
            "failed_auth": int(rng.integers(2, 9)),
            "archetype": "ato",
            "label": 1,
        })
    return events


# ── Feature derivation ──────────────────────────────────────────────────

def apply_noise(
    rng: np.random.Generator,
    events: list[dict],
    noise: float,
) -> list[dict]:
    """Label-preserving per-event feature noise."""
    if noise <= 0:
        return events
    for ev in events:
        ev["amount"] *= float(rng.lognormal(0, 0.5 * noise))
        if rng.random() < noise:
            if ev.get("archetype", "legit") == "legit":
                ev["hour"] = draw_hour(rng, FRAUD_HOUR_WEIGHTS)
            else:
                ev["hour"] = draw_hour(rng, LEGIT_HOUR_WEIGHTS)
        if rng.random() < 0.5 * noise:
            ev["new_device"] = not ev["new_device"]
        if rng.random() < noise:
            ev["unusual_location"] = not ev["unusual_location"]
        if rng.random() < noise:
            ev["unusual_recipient"] = not ev["unusual_recipient"]
        if rng.random() < noise:
            ev["failed_auth"] = max(0, ev["failed_auth"] + int(rng.integers(-1, 2)))
    return events


def derive_features(
    events: list[dict],
    account_start: pd.Timestamp,
    base_devices: int,
    graph: dict | None = None,
) -> list[dict]:
    """Turn raw events into the purpose-limited feature vector."""
    graph = graph or {}
    events = sorted(events, key=lambda e: e["ts"])
    median_amt = float(np.median([x["amount"] for x in events]))
    ratios = [e["amount"] / median_amt for e in events]
    typical_hours = typical_hours_default()

    seen_devices = base_devices
    out: list[dict] = []

    for i, ev in enumerate(events):
        ts = ev["ts"]
        hour = ev["hour"]
        if ev["new_device"]:
            seen_devices += 1

        freq_24h = sum(
            1 for j in range(i)
            if 0 < (ts - events[j]["ts"]).total_seconds() <= 86400
        )

        similar_days = None
        for j in range(i - 1, -1, -1):
            if events[j]["amount"] >= 0.8 * ev["amount"]:
                similar_days = (ts - events[j]["ts"]).total_seconds() / 86400.0
                break
        if similar_days is None:
            similar_days = float((ts - account_start).total_seconds() / 86400.0)

        ratio = ratios[i]
        window = ratios[max(0, i - 9): i + 1]

        recent_amts = [x["amount"] for x in events[max(0, i - 9):i]]
        inter_arrivals = []
        for j in range(max(0, i - 9), i):
            dt_h = (ts - events[j]["ts"]).total_seconds() / 3600.0
            if dt_h > 0:
                inter_arrivals.append(dt_h)

        _hour_dev = hour_deviation_score(hour, typical_hours)
        _amt_z = amount_zscore(events[i]["amount"], median_amt, recent_amts)
        _vel_dev = velocity_deviation_score(freq_24h, float(max(1, freq_24h)))
        _recip_novelty = recipient_novelty_score(
            ev.get("recipient", "R-001"), [], [],
        )
        _txn_reg = txn_regularity_score(inter_arrivals)

        out.append({
            "event_id": uuid.uuid4().hex,
            "fraud_id": ev["fraud_id"],
            "ts": ts,
            "hour_of_day": hour,
            "is_weekend": int(is_weekend(ts)),
            "amount_ratio": round(ratio, 4),
            "txn_amount_bucket": amount_bucket(ratio),
            "txn_freq_last_24h": int(freq_24h),
            "txn_time_unusual": int(not (8 <= hour <= 21)),
            "new_device_flag": int(ev["new_device"]),
            "unusual_location_flag": int(ev["unusual_location"]),
            "unusual_recipient_flag": int(ev["unusual_recipient"]),
            "failed_auth_count_24h": int(ev["failed_auth"]),
            "days_since_last_similar_txn": round(similar_days, 2),
            "gradual_escalation_score": round(escalation_score(window), 4),
            "known_device_count": min(8, seen_devices),
            "account_tenure_days": round(
                (ts - account_start).total_seconds() / 86400.0, 2
            ),
            "shared_device_accounts": int(
                ev.get("shared_device_accounts", graph.get("shared_device_accounts", 0))
            ),
            "shared_recipient_accounts": int(
                ev.get("shared_recipient_accounts", graph.get("shared_recipient_accounts", 0))
            ),
            "mule_ring_score": round(mule_ring_score(
                int(ev.get("shared_device_accounts", graph.get("shared_device_accounts", 0))),
                int(ev.get("shared_recipient_accounts", graph.get("shared_recipient_accounts", 0))),
            ), 4),
            "hour_deviation": _hour_dev,
            "amount_zscore": round(_amt_z, 4),
            "velocity_deviation": round(_vel_dev, 4),
            "recipient_novelty": round(_recip_novelty, 4),
            "txn_regularity": round(_txn_reg, 4),
            "archetype": ev.get("archetype", "legit"),
            "label": int(ev["label"]),
        })
    return out


# ── Main generator ──────────────────────────────────────────────────────

def generate_large(
    n_transactions: int = 1_500_000,
    seed: int = 42,
    noise: float = 0.15,
) -> pd.DataFrame:
    """Generate a large-scale fraud dataset with 8+ distinct archetypes.

    Returns a DataFrame with ML_FEATURES + label + archetype + metadata.
    """
    rng = np.random.default_rng(seed)
    noise_rng = np.random.default_rng(seed + 1) if noise > 0 else rng

    # Target ~1.67% fraud rate
    fraud_rate = 0.0167
    n_fraud_total = int(round(n_transactions * fraud_rate))
    n_legit_total = n_transactions - n_fraud_total

    print(f"Generating {n_transactions:,} transactions ({n_fraud_total:,} fraud, {n_legit_total:,} legit)")

    # ── Fraud archetype allocation ───────────────────────────────────────
    # Weighted allocation across archetypes
    archetype_names = [k for k, v in ARCHETYPE_CONFIG.items() if v["weight"] > 0]
    weights = np.array([ARCHETYPE_CONFIG[k]["weight"] for k in archetype_names])
    weights = weights / weights.sum()

    archetype_fraud_counts = {}
    remaining = n_fraud_total
    for i, name in enumerate(archetype_names[:-1]):
        count = int(round(n_fraud_total * weights[i]))
        archetype_fraud_counts[name] = count
        remaining -= count
    archetype_fraud_counts[archetype_names[-1]] = remaining

    # ── Legitimate accounts ──────────────────────────────────────────────
    # ~30 txns per account on average
    n_legit_accounts = max(100, n_legit_total // 30)
    print(f"  Creating {n_legit_accounts:,} legitimate accounts")

    all_events: list[dict] = []
    legit_accounts: list[dict] = []

    for _ in range(n_legit_accounts):
        fid = gen_fraud_id(rng)
        start = GLOBAL_START + pd.Timedelta(days=float(rng.uniform(0, TIMELINE_DAYS - 30)))
        base_devices = int(rng.poisson(1.8))
        raw = make_legit_events(rng, fid, start, base_devices)
        legit_accounts.append({"fid": fid, "start": start, "base_devices": base_devices, "raw": raw})

    # ATO bursts into ~5% of legitimate accounts
    n_compromised = max(2, int(0.05 * len(legit_accounts)))
    compromised = rng.choice(len(legit_accounts), size=n_compromised, replace=False)
    for idx in compromised:
        acct = legit_accounts[idx]
        burst_day = float(rng.uniform(0.15, 0.9)) * 120.0
        acct["raw"].extend(make_ato_burst_events(rng, acct["fid"], acct["start"], burst_day))

    # Legitimate device sharing (~2%)
    n_share = 2 * max(1, int(0.02 * len(legit_accounts)) // 2)
    if n_share >= 2 and len(legit_accounts) >= 2:
        share_candidates = rng.choice(len(legit_accounts), size=n_share, replace=False)
        for a, b in share_candidates.reshape(-1, 2):
            for idx in (a, b):
                legit_accounts[idx]["graph"] = {
                    "shared_device_accounts": 1,
                    "shared_recipient_accounts": 0,
                }

    for acct in legit_accounts:
        all_events.extend(derive_features(
            apply_noise(noise_rng, acct["raw"], noise),
            acct["start"], acct["base_devices"], acct.get("graph"),
        ))

    # ── Fraud accounts per archetype ─────────────────────────────────────
    for archetype_name in archetype_names:
        config = ARCHETYPE_CONFIG[archetype_name].copy()
        config["_archetype_name"] = archetype_name
        n_events = archetype_fraud_counts[archetype_name]

        if archetype_name == "mule":
            # Mule rings: groups of 2-4 accounts
            avg_ring_size = 3
            avg_txns = np.mean(config["n_txns_range"])
            n_rings = max(2, int(n_events / (avg_ring_size * avg_txns)))
            print(f"  {archetype_name}: {n_rings} rings ({n_events} target events)")

            for _ in range(n_rings):
                size = int(rng.integers(2, 5))
                ring_start = GLOBAL_START + pd.Timedelta(days=float(rng.uniform(0, TIMELINE_DAYS - 60)))
                for fid, events in make_mule_ring_events(rng, size, ring_start):
                    all_events.extend(derive_features(
                        apply_noise(noise_rng, events, noise), ring_start, 1,
                    ))
        else:
            # Individual fraud accounts
            avg_txns = np.mean(config["n_txns_range"])
            n_accounts = max(10, int(n_events / avg_txns))
            print(f"  {archetype_name}: {n_accounts} accounts ({n_events} target events)")

            for _ in range(n_accounts):
                fid = gen_fraud_id(rng)
                start = GLOBAL_START + pd.Timedelta(days=float(rng.uniform(0, TIMELINE_DAYS - 30)))
                is_esc = archetype_name == "gradual_escalation"
                raw = make_fraud_events(rng, fid, start, config, is_escalation=is_esc)
                all_events.extend(derive_features(
                    apply_noise(noise_rng, raw, noise), start, 1,
                ))

    # ── Assemble DataFrame ───────────────────────────────────────────────
    df = pd.DataFrame(all_events)
    df = df.sort_values("ts").reset_index(drop=True)

    # Trim to requested size
    if len(df) > n_transactions:
        drop = rng.choice(len(df), size=len(df) - n_transactions, replace=False)
        keep = np.ones(len(df), dtype=bool)
        keep[drop] = False
        df = df[keep].reset_index(drop=True)

    # Drop raw amount (data minimization)
    if "amount" in df.columns:
        df = df.drop(columns=["amount"])

    return df


def compute_dataset_hash(df: pd.DataFrame) -> str:
    """SHA-256 hash of the dataset for reproducibility tracking."""
    csv_bytes = df.to_csv(index=False).encode()
    return hashlib.sha256(csv_bytes).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-transactions", type=int, default=1_500_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--noise", type=float, default=0.15)
    parser.add_argument("--output", type=str, default="data/transactions_large.csv")
    args = parser.parse_args()

    df = generate_large(args.n_transactions, args.seed, args.noise)

    # Save dataset
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    # Generate summary
    dataset_hash = compute_dataset_hash(df)
    summary = {
        "seed": args.seed,
        "noise": args.noise,
        "n_transactions": int(len(df)),
        "n_fraud": int(df["label"].sum()),
        "n_legit": int((df["label"] == 0).sum()),
        "fraud_rate": round(float(df["label"].mean()), 5),
        "n_accounts": int(df["fraud_id"].nunique()),
        "n_archetypes": int(df["archetype"].nunique()),
        "archetype_counts": {k: int(v) for k, v in df["archetype"].value_counts().items()},
        "date_range": [str(df["ts"].min()), str(df["ts"].max())],
        "timeline_days": int((df["ts"].max() - df["ts"].min()).total_seconds() / 86400),
        "features": ML_FEATURES,
        "n_features": len(ML_FEATURES),
        "dataset_hash": dataset_hash,
        "generator_version": "v2.0",
    }

    summary_path = out_path.parent / "generation_summary_large.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nDataset written: {out_path}")
    print(f"  {len(df):,} transactions, {int(df['label'].sum()):,} fraud ({df['label'].mean():.2%})")
    print(f"  {df['fraud_id'].nunique():,} accounts, {df['archetype'].nunique()} archetypes")
    print(f"  Timeline: {df['ts'].min().date()} to {df['ts'].max().date()}")
    print(f"  Hash: {dataset_hash[:16]}...")
    print(f"  Summary: {summary_path}")


if __name__ == "__main__":
    main()
