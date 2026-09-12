#!/usr/bin/env python3
"""PS-14 Phase 2 - synthetic transaction dataset generator.

Simulates a privacy-first fraud-detection feature store: only derived,
purpose-limited features survive (per section 16 of the architecture doc).
Raw amounts, PII, exact geo, and device IDs never leave the generator - a
concrete demonstration of data minimization at the "Privacy Layer".

Four fraud archetypes are injected:
  1. "Boiling frog" escalation  - amounts ramp gradually over time, driving
     `gradual_escalation_score` (mitigation for Attack Scenario A).
  2. One-off impulse fraud      - large amount, new device, odd hour/location.
  3. Account-takeover bursts    - elevated failed auths, rapid txns, new
     devices and recipients on an otherwise-legitimate account.
  4. Card-not-present testing   - many small, fast transactions to NEW
     merchants, night-heavy hours, few auth failures (card details already
     in hand). Small amounts deliberately defeat amount-based signals, so
     detection must lean on frequency/recipient/time features.
  5. Mule-ring laundering       - small sets of accounts sharing ONE device
     and converging money on a common sink recipient; amounts stay moderate
     so only the cross-account link-analysis features (shared device /
     shared recipient) can flag them.

Label-preserving feature noise (`--noise`, default 0.15): every raw event is
perturbed before feature derivation - binary flags (new device / unusual
location / unusual recipient) flip with probability proportional to the
noise level, hours are occasionally redrawn from the OTHER profile's weight
curve (a legit txn at 3am, a fraud txn at noon), amounts drift, and auth
failures jitter. Labels never change, so the classes overlap in feature
space instead of being crisply separable - the models must learn structure
rather than memorize archetype signatures. `--noise 0` restores the crisp
generator. Link-analysis counts (shared device / recipient) are left exact:
real rings ARE detectable by graph structure.

Outputs (defaults: 10k transactions, ~1.5% fraud, seed 42, noise 0.15):
  data/transactions.csv     - numeric feature matrix with `label` (ML training)
  data/events_sample.jsonl  - first N events in the section-16 "live event"
                              shape (label=null), for the ingestion API later

Usage:
  python src/generate_synthetic_data.py [--n-transactions 10000]
                                        [--fraud-rate 0.015]
                                        [--seed 42] [--noise 0.15]
                                        [--outdir data]
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

# Shared feature-derivation logic lives in the Privacy Layer module so the
# training data and the live ingest path stay consistent.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from privacy_layer.features import (  # noqa: E402
    ML_FEATURES,
    amount_bucket,
    escalation_score,
    mule_ring_score,
)

# Base32-ish alphabet without confusables (no I, O, 0, 1). 16 chars -> 80 bits,
# fits the VARCHAR(16) pseudonym_mapping.fraud_id column in the architecture doc.
FRAUD_ID_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
FRAUD_ID_LENGTH = 16

GLOBAL_START = pd.Timestamp("2025-06-01 00:00:00")

# Hour-of-day weight profiles (24 bins). Derived later into `txn_time_unusual`.
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

def apply_noise(rng: np.random.Generator, events: list[dict], noise: float) -> list[dict]:
    """Label-preserving per-event feature noise (runs BEFORE derivation).

    Perturbs each raw event's fields so fraud and legit distributions overlap
    in feature space, while the label (and the cross-account link-analysis
    counts) stay untouched. `noise` scales every perturbation probability;
    0 returns the events unchanged.
    """
    if noise <= 0:
        return events
    for ev in events:
        # Amount drift: symmetric multiplicative perturbation (lognormal).
        ev["amount"] *= float(rng.lognormal(0, 0.5 * noise))
        # Hour overlap: occasionally redraw from the OTHER profile's weight
        # curve - a legit transaction at 3am or a fraud transaction at noon.
        if rng.random() < noise:
            if ev.get("archetype", "legit") == "legit":
                ev["hour"] = draw_hour(rng, FRAUD_HOUR_WEIGHTS)
            else:
                ev["hour"] = draw_hour(rng, LEGIT_HOUR_WEIGHTS)
        # Binary behavioural flags flip with probability proportional to noise.
        if rng.random() < 0.5 * noise:
            ev["new_device"] = not ev["new_device"]
        if rng.random() < noise:
            ev["unusual_location"] = not ev["unusual_location"]
        if rng.random() < noise:
            ev["unusual_recipient"] = not ev["unusual_recipient"]
        # Auth-failure jitter (a failed login on a legit user, or a lucky
        # attacker who gets in first try).
        if rng.random() < noise:
            ev["failed_auth"] = max(0, ev["failed_auth"] + int(rng.integers(-1, 2)))
    return events


def gen_fraud_id(rng: np.random.Generator) -> str:
    """Cryptographically-shaped random pseudo-ID (CSPRNG in the real system)."""
    chars = rng.choice(list(FRAUD_ID_ALPHABET), size=FRAUD_ID_LENGTH - 1)
    return "F" + "".join(chars.tolist())


def draw_hour(rng: np.random.Generator, weights: np.ndarray) -> int:
    return int(rng.choice(24, p=weights / weights.sum()))


def is_weekend(ts: pd.Timestamp) -> bool:
    return ts.weekday() >= 5


# --------------------------------------------------------------------------
# Account generators. Each returns a list of raw event dicts; raw amounts are
# computed internally and then discarded (only ratios are persisted).
# --------------------------------------------------------------------------

def make_legit_events(rng: np.random.Generator, fraud_id: str, start_ts: pd.Timestamp,
                      base_devices: int) -> list[dict]:
    median_amount = float(rng.lognormal(4.0, 0.6))
    sigma = 0.35
    n_txns = int(rng.integers(18, 48))
    events: list[dict] = []
    t_days = 0.0
    for i in range(n_txns):
        if i == 0 and rng.random() < 0.5:
            # ~half of accounts transact within minutes-to-hours of opening
            # (activation / first swipe) - gives training sub-hour tenure
            # examples so young accounts are not treated as inherently
            # fraudulent (cold-start false-positive fix).
            t_days += float(rng.uniform(0.001, 0.5))
        else:
            t_days += float(rng.exponential(2.2))
        ts = start_ts + pd.Timedelta(days=t_days) + pd.Timedelta(minutes=int(rng.integers(0, 60)))
        hour = draw_hour(rng, LEGIT_HOUR_WEIGHTS)
        weekend = is_weekend(ts)
        amount = median_amount * float(rng.lognormal(0, sigma)) * (1.15 if weekend else 1.0)
        # A fresh account with zero registered devices (base_devices == 0)
        # transacts first from a never-registered device - mirrors the live
        # Privacy Layer, where a brand-new account has an empty known-device
        # list (cold-start false-positive fix).
        is_new_device = rng.random() > 0.92 or (base_devices == 0 and i == 0)
        events.append(
            {
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
            }
        )
    return events


def make_fraud_account_events(rng: np.random.Generator, fraud_id: str, start_ts: pd.Timestamp) -> list[dict]:
    median_amount = float(rng.lognormal(3.9, 0.5))
    sigma = 0.4
    escalating = rng.random() < 0.35
    n_txns = int(rng.integers(4, 9))
    base_mult = 1.0 if escalating else float(rng.uniform(1.4, 2.2))
    ramp_end = float(rng.uniform(2.5, 4.5)) if escalating else None
    events: list[dict] = []
    t_days = 0.0
    for i in range(n_txns):
        t_days += float(rng.exponential(1.5))
        ts = start_ts + pd.Timedelta(days=t_days) + pd.Timedelta(minutes=int(rng.integers(0, 60)))
        if escalating:
            progress = i / max(n_txns - 1, 1)
            mult = 1.0 + (ramp_end - 1.0) * (progress**0.7)
        else:
            mult = base_mult
        amount = median_amount * float(rng.lognormal(0, sigma)) * mult
        events.append(
            {
                "fraud_id": fraud_id,
                "ts": ts,
                "amount": amount,
                "hour": draw_hour(rng, FRAUD_HOUR_WEIGHTS),
                "new_device": rng.random() > 0.25,
                "unusual_location": rng.random() > 0.40,
                "unusual_recipient": rng.random() > 0.50,
                "failed_auth": int(rng.choice([0, 1, 2, 3], p=[0.80, 0.15, 0.04, 0.01])),
                "archetype": "escalation" if escalating else "impulse",
                "label": 1,
            }
        )
    return events


def make_cnp_events(rng: np.random.Generator, fraud_id: str, start_ts: pd.Timestamp) -> list[dict]:
    """Card-not-present testing: rapid-fire small transactions, almost always
    to a NEW merchant token, night-heavy hours, few auth failures.

    Amounts stay near the account's own (small) median so amount_ratio and
    bucket features see nothing unusual - the pattern must be caught by
    frequency, recipient, and time-of-day features.
    """
    median_amount = float(rng.lognormal(2.8, 0.4))  # small-ticket baseline
    sigma = 0.25
    n_txns = int(rng.integers(6, 14))
    events: list[dict] = []
    t_days = 0.0
    for _ in range(n_txns):
        t_days += float(rng.exponential(0.4))  # dense: ~2-3 txns/day on average
        ts = start_ts + pd.Timedelta(days=t_days) + pd.Timedelta(minutes=int(rng.integers(0, 60)))
        amount = median_amount * float(rng.lognormal(0, sigma))
        events.append(
            {
                "fraud_id": fraud_id,
                "ts": ts,
                "amount": amount,
                "hour": draw_hour(rng, ATO_HOUR_WEIGHTS),  # night-leaning
                "new_device": rng.random() > 0.5,
                "unusual_location": rng.random() > 0.5,
                "unusual_recipient": rng.random() > 0.15,  # ~always a new merchant
                "failed_auth": int(rng.choice([0, 1, 2], p=[0.90, 0.08, 0.02])),
                "archetype": "cnp",
                "label": 1,
            }
        )
    return events


def make_ato_burst_events(rng: np.random.Generator, fraud_id: str, start_ts: pd.Timestamp, burst_day: float) -> list[dict]:
    """Short, dense burst of fraudulent txns on an otherwise-legit account."""
    median_amount = float(rng.lognormal(4.0, 0.6))
    n = int(rng.integers(2, 6))
    events: list[dict] = []
    t_days = burst_day
    for _ in range(n):
        t_days += float(rng.uniform(0.01, 0.15))
        ts = start_ts + pd.Timedelta(days=t_days) + pd.Timedelta(minutes=int(rng.integers(0, 60)))
        amount = median_amount * float(rng.lognormal(0, 0.3)) * float(rng.uniform(1.5, 3.5))
        events.append(
            {
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
            }
        )
    return events


def make_mule_ring_events(rng: np.random.Generator, size: int, start_ts: pd.Timestamp) -> list[tuple[str, list[dict]]]:
    """A mule ring: `size` accounts sharing ONE device and sending to a
    common sink recipient (money converges). Amounts stay moderate so the
    pattern is invisible to amount features - detection must come from the
    cross-account link-analysis features (shared device / shared recipient).

    Returns (fraud_id, events) pairs; every event carries the ring's
    shared-account counts (size - 1 OTHER accounts on the device/recipient).
    """
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
            events.append(
                {
                    "fraud_id": fid,
                    "ts": ts,
                    "amount": amount,
                    "hour": draw_hour(rng, FRAUD_HOUR_WEIGHTS),
                    "new_device": True,  # the shared ring device is never this account's own
                    "unusual_location": rng.random() > 0.6,
                    "unusual_recipient": True,  # the sink is never a usual recipient
                    "failed_auth": int(rng.choice([0, 1, 2], p=[0.90, 0.08, 0.02])),
                    "archetype": "mule",
                    "label": 1,
                    "shared_device_accounts": size - 1,
                    "shared_recipient_accounts": size - 1,
                }
            )
        out.append((fid, events))
    return out


def derive_features(events: list[dict], account_start: pd.Timestamp, base_devices: int,
                    graph: dict | None = None) -> list[dict]:
    """Turn raw events into the purpose-limited feature vector (section 16).

    `graph` carries cross-account context ({shared_device_accounts,
    shared_recipient_accounts}) applied to every event of the account;
    defaults to no sharing (both counts 0).
    """
    graph = graph or {}
    events = sorted(events, key=lambda e: e["ts"])
    ratios = [e["amount"] / float(np.median([x["amount"] for x in events])) for e in events]
    # Per-account running stats
    seen_devices = base_devices
    out: list[dict] = []
    for i, ev in enumerate(events):
        ts = ev["ts"]
        hour = ev["hour"]
        if ev["new_device"]:
            seen_devices += 1
        # txn_freq_last_24h: prior events within the last 24h
        freq_24h = sum(1 for j in range(i) if 0 < (ts - events[j]["ts"]).total_seconds() <= 86400)
        # days_since_last_similar_txn: most recent prior txn with amount >= 80% of current
        similar_days = None
        for j in range(i - 1, -1, -1):
            if events[j]["amount"] >= 0.8 * ev["amount"]:
                similar_days = (ts - events[j]["ts"]).total_seconds() / 86400.0
                break
        if similar_days is None:
            similar_days = float((ts - account_start).total_seconds() / 86400.0)
        ratio = ratios[i]
        window = ratios[max(0, i - 9) : i + 1]
        # Deviation features
        from privacy_layer.features import (
            hour_deviation_score, amount_zscore, velocity_deviation_score,
            recipient_novelty_score, txn_regularity_score,
        )
        median_amt = float(np.median([x["amount"] for x in events]))
        recent_amts = [x["amount"] for x in events[max(0, i-9):i]]
        typical_hours = list(range(8, 22))
        # Compute inter-arrival times
        inter_arrivals = []
        for j in range(max(0, i-9), i):
            dt_h = (ts - events[j]["ts"]).total_seconds() / 3600.0
            if dt_h > 0:
                inter_arrivals.append(dt_h)
        _hour_dev = hour_deviation_score(hour, typical_hours)
        _amt_z = amount_zscore(event["amount"] if False else events[i]["amount"], median_amt, recent_amts)
        _vel_dev = velocity_deviation_score(freq_24h, float(max(1, freq_24h)))
        _recip_novelty = recipient_novelty_score(
            ev.get("recipient", "R-001"),
            [],  # no usual recipients for synthetic
            [],
        )
        _txn_reg = txn_regularity_score(inter_arrivals)
        out.append(
            {
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
                # No 1-day floor: young accounts (tenure < 1 day) must appear
                # in training so the models learn they are not inherently
                # fraudulent (cold-start false-positive fix).
                "account_tenure_days": round((ts - account_start).total_seconds() / 86400.0, 2),
                # Cross-account link analysis: counts of OTHER accounts on the
                # device/recipient (per-event when the pattern sets them,
                # else the account-level graph context).
                "shared_device_accounts": int(ev.get("shared_device_accounts", graph.get("shared_device_accounts", 0))),
                "shared_recipient_accounts": int(ev.get("shared_recipient_accounts", graph.get("shared_recipient_accounts", 0))),
                "mule_ring_score": round(mule_ring_score(
                    int(ev.get("shared_device_accounts", graph.get("shared_device_accounts", 0))),
                    int(ev.get("shared_recipient_accounts", graph.get("shared_recipient_accounts", 0))),
                ), 4),
                # Deviation features — continuous measures of anomaly strength.
                "hour_deviation": _hour_dev,
                "amount_zscore": round(_amt_z, 4),
                "velocity_deviation": round(_vel_dev, 4),
                "recipient_novelty": round(_recip_novelty, 4),
                "txn_regularity": round(_txn_reg, 4),
                "archetype": ev.get("archetype", "legit"),
                "label": int(ev["label"]),
            }
        )
    return out


def derive_features_causal(events: list[dict], account_start: pd.Timestamp, base_devices: int,
                           graph: dict | None = None) -> list[dict]:
    """Causal feature derivation - ML-validity rebuild (2026-09-03).

    Identical output schema and semantics to :func:`derive_features` EXCEPT
    every historical statistic uses only information available before the
    current event:

    * ``amount_ratio`` - denominator is the expanding median of PRIOR event
      amounts (1.0 when no prior event exists), never the account-wide
      median (which includes future events, including fraud spikes).
    * ``amount_zscore`` / ``gradual_escalation_score`` - fed by those causal
      ratios instead of future-contaminated ones.

    All other features (24h frequency, last-similar spacing, running device
    count, tenure, hour/weekend, per-event flags) were already prior-only in
    :func:`derive_features` and are unchanged.

    This is the reference implementation used by the ML-validity rebuild;
    the legacy :func:`derive_features` is preserved verbatim so historical
    metrics stay reproducible (and are marked INVALID in the validity
    report because of the future-contamination described above).
    """
    graph = graph or {}
    events = sorted(events, key=lambda e: e["ts"])
    # Prior-only expanding median baseline: median of amounts strictly BEFORE
    # the current event. 1.0 ("typical") when no history exists yet.
    def prior_median(i: int) -> float:
        if i < 1:
            return 1.0
        med = float(np.median([x["amount"] for x in events[:i]]))
        return med if med > 0 else 1.0

    ratios: list[float] = [
        e["amount"] / prior_median(i) for i, e in enumerate(events)
    ]
    seen_devices = base_devices
    out: list[dict] = []
    for i, ev in enumerate(events):
        ts = ev["ts"]
        hour = ev["hour"]
        if ev["new_device"]:
            seen_devices += 1
        freq_24h = sum(1 for j in range(i) if 0 < (ts - events[j]["ts"]).total_seconds() <= 86400)
        similar_days = None
        for j in range(i - 1, -1, -1):
            if events[j]["amount"] >= 0.8 * ev["amount"]:
                similar_days = (ts - events[j]["ts"]).total_seconds() / 86400.0
                break
        if similar_days is None:
            similar_days = float((ts - account_start).total_seconds() / 86400.0)
        ratio = ratios[i]
        window = ratios[max(0, i - 9) : i + 1]
        from privacy_layer.features import (
            hour_deviation_score, amount_zscore, velocity_deviation_score,
            recipient_novelty_score, txn_regularity_score,
        )
        median_amt = prior_median(i)
        recent_amts = [x["amount"] for x in events[max(0, i - 9):i]]
        typical_hours = list(range(8, 22))
        inter_arrivals = []
        for j in range(max(0, i - 9), i):
            dt_h = (ts - events[j]["ts"]).total_seconds() / 3600.0
            if dt_h > 0:
                inter_arrivals.append(dt_h)
        _hour_dev = hour_deviation_score(hour, typical_hours)
        _amt_z = amount_zscore(events[i]["amount"], median_amt, recent_amts)
        _vel_dev = velocity_deviation_score(freq_24h, float(max(1, freq_24h)))
        _recip_novelty = recipient_novelty_score(ev.get("recipient", "R-001"), [], [])
        _txn_reg = txn_regularity_score(inter_arrivals)
        out.append(
            {
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
                "account_tenure_days": round((ts - account_start).total_seconds() / 86400.0, 2),
                "shared_device_accounts": int(ev.get("shared_device_accounts", graph.get("shared_device_accounts", 0))),
                "shared_recipient_accounts": int(ev.get("shared_recipient_accounts", graph.get("shared_recipient_accounts", 0))),
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
            }
        )
    return out


def generate(n_transactions: int, fraud_rate: float, seed: int, noise: float = 0.15,
             derive_fn=derive_features) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    # Noise draws from its OWN stream so the base simulation (row set,
    # archetype mix, trim) is bit-identical between noise levels - the only
    # difference between `--noise 0` and `--noise 0.15` is the feature
    # values, keeping comparisons apples-to-apples.
    noise_rng = np.random.default_rng(seed + 1) if noise > 0 else rng
    n_fraud_total = int(round(n_transactions * fraud_rate))
    n_ato_txns = int(n_fraud_total * 0.55)
    n_impulse_txns = int(n_fraud_total * 0.25)
    n_cnp_txns = n_fraud_total - n_ato_txns - n_impulse_txns
    n_compromised = max(2, round(n_ato_txns / 4.0))
    # Rare archetypes use more, smaller accounts so they spread through the
    # timeline and remain present in the out-of-time test split.
    n_fraud_accounts = max(3, round(n_impulse_txns / 5.0))
    n_cnp_accounts = max(3, round(n_cnp_txns / 7.0))
    n_legit_accounts = max(20, round((n_transactions - n_fraud_total) / 33.0))

    def fraud_start_day(rng_: np.random.Generator) -> float:
        """~1/3 of rare-archetype accounts start late so every archetype is
        present in the out-of-time test window (last ~15% of the timeline) -
        otherwise small archetypes vanish from the test split entirely."""
        if rng_.random() < 0.35:
            return float(rng_.uniform(130, 160))
        return float(rng_.uniform(0, 60))

    all_events: list[dict] = []
    legit_accounts: list[dict] = []

    for _ in range(n_legit_accounts):
        fid = gen_fraud_id(rng)
        start = GLOBAL_START + pd.Timedelta(days=float(rng.uniform(0, 60)))
        # Allow zero known devices: a brand-new account has an empty
        # known-device list, exactly like the live Privacy Layer (its first
        # event is therefore from a new device, count == 1). Without this the
        # model only ever sees known_device_count == 1 on fraud accounts and
        # flags every fresh account's first transaction (cold-start fix).
        base_devices = int(rng.poisson(1.8))
        legit_accounts.append(
            {"fid": fid, "start": start, "base_devices": base_devices,
             "raw": make_legit_events(rng, fid, start, base_devices)}
        )

    # Compromised legit accounts get an ATO burst inserted mid-life; the burst
    # is merged into the account's real history before feature derivation so
    # freq / escalation / spacing features reflect the combined timeline.
    compromised = rng.choice(len(legit_accounts), size=n_compromised, replace=False)
    for idx in compromised:
        acct = legit_accounts[idx]
        burst_day = float(rng.uniform(0.15, 0.9)) * 120.0
        acct["raw"].extend(make_ato_burst_events(rng, acct["fid"], acct["start"], burst_day))

    # Rare legitimate device sharing (~2% of accounts, paired): a family
    # phone used by two legit accounts - gives the models benign examples of
    # shared_device_accounts=1 so sharing is not automatically suspicious.
    share_candidates = rng.choice(len(legit_accounts), size=2 * max(1, int(0.02 * len(legit_accounts)) // 2), replace=False)
    for a, b in share_candidates.reshape(-1, 2):
        for idx in (a, b):
            legit_accounts[idx]["graph"] = {"shared_device_accounts": 1, "shared_recipient_accounts": 0}

    for acct in legit_accounts:
        # Noise runs AFTER the ATO burst is merged, so burst events are
        # perturbed exactly like the rest of the account's history.
        all_events.extend(derive_fn(
            apply_noise(noise_rng, acct["raw"], noise),
            acct["start"], acct["base_devices"], acct.get("graph"),
        ))

    for _ in range(n_fraud_accounts):
        fid = gen_fraud_id(rng)
        start = GLOBAL_START + pd.Timedelta(days=fraud_start_day(rng))
        all_events.extend(derive_fn(
            apply_noise(noise_rng, make_fraud_account_events(rng, fid, start), noise), start, 1,
        ))

    # Card-not-present testing accounts: small fast txns to new merchants.
    for _ in range(n_cnp_accounts):
        fid = gen_fraud_id(rng)
        start = GLOBAL_START + pd.Timedelta(days=fraud_start_day(rng))
        all_events.extend(derive_fn(
            apply_noise(noise_rng, make_cnp_events(rng, fid, start), noise), start, 1,
        ))

    # Mule rings: small sets of accounts converging money on a shared device
    # and a common sink recipient (detectable only via link analysis).
    n_mule_rings = max(2, int(round(n_fraud_total * 0.12)) // 4)
    for _ in range(n_mule_rings):
        size = int(rng.integers(2, 5))  # 2-4 member rings
        ring_start = GLOBAL_START + pd.Timedelta(days=float(rng.uniform(0, 150)))
        for fid, events in make_mule_ring_events(rng, size, ring_start):
            all_events.extend(derive_fn(
                apply_noise(noise_rng, events, noise), ring_start, 1,
            ))

    df = pd.DataFrame(all_events)
    df = df.sort_values("ts").reset_index(drop=True)

    # Trim to the requested size with a seeded random drop (not "earliest N"),
    # so the fraud rate and date range stay representative of the simulation.
    overflow = len(df) - n_transactions
    if overflow > 0:
        drop = rng.choice(len(df), size=overflow, replace=False)
        keep = np.ones(len(df), dtype=bool)
        keep[drop] = False
        df = df[keep].reset_index(drop=True)

    # Drop the raw amount forever - only the ratio survives (data minimization).
    if "amount" in df.columns:
        df = df.drop(columns=["amount"])
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-transactions", type=int, default=10_000)
    ap.add_argument("--fraud-rate", type=float, default=0.015)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--noise", type=float, default=0.15,
                    help="label-preserving feature noise / overlap (0 = crisp archetypes)")
    ap.add_argument("--outdir", default="data")
    ap.add_argument("--causal", action="store_true",
                    help="derive features causally (prior-only history stats; ML-validity "
                         "rebuild) and write transactions_causal.csv instead")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    derive_fn = derive_features_causal if args.causal else derive_features
    out_name = "transactions_causal.csv" if args.causal else "transactions.csv"
    summary_name = "generation_summary_causal.json" if args.causal else "generation_summary.json"
    df = generate(args.n_transactions, args.fraud_rate, args.seed, args.noise, derive_fn)
    df.to_csv(outdir / out_name, index=False)

    # Section-16 "live event" sample: no labels, ready for the ingestion API.
    sample = df.head(250).copy()
    sample["label"] = None
    live = sample.drop(columns=["ts"]).to_dict(orient="records")
    with open(outdir / "events_sample.jsonl", "w", encoding="utf-8") as f:
        for rec in live:
            f.write(json.dumps(rec, default=str) + "\n")

    summary = {
        "seed": args.seed,
        "noise": args.noise,
        "n_transactions": int(len(df)),
        "n_fraud": int(df["label"].sum()),
        "actual_fraud_rate": round(float(df["label"].mean()), 5),
        "n_accounts": int(df["fraud_id"].nunique()),
        "archetype_counts": {k: int(v) for k, v in df["archetype"].value_counts().items()},
        "date_range": [str(df["ts"].min()), str(df["ts"].max())],
        "features": ML_FEATURES,
    }
    summary["derive_fn"] = "derive_features_causal (prior-only)" if args.causal else \
        "derive_features (legacy; future-contaminated amount stats - INVALID for ML validity)"
    summary["file"] = out_name
    with open(outdir / summary_name, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote {len(df):,} transactions to {outdir/out_name}")
    print(f"  fraud rate: {summary['actual_fraud_rate']:.4f} "
          f"({summary['n_fraud']} of {summary['n_transactions']}) over "
          f"{summary['n_accounts']} accounts, {summary['date_range'][0][:10]}..{summary['date_range'][1][:10]}")
    print(f"  label-preserving noise: {args.noise:.2f}")
    print("  archetype mix: " + ", ".join(f"{k}={v}" for k, v in summary["archetype_counts"].items()))
    print(f"  live-event sample: {outdir/'events_sample.jsonl'}")


if __name__ == "__main__":
    main()
