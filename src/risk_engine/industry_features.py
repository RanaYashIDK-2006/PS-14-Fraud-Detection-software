"""Industry-inspired fraud detection features.

Features inspired by Stripe Radar, PayPal fraud detection, Feedzai,
and graph-based fraud detection research (2025-2026).

These are ADDITIVE — they enhance the existing ML_FEATURES without
replacing any core decision logic. The rules engine, ML models, and
privacy layer remain unchanged.

References:
- Stripe Radar: 1,000+ signals, 99.9% accuracy, <100ms latency
- Device fingerprinting: browser, OS, screen, timezone
- Impossible travel: geo-distance / time-difference analysis
- Network analysis: shared entities, community detection
- Behavioral session: transaction patterns within sessions
- Adaptive thresholds: context-dependent risk scoring
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ═══════════════════════════════════════════════════════
# DEVICE FINGERPRINTING (Stripe Radar-inspired)
# ═══════════════════════════════════════════════════════

@dataclass
class DeviceFingerprint:
    """Browser/device fingerprint for fraud detection.
    
    Stripe Radar uses device signals as one of its 1,000+ features.
    This implements a lightweight version for PS-14.
    """
    user_agent: str = ""
    ip_address: str = ""
    screen_width: int = 0
    screen_height: int = 0
    timezone_offset: int = 0  # minutes from UTC
    language: str = ""
    platform: str = ""
    browser: str = ""
    
    def fingerprint_hash(self) -> str:
        """Generate a stable device fingerprint hash."""
        raw = f"{self.user_agent}|{self.screen_width}x{self.screen_height}|{self.timezone_offset}|{self.language}|{self.platform}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]
    
    def is_consistent(self) -> bool:
        """Check if device signals are internally consistent.
        
        Inconsistencies suggest spoofing or automation:
        - Screen size of 0
        - Timezone that doesn't match IP geolocation
        - Missing language/platform
        """
        issues = 0
        if self.screen_width == 0 or self.screen_height == 0:
            issues += 1
        if not self.language:
            issues += 1
        if not self.platform:
            issues += 1
        if abs(self.timezone_offset) > 12 * 60:  # > 12 hours off
            issues += 1
        return issues == 0
    
    def spoofing_score(self) -> float:
        """Score 0-1 for likelihood of device spoofing.
        
        Based on Stripe's signals: headless browsers, automated tools,
        and device farms often have telltale signs.
        """
        score = 0.0
        # Missing basic signals
        if not self.user_agent:
            score += 0.3
        if self.screen_width == 0:
            score += 0.2
        if not self.language:
            score += 0.1
        # Common bot patterns
        ua_lower = self.user_agent.lower()
        if "headless" in ua_lower or "phantom" in ua_lower or "selenium" in ua_lower:
            score += 0.5
        if "python" in ua_lower or "curl" in ua_lower or "wget" in ua_lower:
            score += 0.4
        # Screen too small (mobile emulators)
        if 0 < self.screen_width < 320:
            score += 0.1
        return min(score, 1.0)


# ═══════════════════════════════════════════════════════
# IMPOSSIBLE TRAVEL DETECTION (Stripe/Visa-inspired)
# ═══════════════════════════════════════════════════════

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distance in km between two lat/lon points."""
    R = 6371  # Earth radius in km
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def impossible_travel_score(
    lat1: float, lon1: float, time1: float,
    lat2: float, lon2: float, time2: float,
    max_speed_kmh: float = 900.0,  # ~commercial aircraft
) -> float:
    """Score for impossible travel between two transactions.
    
    If two transactions from the same account are too far apart
    relative to the time between them, it's likely fraud.
    
    Args:
        lat1, lon1: First transaction location
        time1: First transaction timestamp (seconds)
        lat2, lon2: Second transaction location
        time2: Second transaction timestamp (seconds)
        max_speed_kmh: Maximum plausible travel speed
    
    Returns:
        Score 0-1 where 1 = definitely impossible travel
    
    Inspired by: Visa's Advanced Authorization, Stripe Radar
    """
    distance_km = haversine_distance(lat1, lon1, lat2, lon2)
    time_diff_hours = abs(time2 - time1) / 3600.0
    
    if time_diff_hours < 0.001:  # < 3.6 seconds
        return 1.0 if distance_km > 1 else 0.0
    
    required_speed = distance_km / time_diff_hours
    score = min(required_speed / max_speed_kmh, 2.0) / 2.0
    return round(min(score, 1.0), 4)


# ═══════════════════════════════════════════════════════
# NETWORK/GRAPH FEATURES (GNN-inspired)
# ═══════════════════════════════════════════════════════

@dataclass
class EntityNode:
    """A node in the fraud detection graph.
    
    Inspired by graph neural network fraud detection (NVIDIA, AWS).
    Instead of full GNN, we compute lightweight graph statistics.
    """
    entity_id: str
    entity_type: str  # "card", "device", "merchant", "recipient", "ip"
    connections: dict = field(default_factory=dict)  # entity_type -> set of IDs
    fraud_history: float = 0.0  # 0-1 fraud rate for this entity


class NetworkAnalyzer:
    """Lightweight graph analysis for fraud ring detection.
    
    Features inspired by:
    - TigerGraph's PageRank for fraud
    - AWS's near-real-time GNN fraud detection
    - PayPal's network analysis for mule detection
    
    Instead of full GNN, we compute:
    - Entity degree (how many connections)
    - Shared entity count (how many entities share this one)
    - Community density (are connections clustered)
    - Fraud propagation (are connected entities fraudulent)
    """
    
    def __init__(self):
        self._entity_graph: dict[str, dict[str, set]] = {}
        self._fraud_rates: dict[str, float] = {}
    
    def add_transaction(
        self,
        card_id: str,
        device_id: str,
        merchant_id: str,
        recipient_id: str = "",
        ip_address: str = "",
        is_fraud: bool = False,
    ):
        """Record a transaction in the graph."""
        entities = {
            "card": card_id,
            "device": device_id,
            "merchant": merchant_id,
            "recipient": recipient_id,
            "ip": ip_address,
        }
        
        for etype, eid in entities.items():
            if not eid:
                continue
            if eid not in self._entity_graph:
                self._entity_graph[eid] = {"card": set(), "device": set(),
                                           "merchant": set(), "recipient": set(), "ip": set()}
            # Connect this entity to all others in the transaction
            for other_type, other_id in entities.items():
                if other_id and other_id != eid:
                    self._entity_graph[eid][other_type].add(other_id)
    
    def get_features(self, card_id: str, device_id: str, merchant_id: str) -> dict:
        """Get network features for a transaction."""
        features = {}
        
        # Card degree: how many devices/merchants/IPs has this card been used with
        card_node = self._entity_graph.get(card_id, {})
        features["card_device_count"] = len(card_node.get("device", set()))
        features["card_merchant_count"] = len(card_node.get("merchant", set()))
        features["card_ip_count"] = len(card_node.get("ip", set()))
        
        # Device degree: how many cards have used this device
        device_node = self._entity_graph.get(device_id, {})
        features["device_card_count"] = len(device_node.get("card", set()))
        features["device_merchant_count"] = len(device_node.get("merchant", set()))
        
        # Merchant risk: how many different cards use this merchant
        merchant_node = self._entity_graph.get(merchant_id, {})
        features["merchant_card_count"] = len(merchant_node.get("card", set()))
        
        # Shared entity counts (Stripe Radar signal: "cards per IP")
        features["cards_per_device"] = features["device_card_count"]
        features["devices_per_card"] = features["card_device_count"]
        
        # Fraud propagation: average fraud rate of connected entities
        connected_fraud_rates = []
        for etype in ["card", "device", "merchant"]:
            for eid in card_node.get(etype, set()):
                if eid in self._fraud_rates:
                    connected_fraud_rates.append(self._fraud_rates[eid])
        features["connected_fraud_rate"] = (
            np.mean(connected_fraud_rates) if connected_fraud_rates else 0.0
        )
        
        return features
    
    def update_fraud_rate(self, entity_id: str, is_fraud: bool, alpha: float = 0.1):
        """Update entity fraud rate with exponential moving average."""
        current = self._fraud_rates.get(entity_id, 0.0)
        self._fraud_rates[entity_id] = current * (1 - alpha) + float(is_fraud) * alpha


# ═══════════════════════════════════════════════════════
# BEHAVIORAL SESSION FEATURES (PayPal/Feedzai-inspired)
# ═══════════════════════════════════════════════════════

@dataclass
class SessionProfile:
    """User session behavioral profile.
    
    Inspired by PayPal's behavioral analytics and Feedzai's
    session-based fraud detection.
    
    Key insight: Fraudsters behave differently within a session
    than legitimate users — even if individual features look normal.
    """
    session_txns: list = field(default_factory=list)
    session_start: float = 0.0
    session_amounts: list = field(default_factory=list)
    session_merchants: list = field(default_factory=list)
    session_cities: list = field(default_factory=list)
    
    def add_transaction(self, amount: float, merchant: str, city: str, timestamp: float):
        """Add a transaction to the current session."""
        self.session_txns.append(timestamp)
        self.session_amounts.append(amount)
        self.session_merchants.append(merchant)
        self.session_cities.append(city)
        if not self.session_start:
            self.session_start = timestamp
    
    def get_features(self) -> dict:
        """Extract behavioral session features."""
        if not self.session_txns:
            return {"session_length": 0, "session_unique_merchants": 0,
                    "session_amount_escalation": 0, "session_city_switches": 0,
                    "session_speed": 0, "session_amount_cv": 0}
        
        n = len(self.session_txns)
        amounts = np.array(self.session_amounts)
        
        # Session length
        duration = self.session_txns[-1] - self.session_start
        
        # Amount escalation: are amounts increasing? (fraud pattern)
        if n >= 2:
            escalation = float(np.mean(np.diff(amounts)) / (np.mean(amounts) + 1e-8))
        else:
            escalation = 0.0
        
        # City switching: rapid city changes in session = suspicious
        unique_cities = len(set(self.session_cities))
        city_switches = sum(
            1 for i in range(1, len(self.session_cities))
            if self.session_cities[i] != self.session_cities[i - 1]
        )
        
        # Speed: transactions per minute
        speed = n / max(duration / 60, 0.001) if duration > 0 else 0
        
        # Amount CV: coefficient of variation (fraud often has low CV = repeat amounts)
        amount_cv = float(np.std(amounts) / (np.mean(amounts) + 1e-8))
        
        return {
            "session_length": n,
            "session_duration_min": round(duration / 60, 2),
            "session_unique_merchants": len(set(self.session_merchants)),
            "session_unique_cities": unique_cities,
            "session_amount_escalation": round(escalation, 4),
            "session_city_switches": city_switches,
            "session_speed": round(speed, 2),
            "session_amount_cv": round(amount_cv, 4),
            "session_amount_max_ratio": round(
                float(np.max(amounts) / (np.mean(amounts) + 1e-8)), 4
            ),
        }


# ═══════════════════════════════════════════════════════
# ADAPTIVE RISK THRESHOLDS (Stripe Radar-inspired)
# ═══════════════════════════════════════════════════════

class AdaptiveThresholds:
    """Context-dependent risk thresholds.
    
    Inspired by Stripe Radar's approach:
    - Different thresholds for different merchant categories
    - Higher sensitivity for high-risk regions/times
    - Confidence-based escalation
    
    PS-14 uses fixed thresholds. This adds adaptive logic
    on top without changing the core decision rules.
    """
    
    # MCC categories that are higher risk
    HIGH_RISK_MCC = {
        "7995",  # Gambling
        "6051",  # Quasi-cash
        "4829",  # Money transfer
        "6012",  # Financial institutions
        "7994",  # Video games
    }
    
    # Time-based risk multipliers
    NIGHT_RISK_MULTIPLIER = 1.3  # 22:00-06:00
    WEEKEND_RISK_MULTIPLIER = 1.1
    
    @classmethod
    def compute_context_risk(
        cls,
        base_risk: float,
        hour: int = 12,
        mcc: str = "",
        is_new_device: bool = False,
        is_new_merchant: bool = False,
        account_age_days: int = 365,
    ) -> float:
        """Adjust risk score based on transaction context.
        
        This does NOT override the ML score or rules.
        It provides an additional signal that can be used
        for confidence-based escalation.
        
        Returns:
            Adjusted risk score 0-1
        """
        multiplier = 1.0
        
        # Time-of-day adjustment
        if 22 <= hour or hour <= 6:
            multiplier *= cls.NIGHT_RISK_MULTIPLIER
        
        # Weekend adjustment
        # (caller should pass is_weekend if available)
        
        # MCC risk
        if mcc in cls.HIGH_RISK_MCC:
            multiplier *= 1.2
        
        # New entity risk (first-time device/merchant)
        if is_new_device:
            multiplier *= 1.15
        if is_new_merchant:
            multiplier *= 1.1
        
        # Account age risk (new accounts are higher risk)
        if account_age_days < 30:
            multiplier *= 1.2
        elif account_age_days < 90:
            multiplier *= 1.1
        
        adjusted = base_risk * multiplier
        return min(adjusted, 1.0)
    
    @classmethod
    def should_escalate(
        cls,
        ml_score: float,
        rule_score: float,
        context_risk: float,
        confidence: float = 0.9,
    ) -> str:
        """Determine escalation level based on combined signals.
        
        This implements Stripe Radar's approach of:
        - High confidence fraud → BLOCK
        - Medium confidence → REVIEW
        - Low confidence but context risky → STEP_UP
        
        Returns: "allow", "step_up", "review", "block"
        """
        # High ML confidence + high score → block
        if ml_score > 0.8 and confidence > 0.9:
            return "block"
        
        # Rules triggered → at least review
        if rule_score > 0:
            if ml_score > 0.5:
                return "block"
            return "review"
        
        # ML score high but low confidence → step up
        if ml_score > 0.6 and confidence < 0.8:
            return "step_up"
        
        # Context risk high + ML moderate → step up
        if context_risk > 0.7 and ml_score > 0.4:
            return "step_up"
        
        # Low risk → allow
        return "allow"


# ═══════════════════════════════════════════════════════
# CLASS IMBALANCE HANDLING (Industry best practice)
# ═══════════════════════════════════════════════════════

def compute_class_weights(y: np.ndarray) -> dict:
    """Compute balanced class weights for imbalanced datasets.
    
    Fraud datasets are typically 0.1-5% fraud. Standard practice:
    - Scale positive weight inversely with prevalence
    - Use in XGBoost's scale_pos_weight parameter
    
    This is already used in PS-14 but this provides a
    documented, reusable function.
    """
    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    total = len(y)
    
    # Inverse frequency weighting
    weight_pos = total / (2 * n_pos) if n_pos > 0 else 1.0
    weight_neg = total / (2 * n_neg) if n_neg > 0 else 1.0
    
    return {
        "weight_0": round(weight_neg, 4),
        "weight_1": round(weight_pos, 4),
        "scale_pos_weight": round(weight_pos / weight_neg, 4),
        "prevalence": round(n_pos / total, 6),
    }


# ═══════════════════════════════════════════════════════
# COMBINED FEATURE EXTRACTION
# ═══════════════════════════════════════════════════════

def extract_industry_features(
    transaction: dict,
    network: Optional[NetworkAnalyzer] = None,
    session: Optional[SessionProfile] = None,
    prev_txn: Optional[dict] = None,
) -> dict:
    """Extract all industry-inspired features from a transaction.
    
    This is the main entry point. Call this for each transaction
    to get the additional features that enhance the existing
    ML_FEATURES.
    
    Args:
        transaction: Dict with keys like amount, hour, mcc,
                     device_id, card_id, merchant_id, etc.
        network: NetworkAnalyzer instance for graph features
        session: SessionProfile for behavioral features
        prev_txn: Previous transaction for velocity/travel features
    
    Returns:
        Dict of feature_name -> value
    """
    features = {}
    
    # 1. Device fingerprinting
    fp = DeviceFingerprint(
        user_agent=transaction.get("user_agent", ""),
        screen_width=transaction.get("screen_width", 0),
        screen_height=transaction.get("screen_height", 0),
        timezone_offset=transaction.get("timezone_offset", 0),
        language=transaction.get("language", ""),
        platform=transaction.get("platform", ""),
    )
    features["device_spoofing_score"] = fp.spoofing_score()
    features["device_consistent"] = float(fp.is_consistent())
    
    # 2. Impossible travel
    if prev_txn and all(k in transaction for k in ["lat", "lon", "timestamp"]) and \
       all(k in prev_txn for k in ["lat", "lon", "timestamp"]):
        features["impossible_travel_score"] = impossible_travel_score(
            prev_txn["lat"], prev_txn["lon"], prev_txn["timestamp"],
            transaction["lat"], transaction["lon"], transaction["timestamp"],
        )
    else:
        features["impossible_travel_score"] = 0.0
    
    # 3. Network features
    if network:
        nf = network.get_features(
            transaction.get("card_id", ""),
            transaction.get("device_id", ""),
            transaction.get("merchant_id", ""),
        )
        features.update(nf)
    else:
        features["card_device_count"] = 0
        features["card_merchant_count"] = 0
        features["card_ip_count"] = 0
        features["device_card_count"] = 0
        features["device_merchant_count"] = 0
        features["merchant_card_count"] = 0
        features["cards_per_device"] = 0
        features["devices_per_card"] = 0
        features["connected_fraud_rate"] = 0.0
    
    # 4. Session behavioral features
    if session:
        sf = session.get_features()
        features.update(sf)
    else:
        features["session_length"] = 0
        features["session_unique_merchants"] = 0
        features["session_amount_escalation"] = 0
        features["session_city_switches"] = 0
        features["session_speed"] = 0
        features["session_amount_cv"] = 0
    
    # 5. Adaptive context risk
    features["context_risk"] = AdaptiveThresholds.compute_context_risk(
        base_risk=transaction.get("ml_score", 0.5),
        hour=transaction.get("hour", 12),
        mcc=transaction.get("mcc", ""),
        is_new_device=transaction.get("is_new_device", False),
        is_new_merchant=transaction.get("is_new_merchant", False),
        account_age_days=transaction.get("account_age_days", 365),
    )
    
    # 6. Class imbalance info (for training, not inference)
    features["is_high_risk_mcc"] = float(
        transaction.get("mcc", "") in AdaptiveThresholds.HIGH_RISK_MCC
    )
    
    return features


# Industry-inspired feature names for documentation
INDUSTRY_FEATURES = [
    # Device fingerprinting (Stripe Radar)
    "device_spoofing_score",
    "device_consistent",
    
    # Impossible travel (Visa/Stripe)
    "impossible_travel_score",
    
    # Network/Graph (GNN-inspired)
    "card_device_count",
    "card_merchant_count",
    "card_ip_count",
    "device_card_count",
    "device_merchant_count",
    "merchant_card_count",
    "cards_per_device",
    "devices_per_card",
    "connected_fraud_rate",
    
    # Behavioral session (PayPal/Feedzai)
    "session_length",
    "session_unique_merchants",
    "session_amount_escalation",
    "session_city_switches",
    "session_speed",
    "session_amount_cv",
    
    # Adaptive context
    "context_risk",
    "is_high_risk_mcc",
]
