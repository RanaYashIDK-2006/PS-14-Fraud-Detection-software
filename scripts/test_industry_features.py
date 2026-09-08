#!/usr/bin/env python3
"""Test industry-inspired features."""
import sys
sys.path.insert(0, ".")
from src.risk_engine.industry_features import (
    DeviceFingerprint, impossible_travel_score, NetworkAnalyzer,
    SessionProfile, AdaptiveThresholds, extract_industry_features,
    INDUSTRY_FEATURES, compute_class_weights,
)
import numpy as np

print("=" * 60)
print("INDUSTRY FEATURES TEST")
print("=" * 60)

# 1. Device Fingerprinting
print("\n--- 1. Device Fingerprinting ---")
fp_clean = DeviceFingerprint(user_agent="Mozilla/5.0 Chrome/120", screen_width=1920, screen_height=1080, language="en-US", platform="Win32")
fp_bot = DeviceFingerprint(user_agent="python-requests/2.31", screen_width=0, screen_height=0)
fp_headless = DeviceFingerprint(user_agent="Mozilla/5.0 HeadlessChrome/120", screen_width=1920, screen_height=1080, language="en", platform="Linux")

print(f"  Clean device: spoof={fp_clean.spoofing_score():.2f} consistent={fp_clean.is_consistent()}")
print(f"  Bot: spoof={fp_bot.spoofing_score():.2f} consistent={fp_bot.is_consistent()}")
print(f"  Headless: spoof={fp_headless.spoofing_score():.2f}")
assert fp_clean.spoofing_score() < 0.1, "Clean device should have low spoof score"
assert fp_bot.spoofing_score() > 0.3, "Bot should have high spoof score"
assert fp_headless.spoofing_score() >= 0.5, "Headless should be detected"
print("  PASS")

# 2. Impossible Travel
print("\n--- 2. Impossible Travel ---")
# New York to London in 1 hour = impossible
score1 = impossible_travel_score(40.7, -74.0, 0, 51.5, -0.1, 3600)
# New York to London in 8 hours = possible (flight)
score2 = impossible_travel_score(40.7, -74.0, 0, 51.5, -0.1, 8 * 3600)
# Same location in 5 minutes = normal
score3 = impossible_travel_score(40.7, -74.0, 0, 40.7, -74.0, 300)
print(f"  NY→London 1hr: {score1:.4f} (should be high)")
print(f"  NY→London 8hr: {score2:.4f} (should be low)")
print(f"  Same loc 5min: {score3:.4f} (should be 0)")
assert score1 > 0.5, "Impossible travel should score high"
assert score2 < 0.5, "Possible travel should score low"
assert score3 == 0.0, "Same location should be 0"
print("  PASS")

# 3. Network Analysis
print("\n--- 3. Network Analysis ---")
net = NetworkAnalyzer()
net.add_transaction("card_1", "dev_a", "merch_x", "recip_1", "ip_1", is_fraud=False)
net.add_transaction("card_1", "dev_b", "merch_y", "recip_1", "ip_2", is_fraud=False)
net.add_transaction("card_2", "dev_a", "merch_x", "recip_2", "ip_1", is_fraud=True)  # Same device, different card = suspicious
nf = net.get_features("card_1", "dev_a", "merch_x")
print(f"  card_1 uses {nf['card_device_count']} devices, {nf['card_merchant_count']} merchants")
print(f"  dev_a has {nf['device_card_count']} cards (shared device = mule signal)")
print(f"  merchant_x has {nf['merchant_card_count']} cards")
assert nf["device_card_count"] >= 2, "Device shared by 2+ cards"
assert nf["card_device_count"] >= 2, "Card used on 2+ devices"
print("  PASS")

# 4. Session Behavioral
print("\n--- 4. Session Behavioral ---")
sess = SessionProfile()
sess.add_transaction(10.0, "merchant_a", "New York", 1000)
sess.add_transaction(25.0, "merchant_b", "New York", 1060)
sess.add_transaction(100.0, "merchant_c", "London", 1120)
sf = sess.get_features()
print(f"  Session: {sf['session_length']} txns, {sf['session_unique_merchants']} merchants")
print(f"  Amount escalation: {sf['session_amount_escalation']:.4f}")
print(f"  City switches: {sf['session_city_switches']}")
print(f"  Speed: {sf['session_speed']:.2f} txn/min")
assert sf["session_length"] == 3
assert sf["session_city_switches"] >= 1, "Should detect city change"
print("  PASS")

# 5. Adaptive Thresholds
print("\n--- 5. Adaptive Thresholds ---")
r1 = AdaptiveThresholds.compute_context_risk(0.5, hour=3, mcc="7995", is_new_device=True)
r2 = AdaptiveThresholds.compute_context_risk(0.5, hour=12, mcc="5411", is_new_device=False)
print(f"  Night + gambling + new device: {r1:.4f}")
print(f"  Day + grocery + known device: {r2:.4f}")
assert r1 > r2, "High-risk context should score higher"

e1 = AdaptiveThresholds.should_escalate(ml_score=0.9, rule_score=0, context_risk=0.3, confidence=0.95)
e2 = AdaptiveThresholds.should_escalate(ml_score=0.3, rule_score=0.5, context_risk=0.2, confidence=0.9)
e3 = AdaptiveThresholds.should_escalate(ml_score=0.7, rule_score=0, context_risk=0.3, confidence=0.6)
print(f"  High ML + confident: {e1} (should be block)")
print(f"  Rules triggered: {e2} (should be review/block)")
print(f"  High ML + low confidence: {e3} (should be step_up)")
assert e1 == "block", f"Expected block, got {e1}"
assert e3 == "step_up", f"Expected step_up, got {e3}"
print("  PASS")

# 6. Combined extraction
print("\n--- 6. Combined Feature Extraction ---")
tx = {"amount": 50.0, "hour": 3, "mcc": "7995", "device_id": "dev_a",
      "card_id": "card_1", "merchant_id": "merch_x", "user_agent": "Mozilla/5.0",
      "screen_width": 1920, "screen_height": 1080, "language": "en"}
features = extract_industry_features(tx, network=net, session=sess)
print(f"  Extracted {len(features)} features")
for fname in INDUSTRY_FEATURES:
    assert fname in features, f"Missing feature: {fname}"
print("  All industry features present")
print("  PASS")

# 7. Class weights
print("\n--- 7. Class Weights ---")
y = np.array([0]*9900 + [1]*100)  # 1% fraud
cw = compute_class_weights(y)
print(f"  Prevalence: {cw['prevalence']:.4f}")
print(f"  scale_pos_weight: {cw['scale_pos_weight']:.1f}")
assert cw["scale_pos_weight"] > 10, "Should handle heavy imbalance"
print("  PASS")

print(f"\n{'='*60}")
print("ALL 7 TESTS PASSED")
print(f"Industry features: {len(INDUSTRY_FEATURES)} features")
print(f"{'='*60}")
