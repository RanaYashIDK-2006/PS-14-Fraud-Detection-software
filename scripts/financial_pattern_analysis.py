#!/usr/bin/env python3
"""Financial fraud pattern analysis - identifies patterns for improved detection."""
import json
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "transactions.csv"


def analyze_patterns():
    """Analyze fraud patterns to identify detection opportunities."""
    df = pd.read_csv(DATA)
    print(f"Dataset: {len(df)} transactions, {df['label'].sum()} fraud ({df['label'].mean()*100:.2f}%)")
    
    fraud = df[df['label'] == 1]
    legit = df[df['label'] == 0]
    
    print("\n=== FRAUD PATTERNS ===")
    
    # 1. Amount patterns
    print("\n1. AMOUNT PATTERNS:")
    print(f"   Fraud amount_ratio: mean={fraud['amount_ratio'].mean():.2f}, median={fraud['amount_ratio'].median():.2f}")
    print(f"   Legit amount_ratio: mean={legit['amount_ratio'].mean():.2f}, median={legit['amount_ratio'].median():.2f}")
    
    # 2. Time patterns
    print("\n2. TIME PATTERNS:")
    print(f"   Fraud unusual time: {(fraud['txn_time_unusual']==1).mean()*100:.1f}%")
    print(f"   Legit unusual time: {(legit['txn_time_unusual']==1).mean()*100:.1f}%")
    
    # 3. Device patterns
    print("\n3. DEVICE PATTERNS:")
    print(f"   Fraud new device: {(fraud['new_device_flag']==1).mean()*100:.1f}%")
    print(f"   Legit new device: {(legit['new_device_flag']==1).mean()*100:.1f}%")
    
    # 4. Location patterns
    print("\n4. LOCATION PATTERNS:")
    print(f"   Fraud unusual location: {(fraud['unusual_location_flag']==1).mean()*100:.1f}%")
    print(f"   Legit unusual location: {(legit['unusual_location_flag']==1).mean()*100:.1f}%")
    
    # 5. Recipient patterns
    print("\n5. RECIPIENT PATTERNS:")
    print(f"   Fraud unusual recipient: {(fraud['unusual_recipient_flag']==1).mean()*100:.1f}%")
    print(f"   Legit unusual recipient: {(legit['unusual_recipient_flag']==1).mean()*100:.1f}%")
    
    # 6. Frequency patterns
    print("\n6. FREQUENCY PATTERNS:")
    print(f"   Fraud avg freq: {fraud['txn_freq_last_24h'].mean():.1f}")
    print(f"   Legit avg freq: {legit['txn_freq_last_24h'].mean():.1f}")
    
    # 7. Auth failure patterns
    print("\n7. AUTH FAILURE PATTERNS:")
    print(f"   Fraud avg failures: {fraud['failed_auth_count_24h'].mean():.1f}")
    print(f"   Legit avg failures: {legit['failed_auth_count_24h'].mean():.1f}")
    
    # 8. Escalation patterns
    print("\n8. ESCALATION PATTERNS:")
    print(f"   Fraud avg escalation: {fraud['gradual_escalation_score'].mean():.3f}")
    print(f"   Legit avg escalation: {legit['gradual_escalation_score'].mean():.3f}")
    
    # 9. Link analysis patterns
    print("\n9. LINK ANALYSIS PATTERNS:")
    print(f"   Fraud shared devices: {fraud['shared_device_accounts'].mean():.2f}")
    print(f"   Legit shared devices: {legit['shared_device_accounts'].mean():.2f}")
    print(f"   Fraud shared recipients: {fraud['shared_recipient_accounts'].mean():.2f}")
    print(f"   Legit shared recipients: {legit['shared_recipient_accounts'].mean():.2f}")
    print(f"   Fraud mule ring score: {fraud['mule_ring_score'].mean():.3f}")
    print(f"   Legit mule ring score: {legit['mule_ring_score'].mean():.3f}")
    
    # 10. Multi-signal patterns
    print("\n10. MULTI-SIGNAL PATTERNS:")
    fraud_signals = fraud[['new_device_flag', 'unusual_location_flag', 'unusual_recipient_flag', 'txn_time_unusual']].sum(axis=1)
    legit_signals = legit[['new_device_flag', 'unusual_location_flag', 'unusual_recipient_flag', 'txn_time_unusual']].sum(axis=1)
    print(f"   Fraud avg signals: {fraud_signals.mean():.2f}")
    print(f"   Legit avg signals: {legit_signals.mean():.2f}")
    print(f"   Fraud with 3+ signals: {(fraud_signals>=3).mean()*100:.1f}%")
    print(f"   Legit with 3+ signals: {(legit_signals>=3).mean()*100:.1f}%")
    
    # 11. Micro-transaction patterns
    print("\n11. MICRO-TRANSACTION PATTERNS:")
    micro_fraud = fraud[fraud['amount_ratio'] < 0.5]
    print(f"   Fraud with ratio < 0.5: {len(micro_fraud)} ({len(micro_fraud)/len(fraud)*100:.1f}%)")
    print(f"   Micro-fraud avg freq: {micro_fraud['txn_freq_last_24h'].mean():.1f}")
    print(f"   Micro-fraud with new device: {(micro_fraud['new_device_flag']==1).mean()*100:.1f}%")
    
    # 12. Night fraud patterns
    print("\n12. NIGHT FRAUD PATTERNS:")
    night_fraud = fraud[fraud['hour_of_day'].isin([0,1,2,3,4,5])]
    print(f"   Fraud at night (0-5h): {len(night_fraud)} ({len(night_fraud)/len(fraud)*100:.1f}%)")
    print(f"   Night fraud avg amount_ratio: {night_fraud['amount_ratio'].mean():.2f}")
    print(f"   Night fraud with new device: {(night_fraud['new_device_flag']==1).mean()*100:.1f}%")
    
    # 13. Weekend fraud patterns
    print("\n13. WEEKEND FRAUD PATTERNS:")
    weekend_fraud = fraud[fraud['is_weekend'] == 1]
    print(f"   Fraud on weekend: {len(weekend_fraud)} ({len(weekend_fraud)/len(fraud)*100:.1f}%)")
    
    # 14. Account age patterns
    print("\n14. ACCOUNT AGE PATTERNS:")
    print(f"   Fraud avg tenure: {fraud['account_tenure_days'].mean():.1f} days")
    print(f"   Legit avg tenure: {legit['account_tenure_days'].mean():.1f} days")
    young_fraud = fraud[fraud['account_tenure_days'] <= 7]
    print(f"   Fraud with tenure <= 7 days: {len(young_fraud)} ({len(young_fraud)/len(fraud)*100:.1f}%)")
    
    # 15. Correlation analysis
    print("\n15. FEATURE CORRELATIONS WITH FRAUD:")
    features = ['amount_ratio', 'txn_freq_last_24h', 'txn_time_unusual', 'new_device_flag', 
                'unusual_location_flag', 'unusual_recipient_flag', 'failed_auth_count_24h',
                'gradual_escalation_score', 'shared_device_accounts', 'shared_recipient_accounts']
    for f in features:
        corr = df[f].corr(df['label'])
        print(f"   {f}: {corr:.3f}")
    
    # 16. Optimal detection thresholds
    print("\n16. OPTIMAL DETECTION THRESHOLDS:")
    for threshold in [0.3, 0.5, 1.0, 1.5, 2.0]:
        caught = fraud[fraud['amount_ratio'] >= threshold]
        fp = legit[legit['amount_ratio'] >= threshold]
        print(f"   amount_ratio >= {threshold}: {len(caught)}/{len(fraud)} fraud ({len(caught)/len(fraud)*100:.1f}%), {len(fp)} FP")
    
    # Save analysis results
    results = {
        "fraud_count": len(fraud),
        "legit_count": len(legit),
        "fraud_rate": float(df['label'].mean()),
        "patterns": {
            "amount_ratio": {"fraud_mean": float(fraud['amount_ratio'].mean()), "legit_mean": float(legit['amount_ratio'].mean())},
            "new_device": {"fraud_pct": float((fraud['new_device_flag']==1).mean()), "legit_pct": float((legit['new_device_flag']==1).mean())},
            "unusual_time": {"fraud_pct": float((fraud['txn_time_unusual']==1).mean()), "legit_pct": float((legit['txn_time_unusual']==1).mean())},
            "unusual_location": {"fraud_pct": float((fraud['unusual_location_flag']==1).mean()), "legit_pct": float((legit['unusual_location_flag']==1).mean())},
            "unusual_recipient": {"fraud_pct": float((fraud['unusual_recipient_flag']==1).mean()), "legit_pct": float((legit['unusual_recipient_flag']==1).mean())},
            "multi_signal": {"fraud_3plus": float((fraud_signals>=3).mean()), "legit_3plus": float((legit_signals>=3).mean())},
        }
    }
    
    with open(ROOT / "data" / "financial_pattern_analysis.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\n✓ Analysis saved to data/financial_pattern_analysis.json")


if __name__ == "__main__":
    analyze_patterns()
