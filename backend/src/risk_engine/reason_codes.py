"""Category-level reason codes (architecture section 11).

These are the ONLY explanations exposed to users - never model weights,
thresholds, or raw probabilities. Shared by the Risk Engine (which emits the
codes) and the Verification Service (which renders them).
"""

REASON_CODE_TEXT = {
    "AMOUNT_UNUSUAL": "Transaction amount is unusual for this account",
    "NEW_DEVICE": "New device detected",
    "UNUSUAL_TIME": "Unusual time of day for this account",
    "LOCATION_UNUSUAL": "Transaction originated from an unusual location",
    "RECIPIENT_UNUSUAL": "Unusual recipient for this account",
    "AUTH_ANOMALY": "Unusual authentication activity detected",
    "FREQUENCY_ABNORMAL": "Transaction frequency is abnormal",
    "BEHAVIOR_DEVIATION": "Behavior differs from your usual pattern",
    "MULE_RING": "Device or recipient shared with other accounts",
    "ACCOUNT_DAILY_LIMIT": "Too many transactions in 24 hours",
    "DEVICE_DAILY_LIMIT": "Device used for too many transactions in 24 hours",
    "DAILY_SPEND_EXCEEDED": "Daily spending exceeds normal pattern",
    "UNCERTAINTY_ESCALATION": "Low model confidence escalated to verify",
    "UNCERTAINTY_INVESTIGATION": "Low model confidence flagged for investigation",
    "DOMAIN_SHIFT": "Features outside expected training range — domain compatibility uncertain",
    # System-level codes (degraded/blocked paths). These are category-level
    # explanations, not model internals - safe for users, and required so the
    # verification/audit UIs render text instead of the raw code.
    "ML_UNAVAILABLE": "Automated model scoring was unavailable — decision based on standard rule checks",
    "DATA_QUALITY_BLOCKED": "Transaction data did not pass integrity checks — model scoring withheld",
}
