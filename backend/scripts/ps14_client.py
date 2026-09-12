"""PS-14 API Client — easy calling interface for all services.

Usage:
    from scripts.ps14_client import PS14Client

    client = PS14Client()

    # Evaluate a transaction (simple interface)
    result = client.evaluate_transaction(
        amount_ratio=1.5,
        txn_freq_last_24h=2,
        new_device_flag=1,
        unusual_recipient_flag=1,
    )
    print(f"Score: {result['risk_score']}, Decision: {result['decision']}")

    # Batch evaluate
    results = client.batch_evaluate([
        {"amount_ratio": 1.0, "txn_freq_last_24h": 1},
        {"amount_ratio": 5.0, "txn_freq_last_24h": 10},
    ])

    # Quick risk check (returns just score + decision)
    score, decision = client.quick_check(amount_ratio=2.0, new_device_flag=1)

    # Get service health
    health = client.health_all()

CLI mode:
    python scripts/ps14_client.py evaluate --amount-ratio 3.5 --new-device --fraud
    python scripts/ps14_client.py health
    python scripts/ps14_client.py batch --count 10
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

# Load tokens from .env
def _load_tokens() -> dict:
    tokens = {}
    env_path = Path(__file__).resolve().parent.parent.parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                tokens[k.strip()] = v.strip()
    return tokens

_TOKENS = _load_tokens()
INTERNAL_TOKEN = _TOKENS.get("INTERNAL_TOKEN", "ps14-dev-internal-token-replace-in-production")
COMPLIANCE_TOKEN = _TOKENS.get("COMPLIANCE_TOKEN", "ps14-dev-compliance-token-replace-in-prod")

# Default ports
RISK_PORT = 8003
PRIVACY_PORT = 8002
FRONT_PORT = 8000
VERIFY_PORT = 8004
AUDIT_PORT = 8005
IDENTITY_PORT = 8001

SERVICE_PORTS = {
    "risk": RISK_PORT,
    "privacy": PRIVACY_PORT,
    "front": FRONT_PORT,
    "verify": VERIFY_PORT,
    "audit": AUDIT_PORT,
    "identity": IDENTITY_PORT,
}

ML_FEATURES = [
    "amount_ratio", "txn_freq_last_24h", "txn_time_unusual", "new_device_flag",
    "unusual_location_flag", "unusual_recipient_flag", "failed_auth_count_24h",
    "days_since_last_similar_txn", "gradual_escalation_score", "known_device_count",
    "account_tenure_days", "hour_of_day", "is_weekend",
    "shared_device_accounts", "shared_recipient_accounts", "mule_ring_score",
]


def _post(url: str, data: dict, token: str | None = None, timeout: int = 10) -> dict:
    """POST JSON to a service endpoint."""
    body = json.dumps(data).encode()
    headers = {"Content-Type": "application/json"}
    if token:
        headers["X-Internal-Token"] = token
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        try:
            return json.loads(body)
        except Exception:
            return {"error": e.code, "detail": body}


def _get(url: str, token: str | None = None, timeout: int = 5) -> dict:
    """GET from a service endpoint."""
    headers = {}
    if token:
        headers["X-Internal-Token"] = token
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        return {"error": e.code}
    except Exception as e:
        return {"error": str(e)}


class PS14Client:
    """Easy-to-use client for the PS-14 fraud detection system."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        internal_token: str | None = None,
        compliance_token: str | None = None,
    ):
        self.host = host
        self.internal_token = internal_token or INTERNAL_TOKEN
        self.compliance_token = compliance_token or COMPLIANCE_TOKEN

    def _url(self, service: str, path: str) -> str:
        port = SERVICE_PORTS.get(service, 8003)
        return f"http://{self.host}:{port}{path}"

    # ── Risk Engine ──────────────────────────────────────────────

    def evaluate_transaction(self, **features) -> dict:
        """Evaluate a single transaction. Pass ML features as keyword args.

        All features default to safe/benign values if not specified:
            amount_ratio=1.0, txn_freq_last_24h=1, new_device_flag=0, etc.

        Returns full risk assessment with score, decision, reason codes.
        """
        feature_values = {f: 0 for f in ML_FEATURES}
        feature_values["amount_ratio"] = 1.0
        feature_values["account_tenure_days"] = 30.0
        feature_values["days_since_last_similar_txn"] = 5.0
        feature_values["known_device_count"] = 3
        feature_values.update(features)

        # Convert int flags
        for f in ["txn_time_unusual", "new_device_flag", "unusual_location_flag",
                   "unusual_recipient_flag", "is_weekend"]:
            feature_values[f] = int(feature_values[f])

        data = {
            "event_id": f"api-{int(time.time()*1000)}-{random.randint(1000,9999)}",
            "fraud_id": f"F{random.choice('ABCDEFGHJKLMNPQRSTUVWXYZ23456789') * 15}",
            "features": feature_values,
        }
        url = self._url("risk", "/internal/evaluate")
        return _post(url, data, token=self.internal_token)

    def quick_check(self, **features) -> tuple[int, str]:
        """Quick risk check. Returns (risk_score, decision).

        >>> score, decision = client.quick_check(amount_ratio=3.0, new_device_flag=1)
        """
        result = self.evaluate_transaction(**features)
        return result.get("risk_score", 0), result.get("decision", "unknown")

    def batch_evaluate(self, transactions: list[dict], delay: float = 0.01) -> list[dict]:
        """Evaluate multiple transactions. Each dict should have feature key-value pairs.

        >>> results = client.batch_evaluate([
        ...     {"amount_ratio": 1.0},
        ...     {"amount_ratio": 5.0, "new_device_flag": 1},
        ... ])
        """
        results = []
        for i, txn in enumerate(transactions):
            features = {f: txn.get(f, 0) for f in ML_FEATURES}
            features["amount_ratio"] = txn.get("amount_ratio", 1.0)
            features["account_tenure_days"] = txn.get("account_tenure_days", 30.0)
            features["days_since_last_similar_txn"] = txn.get("days_since_last_similar_txn", 5.0)
            features["known_device_count"] = txn.get("known_device_count", 3)
            data = {
                "event_id": f"batch-{int(time.time()*1000)}-{i:04d}",
                "fraud_id": f"F{random.choice('ABCDEFGHJKLMNPQRSTUVWXYZ23456789') * 15}",
                "features": features,
            }
            url = self._url("risk", "/internal/evaluate")
            result = _post(url, data, token=self.internal_token)
            results.append(result)
            if delay > 0:
                time.sleep(delay)
        return results

    # ── Health Checks ────────────────────────────────────────────

    def health(self, service: str = "risk") -> dict:
        """Check health of a specific service."""
        url = self._url(service, "/health")
        return _get(url)

    def health_all(self) -> dict[str, dict]:
        """Check health of all services. Returns {service_name: health_dict}."""
        results = {}
        for name in SERVICE_PORTS:
            results[name] = self.health(name)
        return results

    def monitor_metrics(self) -> dict:
        """Get live monitor metrics."""
        url = self._url("front", "/monitor/metrics")
        return _get(url)

    # ── Fraud Report ─────────────────────────────────────────────

    def fraud_report(self) -> dict:
        """Get the fraud report data."""
        url = self._url("front", "/fraud-report")
        return _get(url)

    # ── Audit ────────────────────────────────────────────────────

    def audit_integrity(self) -> dict:
        """Check audit chain integrity."""
        url = self._url("audit", "/audit/integrity")
        return _get(url, token=self.internal_token)

    # ── Chain Status ─────────────────────────────────────────────

    def chain_status(self) -> dict:
        """Check audit chain status."""
        url = self._url("audit", "/chain-status")
        return _get(url)

    # ── Test Runner ──────────────────────────────────────────────

    def run_tests(self) -> dict:
        """Trigger a test run via the monitor."""
        url = self._url("front", "/monitor/run-tests")
        return _post(url, {}, token=self.internal_token)


# ── CLI ──────────────────────────────────────────────────────────

def _generate_scenario(is_fraud: bool = False) -> dict:
    """Generate a realistic transaction scenario."""
    if is_fraud:
        return {
            "amount_ratio": round(random.uniform(2.0, 8.0), 2),
            "txn_freq_last_24h": random.randint(3, 12),
            "txn_time_unusual": 1,
            "new_device_flag": 1,
            "unusual_location_flag": random.choice([0, 1]),
            "unusual_recipient_flag": 1,
            "failed_auth_count_24h": random.randint(0, 5),
            "days_since_last_similar_txn": round(random.uniform(15, 60), 1),
            "gradual_escalation_score": round(random.uniform(0.3, 0.8), 2),
            "known_device_count": random.randint(0, 2),
            "account_tenure_days": round(random.uniform(0.5, 14), 1),
            "hour_of_day": random.choice([1, 2, 3, 4, 23]),
            "is_weekend": random.choice([0, 1]),
            "shared_device_accounts": random.randint(0, 3),
            "shared_recipient_accounts": random.randint(0, 2),
            "mule_ring_score": round(random.uniform(0.05, 0.5), 2),
        }
    else:
        return {
            "amount_ratio": round(random.uniform(0.3, 2.0), 2),
            "txn_freq_last_24h": random.randint(0, 3),
            "txn_time_unusual": 0,
            "new_device_flag": 0,
            "unusual_location_flag": 0,
            "unusual_recipient_flag": 0,
            "failed_auth_count_24h": 0,
            "days_since_last_similar_txn": round(random.uniform(1, 30), 1),
            "gradual_escalation_score": round(random.uniform(0.0, 0.15), 2),
            "known_device_count": random.randint(2, 8),
            "account_tenure_days": round(random.uniform(30, 365), 1),
            "hour_of_day": random.randint(8, 21),
            "is_weekend": random.choice([0, 1]),
            "shared_device_accounts": 0,
            "shared_recipient_accounts": 0,
            "mule_ring_score": 0.0,
        }


def main():
    parser = argparse.ArgumentParser(description="PS-14 API Client")
    sub = parser.add_subparsers(dest="command")

    # evaluate
    ev = sub.add_parser("evaluate", help="Evaluate a transaction")
    ev.add_argument("--amount-ratio", type=float, default=1.0)
    ev.add_argument("--freq", type=int, default=1, dest="txn_freq")
    ev.add_argument("--new-device", action="store_true")
    ev.add_argument("--unusual-loc", action="store_true")
    ev.add_argument("--unusual-rcpt", action="store_true")
    ev.add_argument("--failed-auth", type=int, default=0)
    ev.add_argument("--fraud", action="store_true", help="Generate fraud scenario")

    # health
    sub.add_parser("health", help="Check all service health")

    # batch
    ba = sub.add_parser("batch", help="Send batch transactions")
    ba.add_argument("--count", type=int, default=20)
    ba.add_argument("--fraud-pct", type=float, default=0.3)

    # quick
    qu = sub.add_parser("quick", help="Quick risk check")
    qu.add_argument("--amount-ratio", type=float, default=1.0)
    qu.add_argument("--new-device", action="store_true")

    args = parser.parse_args()
    client = PS14Client()

    if args.command == "evaluate":
        if args.fraud:
            features = _generate_scenario(is_fraud=True)
            print(f"[GENERATED FRAUD SCENARIO]")
        else:
            features = {
                "amount_ratio": args.amount_ratio,
                "txn_freq_last_24h": args.txn_freq,
                "new_device_flag": int(args.new_device),
                "unusual_location_flag": int(args.unusual_loc),
                "unusual_recipient_flag": int(args.unusual_rcpt),
                "failed_auth_count_24h": args.failed_auth,
            }
        result = client.evaluate_transaction(**features)
        print(json.dumps(result, indent=2))

    elif args.command == "health":
        results = client.health_all()
        for name, h in results.items():
            status = h.get("status", "DEAD")
            icon = "✅" if status == "ok" else "❌"
            print(f"  {icon} {name:10s}: {status}")

    elif args.command == "batch":
        n_fraud = int(args.count * args.fraud_pct)
        n_normal = args.count - n_fraud
        txns = [_generate_scenario(False) for _ in range(n_normal)] + \
               [_generate_scenario(True) for _ in range(n_fraud)]
        random.shuffle(txns)

        print(f"Sending {args.count} transactions ({n_normal} normal, {n_fraud} fraud)...")
        t0 = time.time()
        results = client.batch_evaluate(txns, delay=0.05)
        elapsed = time.time() - t0

        fraud_detected = sum(1 for r in results if r.get("decision") == "verify")
        normal_flagged = 0
        for i, r in enumerate(results):
            if i < n_normal and r.get("decision") == "verify":
                normal_flagged += 1

        print(f"\nResults ({elapsed:.1f}s, {args.count/elapsed:.1f} TPS):")
        print(f"  Fraud detected: {fraud_detected}/{n_fraud} ({fraud_detected/max(n_fraud,1)*100:.1f}%)")
        print(f"  False positives: {normal_flagged}/{n_normal} ({normal_flagged/max(n_normal,1)*100:.1f}%)")
        print(f"  Avg latency: {elapsed/args.count*1000:.1f}ms")

    elif args.command == "quick":
        score, decision = client.quick_check(
            amount_ratio=args.amount_ratio,
            new_device_flag=int(args.new_device),
        )
        print(f"Score: {score}/100 → {decision.upper()}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
