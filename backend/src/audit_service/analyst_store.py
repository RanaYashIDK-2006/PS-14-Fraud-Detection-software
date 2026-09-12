"""Per-analyst credential store for compliance access.

Provides username+passphrase authentication for compliance viewers,
replacing the shared COMPLIANCE_TOKEN approach.

Storage: db/analysts.json (minimal fields only)
- username: analyst identifier
- passphrase_hash: scrypt hash of passphrase
- created_at: timestamp
- last_login: timestamp
- login_count: counter

The first analyst is bootstrapped from ANALYST_PASS env var or a default.
Additional analysts can be added via the admin API.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import time
from pathlib import Path
from typing import Optional

# Scrypt parameters (N=16384, r=8, p=1 — fast enough for auth, slow enough to resist brute-force)
_N = 16384
_R = 8
_P = 1


def _hash_passphrase(passphrase: str, salt: bytes | None = None) -> tuple[bytes, bytes]:
    """Hash a passphrase with scrypt. Returns (hash, salt)."""
    if salt is None:
        salt = os.urandom(16)
    dk = hashlib.scrypt(passphrase.encode(), salt=salt, n=_N, r=_R, p=_P, dklen=32)
    return dk, salt


def _verify_passphrase(passphrase: str, stored_hash: bytes, salt: bytes) -> bool:
    """Verify a passphrase against a stored hash."""
    dk, _ = _hash_passphrase(passphrase, salt)
    return secrets.compare_digest(dk, stored_hash)


class AnalystStore:
    """Simple file-based analyst credential store."""

    def __init__(self, path: Path):
        self.path = path
        self._analysts: dict[str, dict] = {}
        self._load()

    def _load(self):
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                self._analysts = data.get("analysts", {})
            except (json.JSONDecodeError, OSError):
                self._analysts = {}

    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {"analysts": self._analysts}
        self.path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def bootstrap(self, username: str, passphrase: str):
        """Create initial analyst if none exist."""
        if not self._analysts:
            h, salt = _hash_passphrase(passphrase)
            self._analysts[username] = {
                "passphrase_hash": h.hex(),
                "salt": salt.hex(),
                "created_at": time.time(),
                "last_login": None,
                "login_count": 0,
            }
            self._save()

    def verify(self, username: str, passphrase: str) -> bool:
        """Verify analyst credentials. Updates last_login on success."""
        analyst = self._analysts.get(username)
        if not analyst:
            return False
        stored_hash = bytes.fromhex(analyst["passphrase_hash"])
        salt = bytes.fromhex(analyst["salt"])
        if _verify_passphrase(passphrase, stored_hash, salt):
            analyst["last_login"] = time.time()
            analyst["login_count"] = analyst.get("login_count", 0) + 1
            self._save()
            return True
        return False

    def add_analyst(self, username: str, passphrase: str) -> bool:
        """Add a new analyst. Returns False if username exists."""
        if username in self._analysts:
            return False
        h, salt = _hash_passphrase(passphrase)
        self._analysts[username] = {
            "passphrase_hash": h.hex(),
            "salt": salt.hex(),
            "created_at": time.time(),
            "last_login": None,
            "login_count": 0,
        }
        self._save()
        return True

    def list_analysts(self) -> list[dict]:
        """List analysts (without sensitive data)."""
        return [
            {
                "username": u,
                "created_at": a.get("created_at"),
                "last_login": a.get("last_login"),
                "login_count": a.get("login_count", 0),
            }
            for u, a in self._analysts.items()
        ]
