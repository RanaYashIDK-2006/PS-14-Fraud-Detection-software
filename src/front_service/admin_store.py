"""Minimal, encrypted admin store for the front page (data minimization).

ONLY essential fields are persisted (`db/admin.json`):
  * username,
  * scrypt hash of the admin passphrase (the passphrase itself is NEVER
    stored — verifiable but not recoverable),
  * the essential access payload, encrypted with AES-256-GCM under a random
    key that is itself WRAPPED with a key derived from the passphrase
    (scrypt). Nothing in the file is readable without the passphrase.
  * created_at / last_login / login_count (minimal operational fields).

Crypto: AES-256-GCM (authenticated encryption — tamper-evident) with keys
derived via scrypt (memory-hard KDF, salt per purpose). This is the
strongest scheme available from the already-installed `cryptography` lib.
"""

from __future__ import annotations

import base64
import hmac
import json
import os
import secrets
from datetime import datetime, timezone
from pathlib import Path

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt

SCRYPT_N, SCRYPT_R, SCRYPT_P = 2**14, 8, 1
SALT_LEN = 16
NONCE_LEN = 12


def _b64(b: bytes) -> str:
    return base64.b64encode(b).decode()


def _unb64(s: str) -> bytes:
    return base64.b64decode(s)


def _derive_key(passphrase: str, salt: bytes) -> bytes:
    """scrypt(passphrase, salt) -> 32-byte AES-256 key (memory-hard KDF)."""
    return Scrypt(salt=salt, length=32, n=SCRYPT_N, r=SCRYPT_R, p=SCRYPT_P).derive(
        passphrase.encode("utf-8")
    )


def _verify_key(passphrase: str, salt: bytes, expected: bytes) -> bool:
    try:
        return hmac.compare_digest(_derive_key(passphrase, salt), expected)
    except Exception:
        return False


def _aes_gcm(key: bytes, nonce: bytes, data: bytes) -> bytes:
    return AESGCM(key).encrypt(nonce, data, None)


def _aes_gcm_decrypt(key: bytes, nonce: bytes, ct: bytes) -> bytes:
    return AESGCM(key).decrypt(nonce, ct, None)


class AdminStore:
    """File-backed store: minimal fields + an encrypted essentials blob."""

    def __init__(self, path: Path):
        self.path = path

    # ------------------------------------------------------------------ load/save
    def load(self) -> dict | None:
        if not self.path.exists():
            return None
        return json.loads(self.path.read_text(encoding="utf-8"))

    def save(self, record: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(record, indent=2), encoding="utf-8")
        os.replace(tmp, self.path)

    # ------------------------------------------------------------------ lifecycle
    def bootstrap(self, username: str, passphrase: str, essentials: dict) -> dict:
        """First-time setup: create the admin record from the ADMIN_PASS env."""
        blob_key = secrets.token_bytes(32)
        pass_salt = secrets.token_bytes(SALT_LEN)
        wrap_salt = secrets.token_bytes(SALT_LEN)
        wrap_nonce = secrets.token_bytes(NONCE_LEN)
        record = {
            "username": username,
            "pass_salt": _b64(pass_salt),
            "pass_hash": _b64(_derive_key(passphrase, pass_salt)),
            "wrap": {
                "salt": _b64(wrap_salt),
                "nonce": _b64(wrap_nonce),
                "ct": _b64(_aes_gcm(_derive_key(passphrase, wrap_salt),
                                    wrap_nonce, blob_key)),
            },
            "essential": self._encrypt_essentials(blob_key, essentials),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "last_login": None,
            "login_count": 0,
        }
        self.save(record)
        return record

    def verify_passphrase(self, passphrase: str) -> bool:
        record = self.load()
        if record is None:
            return False
        return _verify_key(passphrase, _unb64(record["pass_salt"]),
                           _unb64(record["pass_hash"]))

    def unwrap_blob_key(self, passphrase: str) -> bytes | None:
        """Recover the essentials-blob key with the passphrase (None if wrong)."""
        record = self.load()
        if record is None:
            return None
        wrap = record["wrap"]
        try:
            key = _derive_key(passphrase, _unb64(wrap["salt"]))
            return AESGCM(key).decrypt(_unb64(wrap["nonce"]), _unb64(wrap["ct"]), None)
        except Exception:
            return None

    def decrypt_essentials(self, blob_key: bytes) -> dict:
        record = self.load()
        ess = record["essential"]
        plain = _aes_gcm_decrypt(blob_key, _unb64(ess["nonce"]), _unb64(ess["ct"]))
        return json.loads(plain.decode("utf-8"))

    def touch_login(self) -> None:
        record = self.load()
        if record is None:
            return
        record["last_login"] = datetime.now(timezone.utc).isoformat()
        record["login_count"] = int(record.get("login_count", 0)) + 1
        self.save(record)

    # ------------------------------------------------------------------ TOTP
    def _totp_enc_key(self) -> bytes:
        """Derive an AES-256 key from the system fernet key for TOTP encryption."""
        from src.settings import settings
        fk = settings.fernet_key.encode() if isinstance(settings.fernet_key, str) else settings.fernet_key
        return Scrypt(salt=b'totp-encrypt-v1', length=32, n=SCRYPT_N, r=SCRYPT_R, p=SCRYPT_P).derive(fk)

    def get_totp_secret(self) -> bytes | None:
        """Get the TOTP secret for the admin user (decrypts from AES-256-GCM at rest)."""
        record = self.load()
        if record is None:
            return None
        totp_data = record.get("totp")
        if totp_data is None:
            return None
        raw = totp_data.get("secret")
        if not raw:
            return None
        # Support both encrypted (v1) and legacy plaintext formats
        if totp_data.get("encrypted"):
            key = self._totp_enc_key()
            return _aes_gcm_decrypt(key, _unb64(totp_data["nonce"]), _unb64(raw))
        # Legacy plaintext — re-encrypt on next set_totp_secret call
        return _unb64(raw)

    def set_totp_secret(self, secret: bytes, *, enabled: bool = True) -> None:
        """Store the TOTP secret, encrypted with AES-256-GCM at rest.
        Pass enabled=False during setup so the secret is saved but TOTP
        is not enforced until verify completes."""
        record = self.load()
        if record is None:
            return
        key = self._totp_enc_key()
        nonce = os.urandom(NONCE_LEN)
        ct = _aes_gcm(key, nonce, secret)
        record["totp"] = {
            "secret": _b64(ct),
            "nonce": _b64(nonce),
            "encrypted": True,
            "enabled": enabled,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        self.save(record)

    def is_totp_enabled(self) -> bool:
        """Check if TOTP is enabled for the admin user."""
        record = self.load()
        if record is None:
            return False
        totp_data = record.get("totp")
        return totp_data is not None and totp_data.get("enabled", False)

    def rotate_passphrase(self, current: str, new: str) -> bool:
        """Re-hash and re-wrap the blob key under the NEW passphrase."""
        record = self.load()
        if record is None or not self.verify_passphrase(current):
            return False
        blob_key = self.unwrap_blob_key(current)
        if blob_key is None:
            return False
        record["pass_salt"] = _b64(secrets.token_bytes(SALT_LEN))
        record["pass_hash"] = _b64(_derive_key(new, _unb64(record["pass_salt"])))
        wrap_salt = secrets.token_bytes(SALT_LEN)
        wrap_nonce = secrets.token_bytes(NONCE_LEN)
        record["wrap"] = {
            "salt": _b64(wrap_salt),
            "nonce": _b64(wrap_nonce),
            "ct": _b64(_aes_gcm(_derive_key(new, wrap_salt),
                                wrap_nonce, blob_key)),
        }
        self.save(record)
        return True

    @staticmethod
    def _encrypt_essentials(blob_key: bytes, essentials: dict) -> dict:
        nonce = secrets.token_bytes(NONCE_LEN)
        return {
            "nonce": _b64(nonce),
            "ct": _b64(_aes_gcm(blob_key, nonce, json.dumps(essentials).encode("utf-8"))),
        }
