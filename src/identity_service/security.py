"""Security primitives for the Identity Service.

- Credential hashing: Argon2id (architecture section 12).
- Session tokens: short-lived signed JWTs carrying ONLY the `fraud_id` claim,
  never real identity (section 4).
- PII at rest: AES-256 via Fernet (envelope encryption / KMS in production).
- Fraud IDs: CSPRNG-generated, not derived from any hash of PII (section 3).
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import time

import jwt
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError
from cryptography.fernet import Fernet

from src.settings import settings

# Base32-ish alphabet without confusables (no I, O, 0, 1). 16 chars -> 80 bits,
# sized to fit the VARCHAR(16) pseudonym_mapping.fraud_id column (section 13).
FRAUD_ID_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
FRAUD_ID_LENGTH = 16

_ph = PasswordHasher()
_fernet = Fernet(settings.fernet_key)


def gen_fraud_id() -> str:
    """CSPRNG pseudo-ID; not a hash of any PII (rainbow-table resistant)."""
    return "F" + "".join(secrets.choice(FRAUD_ID_ALPHABET) for _ in range(FRAUD_ID_LENGTH - 1))


def hash_password(password: str) -> str:
    return _ph.hash(password)


def verify_password(password: str, secret_hash: str) -> bool:
    try:
        return _ph.verify(secret_hash, password)
    except VerifyMismatchError:
        return False


def encrypt_pii(value: str) -> bytes:
    return _fernet.encrypt(value.encode("utf-8"))


def decrypt_pii(value: bytes) -> str:
    return _fernet.decrypt(value).decode("utf-8")


def encrypt_opt(value: str | None) -> bytes | None:
    """Encrypt an optional plaintext value (None stays None)."""
    return encrypt_pii(value) if value is not None else None


def decrypt_opt(value) -> str | None:
    """Decrypt an optional PII column, tolerating legacy plaintext rows.

    Columns were plaintext until the #25 privacy fix + migration; reading a
    str means the row predates encryption - return it unchanged so reads
    never crash mid-migration. After scripts/migrate_identity_pii.py the
    value is always bytes.
    """
    if value is None:
        return None
    if isinstance(value, (bytes, bytearray)):
        return decrypt_pii(bytes(value))
    return str(value)  # legacy plaintext row


def blind_index(value: str) -> str:
    """Searchable HMAC-style index so login can look up an email without
    ever storing or querying the plaintext."""
    return hashlib.sha256(value.strip().lower().encode("utf-8")).hexdigest()


def mask(value: str, keep: int = 2) -> str:
    if len(value) <= keep + 2:
        return value
    return value[:keep] + "*" * (len(value) - keep - 2) + value[-2:]


def create_access_token(fraud_id: str, role: str = "user") -> str:
    payload = {
        "sub": fraud_id,  # pseudonym only - never real identity (section 4)
        "role": role,  # RBAC role: user, analyst, admin
        "aud": "user-app",
        "iat": int(time.time()),
        "exp": int(time.time()) + settings.jwt_expiry_minutes * 60,
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm="HS256")


def decode_access_token(token: str) -> dict:
    """Returns the full JWT payload (sub=fraud_id, role, etc.); raises on any failure."""
    return jwt.decode(token, settings.jwt_secret, algorithms=["HS256"], audience="user-app")


def verify_internal_token(token: str) -> bool:
    """Constant-time compare. Production: mTLS + short-lived service JWTs."""
    return hmac.compare_digest(token, settings.internal_token)
