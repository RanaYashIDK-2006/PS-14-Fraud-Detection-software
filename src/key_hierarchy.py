"""Key hierarchy for PII encryption — DEK wrapped by KEK.

In production, the KEK lives in a KMS/HSM and is never exposed to the
application. The DEK is generated per-service and wrapped (encrypted)
by the KEK for storage. This implements the envelope encryption pattern:

    plaintext → encrypt(DEK) → ciphertext
    DEK → encrypt(KEK) → encrypted_DEK (stored alongside ciphertext)

For the prototype, both keys are derived from environment variables,
but the separation is real: rotating the KEK re-wraps all DEKs without
re-encrypting data, and rotating a DEK re-encrypts only that service's data.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import secrets
import time
from pathlib import Path

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives.ciphers.aead import AESGCM


class KeyHierarchy:
    """Envelope encryption with DEK/KEK separation.

    The KEK wraps (encrypts) the DEK. The DEK encrypts actual data.
    In production, the KEK would be a KMS key ARN or HSM handle.
    """

    def __init__(
        self,
        kek_material: str | None = None,
        dek_material: str | None = None,
    ):
        """Initialize key hierarchy.

        Args:
            kek_material: Key Encryption Key material (production: KMS key ID)
            dek_material: Data Encryption Key material (service-specific)
        """
        # KEK: derived from kek_material or PII_KEK env var
        if kek_material is None:
            kek_material = os.environ.get("PII_KEK", "")
        if not kek_material:
            # In dev mode, derive from jwt_secret for backward compatibility
            from src.settings import settings
            kek_material = settings.jwt_secret

        self._kek = hashlib.sha256(f"kek:{kek_material}".encode()).digest()

        # DEK: derived from dek_material or PII_DEK env var
        if dek_material is None:
            dek_material = os.environ.get("PII_DEK", "")
        if not dek_material:
            # In dev mode, derive from pii_encryption_key for backward compat
            from src.settings import settings
            dek_material = settings.pii_encryption_key

        self._dek = hashlib.sha256(f"dek:{dek_material}".encode()).digest()

        # Fernet key derived from DEK (for backward compatibility with existing data)
        self._fernet_key = base64.urlsafe_b64encode(self._dek)

        # Wrapped DEK (encrypted by KEK) — stored in metadata
        self._wrapped_dek = self._wrap_key(self._dek, self._kek)

    @property
    def fernet_key(self) -> bytes:
        """Fernet key for backward-compatible encryption."""
        return self._fernet_key

    @property
    def wrapped_dek(self) -> str:
        """The DEK encrypted by the KEK — for storage in metadata."""
        return self._wrapped_dek

    @property
    def kek_fingerprint(self) -> str:
        """Fingerprint of the KEK for audit trail."""
        return hashlib.sha256(self._kek).hexdigest()[:16]

    @property
    def dek_fingerprint(self) -> str:
        """Fingerprint of the DEK for audit trail."""
        return hashlib.sha256(self._dek).hexdigest()[:16]

    def _wrap_key(self, key_to_wrap: bytes, wrapping_key: bytes) -> str:
        """Wrap (encrypt) a key using AES-256-GCM."""
        nonce = secrets.token_bytes(12)
        aesgcm = AESGCM(wrapping_key)
        ciphertext = aesgcm.encrypt(nonce, key_to_wrap, None)
        wrapped = nonce + ciphertext
        return base64.urlsafe_b64encode(wrapped).decode()

    def _unwrap_key(self, wrapped_key: str, wrapping_key: bytes) -> bytes:
        """Unwrap (decrypt) a key using AES-256-GCM."""
        raw = base64.urlsafe_b64decode(wrapped_key)
        nonce = raw[:12]
        ciphertext = raw[12:]
        aesgcm = AESGCM(wrapping_key)
        return aesgcm.decrypt(nonce, ciphertext, None)

    def re_wrap_dek(self, new_kek_material: str) -> str:
        """Re-wrap the DEK with a new KEK (for KEK rotation).

        This does NOT re-encrypt data — it only re-wraps the DEK.
        Data encrypted with the DEK remains valid.
        """
        new_kek = hashlib.sha256(f"kek:{new_kek_material}".encode()).digest()
        return self._wrap_key(self._dek, new_kek)

    def rotate_dek(self, new_dek_material: str) -> bytes:
        """Rotate the DEK (for DEK rotation).

        Returns the new Fernet key. Data must be re-encrypted with the new key.
        """
        new_dek = hashlib.sha256(f"dek:{new_dek_material}".encode()).digest()
        self._dek = new_dek
        self._fernet_key = base64.urlsafe_b64encode(new_dek)
        self._wrapped_dek = self._wrap_key(new_dek, self._kek)
        return self._fernet_key

    def get_audit_info(self) -> dict:
        """Return key metadata for audit trail (no key material)."""
        return {
            "kek_fingerprint": self.kek_fingerprint,
            "dek_fingerprint": self.dek_fingerprint,
            "wrapped_dek": self.wrapped_dek,
            "encryption": "AES-256-GCM (wrap) + Fernet/AES-256-CBC (data)",
        }


# Singleton
_key_hierarchy: KeyHierarchy | None = None


def get_key_hierarchy() -> KeyHierarchy:
    """Get or create the singleton key hierarchy."""
    global _key_hierarchy
    if _key_hierarchy is None:
        _key_hierarchy = KeyHierarchy()
    return _key_hierarchy
