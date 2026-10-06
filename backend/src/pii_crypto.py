"""Versioned PII field encryption (Phase 4A).

PII ciphertext carries an explicit key-version tag so the key that protects a
row is identifiable without trial decryption:

    v2:gAAAAA...        written by this module under the current key
    gAAAAA...           legacy untagged token (pre-Phase-4A rows)

The PII data key is independent of the JWT signing secret — verified in Phase
4A by decrypting every stored token with each candidate derivation (the
`jwt_secret` derivation decrypts 0 of them, `PII_ENCRYPTION_KEY` decrypts all).
Rotating either secret therefore cannot silently break the other.

Single-key mode (no `PII_KEY_CURRENT` in the environment) keeps writing the
untagged legacy format, so existing callers and ciphertext remain byte
compatible; two-key mode (after rotation) writes tagged tokens and still reads
untagged legacy rows until they are migrated.
"""

from __future__ import annotations

import base64
import hashlib
import re
from collections.abc import Mapping

from cryptography.fernet import Fernet, InvalidToken

#: Version name of the pre-rotation scheme (untagged ciphertext).
LEGACY_VERSION = "v1"

_TAG_RE = re.compile(rb"^v(\d+):")
_VERSION_RE = re.compile(r"^v\d+$")


class UnknownKeyVersion(Exception):
    """Ciphertext names a key version this process holds no key for."""


def derive_fernet_key(material: str) -> bytes:
    """Fernet key from key material: base64(sha256(material)) — the original
    (and only) derivation this codebase has ever used for PII at rest."""
    if not material:
        raise ValueError("empty key material")
    return base64.urlsafe_b64encode(hashlib.sha256(material.encode()).digest())


class PiiCipher:
    """Encrypt/decrypt PII fields under a versioned key set.

    ``key_materials`` maps a version name to key material. ``current_version``
    names the key used for new writes; it must be present.
    """

    def __init__(self, current_version: str, key_materials: Mapping[str, str],
                 *, tag_writes: bool | None = None) -> None:
        self._keys: dict[str, Fernet] = {
            version: Fernet(derive_fernet_key(material))
            for version, material in key_materials.items() if material
        }
        if current_version not in self._keys:
            raise ValueError(f"no key material for current version {current_version!r}")
        self.current_version = current_version
        # Tagging only matters once a distinct legacy key is in play; in
        # single-key mode the historical untagged format is preserved.
        self._tag_writes = (len(self._keys) > 1) if tag_writes is None else tag_writes

    # ---------------------------------------------------------------- writers
    def encrypt(self, plaintext: str | bytes) -> bytes:
        raw = plaintext.encode("utf-8") if isinstance(plaintext, str) else plaintext
        token = self._keys[self.current_version].encrypt(raw)
        return (self.current_version.encode() + b":" + token) if self._tag_writes else token

    # ---------------------------------------------------------------- readers
    def version_of(self, token: bytes | str) -> str | None:
        """Version named by the ciphertext tag, or None for a legacy token."""
        raw = token.encode() if isinstance(token, str) else bytes(token)
        m = _TAG_RE.match(raw)
        return f"v{m.group(1).decode()}" if m else None

    def needs_migration(self, token: bytes | str) -> bool:
        """True when the token is readable but not under the current key.

        Single-key mode writes untagged tokens as its canonical form, so an
        untagged token needs no migration there; in two-key mode it does.
        """
        version = self.version_of(token)
        if version is None and not self._tag_writes:
            return False
        return version != self.current_version

    def key_versions(self) -> tuple[str, ...]:
        return tuple(sorted(self._keys))

    def fingerprint(self, version: str) -> str:
        """Non-secret identifier of a key version (for audit/reporting)."""
        fernet = self._keys.get(version)
        if fernet is None:
            raise UnknownKeyVersion(version)
        return hashlib.sha256(fernet._signing_key + fernet._encryption_key).hexdigest()[:16]

    def decrypt(self, token: bytes | str) -> bytes:
        """Decrypt, dispatching on the tag. Unknown version -> UnknownKeyVersion.

        Untagged (legacy) tokens are read with the legacy key when one is
        registered, otherwise with the current key (single-key mode).
        """
        raw = token.encode() if isinstance(token, str) else bytes(token)
        version = self.version_of(raw)
        if version is None:
            version = LEGACY_VERSION if LEGACY_VERSION in self._keys else self.current_version
            payload = raw
        else:
            if version not in self._keys:
                raise UnknownKeyVersion(
                    f"ciphertext requires {version}; this process holds "
                    f"{sorted(self._keys)}")
            payload = raw.split(b":", 1)[1]  # strip the version tag
        return self._keys[version].decrypt(payload)

    def decrypt_str(self, token: bytes | str) -> str:
        return self.decrypt(token).decode("utf-8")


def default_cipher() -> PiiCipher:
    """Cipher bound to the current settings (env-driven)."""
    from src.settings import settings

    materials: dict[str, str] = {}
    if settings.pii_key_legacy:
        materials[LEGACY_VERSION] = settings.pii_key_legacy
    current = settings.pii_key_version
    if settings.pii_key_current:
        materials[current] = settings.pii_key_current
    if not materials:
        # Pre-rotation configuration: one key, legacy derivation, untagged.
        materials = {LEGACY_VERSION: settings.pii_encryption_key}
        current = LEGACY_VERSION
    elif LEGACY_VERSION not in materials:
        # Rotation without an explicit legacy key: the historical key material
        # (PII_ENCRYPTION_KEY) still reads untagged rows until they migrate.
        materials[LEGACY_VERSION] = settings.pii_encryption_key
    return PiiCipher(current, materials)


_CIPHER: PiiCipher | None = None


def cipher() -> PiiCipher:
    global _CIPHER
    if _CIPHER is None:
        _CIPHER = default_cipher()
    return _CIPHER


def reset_cipher_cache() -> None:
    """Test hook: drop the memoised cipher after settings/env changes."""
    global _CIPHER
    _CIPHER = None


# --------------------------------------------------------------- module API
def encrypt_pii(value: str) -> bytes:
    return cipher().encrypt(value)


def encrypt_opt(value: str | None) -> bytes | None:
    return encrypt_pii(value) if value is not None else None


def decrypt_pii(value: bytes) -> str:
    return cipher().decrypt_str(value)


def decrypt_opt(value) -> str | None:
    """Decrypt an optional PII column, tolerating legacy plaintext rows."""
    if value is None:
        return None
    if isinstance(value, (bytes, bytearray)):
        return decrypt_pii(bytes(value))
    return str(value)  # legacy plaintext row


def needs_migration(value) -> bool:
    if not isinstance(value, (bytes, bytearray)):
        return False
    return cipher().needs_migration(bytes(value))


def key_versions() -> tuple[str, ...]:
    return cipher().key_versions()


def key_fingerprints() -> dict[str, str]:
    """Version -> non-secret fingerprint, for reports and audit lines."""
    c = cipher()
    return {v: c.fingerprint(v) for v in c.key_versions()}


__all__ = [
    "LEGACY_VERSION", "PiiCipher", "UnknownKeyVersion", "InvalidToken",
    "derive_fernet_key", "default_cipher", "cipher", "reset_cipher_cache",
    "encrypt_pii", "encrypt_opt", "decrypt_pii", "decrypt_opt",
    "needs_migration", "key_versions", "key_fingerprints",
]
