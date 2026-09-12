"""Compliance export: the full audit chain as a single signed document a
regulator can validate independently (architecture section 13).

Signature: HMAC-SHA256 over the canonical JSON of the export body. Dev uses
a key derived from the JWT secret (settings.export_signing_key); production
would sign with a KMS/HSM asymmetric key (Ed25519) whose public half is
published so regulators can verify without any shared secret.

The chain re-verification (`verify_export_chain`) intentionally does NOT
trust the export's `integrity` field - a regulator's copy must stand on its
own: genesis -> every entry's prev_hash link -> recomputed entry_hash.
"""

from __future__ import annotations

import hashlib
import hmac

from src.audit_service.writer import canonical


def sign_export(body: dict, key: bytes) -> str:
    """HMAC-SHA256 over the canonical serialization of the export body."""
    return hmac.new(key, canonical(body).encode("utf-8"), hashlib.sha256).hexdigest()


def verify_export_signature(doc: dict, key: bytes) -> bool:
    """True if `doc`'s signature matches its content (authenticity).

    Verifies that the document is exactly what the Audit Service signed;
    it does not attest to the chain's integrity (that is
    `verify_export_chain`'s job).
    """
    sig = doc.get("signature")
    if not sig or not isinstance(sig, str):
        return False
    body = {k: v for k, v in doc.items() if k != "signature"}
    return hmac.compare_digest(sign_export(body, key), sig)


def verify_export_chain(doc: dict, genesis_hash: str) -> dict:
    """Independently recompute the hash chain from the exported events.

    Every event must chain to its predecessor (or the genesis for the first)
    and its `entry_hash` must equal sha256(prev + canonical(payload)). The
    exported `integrity` field is ignored - this recomputes from scratch.
    """
    events = sorted(doc.get("events", []), key=lambda e: e.get("seq", 0))
    prev = genesis_hash
    for e in events:
        try:
            body = canonical(e.get("payload", {}))
        except (TypeError, ValueError):
            return {"ok": False, "first_bad_seq": e.get("seq"), "reason": "payload is not valid JSON"}
        recomputed = hashlib.sha256((prev + body).encode("utf-8")).hexdigest()
        if e.get("prev_hash") != prev or e.get("entry_hash") != recomputed:
            return {"ok": False, "first_bad_seq": e.get("seq"), "reason": "hash mismatch"}
        prev = e.get("entry_hash")
    return {"ok": True, "n_entries": len(events)}
