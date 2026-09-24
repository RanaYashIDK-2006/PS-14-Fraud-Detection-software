"""Phase 110: Authoritative release-manifest contract (single source).

ONE place that defines the identity conventions production manifests,
runtime attestation, reconciliation, and tests all consume.  Two
deliberate layers, documented once:

1. GOVERNANCE layer (RWV / promotion / reconciliation machinery):
       CANONICAL_MODEL_ID       = altman_native
       CANONICAL_RELEASE_ID     = release-altman_native_E_hardneg_cert_20260904

2. DEPLOYED legacy-attestation layer — models/production/release_manifest.json
   (gate_verdict LEGACY_ATTESTED; reconstructed from the actual artifact
   bytes by scripts/reconstruct_legacy_manifest.py, format validated by
   phase50, boot-verified by the risk engine before model load):
       CANONICAL_MODEL_VERSION  = altman_native_E_hardneg_cert_20260904
       CANONICAL_LEGACY_RELEASE_ID = legacy-altman_native_E_hardneg_cert_20260904

Both layers name the SAME certified artifact set; ``legacy-`` vs
``release-`` is metadata about the grandfathered manifest FORMAT and
never substitutes for any feature/contract value inside release
verification — verification stays hash/HMAC-bound against disk
(phase49/phase50 runtime checks are unchanged and remain strict).

The stale historical feature version ``altman_native_v1`` is REJECTED
everywhere: the canonical feature version is ``v1`` (= ML_FEATURE_VERSION).
No alias maps a stale value onto a canonical one.

This module is pure/offline metadata validation.  It holds NO promotion
or RWV authority and grants nothing.

CANONICAL_ARTIFACT_BINDINGS — the manifest fields that bind identity to
bytes: artifact_hash, artifact_files, preprocessing_hash, rule_hash,
evaluation_record_hash (+ training_config_hash provenance).
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping

from src.monitoring.feature_contract import ML_FEATURE_VERSION
from src.monitoring.real_world_evaluation_protocol import (
    MODEL_ID as _PROTOCOL_MODEL_ID,
    PRODUCTION_THRESHOLD as _PROTOCOL_THRESHOLD,
    RELEASE_ID as _PROTOCOL_RELEASE_ID,
)

# ── Canonical values (Part 3) ─────────────────────────────────────────
CANONICAL_MODEL_ID = "altman_native"
CANONICAL_RELEASE_ID = "release-altman_native_E_hardneg_cert_20260904"
CANONICAL_MODEL_VERSION = "altman_native_E_hardneg_cert_20260904"
CANONICAL_LEGACY_RELEASE_ID = "legacy-altman_native_E_hardneg_cert_20260904"
CANONICAL_FEATURE_VERSION = ML_FEATURE_VERSION  # "v1"
CANONICAL_SCHEMA_VERSION = "v1"
CANONICAL_THRESHOLD = _PROTOCOL_THRESHOLD  # 0.018758
ACCEPTED_RELEASE_IDS: tuple[str, ...] = (
    CANONICAL_LEGACY_RELEASE_ID,  # deployed grandfathered format
    CANONICAL_RELEASE_ID,         # future signed-governance format
)
ACCEPTED_GATE_VERDICTS: tuple[str, ...] = ("LEGACY_ATTESTED",)
REJECTED_GATE_VERDICTS: tuple[str, ...] = ("PROMOTION_BLOCKED", "")
STALE_FEATURE_VERSIONS: tuple[str, ...] = ("altman_native_v1",)
CANONICAL_ARTIFACT_BINDINGS: tuple[str, ...] = (
    "artifact_hash",
    "artifact_files",
    "preprocessing_hash",
    "rule_hash",
    "evaluation_record_hash",
)

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_HEX32 = re.compile(r"^[0-9a-f]{32}$")

# Cross-consistency: the governance protocol must agree with this contract.
assert _PROTOCOL_MODEL_ID == CANONICAL_MODEL_ID, "protocol/model contract drift"
assert _PROTOCOL_RELEASE_ID == CANONICAL_RELEASE_ID, "protocol release id drift"
assert CANONICAL_FEATURE_VERSION == "v1", "feature version drift"
assert CANONICAL_THRESHOLD == 0.018758, "threshold drift"


def manifest_identifies_canonical_model(manifest: Mapping[str, Any]) -> bool:
    """The manifest's model_id must be the certified model version, which
    by construction belongs to the canonical altman_native model family."""
    return (
        manifest.get("model_id") == CANONICAL_MODEL_VERSION
        and CANONICAL_MODEL_VERSION.startswith(CANONICAL_MODEL_ID + "_")
    )


def validate_release_manifest(
    manifest: Mapping[str, Any],
    artifact_dir: Path | None = None,
) -> tuple[str, ...]:
    """Pure, offline contract validation.  Returns violations (empty == ok).

    With ``artifact_dir`` given, every artifact_files entry is also checked
    against the actual bytes on disk (read-only), so an altered artifact
    hash is rejected against reality rather than format alone.  There are
    no bypass parameters.
    """
    v: list[str] = []

    rid = manifest.get("release_id")
    if rid not in ACCEPTED_RELEASE_IDS:
        v.append(f"release_id {rid!r} is not a canonical release id")

    if not manifest_identifies_canonical_model(manifest):
        v.append(
            f"model_id {manifest.get('model_id')!r} does not identify "
            f"the canonical {CANONICAL_MODEL_ID} model"
        )
    mv = manifest.get("model_version")
    if mv is not None and mv != CANONICAL_MODEL_VERSION:
        v.append(f"model_version {mv!r} != {CANONICAL_MODEL_VERSION!r}")

    fv = manifest.get("feature_version")
    if fv in STALE_FEATURE_VERSIONS:
        v.append(f"stale feature_version {fv!r} (canonical is "
                 f"{CANONICAL_FEATURE_VERSION!r})")
    elif fv != CANONICAL_FEATURE_VERSION:
        v.append(f"feature_version {fv!r} != {CANONICAL_FEATURE_VERSION!r}")

    sv = manifest.get("schema_version")
    if sv is not None and sv != CANONICAL_SCHEMA_VERSION:
        v.append(f"schema_version {sv!r} != {CANONICAL_SCHEMA_VERSION!r}")

    if "locked_threshold" in manifest and \
            manifest.get("locked_threshold") != CANONICAL_THRESHOLD:
        v.append(f"locked_threshold {manifest.get('locked_threshold')!r} != "
                 f"{CANONICAL_THRESHOLD!r}")

    verdict = manifest.get("gate_verdict")
    if verdict in REJECTED_GATE_VERDICTS or verdict not in ACCEPTED_GATE_VERDICTS:
        v.append(f"gate_verdict {verdict!r} not accepted")

    ah = manifest.get("artifact_hash", "")
    if not isinstance(ah, str) or not _HEX64.fullmatch(ah):
        v.append("artifact_hash missing or not 64-hex")

    files = manifest.get("artifact_files")
    if not isinstance(files, Mapping) or not files:
        v.append("artifact_files missing/empty")
        files = {}
    for name, digest in sorted(files.items()):
        if not isinstance(digest, str) or not _HEX64.fullmatch(digest):
            v.append(f"artifact_files[{name}] not 64-hex")

    prep = manifest.get("preprocessing_hash", "")
    if not isinstance(prep, str) or not _HEX64.fullmatch(prep):
        v.append("preprocessing_hash missing or not 64-hex")
    scaler = files.get("scaler_native.joblib")
    if isinstance(prep, str) and isinstance(scaler, str) and prep != scaler:
        v.append("preprocessing_hash does not bind scaler_native.joblib")

    rh = manifest.get("rule_hash", "")
    if not isinstance(rh, str) or not _HEX32.fullmatch(rh):
        v.append("rule_hash missing or not 32-hex")
    er = manifest.get("evaluation_record_hash", "")
    if not isinstance(er, str) or not _HEX64.fullmatch(er):
        v.append("evaluation_record_hash missing or not 64-hex")
    if not manifest.get("training_config_hash"):
        v.append("training_config_hash missing")

    if artifact_dir is not None:
        import hashlib

        ad = Path(artifact_dir)
        for name, digest in sorted(files.items()):
            p = ad / name
            if not p.exists():
                v.append(f"artifact {name} missing on disk")
                continue
            actual = hashlib.sha256(p.read_bytes()).hexdigest()
            if actual != digest:
                v.append(f"artifact {name} hash mismatch against disk")

    return tuple(v)
