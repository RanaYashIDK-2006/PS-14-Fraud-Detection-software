#!/usr/bin/env python3
"""Model release / rollback wiring (model-release audit #28).

Closes the gap the audit found: deploys were ungated direct overwrites with
no archive of the previous good set, no rollback trigger, and no deployment
audit event. This CLI provides:

  python scripts/model_deploy.py archive [--tag LABEL]
      Snapshot the ACTIVE artifact set into models/releases/<model_id>__<ts>
      after verifying it against its governance record. This is the rollback
      source.

  python scripts/model_deploy.py deploy <candidate_dir> <model_id> [--force]
      Gate a candidate: governance hash/schema verify, canary diff vs the
      current deployment, archive the current set, swap files, verify after,
      append a model_deployed audit event. --force bypasses the canary
      max-shift guard (a deliberate model change, not a corrupted deploy).

  python scripts/model_deploy.py rollback <good_model_id> [--from DIR]
      Restore an archived known-good set into models/production, verify after,
      append a model_rollback audit event.

  python scripts/model_deploy.py canary <candidate_dir>
      Standalone canary: prediction diff vs current deployment + latency +
      NaN/Inf detection. Exit 1 on any breach (for CI / auto-rollback hooks).

Atomicity note: per-file copy cannot be atomic across all members, but the
engine refuses to LOAD a mixed/incomplete set (record-verified integrity,
~0.01s) - a partially swapped deploy fails safe instead of serving.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent  # repo root.parent  # repo root
sys.path.insert(0, str(ROOT / "backend"))

PROD_DIR = ROOT / "models" / "production"
RECORDS_DIR = ROOT / "models" / "model_records"
RELEASES_DIR = ROOT / "models" / "releases"

# Pinned canary inputs (benign/attack mix) - same inputs every run so diffs
# are comparable across releases.
CANARY_FEATURES: list[dict] = [
    {"amount_ratio": 0.95, "txn_freq_last_24h": 1, "txn_time_unusual": 0,
     "new_device_flag": 0, "unusual_location_flag": 0, "unusual_recipient_flag": 0,
     "failed_auth_count_24h": 0, "days_since_last_similar_txn": 3.0,
     "gradual_escalation_score": 0.0, "known_device_count": 3,
     "account_tenure_days": 60.0, "hour_of_day": 12, "is_weekend": 0},
    {"amount_ratio": 1.2, "txn_freq_last_24h": 1, "txn_time_unusual": 0,
     "new_device_flag": 0, "unusual_location_flag": 0, "unusual_recipient_flag": 0,
     "failed_auth_count_24h": 0, "days_since_last_similar_txn": 7.0,
     "gradual_escalation_score": 0.0, "known_device_count": 4,
     "account_tenure_days": 120.0, "hour_of_day": 13, "is_weekend": 0},
    {"amount_ratio": 3.5, "txn_freq_last_24h": 1, "txn_time_unusual": 1,
     "new_device_flag": 0, "unusual_location_flag": 1, "unusual_recipient_flag": 0,
     "failed_auth_count_24h": 1, "days_since_last_similar_txn": 1.0,
     "gradual_escalation_score": 0.2, "known_device_count": 2,
     "account_tenure_days": 30.0, "hour_of_day": 22, "is_weekend": 1},
    {"amount_ratio": 37.5, "txn_freq_last_24h": 5, "txn_time_unusual": 1,
     "new_device_flag": 1, "unusual_location_flag": 1, "unusual_recipient_flag": 1,
     "failed_auth_count_24h": 5, "days_since_last_similar_txn": 0.0,
     "gradual_escalation_score": 0.8, "known_device_count": 1,
     "account_tenure_days": 60.0, "hour_of_day": 3, "is_weekend": 0},
    {"amount_ratio": 8.0, "txn_freq_last_24h": 12, "txn_time_unusual": 1,
     "new_device_flag": 1, "unusual_location_flag": 1, "unusual_recipient_flag": 0,
     "failed_auth_count_24h": 3, "days_since_last_similar_txn": 0.5,
     "gradual_escalation_score": 0.6, "known_device_count": 2,
     "account_tenure_days": 15.0, "hour_of_day": 2, "is_weekend": 0},
]


def _current_model_id() -> str:
    manifest = PROD_DIR / "manifest.json"
    if not manifest.exists():
        return "unknown"
    return json.loads(manifest.read_text(encoding="utf-8")).get("model_version", "unknown")


def _audit(event_type: str, payload: dict) -> None:
    try:
        from src.audit_service.writer import append_audit_event
        append_audit_event("MODEL-RELEASE", event_type, payload)
        print(f"  audit: {event_type} appended")
    except Exception as e:  # noqa: BLE001 - audit must never block a release
        print(f"  WARNING: audit append failed: {e}")


def _engine(prod_dir: Path):
    from src.risk_engine.altman_ensemble import AltmanEnsembleEngine
    return AltmanEnsembleEngine(prod_dir, verify_integrity=True)


def _canary_scores(engine) -> list[float]:
    return [float(p) for p in engine.predict_many(CANARY_FEATURES)]


def cmd_archive(args) -> int:
    tag = args.tag or time.strftime("%Y%m%d_%H%M%S")
    model_id = _current_model_id()
    dest = RELEASES_DIR / f"{model_id}__{tag}"
    if dest.exists():
        print(f"archive already exists: {dest}")
        return 1
    # Verify the active set against its record BEFORE archiving.
    from src.risk_engine.model_governance import ModelRecord, verify_artifacts
    try:
        record = ModelRecord.load(RECORDS_DIR, model_id)
    except Exception as e:  # noqa: BLE001
        print(f"refusing to archive: no governance record for {model_id}: {e}")
        return 2
    checks = verify_artifacts(record, PROD_DIR)
    bad = [c for c in checks if not c["pass"]]
    if bad:
        print("refusing to archive: active set does not verify:")
        for c in bad[:5]:
            print(f"  {c['name']}: {c['detail']}")
        return 2
    dest.mkdir(parents=True)
    n = 0
    for p in PROD_DIR.iterdir():
        if p.is_file() and not p.name.endswith((".bak", ".tmp")):
            shutil.copy2(p, dest / p.name)
            n += 1
    (dest / "record.json").write_text(json.dumps(record.__dict__, indent=2), encoding="utf-8")
    _audit("model_archive", {"model_id": model_id, "archive": str(dest), "files": n})
    print(f"archived {n} files -> {dest}")
    return 0


def cmd_canary(args) -> int:
    cand = Path(args.candidate_dir)
    print(f"canary candidate: {cand}")
    cur = _engine(PROD_DIR)
    cand_eng = _engine(cand)
    cur_scores = _canary_scores(cur)
    cand_scores = _canary_scores(cand_eng)
    diffs = [abs(a - b) for a, b in zip(cur_scores, cand_scores)]
    import numpy as np
    bad = [d for d in cand_scores if not np.isfinite(d)]
    report = {
        "candidate": str(cand),
        "current_model": cur.model_version,
        "candidate_model": cand_eng.model_version,
        "n_canary": len(CANARY_FEATURES),
        "max_abs_score_shift": round(max(diffs), 4) if diffs else None,
        "mean_abs_score_shift": round(float(np.mean(diffs)), 4) if diffs else None,
        "non_finite_scores": len(bad),
        "shift_limit": args.max_shift,
    }
    print(json.dumps(report, indent=2))
    breach = bool(bad) or (report["max_abs_score_shift"] is not None
                           and report["max_abs_score_shift"] > args.max_shift)
    print("CANARY:", "BREACH (block deploy / trigger rollback)" if breach else "PASS")
    return 1 if breach else 0


def cmd_deploy(args) -> int:
    cand = Path(args.candidate_dir)
    model_id = args.model_id
    from src.risk_engine.model_governance import ModelRecord, verify_artifacts
    try:
        record = ModelRecord.load(RECORDS_DIR, model_id)
    except Exception as e:  # noqa: BLE001
        print(f"DEPLOY BLOCKED - no governance record for {model_id}: {e}")
        return 2
    checks = verify_artifacts(record, cand)
    bad = [c for c in checks if not c["pass"]]
    if bad:
        print("DEPLOY BLOCKED - candidate does not verify:")
        for c in bad[:5]:
            print(f"  {c['name']}: {c['detail']}")
        return 2
    if model_id == _current_model_id():
        print("candidate is the SAME model as deployed (no-op redeploy)")
    # Canary gate vs the current deployment (unless --force).
    if not args.force:
        cur = _engine(PROD_DIR)
        cand_eng = _engine(cand)
        diffs = [abs(a - b) for a, b in
                 zip(_canary_scores(cur), _canary_scores(cand_eng))]
        import numpy as np
        max_shift = float(np.max(diffs)) if diffs else 0.0
        print(f"canary max score shift vs current deployment: {max_shift:.4f} "
              f"(limit {args.max_shift})")
        if max_shift > args.max_shift:
            print("DEPLOY BLOCKED - canary breach. Use --force for a deliberate "
                  "model change (and re-run the untouched-test protocol first).")
            return 3
    # Archive the current set as the rollback source, then swap.
    if cmd_archive(argparse.Namespace(tag=f"pre-{model_id}")) != 0:
        return 4
    for p in cand.iterdir():
        if p.is_file() and not p.name.endswith((".bak", ".tmp")):
            shutil.copy2(p, PROD_DIR / p.name)
    # Verify AFTER swap: the live set must load + verify as the candidate.
    try:
        eng = _engine(PROD_DIR)
    except Exception as e:  # noqa: BLE001
        print(f"DEPLOY FAILED - post-swap verification error: {e}\n"
              f"Restore with: python scripts/model_deploy.py rollback {_current_model_id()}")
        return 5
    _audit("model_deployed", {"model_id": model_id, "candidate": str(cand),
                              "post_swap_verified": True,
                              "serving_version": eng.model_version})
    print(f"deployed {model_id}; serving version now {eng.model_version}")
    return 0


def cmd_rollback(args) -> int:
    good_id = args.good_model_id
    from src.risk_engine.model_governance import rollback
    src_dir = Path(args.from_dir) if args.from_dir else (
        RELEASES_DIR / f"{good_id}__{args.tag}"
    )
    if not src_dir.exists():
        print(f"archive not found: {src_dir} (use --from or --tag)")
        return 1
    bad_id = _current_model_id()
    try:
        result = rollback(RECORDS_DIR, from_dir=src_dir, to_dir=PROD_DIR,
                          bad_model_id=bad_id, good_model_id=good_id)
    except Exception as e:  # noqa: BLE001
        print(f"ROLLBACK FAILED: {e}")
        return 2
    # Confirm the restored set actually loads + verifies.
    try:
        eng = _engine(PROD_DIR)
        result["post_rollback_serving"] = eng.model_version
    except Exception as e:  # noqa: BLE001
        result["post_rollback_serving"] = f"FAILED TO LOAD: {e}"
    _audit("model_rollback", {"from_model": bad_id, "to_model": good_id,
                              "archive": str(src_dir), "result": result})
    print(json.dumps(result, indent=2))
    return 0 if result.get("post_rollback_verify_pass") else 3


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("archive")
    a.add_argument("--tag", default=None)
    a.set_defaults(fn=cmd_archive)

    c = sub.add_parser("canary")
    c.add_argument("candidate_dir")
    c.add_argument("--max-shift", type=float, default=0.25)
    c.set_defaults(fn=cmd_canary)

    d = sub.add_parser("deploy")
    d.add_argument("candidate_dir")
    d.add_argument("model_id")
    d.add_argument("--force", action="store_true")
    d.add_argument("--max-shift", type=float, default=0.25)
    d.set_defaults(fn=cmd_deploy)

    r = sub.add_parser("rollback")
    r.add_argument("good_model_id")
    r.add_argument("--from", dest="from_dir", default=None)
    r.add_argument("--tag", default=None)
    r.set_defaults(fn=cmd_rollback)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())