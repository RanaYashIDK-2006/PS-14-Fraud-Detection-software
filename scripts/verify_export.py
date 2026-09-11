#!/usr/bin/env python3
"""Regulator-side verifier for a PS-14 compliance export.

Validates, independently of the Audit Service:

  1. AUTHENTICITY - the HMAC-SHA256 signature matches the document (it is
     exactly what the Audit Service signed), and
  2. INTEGRITY - the hash chain recomputed from the exported events: every
     entry links to its predecessor (or the documented genesis for the
     first) and its entry_hash equals sha256(prev + canonical(payload)).
     The export's own `integrity` field is NOT trusted.

Usage:
  python scripts/verify_export.py export.json [--key HEX]
  python scripts/verify_export.py --url http://127.0.0.1:8005/audit/export \\
      [--internal-token TOKEN] [--key HEX]

The key defaults to the same derivation the dev Audit Service uses (from
JWT_SECRET / the dev default in src/settings.py); a production regulator
would instead verify an asymmetric signature with the published public key.
The chain's genesis hash is printed so it can be compared against the
documented genesis.

Run from the project root. Exit 0 = verified, 1 = verification failed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

# The verifier must not touch any real store: the shared audit writer
# creates its (empty) schema on import, so point it at a throwaway dir.
os.environ["DB_DIR"] = tempfile.mkdtemp(prefix="ps14-verify-export-")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.audit_service.export import verify_export_chain, verify_export_signature  # noqa: E402
from src.settings import get_settings, load_dotenv_and_patch  # noqa: E402

# Same key derivation as the signing Audit Service: it patches settings from
# .env at startup, so the verifier must too (the docstring's "same derivation
# the dev Audit Service uses"). No-op when .env is absent (CI).
load_dotenv_and_patch()


def load_export(args) -> dict:
    if args.url:
        import httpx

        headers = {"X-Internal-Token": args.internal_token} if args.internal_token else {}
        r = httpx.get(args.url, headers=headers, timeout=30.0)
        r.raise_for_status()
        return r.json()
    return json.loads(Path(args.file).read_text(encoding="utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser(description="Verify a PS-14 compliance export independently.")
    ap.add_argument("file", nargs="?", default=None, help="path to the exported JSON document")
    ap.add_argument("--url", default=None, help="fetch the export from the Audit Service instead")
    ap.add_argument("--internal-token", default=None, help="X-Internal-Token for --url (compliance role)")
    ap.add_argument("--key", default=None, help="HMAC key (hex) - defaults to the dev derivation")
    args = ap.parse_args()

    if not args.file and not args.url:
        ap.error("provide an export file or --url")

    doc = load_export(args)

    format_ok = doc.get("format") == "ps14-audit-export-v1"
    print(f"format           : {doc.get('format')} {'(expected ps14-audit-export-v1)' if not format_ok else ''}")
    print(f"exported_at      : {doc.get('exported_at')}")
    print(f"genesis_hash     : {doc.get('genesis_hash')}")
    print(f"events           : {len(doc.get('events', []))}")

    if args.key:
        key = bytes.fromhex(args.key)
    else:
        key = get_settings().export_signing_key

    auth = verify_export_signature(doc, key)
    print(f"signature        : {'OK' if auth else 'FAIL'} (hmac-sha256)")

    chain = verify_export_chain(doc, doc.get("genesis_hash", ""))
    if chain["ok"]:
        print(f"chain            : OK - {chain['n_entries']} entries verified from genesis")
    else:
        print(f"chain            : FAIL - first bad entry at seq {chain.get('first_bad_seq')} ({chain.get('reason')})")

    # The export's own report, shown for reference only (not trusted).
    reported = doc.get("integrity", {})
    print(f"reported status  : ok={reported.get('ok')} entries={reported.get('n_entries')} "
          f"first_bad_seq={reported.get('first_bad_seq')}")

    ok = format_ok and auth and chain["ok"]
    print("\n" + ("EXPORT VERIFIED" if ok else "EXPORT REJECTED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
