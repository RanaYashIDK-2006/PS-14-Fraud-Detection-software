#!/usr/bin/env python3
"""Encrypt / decrypt / rekey the private access document.

The plaintext master (`docs/ACCESS.md`) is gitignored; the committed copy is
`docs/ACCESS.md.enc`, a Fernet token. The key comes from the `ACCESS_DOC_KEY`
environment variable, `--key`, or an interactive prompt — it is NEVER the
JWT/internal/compliance dev token, and the known dev defaults are rejected.

Usage:
  python scripts/access_doc.py encrypt [--key ...] [--in docs/ACCESS.md] [--out docs/ACCESS.md.enc]
  python scripts/access_doc.py view    [--key ...] [--in docs/ACCESS.md.enc] [-o ACCESS.md]
  python scripts/access_doc.py rekey   [--old-key ...] [--new-key ...] [--in ...] [--out ...]
  python scripts/access_doc.py template  # print the plaintext master

Exit codes: 0 ok, 1 bad key / tampered ciphertext / other error.
"""

from __future__ import annotations

import argparse
import base64
import getpass
import hashlib
import os
import sys
from pathlib import Path

from cryptography.fernet import Fernet, InvalidToken

ROOT = Path(__file__).resolve().parent.parent.parent  # repo root
DEFAULT_IN = ROOT / "docs" / "ACCESS.md"
DEFAULT_OUT = ROOT / "docs" / "ACCESS.md.enc"

# Known dev defaults that must never be used as the document key.
FORBIDDEN_KEYS = {
    "ps14-dev-secret-change-me",
    "ps14-dev-internal-token-change-me",
    "ps14-dev-compliance-token-change-me",
}


def env_file_key() -> str | None:
    """ACCESS_DOC_KEY from the gitignored .env (only that line is read -
    the file may carry other secrets; we never print them)."""
    env_path = ROOT / ".env"
    if not env_path.exists():
        return None
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("ACCESS_DOC_KEY=") and not line.startswith("#"):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


def resolve_key(cli_key: str | None, prompt: bool) -> bytes:
    """Key from --key, ACCESS_DOC_KEY (env or gitignored .env), or a prompt."""
    raw = cli_key or os.environ.get("ACCESS_DOC_KEY") or env_file_key()
    if not raw and prompt and sys.stdin.isatty():
        raw = getpass.getpass("ACCESS document passphrase: ")
    if not raw:
        sys.exit("no key: pass --key, set ACCESS_DOC_KEY, or run interactively")
    if raw in FORBIDDEN_KEYS:
        sys.exit("refusing the known dev secret as the document key - pick a fresh passphrase")
    digest = hashlib.sha256(raw.encode("utf-8")).digest()
    return base64.urlsafe_b64encode(digest)  # 32 bytes -> valid Fernet key


def encrypt(in_path: Path, out_path: Path, key: bytes) -> None:
    token = Fernet(key).encrypt(in_path.read_bytes())
    out_path.write_bytes(token)
    print(f"encrypted {in_path.name} -> {out_path.name} ({len(token)} bytes)")


def decrypt(in_path: Path, key: bytes) -> bytes:
    try:
        return Fernet(key).decrypt(in_path.read_bytes())
    except (InvalidToken, ValueError) as e:
        sys.exit(f"cannot decrypt {in_path.name} (wrong key or tampered): {e.__class__.__name__}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("encrypt")
    p.add_argument("--key", default=None)
    p.add_argument("--in", dest="in_path", default=str(DEFAULT_IN))
    p.add_argument("--out", dest="out_path", default=str(DEFAULT_OUT))

    p = sub.add_parser("view")
    p.add_argument("--key", default=None)
    p.add_argument("--in", dest="in_path", default=str(DEFAULT_OUT))
    p.add_argument("-o", "--out", dest="out_path", default=None, help="write plaintext to a file instead of stdout")

    p = sub.add_parser("rekey")
    p.add_argument("--old-key", default=None)
    p.add_argument("--new-key", default=None)
    p.add_argument("--in", dest="in_path", default=str(DEFAULT_OUT))
    p.add_argument("--out", dest="out_path", default=str(DEFAULT_OUT))

    p = sub.add_parser("template")
    args = ap.parse_args()

    if args.cmd == "template":
        print((ROOT / "docs" / "ACCESS.md").read_text(encoding="utf-8"))
        return 0

    in_path = Path(args.in_path)
    out_path = Path(args.out_path) if args.out_path else None

    if args.cmd == "encrypt":
        if not in_path.exists():
            sys.exit(f"plaintext {in_path} not found")
        encrypt(in_path, out_path, resolve_key(args.key, prompt=True))
    elif args.cmd == "view":
        plain = decrypt(in_path, resolve_key(args.key, prompt=True))
        if out_path is not None:
            out_path.write_bytes(plain)
            print(f"decrypted -> {out_path}")
        else:
            sys.stdout.buffer.write(plain)
            if not plain.endswith(b"\n"):
                sys.stdout.buffer.write(b"\n")
    elif args.cmd == "rekey":
        if out_path is None:
            sys.exit("rekey needs --out")
        old = resolve_key(args.old_key, prompt=True)
        new_raw = args.new_key or os.environ.get("ACCESS_DOC_KEY")
        if not new_raw:
            new_raw = getpass.getpass("New ACCESS document passphrase: ")
        new = resolve_key(new_raw, prompt=False)
        plain = decrypt(in_path, old)
        out_path.write_bytes(Fernet(new).encrypt(plain))
        print(f"rekeyed {in_path.name} -> {out_path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
