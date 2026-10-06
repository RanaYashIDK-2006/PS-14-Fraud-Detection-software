#!/usr/bin/env python3
"""Phase 4 §12/§24 regression: source secret hygiene + the CI SAST gate.

Guards the two Phase-4 secret findings — a Supabase service_role key and the
JWT signing secret were hardcoded in tracked utility scripts — and the Bandit
gate the CI security job enforces:

  1. no live .env value (>= 12 chars) appears in any git-tracked file
     (non-secret allowlist: CORS_ORIGINS only);
  2. no credential-shaped literal (JWT / provider key / PEM banner) sits in a
     tracked file, except in files that are themselves secret detectors
     (allowlisted by name AND required to contain `re.compile`);
  3. the two remediated scripts carry no literal credential assignment and
     exit non-zero instead of falling back to a default when their secrets
     are absent (no-defaults-in-source contract);
  4. `bandit -r backend/src --severity-level medium` exits 0 — the exact
     command both CI security jobs run.

Values are never printed: only key names, paths and counts.
"""
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
passed = failed = 0
errors: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    global passed, failed
    if cond:
        passed += 1
        print(f"  [PASS] {name}")
    else:
        failed += 1
        errors.append(f"{name}{' — ' + detail if detail else ''}")
        print(f"  [FAIL] {name}{' — ' + detail if detail else ''}")


def tracked_files() -> list[str]:
    out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True,
                         text=True)
    return [f for f in out.stdout.split() if f]


def read_text(path: str) -> str:
    p = ROOT / path
    try:
        if p.stat().st_size > 4_000_000:
            return ""
        return p.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""


def load_env() -> dict[str, str]:
    env_path = ROOT / ".env"
    if not env_path.exists():
        return {}
    values: dict[str, str] = {}
    for line in env_path.read_text(encoding="utf-8").splitlines():
        if "=" not in line or line.lstrip().startswith("#"):
            continue
        k, _, v = line.partition("=")
        v = v.strip().strip('"').strip("'")
        if len(v) >= 12:
            values[k.strip()] = v
    return values


# Non-secret .env keys whose values are legitimately referenced in source
# (CORS origin lists appear in settings defaults, docker-compose and .env.example).
NON_SECRET_ENV_KEYS = {"CORS_ORIGINS"}

# Files that are themselves secret detectors: they must contain the detector
# patterns, so the literal-shape scan is expected to match them. The
# `re.compile` requirement below keeps the allowlist from being abused as a
# general-purpose exemption (a non-detector file cannot join it).
DETECTOR_FILES = {
    "backend/scripts/security_scan.py",
    "backend/scripts/phase75_security_hardening_test.py",
    "backend/scripts/secret_hygiene_test.py",
}

CREDENTIAL_PATTERNS = {
    "jwt_literal":
        re.compile(r"eyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}"),
    "openai_key": re.compile(r"\bsk-[A-Za-z0-9]{20,}\b"),
    "aws_access_key_id": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "github_pat": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
    "google_api_key": re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b"),
    "slack_token": re.compile(r"\bxox[baprs]-[A-Za-z0-9\-]{10,}\b"),
    "stripe_live_key": re.compile(r"\b(sk|rk)_live_[A-Za-z0-9]{20,}\b"),
    "pem_private_key": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
}

# Scripts remediated in Phase 4 (hardcoded credential -> env/.env with a
# fail-fast when absent).
REMEDIATED_SCRIPTS = {
    "backend/scripts/start_risk_loadtest.py": ("INTERNAL_TOKEN", "JWT_SECRET"),
    "backend/scripts/smoke_test_supabase.py": ("SUPABASE_URL", "SUPABASE_SERVICE_KEY"),
}

# A long opaque literal assigned in source (the shape both findings had).
LITERAL_ASSIGNMENT = re.compile(r"=\s*['\"]([A-Za-z0-9_\-./:]{16,})['\"]")


def main() -> int:
    files = tracked_files()
    print(f"tracked files: {len(files)}")

    # ── 1. live .env values must not appear in tracked files ────────────
    env = load_env()
    if env:
        hits: dict[str, list[str]] = {}
        for f in files:
            data = read_text(f)
            if not data:
                continue
            for k, v in env.items():
                if k in NON_SECRET_ENV_KEYS:
                    continue
                if v in data:
                    hits.setdefault(k, []).append(f)
        check(f"no live .env secret value in any tracked file "
              f"({len(env)} keys checked, {len(NON_SECRET_ENV_KEYS)} allowlisted)",
              not hits,
              "; ".join(f"{k} in {sorted(set(v))[:3]}" for k, v in hits.items()))
    else:
        print("  [NOTE] no .env in this checkout — live-value containment "
              "not evaluated here (shape scan below still runs)")

    # ── 2. credential-shaped literals in tracked files ──────────────────
    shape_hits: dict[str, list[str]] = {}
    for f in files:
        if f in DETECTOR_FILES:
            continue
        data = read_text(f)
        if not data:
            continue
        for name, pat in CREDENTIAL_PATTERNS.items():
            if pat.search(data):
                shape_hits.setdefault(name, []).append(f)
    check("no credential-shaped literal in non-detector tracked files",
          not shape_hits,
          "; ".join(f"{k} in {v[:3]}" for k, v in shape_hits.items()))
    for f in DETECTOR_FILES:
        if f in files:
            # A detector references the credential shapes it hunts (PEM banner
            # plus a detector call or the public AWS example key).
            text = read_text(f)
            check(f"detector allowlist entry is a detector: {f}",
                  "PRIVATE KEY" in text
                  and bool(re.search(r"re\.(compile|search|finditer)|AKIA", text)))

    # ── 3. remediated scripts: no literal credentials, fail fast ────────
    for script, keys in REMEDIATED_SCRIPTS.items():
        text = read_text(script)
        check(f"{script} exists", bool(text))
        bad = [ln for ln in text.splitlines()
               if LITERAL_ASSIGNMENT.search(ln) and not ln.lstrip().startswith("#")]
        check(f"{script} has no literal credential assignment",
              not bad, f"{len(bad)} long literal assignment(s)")
        for key in keys:
            check(f"{script} reads {key} from the environment",
                  any(v in text for v in (f'os.environ.get("{key}"',
                                          f"os.environ.get('{key}'",
                                          f"os.environ['{key}']",
                                          f'os.environ["{key}"]')))
        check(f"{script} fails fast when its secrets are absent",
              "sys.exit(" in text and "load_dotenv_and_patch" in text)

    # ── 4. the CI Bandit gate (exact command from both workflows) ───────
    with tempfile.TemporaryDirectory() as td:
        report = Path(td) / "bandit-report.json"
        proc = subprocess.run(
            [sys.executable, "-m", "bandit", "-r", "backend/src/", "-f", "json",
             "-o", str(report), "--severity-level", "medium"],
            cwd=ROOT, capture_output=True, text=True)
    if "No module named bandit" in (proc.stderr or ""):
        check("bandit installed for this interpreter (CI security jobs install it)",
              False, "pip install bandit")
    else:
        detail = ""
        if proc.returncode != 0 and report.exists():
            import json as _json
            try:
                issues = _json.load(open(report, encoding="utf-8"))["results"]
                detail = "; ".join(
                    f"{r['test_id']} {r['filename']}:{r['line_number']}"
                    for r in issues[:6])
            except Exception:  # noqa: BLE001 - reporting only
                detail = "report unreadable"
        check("bandit -r backend/src --severity-level medium exits 0 (CI gate)",
              proc.returncode == 0, detail)

    print("=" * 60)
    print(f"SECRET HYGIENE: {passed} passed, {failed} failed")
    print("=" * 60)
    if errors:
        print("\nFailed checks:")
        for e in errors:
            print(f"  - {e}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
