#!/usr/bin/env python3
"""Build PS-14 complete archive — source code, reports, config, scripts.

Excludes large binary files (CSVs, joblib models, DBs, venv, caches).
Those live in the main checkout and are not needed in a code archive.
"""
import zipfile
import os
import json
import hashlib
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent.parent.parent  # repo root

SKIP_EXTS = {'.pyc', '.pyo', '.db', '.db-shm', '.db-wal', '.db-journal',
             '.npy', '.npz', '.log', '.zip', '.csv', '.tsv',
             '.enc', '.pem', '.key', '.crt'}
SKIP_DIRS = {'.venv', '__pycache__', '.git', 'node_modules', '.idea',
             '.freebuff', 'cache', 'models/cache'}
SKIP_FILES = {'.env', '.env.local', 'ps14_COMPLETE_ARCHIVE.zip',
              'ps14_full_audit.zip', 'ps14_security_audit.zip', 'ps14_AUDIT.zip'}

# Include essential model artifacts (joblib) even if large.
# Skip other large binaries (>10MB) to keep archive manageable.
MAX_FILE_SIZE = 10 * 1024 * 1024
# Joblib models are essential for reproducibility — no size limit
SKIP_EXT_FOR_MODELS = set()  # include all .joblib in models/artifacts/

archive_name = str(ROOT / 'ps14_COMPLETE_ARCHIVE.zip')
print(f"Building {archive_name}...")

count = 0
total_bytes = 0
included_dirs = {}

with zipfile.ZipFile(archive_name, 'w', zipfile.ZIP_DEFLATED) as zf:
    for root, dirs, files in os.walk(ROOT):
        # Filter out skip dirs
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and d not in SKIP_FILES]

        for f in files:
            if f in SKIP_FILES:
                continue
            if any(f.endswith(e) for e in SKIP_EXTS):
                continue

            fp = Path(root) / f
            fsz = fp.stat().st_size
            # Allow .joblib model files regardless of size (essential for reproducibility)
            if fsz > MAX_FILE_SIZE and not (f.endswith('.joblib') and 'artifacts' in str(fp)):
                continue

            arcname = str(fp.relative_to(ROOT))
            zf.write(str(fp), arcname)
            count += 1
            total_bytes += fsz

            top = arcname.split(os.sep)[0] if os.sep in arcname else '.'
            if top not in included_dirs:
                included_dirs[top] = {'files': 0, 'bytes': 0}
            included_dirs[top]['files'] += 1
            included_dirs[top]['bytes'] += fsz

    # Add manifest
    manifest = {
        'built_at': datetime.now(timezone.utc).isoformat(),
        'total_files': count,
        'total_bytes': total_bytes,
        'directories': included_dirs,
    }
    zf.writestr('ARCHIVE_MANIFEST.json', json.dumps(manifest, indent=2))

sz = os.path.getsize(archive_name)
print(f"\nDone: {archive_name}")
print(f"  Files: {count}")
print(f"  Size: {sz / 1024:.0f} KB ({sz / (1024*1024):.1f} MB)")
for d, info in sorted(included_dirs.items()):
    print(f"  {d}: {info['files']} files ({info['bytes']/1024:.0f} KB)")
