"""Build the full audit zip for PS-14."""
import zipfile
import os

zip_name = "ps14_full_audit.zip"
# Will skip data files > 1MB in the loop below
skip_dirs = {".venv", "__pycache__", ".git", "node_modules", ".freebuff", "db", "certs"}
skip_ext = {".db", ".pyc", ".key", ".pem"}

with zipfile.ZipFile(zip_name, "w", zipfile.ZIP_DEFLATED) as zf:
    count = 0
    for root, dirs, files in os.walk("."):
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        for f in files:
            fp = os.path.join(root, f)
            rel = os.path.relpath(fp, ".").replace(os.sep, "/")

            if any(f.endswith(ext) for ext in skip_ext):
                continue
            if f in (".env", ".env.local") and ".example" not in f:
                continue
            if "__pycache__" in rel:
                continue
            # Skip large data files (>1MB)
            try:
                if os.path.getsize(fp) > 1_000_000:
                    continue
            except OSError:
                pass
            if f == zip_name:
                continue

            try:
                zf.write(fp, rel)
                count += 1
            except Exception as e:
                print(f"SKIP {rel}: {e}")

    # Add MANIFEST
    lines = [f"PS-14 Full Audit Zip - {count} files", ""]
    lines.append("EXCLUDED: large CSVs (>1MB), .venv, db/, models/artifacts/, .env")
    lines.append(f"{'Size':>8s}  Path")
    lines.append("")
    for info in zf.infolist():
        if info.filename != "MANIFEST.txt":
            lines.append(f"  {info.file_size:>8d}  {info.filename}")
    zf.writestr("MANIFEST.txt", "\n".join(lines))

sz = os.path.getsize(zip_name)
print(f"Created {zip_name}: {count} files, {sz / 1024:.0f} KB")
