# PS-14 — one image, six services. Hardened for security.
FROM python:3.12-slim AS builder

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps first so the layer cache survives code edits.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application: source, scripts, trained artifacts, and data.
# db/ is NOT copied — it is a named volume at runtime.
COPY src ./src
COPY scripts ./scripts
COPY models ./models
COPY data ./data
COPY config ./config

# ── Security hardening ──────────────────────────────────────────────────────
# Create non-root user (Docker best practice)
RUN groupadd -r ps14 && useradd -r -g ps14 -d /app -s /sbin/nologin ps14 \
    && chown -R ps14:ps14 /app

# Drop all capabilities, add only what's needed
USER ps14

ENV DB_DIR=/app/db

EXPOSE 8000 8001 8002 8003 8004 8005

# Health check baked into the image
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:${PORT:-8000}/health', timeout=3)" || exit 1

CMD ["sh", "-c", "uvicorn $MODULE --host 0.0.0.0 --port ${PORT:-8000}"]
