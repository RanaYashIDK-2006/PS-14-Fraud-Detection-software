#!/usr/bin/env python3
"""Start risk engine for load testing.

Secrets are never stored in source: INTERNAL_TOKEN and JWT_SECRET come from
the environment, falling back to the repo-root .env via the shared loader.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src.settings import load_dotenv_and_patch  # noqa: E402

load_dotenv_and_patch()
os.environ['PS14_MODE'] = 'development'
os.environ['DB_DIR'] = os.path.join(os.getcwd(), 'db')
if not os.environ.get('INTERNAL_TOKEN') or not os.environ.get('JWT_SECRET'):
    sys.exit("INTERNAL_TOKEN and JWT_SECRET are required "
             "(environment or repo-root .env) — no defaults in source")

import uvicorn
uvicorn.run('src.risk_engine.main:app', host='0.0.0.0', port=8003)
