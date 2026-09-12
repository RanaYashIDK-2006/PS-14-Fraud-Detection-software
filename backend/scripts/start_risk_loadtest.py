#!/usr/bin/env python3
"""Start risk engine with known INTERNAL_TOKEN for load testing."""
import os
os.environ['INTERNAL_TOKEN'] = 'loadtest-prod-token-2026'
os.environ['PS14_MODE'] = 'development'
os.environ['DB_DIR'] = os.path.join(os.getcwd(), 'db')
os.environ['JWT_SECRET'] = 'QwHGkFh8Es7e4wdKr2o0G0jVpMNjspOVOrm6USv0Uz0'

import uvicorn
uvicorn.run('src.risk_engine.main:app', host='0.0.0.0', port=8003)
