# PS14 Security Hardening Guide for Production Deployment

## Executive Summary

This guide provides comprehensive security hardening procedures for deploying the PS14 Privacy-First AI Fraud Detection System in production. It covers all critical security controls, compliance requirements, and operational best practices.

**Target Audience:** DevOps engineers, security architects, and system administrators
**Compliance Targets:** DPDP Act 2023, RBI guidelines, GDPR-equivalent controls

---

## Table of Contents

1. [Pre-Deployment Checklist](#1-pre-deployment-checklist)
2. [Secrets Management](#2-secrets-management)
3. [Network Security](#3-network-security)
4. [Authentication & Authorization](#4-authentication--authorization)
5. [Data Protection](#5-data-protection)
6. [Container Security](#6-container-security)
7. [Monitoring & Logging](#7-monitoring--logging)
8. [Incident Response](#8-incident-response)
9. [Compliance & Auditing](#9-compliance--auditing)
10. [Operational Security](#10-operational-security)

---

## 1. Pre-Deployment Checklist

### Critical (Must Complete)

- [ ] **Generate cryptographically random secrets** for all services
- [ ] **Replace default CORS origins** with your actual domains
- [ ] **Enable HTTPS** with TLS 1.3 for all endpoints
- [ ] **Configure mTLS** for service-to-service communication
- [ ] **Set up secrets manager** (HashiCorp Vault, AWS Secrets Manager, etc.)
- [ ] **Enable audit logging** to external SIEM system
- [ ] **Run security scan** and resolve all CRITICAL/HIGH issues
- [ ] **Conduct penetration testing** by qualified security team

### High Priority

- [ ] **Implement rate limiting** on all public endpoints
- [ ] **Configure WAF** (Web Application Firewall) rules
- [ ] **Set up DDoS protection** (Cloudflare, AWS Shield, etc.)
- [ ] **Enable database encryption** at rest
- [ ] **Configure backup encryption** and test restore procedures
- [ ] **Set up monitoring alerts** for security events
- [ ] **Review and update** all environment-specific configurations

### Medium Priority

- [ ] **Implement CSP headers** for all web interfaces
- [ ] **Configure HSTS** with long max-age
- [ ] **Set up vulnerability scanning** in CI/CD pipeline
- [ ] **Enable security headers** on all endpoints
- [ ] **Configure log retention** per compliance requirements
- [ ] **Document runbooks** for common security incidents

---

## 2. Secrets Management

### 2.1 Secret Rotation Schedule

| Secret | Rotation Frequency | Method |
|--------|-------------------|--------|
| JWT_SECRET | Quarterly | Key derivation + re-encrypt PII |
| PII_ENCRYPTION_KEY | Quarterly | Re-encrypt all PII in DB-1 |
| EXPORT_SIGNING_KEY | Quarterly | Re-sign audit exports |
| INTERNAL_TOKEN | Monthly | Rolling update across services |
| COMPLIANCE_TOKEN | Monthly | Rolling update across services |
| ADMIN_PASS | On personnel change | Immediate rotation |

### 2.2 Secret Generation

```bash
# Generate cryptographically random secrets (32+ bytes)
python -c "import secrets; print(secrets.token_urlsafe(32))"

# Generate for each purpose
JWT_SECRET=$(python -c "import secrets; print(secrets.token_urlsafe(32))")
PII_ENCRYPTION_KEY=$(python -c "import secrets; print(secrets.token_urlsafe(32))")
EXPORT_SIGNING_KEY=$(python -c "import secrets; print(secrets.token_urlsafe(32))")
INTERNAL_TOKEN=$(python -c "import secrets; print(secrets.token_urlsafe(32))")
COMPLIANCE_TOKEN=$(python -c "import secrets; print(secrets.token_urlsafe(32))")
ADMIN_PASS=$(python -c "import secrets; print(secrets.token_urlsafe(24))")
```

### 2.3 Secrets Manager Integration

#### HashiCorp Vault Example

```bash
# Store secrets in Vault
vault kv put secret/ps14/jwt-secret value="$JWT_SECRET"
vault kv put secret/ps14/pii-key value="$PII_ENCRYPTION_KEY"
vault kv put secret/ps14/export-key value="$EXPORT_SIGNING_KEY"
vault kv put secret/ps14/internal-token value="$INTERNAL_TOKEN"
vault kv put secret/ps14/compliance-token value="$COMPLIANCE_TOKEN"
vault kv put secret/ps14/admin-pass value="$ADMIN_PASS"
```

#### Environment Variables (Production)

```env
# NEVER commit .env to version control
# Use secrets manager or encrypted environment variables

JWT_SECRET=vault:secret/ps14/jwt-secret
PII_ENCRYPTION_KEY=vault:secret/ps14/pii-key
EXPORT_SIGNING_KEY=vault:secret/ps14/export-key
INTERNAL_TOKEN=vault:secret/ps14/internal-token
COMPLIANCE_TOKEN=vault:secret/ps14/compliance-token
ADMIN_PASS=vault:secret/ps14/admin-pass
```

### 2.4 Key Derivation Security

**Current Implementation:**
- JWT_SECRET → JWT signing (HS256)
- PII_ENCRYPTION_KEY → PII encryption (AES-256 via Fernet)
- EXPORT_SIGNING_KEY → Export signatures (HMAC-SHA256)

**Production Recommendations:**
- Use separate keys for each purpose (already implemented)
- Consider HSM-backed keys for critical operations
- Implement key versioning for rotation
- Use asymmetric keys for export signing (Ed25519)

---

## 3. Network Security

### 3.1 TLS Configuration

```nginx
# Nginx TLS configuration (nginx.conf)
server {
    listen 443 ssl http2;
    server_name api.yourdomain.com;
    
    # TLS 1.3 only
    ssl_protocols TLSv1.3;
    ssl_ciphers TLS_AES_256_GCM_SHA384:TLS_CHACHA20_POLY1305_SHA256;
    ssl_prefer_server_ciphers on;
    
    # Certificate
    ssl_certificate /etc/ssl/certs/ps14.pem;
    ssl_certificate_key /etc/ssl/private/ps14.key;
    
    # OCSP Stapling
    ssl_stapling on;
    ssl_stapling_verify on;
    
    # Security Headers
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-Frame-Options "DENY" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header Content-Security-Policy "default-src 'self'" always;
    
    # Proxy to application
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### 3.2 CORS Configuration

**Production CORS (restrictive):**

```python
# src/settings.py
cors_origins: str = "https://app.yourdomain.com,https://admin.yourdomain.com"
```

**Environment Variable:**

```env
CORS_ORIGINS=https://app.yourdomain.com,https://admin.yourdomain.com
```

### 3.3 mTLS for Service-to-Service

```yaml
# docker-compose.yml (production)
services:
  identity:
    environment:
      - INTERNAL_TOKEN=${INTERNAL_TOKEN}  # Short-lived token
    # In production, also configure mTLS certificates
    # volumes:
    #   - ./certs/identity.crt:/app/certs/service.crt
    #   - ./certs/identity.key:/app/certs/service.key
    #   - ./certs/ca.crt:/app/certs/ca.crt
```

### 3.4 Network Segmentation

```yaml
# Docker network configuration
networks:
  frontend:
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.0.0/24
  backend:
    driver: bridge
    internal: true  # No external access
    ipam:
      config:
        - subnet: 172.20.1.0/24
  database:
    driver: bridge
    internal: true  # No external access
    ipam:
      config:
        - subnet: 172.20.2.0/24

services:
  front:
    networks:
      - frontend
  
  identity:
    networks:
      - frontend
      - backend
      - database
  
  privacy:
    networks:
      - backend
      - database
  
  risk:
    networks:
      - backend
      - database
  
  verify:
    networks:
      - frontend
      - backend
      - database
  
  audit:
    networks:
      - backend
      - database
```

---

## 4. Authentication & Authorization

### 4.1 Rate Limiting Configuration

```python
# src/middleware/rate_limiter.py
class RateLimitConfig:
    # Production settings
    max_requests: int = 3  # Stricter than dev (5)
    window_seconds: int = 60
    block_duration_seconds: int = 600  # 10 minutes
    
# Per-endpoint configuration
endpoint_configs = {
    "/auth/login": RateLimitConfig(
        max_requests=3,           # 3 attempts per minute
        window_seconds=60,
        block_duration_seconds=600  # 10 minute block
    ),
    "/auth/register": RateLimitConfig(
        max_requests=2,           # 2 registrations per 5 minutes
        window_seconds=300,
        block_duration_seconds=1800  # 30 minute block
    ),
    "/internal/evaluate": RateLimitConfig(
        max_requests=100,         # Higher limit for internal APIs
        window_seconds=60,
        block_duration_seconds=60
    ),
}
```

### 4.2 JWT Configuration

```python
# Production JWT settings
jwt_expiry_minutes: int = 15      # Short-lived tokens
jwt_refresh_enabled: bool = True  # Enable refresh tokens
jwt_algorithm: str = "HS256"      # Or RS256 for asymmetric

# Refresh token settings
refresh_token_expiry_days: int = 7
refresh_token_rotation: bool = True  # Rotate on use
```

### 4.3 Password Policy

```python
# Identity Service password requirements
class RegisterRequest(BaseModel):
    password: str = Field(
        min_length=12,           # Minimum 12 characters
        max_length=128,
        pattern=r"^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{12,}$"
    )
    # Requirements:
    # - At least 12 characters
    # - At least 1 uppercase letter
    # - At least 1 lowercase letter
    # - At least 1 number
    # - At least 1 special character (@$!%*?&)
```

### 4.4 Break-Glass Access

```python
# Production break-glass configuration
BREAK_GLASS_CONFIG = {
    "require_mfa": True,           # Require MFA for break-glass
    "notify_security_team": True,  # Email security team
    "notify_ciso": True,           # Email CISO
    "log_to_siem": True,           # Send to SIEM
    "require_justification": True, # Require detailed justification
    "max_session_minutes": 15,     # Short session for break-glass
    "ip_whitelist": [              # Only from specific IPs
        "10.0.0.0/8",             # Internal network
        "172.16.0.0/12",          # VPN range
    ],
}
```

---

## 5. Data Protection

### 5.1 Database Encryption

```yaml
# PostgreSQL encryption at rest
services:
  postgres:
    environment:
      - POSTGRES_PASSWORD=${DB_PASSWORD}
    volumes:
      - pgdata:/var/lib/postgresql/data
    # Enable encryption
    command: >
      postgres
      -c ssl=on
      -c ssl_cert_file=/etc/ssl/certs/server.crt
      -c ssl_key_file=/etc/ssl/private/server.key
      -c ssl_ca_file=/etc/ssl/certs/ca.crt
```

### 5.2 Backup Encryption

```bash
#!/bin/bash
# backup.sh - Encrypted backup script

BACKUP_DIR="/backups/ps14"
DATE=$(date +%Y%m%d_%H%M%S)
GPG_RECIPIENT="backup@yourdomain.com"

# Create encrypted backup
pg_dump ps14 | gzip | gpg --encrypt --recipient $GPG_RECIPIENT > "$BACKUP_DIR/ps14_$DATE.sql.gz.gpg"

# Verify backup integrity
gpg --decrypt "$BACKUP_DIR/ps14_$DATE.sql.gz.gpg" | gunzip | pg_restore --list

# Upload to secure storage
aws s3 cp "$BACKUP_DIR/ps14_$DATE.sql.gz.gpg" s3://ps14-backups/ --sse aws:kms

# Cleanup local backup
rm "$BACKUP_DIR/ps14_$DATE.sql.gz.gpg"
```

### 5.3 PII Handling

```python
# Data minimization principles
class DataMinimization:
    """
    1. Collect only what's necessary
    2. Store only derived features (never raw amounts)
    3. Encrypt all PII at rest
    4. Pseudonymize for fraud detection
    5. Implement right to erasure
    """
    
    # Never store raw transaction amounts
    # Only store amount_ratio = amount / median_amount
    
    # PII encryption
    def encrypt_pii(self, value: str) -> bytes:
        """AES-256 encryption via Fernet"""
        return self.fernet.encrypt(value.encode("utf-8"))
    
    # Right to erasure
    def erase_user_data(self, fraud_id: str) -> dict:
        """GDPR Article 17 - Right to Erasure"""
        # 1. Remove from Identity Service (DB-1)
        # 2. Remove from Privacy Layer (DB-2)
        # 3. Remove from Risk Engine (DB-3)
        # 4. Keep audit trail (DB-4) for compliance
        # 5. Log the erasure request
        pass
```

---

## 6. Container Security

### 6.1 Docker Security

```dockerfile
# Dockerfile with security best practices
FROM python:3.12-slim as builder

# Security: Run as non-root
RUN groupadd -r ps14 && useradd -r -g ps14 ps14

# Security: Install only necessary packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Security: Copy requirements first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Security: Copy application
COPY . /app
WORKDIR /app

# Security: Change ownership
RUN chown -R ps14:ps14 /app

# Security: Switch to non-root user
USER ps14

# Security: Expose only necessary ports
EXPOSE 8000

# Security: Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)"

# Security: Run with limited capabilities
CMD ["python", "-m", "uvicorn", "src.front_service.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 6.2 Docker Compose Security

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  front:
    build:
      context: .
      dockerfile: Dockerfile
    # Security: Read-only filesystem
    read_only: true
    # Security: No new privileges
    security_opt:
      - no-new-privileges:true
    # Security: Limit capabilities
    cap_drop:
      - ALL
    cap_add:
      - NET_BIND_SERVICE
    # Security: Resource limits
    deploy:
      resources:
        limits:
          cpus: '0.5'
          memory: 512M
        reservations:
          cpus: '0.25'
          memory: 256M
    # Security: tmpfs for temporary files
    tmpfs:
      - /tmp
    # Security: Environment from secrets
    environment:
      - JWT_SECRET_FILE=/run/secrets/jwt_secret
    secrets:
      - jwt_secret
    # Security: Health check
    healthcheck:
      test: ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)"]
      interval: 30s
      timeout: 3s
      retries: 3
      start_period: 10s

secrets:
  jwt_secret:
    file: ./secrets/jwt_secret.txt
  pii_key:
    file: ./secrets/pii_key.txt
  export_key:
    file: ./secrets/export_key.txt
```

---

## 7. Monitoring & Logging

### 7.1 Security Event Logging

```python
# Security events to log
SECURITY_EVENTS = {
    # Authentication events
    "auth.login.success": "INFO",
    "auth.login.failure": "WARNING",
    "auth.login.blocked": "CRITICAL",
    "auth.register.success": "INFO",
    "auth.register.failure": "WARNING",
    
    # Authorization events
    "auth.break_glass.access": "CRITICAL",
    "auth.token.expired": "INFO",
    "auth.token.invalid": "WARNING",
    
    # Rate limiting events
    "rate_limit.exceeded": "WARNING",
    "rate_limit.blocked": "CRITICAL",
    
    # Data events
    "data.pii.access": "INFO",
    "data.pii.encrypted": "INFO",
    "data.pii.decrypted": "INFO",
    "data.export.requested": "INFO",
    "data.export.completed": "INFO",
    
    # Security events
    "security.cors.violation": "WARNING",
    "security.header.missing": "INFO",
    "security.tls.error": "WARNING",
    
    # Audit events
    "audit.chain.tampered": "CRITICAL",
    "audit.export.signed": "INFO",
    "audit.export.verified": "INFO",
}
```

### 7.2 SIEM Integration

```python
# SIEM integration example (Splunk)
import splunklib.client as client

class SIEMLogger:
    def __init__(self, host, port, token):
        self.service = client.connect(
            host=host,
            port=port,
            token=token
        )
    
    def log_security_event(self, event_type, details, severity="INFO"):
        """Log security event to SIEM"""
        event = {
            "time": datetime.now(timezone.utc).isoformat(),
            "source": "ps14-fraud-detection",
            "sourcetype": "ps14:security",
            "event_type": event_type,
            "severity": severity,
            "details": details,
            "host": os.environ.get("HOSTNAME", "unknown"),
        }
        
        # Send to Splunk
        self.service.indexes["main"].submit(
            json.dumps(event),
            sourcetype="ps14:security"
        )
        
        # Also log locally
        logger.log(getattr(logging, severity), f"{event_type}: {details}")
```

### 7.3 Alerting Rules

```yaml
# prometheus/alerting_rules.yml
groups:
  - name: ps14_security
    rules:
      # High rate of failed logins
      - alert: HighLoginFailureRate
        expr: rate(ps14_auth_login_failures_total[5m]) > 0.1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High login failure rate"
          description: "More than 10% of login attempts are failing"
      
      # Rate limit triggered
      - alert: RateLimitTriggered
        expr: increase(ps14_rate_limit_exceeded_total[5m]) > 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Rate limit triggered"
          description: "Rate limiting has been triggered - possible brute force attack"
      
      # Audit chain tampered
      - alert: AuditChainTampered
        expr: ps14_audit_chain_integrity == 0
        for: 0m
        labels:
          severity: critical
        annotations:
          summary: "Audit chain integrity compromised"
          description: "The audit chain has been tampered with"
      
      # Service down
      - alert: ServiceDown
        expr: up{job="ps14"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "PS14 service down"
          description: "{{ $labels.instance }} is down"
```

---

## 8. Incident Response

### 8.1 Incident Response Plan

```markdown
# PS14 Incident Response Plan

## Severity Levels

### P1 - Critical
- Data breach / PII exposure
- System compromise
- Audit chain tampered
- Complete service outage

### P2 - High
- Partial service outage
- Rate limit abuse
- Suspicious activity detected
- Security control failure

### P3 - Medium
- Performance degradation
- Non-critical service issues
- Configuration drift

### P4 - Low
- Minor issues
- Documentation gaps

## Response Procedures

### P1 - Critical

1. **Immediate Actions (0-15 minutes)**
   - Activate incident response team
   - Isolate affected systems
   - Preserve evidence
   - Notify CISO and legal

2. **Short-term (15-60 minutes)**
   - Assess scope of breach
   - Implement containment measures
   - Begin forensic analysis
   - Notify affected parties (if required)

3. **Long-term (1-24 hours)**
   - Complete forensic investigation
   - Implement remediation
   - Update security controls
   - Conduct post-incident review

### P2 - High

1. **Immediate Actions (0-30 minutes)**
   - Investigate alert
   - Assess impact
   - Implement containment

2. **Short-term (30 minutes - 4 hours)**
   - Root cause analysis
   - Implement fixes
   - Update monitoring

## Contact Information

- **Security Team:** security@yourdomain.com
- **CISO:** ciso@yourdomain.com
- **Legal:** legal@yourdomain.com
- **External Forensics:** [Forensics Firm Contact]
```

### 8.2 Playbooks

```markdown
# Brute Force Attack Playbook

## Detection
- Rate limit alerts
- High login failure rate
- Multiple accounts affected

## Response

1. **Verify Alert**
   ```bash
   # Check rate limit logs
   grep "rate_limit" /var/log/ps14/auth.log | tail -100
   
   # Check affected IPs
   awk '{print $1}' /var/log/ps14/auth.log | sort | uniq -c | sort -rn | head -20
   ```

2. **Contain**
   ```bash
   # Block attacking IPs
   iptables -A INPUT -s <attacking_ip> -j DROP
   
   # Or via cloud WAF
   aws wafv2 update-ip-set --scope CLOUDFRONT --id <ip_set_id> --addresses <attacking_ip>
   ```

3. **Investigate**
   - Check for successful logins from attacking IPs
   - Review affected accounts
   - Check for data exfiltration

4. **Remediate**
   - Force password reset for affected accounts
   - Review and update rate limiting rules
   - Update WAF rules
```

---

## 9. Compliance & Auditing

### 9.1 DPDP Act 2023 Compliance

```python
# Data Protection Impact Assessment (DPIA) Requirements

class DPDPCompliance:
    """
    India's Digital Personal Data Protection Act 2023
    """
    
    REQUIREMENTS = {
        "purpose_limitation": {
            "description": "Collect data only for specified purposes",
            "implementation": "Fraud detection only, no secondary use",
            "evidence": "Privacy policy, data flow diagrams",
        },
        "data_minimization": {
            "description": "Collect only necessary data",
            "implementation": "Derived features only, no raw PII in fraud engine",
            "evidence": "Feature schema, data flow audit",
        },
        "consent": {
            "description": "Obtain explicit consent",
            "implementation": "User registration consent, privacy policy acceptance",
            "evidence": "Consent records, privacy policy",
        },
        "data_principal_rights": {
            "description": "Right to access, correct, erase",
            "implementation": "User dashboard, data export, erasure API",
            "evidence": "API documentation, user guides",
        },
        "security_safeguards": {
            "description": "Implement reasonable security",
            "implementation": "Encryption, access controls, audit logging",
            "evidence": "Security controls documentation, audit logs",
        },
        "breach_notification": {
            "description": "Notify in case of breach",
            "implementation": "Incident response plan, notification templates",
            "evidence": "IR plan, notification procedures",
        },
    }
```

### 9.2 RBI Fraud Risk Management

```python
# RBI Guidelines Compliance

class RBICompliance:
    """
    Reserve Bank of India Fraud Risk Management Guidelines
    """
    
    REQUIREMENTS = {
        "risk_assessment": {
            "description": "Regular risk assessment",
            "implementation": "Quarterly security scans, annual penetration testing",
            "evidence": "Scan reports, pentest reports",
        },
        "internal_controls": {
            "description": "Adequate internal controls",
            "implementation": "Role-based access, segregation of duties",
            "evidence": "Access control matrix, audit logs",
        },
        "monitoring": {
            "description": "Continuous monitoring",
            "implementation": "Real-time alerting, SIEM integration",
            "evidence": "Monitoring dashboards, alert logs",
        },
        "incident_response": {
            "description": "Incident response capability",
            "implementation": "IR plan, playbooks, training",
            "evidence": "IR plan, training records, incident reports",
        },
        "reporting": {
            "description": "Regular reporting to board",
            "implementation": "Monthly security reports, quarterly risk reviews",
            "evidence": "Board reports, risk register",
        },
    }
```

### 9.3 Audit Trail Requirements

```python
# Audit trail requirements for compliance

class AuditTrailRequirements:
    """
    What must be logged and retained
    """
    
    RETENTION_PERIODS = {
        "auth_events": "7 years",        # Authentication logs
        "pii_access": "7 years",         # PII access logs
        "break_glass": "10 years",       # Break-glass access
        "data_exports": "7 years",       # Data export logs
        "security_events": "7 years",    # Security event logs
        "system_logs": "1 year",         # System operational logs
    }
    
    REQUIRED_FIELDS = {
        "timestamp": "ISO 8601 format",
        "actor": "Pseudonymous identifier",
        "action": "Specific action performed",
        "resource": "Resource accessed/modified",
        "outcome": "Success/failure",
        "ip_address": "Source IP (if available)",
        "user_agent": "Client user agent",
    }
```

---

## 10. Operational Security

### 10.1 Access Control

```yaml
# Role-Based Access Control (RBAC) matrix

roles:
  admin:
    description: "Full system access"
    permissions:
      - "system:*"
      - "user:*"
      - "data:*"
      - "audit:*"
    mfa_required: true
    
  compliance:
    description: "Compliance and audit access"
    permissions:
      - "audit:read"
      - "audit:export"
      - "compliance:*"
    mfa_required: true
    
  analyst:
    description: "Fraud analysis access"
    permissions:
      - "alerts:read"
      - "alerts:investigate"
      - "risk:read"
    mfa_required: false
    
  user:
    description: "Regular user access"
    permissions:
      - "alerts:read_own"
      - "alerts:verify_own"
    mfa_required: false
```

### 10.2 Change Management

```markdown
# Change Management Process

## Changes Requiring Approval

1. **Security Changes**
   - Firewall rules
   - Access control changes
   - Encryption key rotation
   - Security configuration updates

2. **Infrastructure Changes**
   - Network changes
   - Database schema changes
   - Service configuration changes
   - Deployment procedures

3. **Code Changes**
   - Security-related code
   - Authentication/authorization changes
   - Data handling changes

## Approval Process

1. Submit change request
2. Security review (for security changes)
3. Testing in staging environment
4. Approval from security team
5. Deployment to production
6. Verification and monitoring
```

### 10.3 Backup and Recovery

```bash
#!/bin/bash
# backup_and_recovery.sh

# Daily backup script
backup_database() {
    DATE=$(date +%Y%m%d)
    
    # Backup each database
    for DB in identity features risk audit; do
        pg_dump $DB | gzip | gpg --encrypt --recipient backup@yourdomain.com > \
            "/backups/ps14/${DB}_${DATE}.sql.gz.gpg"
    done
    
    # Upload to secure storage
    aws s3 sync /backups/ps14/ s3://ps14-backups/ --sse aws:kms
    
    # Verify backup integrity
    for DB in identity features risk audit; do
        gpg --decrypt "/backups/ps14/${DB}_${DATE}.sql.gz.gpg" | gunzip | pg_restore --list
    done
}

# Recovery procedure
restore_database() {
    DB=$1
    DATE=$2
    
    # Download backup
    aws s3 cp "s3://ps14-backups/${DB}_${DATE}.sql.gz.gpg" /tmp/
    
    # Decrypt and restore
    gpg --decrypt "/tmp/${DB}_${DATE}.sql.gz.gpg" | gunzip | psql $DB
    
    # Verify restoration
    psql $DB -c "SELECT COUNT(*) FROM $DB;"
}
```

---

## Appendix A: Security Configuration Checklist

```yaml
# security_config.yml

cors:
  origins: "https://app.yourdomain.com"
  methods: "GET,POST"
  headers: "Authorization,Content-Type,X-Internal-Token"
  credentials: true

rate_limiting:
  login:
    max_requests: 3
    window_seconds: 60
    block_duration: 600
  register:
    max_requests: 2
    window_seconds: 300
    block_duration: 1800
  internal:
    max_requests: 100
    window_seconds: 60
    block_duration: 60

security_headers:
  x_content_type_options: "nosniff"
  x_frame_options: "DENY"
  x_xss_protection: "1; mode=block"
  strict_transport_security: "max-age=31536000; includeSubDomains"
  content_security_policy: "default-src 'self'"
  referrer_policy: "strict-origin-when-cross-origin"

encryption:
  pii: "AES-256-GCM"
  backups: "AES-256-CBC"
  tls: "TLSv1.3"

logging:
  auth_events: true
  pii_access: true
  break_glass: true
  security_events: true
  audit_trail: true
```

---

## Appendix B: Security Testing Commands

```bash
# Run security scanner
python scripts/security_scan.py --full

# Run security tests
python scripts/security_test.py

# Run penetration test
python scripts/penetration_test.py

# Run batch test suite
python scripts/batch_cases.py
python scripts/real_cases.py

# Check service health
for p in 8000 8001 8002 8003 8004 8005; do
    curl -s -o /dev/null -w "Port $p: %{http_code}\n" http://127.0.0.1:$p/health
done

# Check security headers
curl -I https://api.yourdomain.com/health

# Verify TLS configuration
openssl s_client -connect api.yourdomain.com:443 -tls1_3
```

---

## Appendix C: Emergency Procedures

```markdown
# Emergency Procedures

## Data Breach Response

1. **Contain**
   - Isolate affected systems
   - Block compromised accounts
   - Preserve evidence

2. **Notify**
   - Security team
   - CISO
   - Legal team
   - Affected users (if required)

3. **Investigate**
   - Forensic analysis
   - Root cause identification
   - Scope assessment

4. **Remediate**
   - Patch vulnerabilities
   - Update security controls
   - Implement monitoring

## System Compromise

1. **Isolate**
   - Disconnect from network
   - Preserve forensic evidence
   - Activate incident response

2. **Assess**
   - Determine scope of compromise
   - Identify affected data
   - Evaluate business impact

3. **Recover**
   - Restore from clean backups
   - Rebuild compromised systems
   - Verify integrity

## Audit Chain Tampered

1. **Investigate**
   - Identify tampered entries
   - Determine scope of tampering
   - Identify attacker

2. **Restore**
   - Restore from verified backup
   - Rebuild chain from genesis
   - Verify integrity

3. **Harden**
   - Review access controls
   - Update monitoring
   - Implement additional controls
```

---

**Document Version:** 1.0
**Last Updated:** August 18, 2026
**Classification:** Confidential
**Owner:** Security Team