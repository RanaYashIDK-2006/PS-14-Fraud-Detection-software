#!/usr/bin/env bash
# Generate self-signed TLS certificates for development/testing.
# For production, replace with certificates from a real CA (Let's Encrypt, etc.)
set -euo pipefail

CERT_DIR="certs"
mkdir -p "$CERT_DIR"

echo "Generating self-signed TLS certificates..."

# Generate CA key and cert
openssl req -x509 -newkey rsa:2048 -nodes \
    -keyout "$CERT_DIR/ca.key" \
    -out "$CERT_DIR/ca.crt" \
    -days 365 \
    -subj "/C=US/ST=Dev/L=Dev/O=PS14-Dev/CN=PS14-CA" 2>/dev/null

# Generate server key and CSR
openssl req -newkey rsa:2048 -nodes \
    -keyout "$CERT_DIR/key.pem" \
    -out "$CERT_DIR/server.csr" \
    -subj "/C=US/ST=Dev/L=Dev/O=PS14-Dev/CN=localhost" 2>/dev/null

# Sign server cert with CA
openssl x509 -req \
    -in "$CERT_DIR/server.csr" \
    -CA "$CERT_DIR/ca.crt" \
    -CAkey "$CERT_DIR/ca.key" \
    -CAcreateserial \
    -out "$CERT_DIR/cert.pem" \
    -days 365 \
    -extfile <(echo "subjectAltName=DNS:localhost,DNS:*.localhost,IP:127.0.0.1") 2>/dev/null

# Cleanup
rm -f "$CERT_DIR/server.csr" "$CERT_DIR/ca.srl"

echo "Certificates generated:"
echo "  $CERT_DIR/cert.pem  (server certificate)"
echo "  $CERT_DIR/key.pem   (server private key)"
echo "  $CERT_DIR/ca.crt    (CA certificate)"
echo ""
echo "To enable TLS in production:"
echo "  1. Uncomment the TLS server block in config/nginx.prod.conf"
echo "  2. Replace self-signed certs with real CA certificates"
echo "  3. Set CORS_ORIGINS to your production domain"
