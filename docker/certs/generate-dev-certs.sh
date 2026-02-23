#!/usr/bin/env bash
# Generate self-signed development certificates for PIANO services
# Usage: ./generate-dev-certs.sh [output_dir]
set -euo pipefail

OUTPUT_DIR="${1:-$(dirname "$0")}"
mkdir -p "$OUTPUT_DIR"

echo "=== Generating PIANO development certificates ==="
echo "Output directory: $OUTPUT_DIR"

# --- CA ---
echo "--- Generating CA certificate ---"
openssl genrsa -out "$OUTPUT_DIR/ca.key" 4096
openssl req -x509 -new -nodes -key "$OUTPUT_DIR/ca.key" \
  -sha256 -days 365 -out "$OUTPUT_DIR/ca.crt" \
  -subj "/CN=PIANO Dev CA/O=Project Sid"

# --- Redis ---
echo "--- Generating Redis certificates ---"
openssl genrsa -out "$OUTPUT_DIR/redis.key" 2048
openssl req -new -key "$OUTPUT_DIR/redis.key" \
  -out "$OUTPUT_DIR/redis.csr" \
  -subj "/CN=redis/O=Project Sid"
openssl x509 -req -in "$OUTPUT_DIR/redis.csr" \
  -CA "$OUTPUT_DIR/ca.crt" -CAkey "$OUTPUT_DIR/ca.key" \
  -CAcreateserial -out "$OUTPUT_DIR/redis.crt" \
  -days 365 -sha256

# --- Client certificate (for services connecting to Redis) ---
echo "--- Generating client certificates ---"
openssl genrsa -out "$OUTPUT_DIR/client.key" 2048
openssl req -new -key "$OUTPUT_DIR/client.key" \
  -out "$OUTPUT_DIR/client.csr" \
  -subj "/CN=piano-client/O=Project Sid"
openssl x509 -req -in "$OUTPUT_DIR/client.csr" \
  -CA "$OUTPUT_DIR/ca.crt" -CAkey "$OUTPUT_DIR/ca.key" \
  -CAcreateserial -out "$OUTPUT_DIR/client.crt" \
  -days 365 -sha256

# --- Cleanup CSR files ---
rm -f "$OUTPUT_DIR"/*.csr "$OUTPUT_DIR"/*.srl

echo ""
echo "=== Certificates generated successfully ==="
echo "CA:     $OUTPUT_DIR/ca.crt, $OUTPUT_DIR/ca.key"
echo "Redis:  $OUTPUT_DIR/redis.crt, $OUTPUT_DIR/redis.key"
echo "Client: $OUTPUT_DIR/client.crt, $OUTPUT_DIR/client.key"
echo ""
echo "NOTE: These are self-signed development certificates."
echo "Do NOT use in production."
