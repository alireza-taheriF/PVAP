#!/usr/bin/env bash
# Self-signed HTTPS certificate for calling over your local network.
# Phones will warn once ("not private") — choose "visit anyway".
set -euo pipefail
IP=${1:-$(ipconfig getifaddr en0 2>/dev/null || hostname -I | awk '{print $1}')}
openssl req -x509 -newkey rsa:2048 -nodes -days 365 -keyout key.pem -out cert.pem \
  -subj "/CN=$IP" -addext "subjectAltName=IP:$IP,DNS:localhost"
echo "wrote cert.pem / key.pem for $IP — run: python -m pvap.web --cert cert.pem --key key.pem"
