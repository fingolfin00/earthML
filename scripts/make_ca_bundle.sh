#!/usr/bin/env bash
set -euo pipefail

OUT="$HOME/certs/earthml-ca-bundle.pem"
SYS="/etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem"
HARICA="$HOME/certs/harica_tls_rsa_root_ca_2021.pem"

mkdir -p "$HOME/certs"
cat "$SYS" "$HARICA" > "$OUT"
echo "Wrote $OUT"

