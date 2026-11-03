#!/usr/bin/env bash
# One-time root setup for the homellm main server.
#   sudo bash scripts/server-root-setup.sh
# Installs Postgres for LiteLLM, opens the LAN firewall ports and enables lingering
# so the user services keep running with nobody logged in.
set -euo pipefail

[ "$(id -u)" -eq 0 ] || { echo "run with sudo"; exit 1; }
TARGET_USER="${SUDO_USER:?run via sudo from your normal user}"
HOME_DIR=$(getent passwd "$TARGET_USER" | cut -d: -f6)
SECRETS="$HOME_DIR/.config/homellm/secrets.env"
[ -f "$SECRETS" ] || { echo "missing $SECRETS - run scripts/install-server.sh first"; exit 1; }
PG_PASSWORD=$(grep '^HOMELLM_PG_PASSWORD=' "$SECRETS" | cut -d= -f2-)
LAN_CIDR="${LAN_CIDR:-$(ip -4 route get 1.1.1.1 | awk '{for(i=1;i<=NF;i++) if($i=="src") print $(i+1)}' | sed 's/\.[0-9]*$/.0\/24/')}"

echo "==> Postgres"
apt-get update -qq
apt-get install -y -qq postgresql >/dev/null
systemctl enable --now postgresql
sudo -u postgres psql -v ON_ERROR_STOP=1 -q <<SQL
DO \$\$ BEGIN
  IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'litellm') THEN
    CREATE ROLE litellm LOGIN PASSWORD '$PG_PASSWORD';
  ELSE
    ALTER ROLE litellm WITH LOGIN PASSWORD '$PG_PASSWORD';
  END IF;
END \$\$;
SQL
sudo -u postgres psql -tAc "SELECT 1 FROM pg_database WHERE datname='litellm'" | grep -q 1 \
  || sudo -u postgres createdb -O litellm litellm

echo "==> Firewall (LAN $LAN_CIDR)"
if command -v ufw >/dev/null && ufw status | grep -q "Status: active"; then
  ufw allow from "$LAN_CIDR" to any port 4000,8765,8780 proto tcp comment 'homellm' >/dev/null
  ufw allow from "$LAN_CIDR" to any port 5353 proto udp comment 'homellm mDNS' >/dev/null
  ufw status | grep -E 'homellm|1234'
else
  echo "ufw not active, skipping"
fi

echo "==> Linger for $TARGET_USER"
loginctl enable-linger "$TARGET_USER"

echo "done"
