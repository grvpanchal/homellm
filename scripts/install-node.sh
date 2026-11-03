#!/usr/bin/env bash
# Install the homellm node agent on this machine (no root needed).
#
#   ./scripts/install-node.sh --token <HOMELLM_CLUSTER_TOKEN>
#
# On the main server, install-server.sh calls this for you with the token from secrets.env.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
SHARE="$HOME/.local/share/homellm"
CONF="$HOME/.config/homellm"
UNITS="$HOME/.config/systemd/user"
TOKEN=""

while [ $# -gt 0 ]; do
  case "$1" in
    --token) TOKEN="$2"; shift 2 ;;
    *) echo "unknown option $1"; exit 1 ;;
  esac
done

if [ -z "$TOKEN" ] && [ -f "$CONF/secrets.env" ]; then
  TOKEN=$(grep '^HOMELLM_CLUSTER_TOKEN=' "$CONF/secrets.env" | cut -d= -f2-)
fi
if [ -z "$TOKEN" ] && [ -f "$CONF/node.env" ]; then
  TOKEN=$(grep '^HOMELLM_CLUSTER_TOKEN=' "$CONF/node.env" | cut -d= -f2-)
fi
[ -n "$TOKEN" ] || { echo "need --token (HOMELLM_CLUSTER_TOKEN from the main server's ~/.config/homellm/secrets.env)"; exit 1; }

UV="$(command -v uv || echo "$HOME/.local/bin/uv")"
if [ ! -x "$UV" ]; then
  echo "==> installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | env UV_NO_MODIFY_PATH=1 sh
  UV="$HOME/.local/bin/uv"
fi

echo "==> python env"
mkdir -p "$SHARE" "$CONF" "$UNITS"
[ -d "$SHARE/venv" ] || "$UV" venv -q --python 3.12 "$SHARE/venv"
VIRTUAL_ENV="$SHARE/venv" "$UV" pip install -q --reinstall-package homellm "$REPO"

echo "==> config"
umask 077
printf 'HOMELLM_CLUSTER_TOKEN=%s\n' "$TOKEN" > "$CONF/node.env"
umask 022
"$SHARE/venv/bin/homellm" init node

echo "==> systemd user service"
cp "$REPO/deploy/systemd/homellm-node.service" "$UNITS/"
# the homellm node agent replaces the standalone watchdog from earlier
if systemctl --user list-unit-files llm-watchdog.service >/dev/null 2>&1 \
   && systemctl --user is-enabled llm-watchdog.service >/dev/null 2>&1; then
  echo "    disabling old llm-watchdog.service (homellm-node does its job now)"
  systemctl --user disable --now llm-watchdog.service || true
fi
systemctl --user daemon-reload
systemctl --user enable --now homellm-node.service
systemctl --user restart homellm-node.service

if [ "$(loginctl show-user "$USER" -p Linger --value 2>/dev/null)" != "yes" ]; then
  echo
  echo "NOTE: run 'sudo loginctl enable-linger $USER' so the agent keeps running when you're logged out."
fi
if command -v ufw >/dev/null; then
  echo "NOTE: if ufw is active, let the main server reach this node:"
  echo "      sudo ufw allow from <server-ip> to any port 8765,1234,1236 proto tcp"
fi
echo
echo "node agent: $(systemctl --user is-active homellm-node.service)  ->  edit $CONF/node.yaml for profiles"
