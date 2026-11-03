#!/usr/bin/env bash
# Expose LiteLLM (API + admin UI) on a Cloudflare named tunnel.
#
#   ./scripts/setup-tunnel.sh llm.example.com [--dashboard dash.example.com]
#
# Needs a domain on your Cloudflare account. The first run opens a browser for `cloudflared tunnel login`.
# Everything behind the tunnel is protected by LiteLLM keys. The dashboard is NOT exposed unless you pass
# --dashboard; if you do, put it behind Cloudflare Access (Zero Trust -> Access -> Applications).
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
CONF="$HOME/.config/homellm"
UNITS="$HOME/.config/systemd/user"
NAME=homellm
HOST="${1:-}"; shift || true
DASH=""
while [ $# -gt 0 ]; do
  case "$1" in
    --dashboard) DASH="$2"; shift 2 ;;
    *) echo "unknown option $1"; exit 1 ;;
  esac
done
[ -n "$HOST" ] || { echo "usage: $0 <hostname, e.g. llm.example.com> [--dashboard <hostname>]"; exit 1; }

CF="$(command -v cloudflared || echo "$HOME/.local/bin/cloudflared")"
if [ ! -x "$CF" ]; then
  echo "==> installing cloudflared"
  mkdir -p "$HOME/.local/bin"
  curl -fsSL -o "$HOME/.local/bin/cloudflared" \
    https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
  chmod +x "$HOME/.local/bin/cloudflared"
  CF="$HOME/.local/bin/cloudflared"
fi

if [ ! -f "$HOME/.cloudflared/cert.pem" ]; then
  echo "==> log in to Cloudflare (a browser window opens; pick the zone for $HOST)"
  "$CF" tunnel login
fi

if ! "$CF" tunnel list -n "$NAME" -o json | grep -q '"id"'; then
  "$CF" tunnel create "$NAME"
fi
ID=$("$CF" tunnel list -n "$NAME" -o json | python3 -c 'import sys,json; print(json.load(sys.stdin)[0]["id"])')

"$CF" tunnel route dns --overwrite-dns "$NAME" "$HOST"
[ -n "$DASH" ] && "$CF" tunnel route dns --overwrite-dns "$NAME" "$DASH"

{
  echo "tunnel: $ID"
  echo "credentials-file: $HOME/.cloudflared/$ID.json"
  echo "ingress:"
  echo "  - hostname: $HOST"
  echo "    service: http://127.0.0.1:4000"
  if [ -n "$DASH" ]; then
    echo "  - hostname: $DASH"
    echo "    service: http://127.0.0.1:8780"
  fi
  echo "  - service: http_status:404"
} > "$CONF/cloudflared.yml"

sed "s|CLOUDFLARED|$CF|" "$REPO/deploy/systemd/homellm-tunnel.service" > "$UNITS/homellm-tunnel.service"
systemctl --user daemon-reload
systemctl --user enable --now homellm-tunnel.service
systemctl --user restart homellm-tunnel.service
"$HOME/.local/share/homellm/venv/bin/homellm" set-public-url "https://$HOST"
systemctl --user restart homellm-controller.service

echo
echo "tunnel: $(systemctl --user is-active homellm-tunnel.service)"
echo "API   : https://$HOST/v1   (create keys in https://$HOST/ui)"
[ -n "$DASH" ] && echo "dash  : https://$DASH   <- protect with Cloudflare Access"
