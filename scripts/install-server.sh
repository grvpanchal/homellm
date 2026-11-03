#!/usr/bin/env bash
# Install the homellm main server: node agent + LiteLLM gateway + controller dashboard.
#
#   ./scripts/install-server.sh
#   sudo bash scripts/server-root-setup.sh     # Postgres, firewall, linger (once)
#   ./scripts/setup-tunnel.sh llm.example.com  # optional, public access
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
SHARE="$HOME/.local/share/homellm"
CONF="$HOME/.config/homellm"
UNITS="$HOME/.config/systemd/user"
SECRETS="$CONF/secrets.env"
mkdir -p "$SHARE" "$CONF" "$UNITS"

gen() { python3 -c 'import secrets; print(secrets.token_urlsafe(32))'; }
if [ ! -f "$SECRETS" ]; then
  echo "==> generating secrets ($SECRETS)"
  umask 077
  PG=$(python3 -c 'import secrets; print(secrets.token_hex(16))')
  cat > "$SECRETS" <<EOF
HOMELLM_CLUSTER_TOKEN=$(gen)
HOMELLM_ADMIN_TOKEN=$(gen)
LITELLM_MASTER_KEY=sk-$(gen)
LITELLM_SALT_KEY=sk-$(gen)
HOMELLM_PG_PASSWORD=$PG
DATABASE_URL=postgresql://litellm:$PG@127.0.0.1:5432/litellm
EOF
  umask 022
fi

"$REPO/scripts/install-node.sh"

UV="$(command -v uv || echo "$HOME/.local/bin/uv")"
echo "==> LiteLLM"
[ -d "$SHARE/litellm-venv" ] || "$UV" venv -q --python 3.12 "$SHARE/litellm-venv"
VIRTUAL_ENV="$SHARE/litellm-venv" "$UV" pip install -q 'litellm[proxy]' prisma
[ -f "$CONF/litellm.yaml" ] || cp "$REPO/deploy/litellm.yaml" "$CONF/litellm.yaml"
# generate the Prisma client for LiteLLM's schema (needs network once)
SCHEMA=$(ls "$SHARE"/litellm-venv/lib/python3*/site-packages/litellm/proxy/schema.prisma)
( cd "$(dirname "$SCHEMA")" && PATH="$SHARE/litellm-venv/bin:$PATH" "$SHARE/litellm-venv/bin/prisma" generate >/dev/null )

echo "==> controller"
"$SHARE/venv/bin/homellm" init controller

echo "==> systemd user services"
cp "$REPO/deploy/systemd/homellm-litellm.service" "$REPO/deploy/systemd/homellm-controller.service" "$UNITS/"
systemctl --user daemon-reload
systemctl --user enable homellm-litellm.service homellm-controller.service >/dev/null

if (exec 3<>/dev/tcp/127.0.0.1/5432) 2>/dev/null; then
  systemctl --user restart homellm-litellm.service homellm-controller.service
else
  systemctl --user restart homellm-controller.service
  echo
  echo "Postgres isn't running yet. Run once:  sudo bash $REPO/scripts/server-root-setup.sh"
  echo "then:                                 systemctl --user restart homellm-litellm"
fi

IP=$(ip -4 route get 1.1.1.1 | awk '{for(i=1;i<=NF;i++) if($i=="src") print $(i+1)}')
echo
echo "dashboard : http://$IP:8780   (admin token: HOMELLM_ADMIN_TOKEN in $SECRETS)"
echo "LiteLLM   : http://$IP:4000/v1   UI http://$IP:4000/ui  (user admin, password LITELLM_MASTER_KEY)"
echo "add nodes : on each LAN machine run"
echo "            git clone https://github.com/grvpanchal/homellm && homellm/scripts/install-node.sh --token \$HOMELLM_CLUSTER_TOKEN"
