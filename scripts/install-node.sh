#!/usr/bin/env bash
# Install the homellm node agent on this machine (no root needed).
#
#   ./scripts/install-node.sh --token <HOMELLM_CLUSTER_TOKEN> [--flavor cuda-12|cuda-13|rocm|vulkan|cpu]
#                             [--engine-version bNNNN] [--no-engine] [--hf-token <token>]
#
# Installs homellm's own llama-server (official llama.cpp release build matching this machine's GPU) under
# ~/.local/share/homellm/llama.cpp. On the main server, install-server.sh calls this with the token from secrets.env.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
SHARE="$HOME/.local/share/homellm"
CONF="$HOME/.config/homellm"
UNITS="$HOME/.config/systemd/user"
TOKEN=""
FLAVOR=""
ENGINE_VERSION=""
ENGINE=1
HF_TOKEN_ARG=""

while [ $# -gt 0 ]; do
  case "$1" in
    --token) TOKEN="$2"; shift 2 ;;
    --flavor) FLAVOR="$2"; shift 2 ;;
    --engine-version) ENGINE_VERSION="$2"; shift 2 ;;
    --no-engine) ENGINE=0; shift ;;
    --hf-token) HF_TOKEN_ARG="$2"; shift 2 ;;
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

HOMELLM="$SHARE/venv/bin/homellm"
if [ "$ENGINE" = 1 ]; then
  echo "==> llama-server (llama.cpp release build)"
  ARGS=(engine install)
  [ -n "$FLAVOR" ] && ARGS+=(--flavor "$FLAVOR")
  [ -n "$ENGINE_VERSION" ] && ARGS+=(--version "$ENGINE_VERSION")
  if ! "$HOMELLM" "${ARGS[@]}"; then
    echo "    engine install failed; retry later with: $HOMELLM engine install"
  fi
fi

echo "==> config"
# keep an HF token set earlier unless a new one is given
OLD_HF=""
[ -f "$CONF/node.env" ] && OLD_HF=$(grep '^HF_TOKEN=' "$CONF/node.env" | cut -d= -f2- || true)
HF="${HF_TOKEN_ARG:-$OLD_HF}"
umask 077
{
  printf 'HOMELLM_CLUSTER_TOKEN=%s\n' "$TOKEN"
  if [ -n "$HF" ]; then printf 'HF_TOKEN=%s\n' "$HF"; fi
} > "$CONF/node.env"
umask 022
"$HOMELLM" init node

echo "==> LM Studio"
# homellm runs the models straight from ~/.lmstudio/models with its own llama-server, so LM Studio must not hold
# the VRAM. Unload and quit it, and keep it from starting at login. Match process names exactly: llama-server's
# command line contains ".lmstudio/models" and must survive.
LMS="$HOME/.lmstudio/bin/lms"
if [ -x "$LMS" ]; then
  "$LMS" unload --all >/dev/null 2>&1 || true
  "$LMS" server stop >/dev/null 2>&1 || true
  "$LMS" daemon down >/dev/null 2>&1 || true
fi
# by name, then by install path (the app, its node helper and its bundled llama-server); homellm's own llama-server
# lives under ~/.local/share/homellm and is not matched
LMS_PROCS='/LM-Studio/|/\.lmstudio/\.internal/|/\.lmstudio/extensions/backends/|LM_Studio.*\.AppImage'
quit=0
pkill -i -x 'lm-studio|lm studio|lmstudio|llmster' 2>/dev/null && quit=1
pkill -f "$LMS_PROCS" 2>/dev/null && quit=1
if [ "$quit" = 1 ]; then
  for _ in 1 2 3 4 5 6 7 8 9 10; do pgrep -f "$LMS_PROCS" >/dev/null || break; sleep 1; done
  pkill -KILL -f "$LMS_PROCS" 2>/dev/null || true
  echo "    quit LM Studio"
fi
for f in "$HOME"/.config/autostart/*[Ll][Mm]*[Ss]tudio*.desktop "$HOME"/.config/autostart/*lm-studio*.desktop; do
  [ -f "$f" ] || continue
  mv "$f" "$f.disabled-by-homellm" && echo "    disabled LM Studio autostart ($(basename "$f"))"
done
# LM Studio's "Run LLM server on login" (enableLocalService) relaunches it headless at every login; turn it off.
# Done after quitting it, since LM Studio rewrites settings.json on exit.
LMS_SETTINGS="$HOME/.lmstudio/settings.json"
if [ -f "$LMS_SETTINGS" ]; then
  sleep 2
  "$SHARE/venv/bin/python" - "$LMS_SETTINGS" <<'PY'
import json, shutil, sys
path = sys.argv[1]
with open(path) as f:
    s = json.load(f)
if s.get("enableLocalService"):
    shutil.copy2(path, path + ".bak-homellm")
    s["enableLocalService"] = False
    with open(path, "w") as f:
        json.dump(s, f, indent=2)
    print("    turned off LM Studio's 'Run LLM server on login'")
PY
fi
mkdir -p "$HOME/.lmstudio/models"

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

# linger starts user services at boot, before anyone logs in: the node agent then reloads the last active model
if [ "$(loginctl show-user "$USER" -p Linger --value 2>/dev/null)" != "yes" ]; then
  if sudo -n loginctl enable-linger "$USER" 2>/dev/null; then
    echo "    enabled linger: the node and its model start at boot"
  else
    echo
    echo "NOTE: run 'sudo loginctl enable-linger $USER' so the node and its model start at boot without a login."
  fi
fi
if command -v ufw >/dev/null; then
  echo "NOTE: if ufw is active, let the main server reach this node:"
  echo "      sudo ufw allow from <server-ip> to any port 8765,1236 proto tcp"
fi
echo
echo "node agent: $(systemctl --user is-active homellm-node.service)  ->  edit $CONF/node.yaml for profiles"
echo "models    : $HOMELLM catalog   then   $HOMELLM pull <name>   (or the dashboard's Models dialog)"
