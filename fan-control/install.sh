#!/bin/bash
# Install fanctl as the gpu-fan-curve system service. Idempotent: re-run it after changing the script or a config.
# Usage: sudo ./install.sh [config]    (default: hosts/$(hostname -s).conf)
# Roll back: sudo systemctl disable --now gpu-fan-curve   (stopping hands every controlled header to the BIOS)
set -euo pipefail

[[ $EUID -eq 0 ]] || { echo "install.sh: run as root: sudo $0 $*" >&2; exit 1; }
here=$(cd "$(dirname "$0")" && pwd)
conf=${1:-$here/hosts/$(hostname -s).conf}
[[ -r $conf ]] || { echo "install.sh: no config at $conf; copy hosts/ui-pc.conf and fill it in" >&2; exit 1; }

module=$(bash -c '. "$1"; echo "${MODULE:-}"' _ "$conf")
[[ $module =~ ^[a-z0-9_]+$ ]] || { echo "install.sh: $conf: MODULE is not set" >&2; exit 1; }
modprobe "$module"
bash "$here/fanctl.sh" --check "$conf"   # refuses incomplete configs and missing headers before anything changes

install -m 755 "$here/fanctl.sh" /usr/local/bin/fanctl
install -m 644 "$conf" /etc/fanctl.conf
echo "$module" > /etc/modules-load.d/fanctl.conf
sed "s/@MODULE@/$module/" "$here/gpu-fan-curve.service" > /etc/systemd/system/gpu-fan-curve.service
systemctl daemon-reload
systemctl enable gpu-fan-curve
systemctl restart gpu-fan-curve   # restart rather than --now, so a re-install picks up the new script

sleep 5
if systemctl is-active --quiet gpu-fan-curve; then
  echo "install.sh: gpu-fan-curve is running"
else
  echo "install.sh: gpu-fan-curve failed to start" >&2
fi
journalctl -u gpu-fan-curve -n 10 --no-pager -o cat
[[ -e /usr/local/bin/gpu-fan-curve.sh ]] && echo "install.sh: the old /usr/local/bin/gpu-fan-curve.sh is unused now; remove it when happy"
exit 0
