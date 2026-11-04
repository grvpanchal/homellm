#!/bin/bash
# Find which fan header drives which fan without listening: change ONE header at a time and measure.
#   1. RPM test: header to 100%, then 30%; the tach whose RPM follows is on that header.
#   2. GPU test: header to 100%, then 0%; if the GPU warms up while it is off, the header cools the GPU.
# Every header is restored before the next one is touched, and at once if a GPU passes MAX_GPU.
#
# Usage: sudo ./probe.sh [CHIP...]   (default: every it87/nct67xx chip with pwm headers)
# Keep the GPUs idle while it runs. Takes about 1.5 minutes per header.
set -u

MAX_GPU=${MAX_GPU:-65}
RPM_WAIT=${RPM_WAIT:-8}
GPU_WAIT=${GPU_WAIT:-30}
HWMON_ROOT=${HWMON_ROOT:-/sys/class/hwmon}
[[ $EUID -eq 0 ]] || { echo "probe.sh: run as root" >&2; exit 1; }
if systemctl is-active --quiet gpu-fan-curve 2>/dev/null; then
  echo "probe.sh: stop gpu-fan-curve first: sudo systemctl stop gpu-fan-curve" >&2; exit 1
fi

chips=()
for h in "$HWMON_ROOT"/hwmon*; do
  n=$(cat "$h/name" 2>/dev/null)
  [[ -e $h/pwm1 ]] || continue
  if (( $# )); then
    for want in "$@"; do [[ $n == "$want" ]] && chips+=("$h"); done
  elif [[ $n =~ ^(it8|nct6) ]]; then
    chips+=("$h")
  fi
done
(( ${#chips[@]} )) || { echo "probe.sh: no fan chip with pwm headers found (is the driver loaded?)" >&2; exit 1; }

gpu() { timeout 10 nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null | sort -n | tail -1; }
rpms() { # "hwmonX/fanN=RPM ..." across all probed chips
  local h f
  for h in "${chips[@]}"; do
    for f in "$h"/fan[0-9]*_input; do [[ -e $f ]] && echo -n "$(basename "$h")/$(basename "$f" _input)=$(cat "$f") "; done
  done
}
# wait N seconds, but give the header back early if a GPU gets too hot; prints the last GPU temp
hold() {
  local i g=""
  for (( i = 0; i < $1; i += 2 )); do
    sleep 2; g=$(gpu)
    if [[ $g =~ ^[0-9]+$ ]] && (( g >= MAX_GPU )); then echo "$g!"; return 1; fi
  done
  echo "${g:-?}"
}

H=""; P=""; saved_en=""; saved_val=""
restore() {
  [[ -n $P ]] || return 0
  # a header that was manual gets its value back; one in an auto mode just gets its mode back
  if [[ $saved_en == 1 ]]; then echo "$saved_val" > "$H/pwm$P"; else echo "$saved_en" > "$H/pwm${P}_enable"; fi
  P=""
}
trap 'restore; exit 130' INT TERM
trap restore EXIT

echo "probe.sh: chips: ${chips[*]}"
echo "baseline: gpu $(gpu)C, $(rpms)"
for H in "${chips[@]}"; do
  echo "== $(basename "$H") ($(cat "$H/name"))"
  for f in "$H"/pwm[0-9]; do
    p=${f##*pwm}
    g=$(gpu)
    if [[ $g =~ ^[0-9]+$ ]] && (( g >= MAX_GPU - 5 )); then
      echo "  GPU at ${g}C, waiting for it to cool below $(( MAX_GPU - 5 ))C"
      until g=$(gpu); [[ $g =~ ^[0-9]+$ ]] && (( g < MAX_GPU - 5 )); do sleep 10; done
    fi
    saved_en=$(cat "$H/pwm${p}_enable"); saved_val=$(cat "$H/pwm$p"); P=$p
    if ! echo 1 > "$H/pwm${p}_enable" 2>/dev/null; then
      echo "  pwm$p: can't take manual control (enable=$saved_en), skipped"; P=""; continue
    fi
    echo 255 > "$H/pwm$p"; sleep "$RPM_WAIT"; hi=$(rpms)
    echo 77 > "$H/pwm$p"; sleep "$RPM_WAIT"; lo=$(rpms)
    echo 255 > "$H/pwm$p"; g_on=$(hold "$GPU_WAIT")
    echo 0 > "$H/pwm$p"; g_off=$(hold "$GPU_WAIT"); hot=$?
    restore
    # tachs that moved by more than 25% between 100% and 30%
    moved=""
    read -ra a <<< "$hi"; read -ra b <<< "$lo"
    for i in "${!a[@]}"; do
      x=${a[i]#*=}; y=${b[i]#*=}
      (( x > 0 && (x - y) * 4 > x )) && moved+="${a[i]%%=*} ${x}->${y}rpm "
    done
    verdict=""
    (( hot )) && verdict="  <-- COOLS THE GPU (hit ${MAX_GPU}C with this header off)"
    if [[ ${g_on%!} =~ ^[0-9]+$ && ${g_off%!} =~ ^[0-9]+$ ]] && (( ${g_off%!} - ${g_on%!} >= 3 )); then
      verdict="  <-- COOLS THE GPU"
    fi
    echo "  pwm$p: tach ${moved:-none moved}; gpu ${g_on}C at 100% -> ${g_off}C at 0%$verdict"
  done
done
echo "probe.sh: done; every header is back to how it was"
