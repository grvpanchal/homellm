#!/bin/bash
# Find out what each fan header drives, ONE header at a time: change it, let a human listen, record the answer,
# put it back, move on. Never sweep all headers at once; listening tests misled us twice that way.
#
# Usage: sudo ./identify.sh CHIP [HEADER...]     e.g. sudo ./identify.sh nct6798        (all headers)
#                                                     sudo ./identify.sh nct6798 3 6    (just pwm3 and pwm6)
# Keep the GPUs idle while it runs: a header turned to 0% may be the only thing cooling a passive GPU. If any GPU
# goes above MAX_GPU (default 70C) the header under test is put back at once.
set -u

MAX_GPU=${MAX_GPU:-70}
HWMON_ROOT=${HWMON_ROOT:-/sys/class/hwmon}
[[ $EUID -eq 0 ]] || { echo "identify.sh: run as root" >&2; exit 1; }
chip=${1:?usage: identify.sh CHIP [HEADER...]}; shift
if systemctl is-active --quiet gpu-fan-curve 2>/dev/null; then
  echo "identify.sh: gpu-fan-curve is running and would fight this; stop it first:" >&2
  echo "  sudo systemctl stop gpu-fan-curve" >&2
  exit 1
fi

H=""
for h in "$HWMON_ROOT"/hwmon*; do [[ $(cat "$h/name" 2>/dev/null) == "$chip" ]] && H=$h; done
[[ -n $H ]] || { echo "identify.sh: no hwmon named $chip" >&2; exit 1; }
headers=("$@")
if (( ${#headers[@]} == 0 )); then
  for f in "$H"/pwm[0-9]*; do [[ $f =~ /pwm([0-9]+)$ ]] && headers+=("${BASH_REMATCH[1]}"); done
fi
log=identify-$(hostname -s)-$(date +%Y%m%d-%H%M).log
echo "identify.sh: $chip at $H, headers: ${headers[*]}, logging to $log"

gpu_temps() { timeout 10 nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null | paste -sd, -; }
max_gpu() { gpu_temps | tr , '\n' | sort -n | tail -1; }
rpms() {
  local f out=""
  for f in "$H"/fan[0-9]*_input; do [[ -e $f ]] && out+="$(basename "$f" _input)=$(cat "$f") "; done
  echo "${out:-none}"
}

testing=""; saved_en=""; saved_val=""
restore() {
  [[ -n $testing ]] || return 0
  if [[ $saved_en == 1 ]]; then echo "$saved_val" > "$H/pwm$testing"; else echo "$saved_en" > "$H/pwm${testing}_enable"; fi
  echo "  pwm$testing restored (enable=$saved_en pwm=$saved_val)"
  testing=""
}
trap 'restore; exit 130' INT TERM
trap restore EXIT

for p in "${headers[@]}"; do
  [[ -e $H/pwm$p ]] || { echo "pwm$p: not present, skipped"; continue; }
  g=$(max_gpu)
  if [[ $g =~ ^[0-9]+$ ]] && (( g > MAX_GPU )); then
    echo "identify.sh: a GPU is at ${g}C (> ${MAX_GPU}C); let it cool and re-run from pwm$p" >&2; exit 1
  fi
  echo; echo "== pwm$p (enable=$(cat "$H/pwm${p}_enable"), pwm=$(cat "$H/pwm$p"))"
  read -rp "Set pwm$p to [1] 100%, [0] 0%, [s] skip, [q] quit? [1] " choice
  case ${choice:-1} in
    q) break ;;
    s) continue ;;
    0) val=0 ;;
    *) val=255 ;;
  esac
  before=$(rpms); gpu_before=$(gpu_temps)
  saved_en=$(cat "$H/pwm${p}_enable"); saved_val=$(cat "$H/pwm$p"); testing=$p
  echo 1 > "$H/pwm${p}_enable"; echo "$val" > "$H/pwm$p"
  echo "  pwm$p is now at $(( val * 100 / 255 ))%. Listen, then press Enter."
  aborted=""
  until read -rt 3 _; do
    g=$(max_gpu)
    if [[ $g =~ ^[0-9]+$ ]] && (( g > MAX_GPU )); then
      aborted="gpu ${g}C"; echo "  GPU at ${g}C, putting pwm$p back now"; break
    fi
    echo -ne "  rpm: $(rpms) gpu: $(gpu_temps)   \r"
  done
  after=$(rpms); gpu_after=$(gpu_temps)
  restore
  read -rp "  Did you hear a change? [y/n] " heard
  read -rp "  What does it drive (e.g. 'GPU1 fans', blank if unsure)? " note
  printf '%s pwm%s set=%s heard=%s note="%s" rpm_before="%s" rpm_after="%s" gpu_before=%s gpu_after=%s%s\n' \
    "$(date +%T)" "$p" "$val" "${heard:-?}" "$note" "$before" "$after" "$gpu_before" "$gpu_after" \
    "${aborted:+ aborted=\"$aborted\"}" | tee -a "$log"
done
echo; echo "identify.sh: done, results in $log. Put the confirmed map in hosts/$(hostname -s).conf."
