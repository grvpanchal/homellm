#!/bin/bash
# fanctl: drive add-on fans on passive datacenter GPUs from GPU temperature, as quietly as possible,
# with fail-safes so a GPU can never overheat. All host specifics live in a config file (hosts/*.conf).
#
# Usage: fanctl [--check] [config]
#   config defaults to /etc/fanctl.conf, then hosts/$(hostname -s).conf next to this script.
#   --check validates the config and finds the hwmon chips, then exits without touching any fan.
#
# HWMON_ROOT and NVIDIA_SMI can be overridden from the environment (used by the tests).

check=0
[[ ${1:-} == --check ]] && { check=1; shift; }

conf=${1:-}
if [[ -z $conf ]]; then
  for c in /etc/fanctl.conf "$(dirname "$(readlink -f "$0")")/hosts/$(hostname -s).conf"; do
    [[ -r $c ]] && { conf=$c; break; }
  done
fi
die() { echo "fanctl: $*" >&2; exit 1; }
[[ -n $conf && -r $conf ]] || die "no config (tried /etc/fanctl.conf and hosts/$(hostname -s).conf)"

# Defaults. The host config overrides them; CHIP, GPU_HEADERS and both curves have no default on purpose.
INTERVAL=3
CHIP=""; CPU_CHIP=k10temp; CPU_TEMP=temp1_input; NGPU=1
GPU_HEADERS=""; CPU_HEADERS=""; MIXED_HEADERS=""; BIOS_HEADERS=""; FIXED=""
AUTO_ENABLE=5                       # pwmN_enable value that hands a header back to the BIOS (nct6775: 5)
GPU_MIN=77; CPU_MIN=64              # floors (77 = 30%, 64 = 25%)
GPU_RAMP_DOWN=1; CPU_RAMP_DOWN=5    # pwm units per interval when slowing down; speeding up is instant
GPU_CURVE=(); CPU_CURVE=()          # TEMP:PWM points, linear in between, 255 past the last point
CPU_PANIC=93                        # instantaneous CPU temp that forces CPU headers to 255
CPU_AVG_SAMPLES=5                   # CPU curve uses this many samples of moving average (5 x 3 s = 15 s)
LOG_EVERY=100                       # print a status line every N intervals (0 = never)
# shellcheck source=/dev/null
. "$conf"
MIXED_MIN=${MIXED_MIN:-$CPU_MIN}
MIXED_RAMP_DOWN=${MIXED_RAMP_DOWN:-$CPU_RAMP_DOWN}
HWMON_ROOT=${HWMON_ROOT:-/sys/class/hwmon}
NVIDIA_SMI=${NVIDIA_SMI:-nvidia-smi}

read -ra GH <<< "$GPU_HEADERS"
read -ra CPH <<< "$CPU_HEADERS"
read -ra MH <<< "$MIXED_HEADERS"
read -ra BH <<< "$BIOS_HEADERS"
read -ra FX <<< "$FIXED"
CONTROLLED=("${GH[@]}" "${CPH[@]}" "${MH[@]}")

[[ -n $CHIP ]] || die "$conf: CHIP is not set"
(( ${#GH[@]} > 0 )) || die "$conf: GPU_HEADERS is empty (headers not identified yet?)"
(( ${#GPU_CURVE[@]} > 0 && ${#CPU_CURVE[@]} > 0 )) || die "$conf: GPU_CURVE and CPU_CURVE are required"

# hwmonN changes between boots, so find chips by their name file
H=""; K=""
for h in "$HWMON_ROOT"/hwmon*; do
  # shellcheck disable=SC2254  # CHIP may be a glob, e.g. it8689*
  case $(cat "$h/name" 2>/dev/null) in
    $CHIP) H=$h ;;
    "$CPU_CHIP") K=$h ;;
  esac
done
[[ -n $H && -n $K ]] || die "hwmon not found ($CHIP=${H:-missing} $CPU_CHIP=${K:-missing})"
for f in "${FX[@]}"; do [[ $f =~ ^[0-9]+:[0-9]+$ ]] || die "$conf: FIXED entry '$f' is not HEADER:PWM"; done
for p in "${CONTROLLED[@]}" "${BH[@]}" "${FX[@]%%:*}"; do
  [[ -e $H/pwm$p && -e $H/pwm${p}_enable ]] || die "$H has no pwm$p"
done
[[ -e $K/$CPU_TEMP ]] || die "$K has no $CPU_TEMP"
if (( check )); then
  echo "fanctl: config $conf ok: $CHIP at $H, $CPU_CHIP at $K; gpu=[${GH[*]}] cpu=[${CPH[*]}] mixed=[${MH[*]}]"
  exit 0
fi

sleeper=""
cleanup() {
  trap - EXIT INT TERM   # run once: a second pass makes the writes fail with EBUSY
  [[ -n $sleeper ]] && kill "$sleeper" 2>/dev/null
  for p in "${CONTROLLED[@]}"; do
    echo 255 > "$H/pwm$p"
    echo "$AUTO_ENABLE" > "$H/pwm${p}_enable"
  done
  exit 0
}
trap cleanup EXIT INT TERM

for p in "${BH[@]}"; do echo "$AUTO_ENABLE" > "$H/pwm${p}_enable"; done
for f in "${FX[@]}"; do echo 1 > "$H/pwm${f%%:*}_enable"; echo "${f#*:}" > "$H/pwm${f%%:*}"; done
for p in "${CONTROLLED[@]}"; do echo 1 > "$H/pwm${p}_enable"; done
echo "fanctl: $conf: gpu=[${GH[*]}] cpu=[${CPH[*]}] mixed=[${MH[*]}] bios=[${BH[*]}] fixed=[${FX[*]}] on $H"

interp() { # interp TEMP POINT... -> pwm
  local t=$1 pt pp ct cp p; shift
  IFS=: read -r pt pp <<< "$1"; shift
  (( t <= pt )) && { echo "$pp"; return; }
  for p in "$@"; do
    IFS=: read -r ct cp <<< "$p"
    (( t <= ct )) && { echo $(( pp + (t - pt) * (cp - pp) / (ct - pt) )); return; }
    pt=$ct; pp=$cp
  done
  echo 255
}

ramp() { # ramp CURRENT WANT FLOOR DOWN_STEP -> next pwm: up at once, down by at most DOWN_STEP
  local cur=$1 want=$2 min=$3 down=$4
  (( want < min )) && want=$min
  if (( want > cur )); then echo "$want"
  elif (( want < cur )); then echo $(( cur - down < want ? want : cur - down ))
  else echo "$cur"; fi
}

# GPU headers start at 0 so the first pass sets them straight to their curve value: starting at 255
# with the slow ramp-down would keep them loud for ~9 minutes. CPU and mixed headers start at 255.
declare -A cur
for p in "${GH[@]}"; do cur[$p]=0; done
for p in "${CPH[@]}" "${MH[@]}"; do cur[$p]=255; done
CH=(); failsafe=0; n=0
while true; do
  T=$(timeout 10 "$NVIDIA_SMI" --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null)
  G=""
  if (( $(grep -cE '^[0-9]+$' <<< "$T") == NGPU )); then G=$(sort -n <<< "$T" | tail -1); fi
  C=0
  raw=$(cat "$K/$CPU_TEMP" 2>/dev/null)
  if [[ $raw =~ ^[0-9]+$ ]]; then C=$(( raw / 1000 )); fi

  if [[ -n $G ]] && (( C > 0 )); then
    if (( failsafe )); then echo "fanctl: sensors back (gpu ${G}C, cpu ${C}C), leaving fail-safe"; failsafe=0; fi
    CH+=("$C"); (( ${#CH[@]} > CPU_AVG_SAMPLES )) && CH=("${CH[@]:1}")
    s=0; for x in "${CH[@]}"; do s=$(( s + x )); done; Ca=$(( s / ${#CH[@]} ))
    g=$(interp "$G" "${GPU_CURVE[@]}"); c=$(interp "$Ca" "${CPU_CURVE[@]}")
    for p in "${GH[@]}"; do cur[$p]=$(ramp "${cur[$p]}" "$g" "$GPU_MIN" "$GPU_RAMP_DOWN"); done
    for p in "${CPH[@]}"; do
      cur[$p]=$(ramp "${cur[$p]}" "$c" "$CPU_MIN" "$CPU_RAMP_DOWN")
      (( C >= CPU_PANIC )) && cur[$p]=255
    done
    for p in "${MH[@]}"; do cur[$p]=$(ramp "${cur[$p]}" $(( g > c ? g : c )) "$MIXED_MIN" "$MIXED_RAMP_DOWN"); done
  else
    if (( ! failsafe )); then
      echo "fanctl: FAIL-SAFE: gpu temps '${T//$'\n'/ }' (want $NGPU), cpu ${C}C; all controlled headers to 255" >&2
      failsafe=1
    fi
    for p in "${CONTROLLED[@]}"; do cur[$p]=255; done
  fi

  for p in "${CONTROLLED[@]}"; do
    [[ -e $H/pwm$p ]] || die "$H/pwm$p vanished (driver reloaded?); exiting so systemd restarts us"
    echo "${cur[$p]}" > "$H/pwm$p"
  done
  if (( LOG_EVERY > 0 && ++n % LOG_EVERY == 0 )); then
    line="gpu ${G:-?}C cpu ${C}C (avg ${Ca:-?}C)"
    for p in "${CONTROLLED[@]}"; do line+=" pwm$p=${cur[$p]}"; done
    echo "fanctl: $line"
  fi
  sleep "$INTERVAL" & sleeper=$!
  wait "$sleeper"   # unlike a foreground sleep, wait lets SIGTERM interrupt at once
done
