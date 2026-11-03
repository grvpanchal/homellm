# fan-control

Runs the add-on fans on passive datacenter GPUs (Tesla P100/P40) from GPU temperature, as quietly as possible,
with fail-safes so a GPU can never overheat. It is a plain bash script plus a per-host config, run as a root
systemd service. It is separate from the homellm node agent on purpose: restarting or crashing the LLM side never
touches the fans.

| File | What it is |
|---|---|
| `fanctl.sh` | The control loop. Installed as `/usr/local/bin/fanctl`. |
| `hosts/<hostname>.conf` | Per-host header map and curves. Installed as `/etc/fanctl.conf`. |
| `gpu-fan-curve.service` | systemd unit (`Restart=always`, `RestartSec=5`). |
| `install.sh` | Idempotent installer, needs root. |
| `identify.sh` | Interactive helper that finds what each fan header drives, one header at a time. |

## Install

```bash
sudo fan-control/install.sh                      # uses hosts/$(hostname -s).conf
sudo fan-control/install.sh hosts/ui-pc.conf     # or name a config
```

It loads the kernel driver (`MODULE`) and runs `fanctl --check` on the config, so an incomplete config or a missing
header is refused before anything changes. Then it installs the script, the config, the unit and
`/etc/modules-load.d/fanctl.conf`, and enables and restarts the service. Re-run it after editing a config.

**Roll back:** `sudo systemctl disable --now gpu-fan-curve`. Stopping the service sets every controlled header to
255 and then hands it back to the BIOS (`pwmN_enable=5` on nct6775).

## ui-pc (192.168.1.212)

ASUS TUF GAMING B550-PLUS WIFI II with a Nuvoton NCT6798D (driver `nct6775`, hwmon `nct6798`), Ryzen 7 5700G
(`k10temp`, temp1 = Tctl, max 95°C), and 2x Tesla P100 16GB (slow down at 82°C, shut off at 85°C).

| Header | Drives | Control |
|---|---|---|
| pwm3 | 1st P100 fans (no RPM sensor, stall below 30%) | GPU curve |
| pwm6 | 2nd P100 fans (no RPM sensor) | GPU curve |
| pwm2 | CPU cooler (RPM on fan2 and fan7) | CPU curve |
| pwm4 | case fan (RPM on fan4) | the higher of the GPU and CPU curves |
| pwm1 | nothing audible | left to BIOS |
| pwm5 | nothing audible | fixed at manual 153 |

All headers are PWM-only (writing 0 to `pwmN_mode` fails with EINVAL). The fans on pwm2 and pwm4 stop slowing
down below 25% (620–660 rpm).

## Curves and behaviour

- **GPU curve** `45:77 55:128 65:175 72:225 76:255` on the hottest GPU. It reaches 255 at 76°C, safely below the
  P100's 82°C slowdown. The floor is 77 (30%): a 50% floor made the fans cycle on and off.
- **CPU curve** `72:64 85:110 91:190 94:255` on a ~15 s moving average of Tctl (5 samples), because Ryzen Tctl jumps
  around. An instantaneous Tctl of 93°C or more still forces the CPU headers to 255. The floor is 64 (25%).
- **Speeding up is instant; slowing down is gradual**: 1 pwm unit per 3 s step for GPU headers, 5 for CPU and case
  headers. This stops the fans from hunting.
- At start the GPU headers go **straight to their curve value**. Starting them at 255 with the slow ramp-down would
  keep them loud for about 9 minutes.

## Fail-safes

- If nvidia-smi doesn't return exactly `NGPU` numeric temperatures, or the CPU temperature can't be read, every
  controlled header goes to 255 on that same interval. A line starting `FAIL-SAFE` goes to the journal, and another
  line when the sensors come back.
- nvidia-smi runs under `timeout 10`, so a driver hang can't freeze the loop.
- On exit or SIGTERM: controlled headers go to 255, then to BIOS auto. The trap is cleared first so cleanup runs
  exactly once (a second pass fails with EBUSY).
- If the chip disappears (driver reloaded), the script exits and systemd restarts it in 5 s. hwmon devices are always
  found by their `name` file, because the hwmonN number changes between boots.

Check it: `journalctl -u gpu-fan-curve -f`. A status line is printed every `LOG_EVERY` intervals (100 = 5 min).

## Tuning

Edit `hosts/<host>.conf` and re-run `install.sh`. Curve points are `TEMP:PWM` (PWM 0–255), linear in between, and 255
past the last point. Keep the last GPU point at least 6°C below the GPU's slowdown temperature (see
`nvidia-smi -q -d TEMPERATURE`). Other settings: `GPU_MIN`/`CPU_MIN`/`MIXED_MIN` (floors),
`GPU_RAMP_DOWN`/`CPU_RAMP_DOWN`/`MIXED_RAMP_DOWN`, `CPU_PANIC`, `CPU_AVG_SAMPLES`, `INTERVAL`.

## Identifying headers on a new machine

Listening tests misled us twice when several headers changed at once. `identify.sh` changes **one** header at a time
and waits for you before moving on:

```bash
sudo systemctl stop gpu-fan-curve        # if it is installed; it would fight the test
sudo fan-control/identify.sh nct6798     # the chip's hwmon name; add header numbers to test only those
```

For each header it asks for 100% or 0%, prints live RPM and GPU temperatures, waits for Enter, then asks whether you
heard a change and what the header drives. Then it restores the header and logs the answer, RPMs and GPU temperatures
to `identify-<host>-<time>.log`. Keep the GPUs idle. If any GPU goes above `MAX_GPU` (70°C) the header is put back at
once. Where you can, confirm a GPU header under light load: if slowing it doesn't change the GPU temperature, it
isn't a GPU fan.

## ai2-pc (192.168.1.210, Tesla P40): TODO

`hosts/ai2-pc.conf` is a stub, and fanctl refuses to start while `GPU_HEADERS` is empty. The board (Gigabyte B550M
GAMING X WIFI6) exposes no PWM headers yet: only `gigabyte_wmi` temperatures load. Its fan chip needs the `it87`
driver. Once the driver loads, find the headers with `identify.sh` and fill in the config. The P40 slows down at
92°C and shuts off at 95°C.

## Tests

`tests/test_fanctl.py` runs the script against a fake hwmon tree and a fake nvidia-smi. It covers the start
behaviour, ramping, every fail-safe, CPU panic and averaging, SIGTERM cleanup and `--check`. CI also runs
`shellcheck` and `bash -n`.
