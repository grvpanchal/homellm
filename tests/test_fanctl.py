"""Run fan-control/fanctl.sh against a fake hwmon tree and a fake nvidia-smi."""

import os
import signal
import subprocess
import time
from pathlib import Path

import pytest

FAN = Path(__file__).resolve().parent.parent / "fan-control"
UI_PC = (FAN / "hosts" / "ui-pc.conf").read_text()


@pytest.fixture
def rig(tmp_path):
    hw = tmp_path / "hw"
    (hw / "hwmon0").mkdir(parents=True)
    (hw / "hwmon4").mkdir()
    (hw / "hwmon0" / "name").write_text("k10temp\n")
    (hw / "hwmon0" / "temp1_input").write_text("60000\n")
    (hw / "hwmon4" / "name").write_text("nct6798\n")
    for p in range(1, 7):
        (hw / "hwmon4" / f"pwm{p}").write_text("128\n")
        (hw / "hwmon4" / f"pwm{p}_enable").write_text("5\n")
    temps = tmp_path / "gpu_temps"
    temps.write_text("50\n48\n")
    smi = tmp_path / "nvidia-smi"
    smi.write_text(f"#!/bin/bash\ncat {temps}\n")
    smi.chmod(0o755)
    conf = tmp_path / "ui-pc.conf"
    conf.write_text(UI_PC + "\nINTERVAL=0.1\n")

    class Rig:
        def __init__(self):
            self.tmp, self.conf, self.temps, self.smi = tmp_path, conf, temps, smi
            self.env = {**os.environ, "HWMON_ROOT": str(hw), "NVIDIA_SMI": str(smi)}
            self.proc = None

        def pwm(self, p):
            return int((hw / "hwmon4" / f"pwm{p}").read_text())

        def enable(self, p):
            return int((hw / "hwmon4" / f"pwm{p}_enable").read_text())

        def cpu(self, c):
            (hw / "hwmon0" / "temp1_input").write_text(f"{c * 1000}\n")

        def start(self, *args):
            self.proc = subprocess.Popen(["bash", str(FAN / "fanctl.sh"), *args, str(self.conf)], env=self.env,
                                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        def until(self, cond, timeout=5):
            end = time.time() + timeout
            while time.time() < end:
                if cond():
                    return True
                time.sleep(0.02)
            return False

    r = Rig()
    yield r
    if r.proc and r.proc.poll() is None:
        r.proc.kill()


def test_start_sets_gpu_headers_straight_to_curve_and_stop_hands_back_to_bios(rig):
    rig.start()
    # 50C on the GPU curve (45:77 55:128) is 102; GPU headers jump there, CPU and case headers start at 255
    assert rig.until(lambda: rig.pwm(3) == 102 and rig.pwm(6) == 102)
    assert rig.enable(3) == 1 and rig.enable(1) == 5 and rig.enable(5) == 1 and rig.pwm(5) == 153
    assert rig.until(lambda: rig.pwm(2) <= 245)  # CPU fan slows by 5 per interval
    rig.proc.send_signal(signal.SIGTERM)
    out, err = rig.proc.communicate(timeout=5)
    assert rig.proc.returncode == 0 and err == ""
    for p in (2, 3, 4, 6):
        assert rig.pwm(p) == 255 and rig.enable(p) == 5
    assert rig.enable(5) == 1 and rig.pwm(5) == 153  # fixed header is left as configured


def test_sigterm_does_not_wait_for_a_long_interval(rig):
    rig.conf.write_text(UI_PC)  # real 3 s interval
    rig.start()
    assert rig.until(lambda: rig.pwm(3) == 102)
    t = time.time()
    rig.proc.send_signal(signal.SIGTERM)
    rig.proc.communicate(timeout=5)
    assert time.time() - t < 1.5 and rig.enable(3) == 5


def test_gpu_slows_down_one_unit_per_interval(rig):
    rig.temps.write_text("76\n70\n")
    rig.start()
    assert rig.until(lambda: rig.pwm(3) == 255)
    rig.temps.write_text("40\n40\n")
    assert rig.until(lambda: rig.pwm(3) < 255)
    time.sleep(0.35)
    assert 245 <= rig.pwm(3) < 255  # not straight down to the 77 floor


@pytest.mark.parametrize("failure", ["missing_gpu", "smi_fails", "cpu_unreadable"])
def test_failsafe_drives_all_controlled_headers_to_255(rig, failure):
    rig.start()
    assert rig.until(lambda: rig.pwm(3) == 102 and rig.pwm(2) < 255)
    if failure == "missing_gpu":
        rig.temps.write_text("50\n")
    elif failure == "smi_fails":
        rig.smi.write_text("#!/bin/bash\nexit 1\n")
    else:
        (rig.tmp / "hw" / "hwmon0" / "temp1_input").unlink()
    assert rig.until(lambda: all(rig.pwm(p) == 255 for p in (2, 3, 4, 6)), timeout=1)
    rig.proc.send_signal(signal.SIGTERM)
    _, err = rig.proc.communicate(timeout=5)
    assert "FAIL-SAFE" in err


def test_cpu_panic_and_moving_average(rig):
    rig.cpu(60)
    rig.start()
    assert rig.until(lambda: rig.pwm(2) == 245)
    rig.cpu(85)  # one hot sample: the 15 s average barely moves, so no jump to 110
    assert rig.until(lambda: rig.pwm(2) == 230)
    rig.cpu(93)  # instantaneous panic threshold
    assert rig.until(lambda: rig.pwm(2) == 255)


def test_check_refuses_incomplete_config_and_missing_headers(rig):
    stub = FAN / "hosts" / "ai2-pc.conf"
    r = subprocess.run(["bash", str(FAN / "fanctl.sh"), "--check", str(stub)], env=rig.env, capture_output=True,
                       text=True)
    assert r.returncode != 0 and "CHIP" in r.stderr
    rig.conf.write_text(UI_PC.replace('FIXED="5:153"', 'FIXED="7:153"'))
    r = subprocess.run(["bash", str(FAN / "fanctl.sh"), "--check", str(rig.conf)], env=rig.env, capture_output=True,
                       text=True)
    assert r.returncode != 0 and "no pwm7" in r.stderr
    rig.conf.write_text(UI_PC)
    r = subprocess.run(["bash", str(FAN / "fanctl.sh"), "--check", str(rig.conf)], env=rig.env, capture_output=True,
                       text=True)
    assert r.returncode == 0 and rig.enable(3) == 5  # --check touches nothing
