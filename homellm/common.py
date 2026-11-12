"""Shared helpers: config paths, YAML loading, auth, host stats."""

from __future__ import annotations

import hashlib
import os
import secrets
import shutil
import socket
import subprocess
from pathlib import Path
from typing import Any

import psutil
import yaml
from fastapi import Header, HTTPException

CONFIG_DIR = Path(os.environ.get("HOMELLM_CONFIG_DIR", Path.home() / ".config" / "homellm"))
STATE_DIR = Path(os.environ.get("HOMELLM_STATE_DIR", Path.home() / ".local" / "state" / "homellm"))
# GGUFs live in LM Studio's folder (<publisher>/<repo>/<file>.gguf) so LM Studio and homellm share one copy
MODELS_DIR = Path.home() / ".lmstudio" / "models"
DEFAULT_CONTEXT = 65536


def load_yaml(path: Path, default: dict | None = None) -> dict:
    if not path.exists():
        return dict(default or {})
    with path.open() as f:
        return yaml.safe_load(f) or {}


def save_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    tmp.replace(path)


def expand(p: str | None) -> str | None:
    return os.path.expanduser(os.path.expandvars(p)) if p else p


def bearer_guard(token: str | None):
    """FastAPI dependency factory that requires `Authorization: Bearer <token>`."""

    def check(authorization: str | None = Header(default=None)) -> None:
        if not token:
            raise HTTPException(500, "server has no token configured")
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(401, "missing bearer token")
        if not secrets.compare_digest(authorization[7:], token):
            raise HTTPException(403, "bad token")

    return check


def primary_ip() -> str:
    """LAN address used for the default route (no packets are sent)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("1.1.1.1", 80))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def node_id() -> str:
    try:
        raw = Path("/etc/machine-id").read_text().strip()
    except OSError:
        raw = socket.gethostname()
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


def memory_stats() -> dict[str, int]:
    vm, sw = psutil.virtual_memory(), psutil.swap_memory()
    mb = 1024 * 1024
    return {
        "total_mb": vm.total // mb,
        "available_mb": vm.available // mb,
        "swap_total_mb": sw.total // mb,
        "swap_used_mb": sw.used // mb,
    }


def gpu_stats() -> list[dict[str, Any]]:
    gpus: list[dict[str, Any]] = []
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10,
            ).stdout
            for line in out.strip().splitlines():
                idx, name, used, total, util, temp = [x.strip() for x in line.split(",")]
                gpus.append({
                    "vendor": "nvidia", "index": int(idx), "name": name,
                    "vram_used_mb": int(used), "vram_total_mb": int(total),
                    "util_pct": _int(util), "temp_c": _int(temp),
                })
        except (subprocess.SubprocessError, ValueError):
            pass
    for dev in sorted(Path("/sys/class/drm").glob("card[0-9]*/device")):
        used, total = dev / "mem_info_vram_used", dev / "mem_info_vram_total"
        if not total.exists():
            continue
        try:
            name = (dev / "product_name").read_text().strip() if (dev / "product_name").exists() else "AMD GPU"
            gpus.append({
                "vendor": "amd", "index": len(gpus), "name": name,
                "vram_used_mb": int(used.read_text()) // 2**20, "vram_total_mb": int(total.read_text()) // 2**20,
                "util_pct": _int((dev / "gpu_busy_percent").read_text()) if (dev / "gpu_busy_percent").exists()
                else None,
                "temp_c": None,
            })
        except (OSError, ValueError):
            pass
    return gpus


def gpu_processes() -> list[dict[str, Any]]:
    """Processes holding NVIDIA GPU memory: [{pid, name, used_mb}]."""
    if not shutil.which("nvidia-smi"):
        return []
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout
    except (subprocess.SubprocessError, OSError):
        return []
    procs = []
    for line in out.strip().splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) == 3 and parts[0].isdigit():
            procs.append({"pid": int(parts[0]), "name": parts[1], "used_mb": _int(parts[2]) or 0})
    return procs


def _int(s: str) -> int | None:
    try:
        return int(float(s))
    except ValueError:
        return None


def notify(title: str, body: str) -> None:
    """Best-effort desktop notification."""
    if shutil.which("notify-send"):
        subprocess.run(["notify-send", "-u", "critical", title, body], capture_output=True, timeout=5)
