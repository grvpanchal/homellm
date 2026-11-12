"""Inference backends a node can run models on.

A *profile* is a named, ready-to-run model preset, e.g.

    gemma-4-26b-a4b:
      backend: llamacpp
      model: gemma-4-26B-A4B-it-GGUF/gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf   # relative to models_dir (~/.lmstudio/models)
      context: 50176
      parallel: 2
      args: [--flash-attn, "on"]

The served model id (what LiteLLM and clients call) is the profile name.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import httpx
import psutil

from .. import engine
from ..common import MODELS_DIR, expand

log = logging.getLogger("homellm.node")

# llama.cpp's server keeps a RAM prompt cache (default ~8 GB) that OOM-killed the engine on a 13 GB box.
NO_PROMPT_CACHE_ENV = {"LLAMA_ARG_CACHE_RAM": "0"}


def anon_rss_mb(pid: int) -> int | None:
    """Private memory of a process. Total RSS also counts mmapped model pages, which the kernel can drop."""
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("RssAnon:"):
                return int(line.split()[1]) // 1024
    except (OSError, ValueError):
        pass
    return None


class BackendError(RuntimeError):
    pass


class Backend:
    name = "base"

    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg

    def available(self) -> bool:
        raise NotImplementedError

    def port(self) -> int:
        raise NotImplementedError

    def list_models(self) -> list[dict[str, Any]]:
        """Models present on disk that a profile can point at."""
        raise NotImplementedError

    def is_ready(self, profile_name: str, profile: dict[str, Any]) -> bool:
        raise NotImplementedError

    def start(self, profile_name: str, profile: dict[str, Any]) -> None:
        raise NotImplementedError

    def stop(self) -> None:
        raise NotImplementedError

    def server_pid(self) -> int | None:
        return None

    def unavailable_reason(self) -> str:
        return f"backend {self.name} not available on this node"

    def engine_info(self) -> dict[str, Any]:
        return {}

    def health(self) -> dict[str, Any]:
        """Extra diagnostics: RSS of the engine process, prompt-cache env, warnings."""
        pid = self.server_pid()
        out: dict[str, Any] = {"pid": pid, "warnings": []}
        if pid:
            try:
                p = psutil.Process(pid)
                out["rss_mb"] = anon_rss_mb(pid)
                env = p.environ()
                out["prompt_cache_disabled"] = env.get("LLAMA_ARG_CACHE_RAM") == "0" or "--cache-ram" in " ".join(
                    p.cmdline())
            except (psutil.Error, OSError):
                pass
        return out

    def openai_ok(self, model_id: str) -> bool:
        try:
            r = httpx.get(f"http://127.0.0.1:{self.port()}/v1/models", timeout=5)
            return r.status_code == 200 and any(m.get("id") == model_id for m in r.json().get("data", []))
        except (httpx.HTTPError, ValueError):
            return False


class LlamaCppBackend(Backend):
    """Runs upstream llama.cpp `llama-server` as a transient systemd user unit (own cgroup, survives agent
    restarts, memory limits enforced by systemd)."""

    name = "llamacpp"
    UNIT = "homellm-llamacpp"

    def __init__(self, cfg: dict[str, Any]):
        super().__init__(cfg)
        # no server_bin (or "managed"): use homellm's own copy from `homellm engine install`
        custom = cfg.get("server_bin")
        self.managed = not custom or custom == "managed"
        self._bin = None if self.managed else expand(custom)
        self._ld = [expand(p) for p in cfg.get("ld_library_path", [])]
        self.models_dir = Path(expand(cfg.get("models_dir", str(MODELS_DIR))))
        self.extra_dirs = [Path(expand(p)) for p in cfg.get("extra_models_dirs", [])]

    @property
    def bin(self) -> str | None:
        if not self.managed:
            return self._bin
        cur = engine.current()
        return cur["server_bin"] if cur else None

    @property
    def ld_library_path(self) -> list[str]:
        if self.managed:
            cur = engine.current()
            return [cur["dir"]] if cur else []
        return self._ld

    def engine_info(self) -> dict[str, Any]:
        if self.managed:
            return {"managed": True, **(engine.current() or {"missing": True})}
        return {"managed": False, "server_bin": self._bin}

    def available(self) -> bool:
        return bool(self.bin) and Path(self.bin).exists() and shutil.which("systemd-run") is not None

    def unavailable_reason(self) -> str:
        if not shutil.which("systemd-run"):
            return "systemd-run not found"
        if self.managed:
            return "llama-server not installed; run `homellm engine install` or press Update engine"
        return f"llama-server not found at {self._bin}"

    def port(self) -> int:
        return int(self.cfg.get("port", 1236))

    def list_models(self) -> list[dict[str, Any]]:
        """GGUF files in models_dir (as paths relative to it) and extra_models_dirs (absolute paths).
        Only the first part of a split model is listed; llama-server finds the rest next to it."""
        out = []
        for root in [self.models_dir, *self.extra_dirs]:
            if not root.is_dir():
                continue
            for p in sorted(root.rglob("*.gguf")):
                n = p.name.lower()
                if "mmproj" in n or "/mtp/" in str(p).lower() or re.search(r"-0000[2-9]-of-|-000[1-9]\d-of-", n):
                    continue
                size = sum(q.stat().st_size for q in p.parent.glob(re.sub(r"-00001-of-", "-*-of-", p.name)))
                model = str(p.relative_to(self.models_dir)) if root == self.models_dir else str(p)
                out.append({"backend": self.name, "model": model, "name": p.stem, "size_gb": round(size / 1e9, 2),
                            "mmproj": [str(m) for m in sorted(p.parent.glob("*mmproj*.gguf"))]})
        return out

    def _active(self) -> bool:
        r = subprocess.run(["systemctl", "--user", "is-active", self.UNIT], capture_output=True, text=True)
        return r.stdout.strip() in ("active", "activating")

    def is_ready(self, profile_name: str, profile: dict[str, Any]) -> bool:
        return self._active() and self.openai_ok(profile_name)

    def is_running(self) -> bool:
        return self._active()

    def start(self, profile_name: str, profile: dict[str, Any]) -> None:
        self.stop()
        model = Path(profile["model"])
        if not model.is_absolute():
            model = self.models_dir / model
        if not model.exists():
            raise BackendError(f"model file not found: {model}")
        server_bin = os.path.realpath(self.bin or "")
        cmd = [server_bin, "--model", str(model), "--alias", profile_name, "--host", "0.0.0.0",
               "--port", str(self.port()), "-ngl", str(profile.get("gpu_layers", 999)), "--jinja"]
        if profile.get("context"):
            cmd += ["-c", str(profile["context"])]
        if profile.get("parallel"):
            cmd += ["-np", str(profile["parallel"])]
        args = list(profile.get("args", []))
        if "--cache-ram" not in args:
            args += ["--cache-ram", "0"]
        cmd += [str(a) for a in args]
        run = ["systemd-run", "--user", f"--unit={self.UNIT}", "--collect", "--quiet",
               "-p", "Restart=no", "--setenv=LLAMA_ARG_CACHE_RAM=0"]
        if self.ld_library_path:
            run.append(f"--setenv=LD_LIBRARY_PATH={':'.join(os.path.realpath(p) for p in self.ld_library_path)}")
        for key in ("memory_high", "memory_max"):
            if self.cfg.get(key):
                run += ["-p", f"{'MemoryHigh' if key == 'memory_high' else 'MemoryMax'}={self.cfg[key]}"]
        r = subprocess.run(run + cmd, capture_output=True, text=True, timeout=30)
        if r.returncode != 0:
            raise BackendError(f"systemd-run failed: {r.stderr[-400:]}")

    def stop(self) -> None:
        if shutil.which("systemctl"):
            subprocess.run(["systemctl", "--user", "stop", self.UNIT], capture_output=True, timeout=60)
            subprocess.run(["systemctl", "--user", "reset-failed", self.UNIT], capture_output=True, timeout=10)

    def server_pid(self) -> int | None:
        r = subprocess.run(["systemctl", "--user", "show", self.UNIT, "-p", "MainPID", "--value"],
                           capture_output=True, text=True)
        pid = r.stdout.strip()
        return int(pid) if pid.isdigit() and pid != "0" else None

    def logs(self, lines: int = 50) -> str:
        r = subprocess.run(["journalctl", "--user", "-u", self.UNIT, "-n", str(lines), "--no-pager", "-o", "cat"],
                           capture_output=True, text=True)
        return r.stdout


BACKENDS = {"llamacpp": LlamaCppBackend}


def build_backends(cfg: dict[str, Any]) -> dict[str, Backend]:
    out = {}
    for name, bcfg in (cfg.get("backends") or {}).items():
        if name in BACKENDS and (bcfg or {}).get("enabled", True):
            out[name] = BACKENDS[name](bcfg or {})
    return out
