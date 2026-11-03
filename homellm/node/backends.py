"""Inference backends a node can run models on.

A *profile* is a named, ready-to-run model preset, e.g.

    gemma-4-26b:
      backend: lmstudio
      model: google/gemma-4-26b-a4b
      context: 50000
      parallel: 2

The served model id (what LiteLLM and clients call) is the profile name.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import httpx
import psutil

from ..common import expand

log = logging.getLogger("homellm.node")

# llama.cpp's server keeps a RAM prompt cache (default ~8 GB) that OOM-killed LM Studio on a 13 GB box.
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


class LMStudioBackend(Backend):
    """Drives LM Studio through its `lms` CLI. LM Studio serves an OpenAI API on its own port."""

    name = "lmstudio"

    def __init__(self, cfg: dict[str, Any]):
        super().__init__(cfg)
        self.lms = expand(cfg.get("lms", "~/.lmstudio/bin/lms"))
        self.server_cfg = Path(expand("~/.lmstudio/.internal/http-server-config.json"))

    def _run(self, *args: str, timeout: int = 60) -> subprocess.CompletedProcess:
        env = {**os.environ, **NO_PROMPT_CACHE_ENV}
        return subprocess.run([self.lms, *args], capture_output=True, text=True, timeout=timeout, env=env)

    def available(self) -> bool:
        return bool(self.lms) and Path(self.lms).exists()

    def port(self) -> int:
        try:
            return int(json.loads(self.server_cfg.read_text()).get("port", 1234))
        except (OSError, ValueError):
            return int(self.cfg.get("port", 1234))

    def network_interface(self) -> str:
        try:
            return json.loads(self.server_cfg.read_text()).get("networkInterface", "127.0.0.1")
        except (OSError, ValueError):
            return "unknown"

    def ensure_server(self) -> None:
        r = self._run("server", "status")
        if "running" not in (r.stdout + r.stderr).lower() or "not running" in (r.stdout + r.stderr).lower():
            self._run("server", "start", "--port", str(self.port()), timeout=120)

    def _json(self, *args: str) -> list[dict[str, Any]]:
        r = self._run(*args, "--json")
        try:
            return json.loads(r.stdout or "[]")
        except ValueError:
            return []

    def list_models(self) -> list[dict[str, Any]]:
        return [
            {
                "backend": self.name,
                "model": m.get("modelKey"),
                "name": m.get("displayName") or m.get("modelKey"),
                "size_gb": round((m.get("sizeBytes") or 0) / 1e9, 2),
                "max_context": m.get("maxContextLength"),
                "quant": (m.get("quantization") or {}).get("name"),
            }
            for m in self._json("ls")
            if m.get("type") == "llm"
        ]

    def loaded(self) -> list[dict[str, Any]]:
        return self._json("ps")

    def is_ready(self, profile_name: str, profile: dict[str, Any]) -> bool:
        for m in self.loaded():
            if m.get("identifier") == profile_name and m.get("modelKey") == profile["model"]:
                return str(m.get("status", "")).lower() in ("idle", "generating", "processingprompt", "")
        return False

    def start(self, profile_name: str, profile: dict[str, Any]) -> None:
        self.ensure_server()
        self._run("unload", "--all")
        args = ["load", profile["model"], "--gpu", str(profile.get("gpu", "max")),
                "--identifier", profile_name, "-y"]
        if profile.get("context"):
            args += ["--context-length", str(profile["context"])]
        if profile.get("parallel"):
            args += ["--parallel", str(profile["parallel"])]
        r = self._run(*args, timeout=900)
        if r.returncode != 0:
            raise BackendError(f"lms load failed: {(r.stderr or r.stdout)[-400:]}")

    def stop(self) -> None:
        if self.available():
            self._run("unload", "--all")

    def server_pid(self) -> int | None:
        best = None
        for p in psutil.process_iter(["pid", "exe", "cmdline", "create_time"]):
            try:
                cmd = p.info["cmdline"] or []
                if cmd and cmd[0].endswith("llama-server") and ".lmstudio" in cmd[0]:
                    if best is None or p.info["create_time"] > best[1]:
                        best = (p.info["pid"], p.info["create_time"])
            except (psutil.Error, TypeError):
                continue
        return best[0] if best else None

    def download(self, name: str) -> subprocess.Popen:
        env = {**os.environ, **NO_PROMPT_CACHE_ENV}
        return subprocess.Popen([self.lms, "get", name, "-y"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, env=env)

    def health(self) -> dict[str, Any]:
        out = super().health()
        if self.network_interface() in ("127.0.0.1", "localhost"):
            out["warnings"].append("LM Studio server only listens on localhost; enable 'Serve on local network'")
        if out.get("pid") and out.get("prompt_cache_disabled") is False:
            out["warnings"].append("LM Studio was started without LLAMA_ARG_CACHE_RAM=0; RAM can climb")
        return out


class LlamaCppBackend(Backend):
    """Runs upstream llama.cpp `llama-server` as a transient systemd user unit (own cgroup, survives agent
    restarts, memory limits enforced by systemd)."""

    name = "llamacpp"
    UNIT = "homellm-llamacpp"

    def __init__(self, cfg: dict[str, Any]):
        super().__init__(cfg)
        self.bin = expand(cfg.get("server_bin"))
        self.models_dir = Path(expand(cfg.get("models_dir", "~/models")))
        self.ld_library_path = [expand(p) for p in cfg.get("ld_library_path", [])]

    def available(self) -> bool:
        return bool(self.bin) and Path(self.bin).exists() and shutil.which("systemd-run") is not None

    def port(self) -> int:
        return int(self.cfg.get("port", 1236))

    def list_models(self) -> list[dict[str, Any]]:
        if not self.models_dir.is_dir():
            return []
        return [
            {"backend": self.name, "model": p.name, "name": p.stem, "size_gb": round(p.stat().st_size / 1e9, 2)}
            for p in sorted(self.models_dir.rglob("*.gguf"))
            if "mmproj" not in p.name.lower()
        ]

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
        cmd = [self.bin, "--model", str(model), "--alias", profile_name, "--host", "0.0.0.0",
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
            run.append(f"--setenv=LD_LIBRARY_PATH={':'.join(self.ld_library_path)}")
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


BACKENDS = {"lmstudio": LMStudioBackend, "llamacpp": LlamaCppBackend}


def build_backends(cfg: dict[str, Any]) -> dict[str, Backend]:
    out = {}
    for name, bcfg in (cfg.get("backends") or {}).items():
        if name in BACKENDS and (bcfg or {}).get("enabled", True):
            out[name] = BACKENDS[name](bcfg or {})
    return out
