"""Write starter config files by looking at what is installed on this machine."""

from __future__ import annotations

import json
import re
import socket
import subprocess
from pathlib import Path

from .common import CONFIG_DIR, load_yaml, save_yaml


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9.-]+", "-", s.lower()).strip("-")[:48]


def detect_lmstudio() -> dict | None:
    lms = Path.home() / ".lmstudio" / "bin" / "lms"
    return {"enabled": True, "lms": "~/.lmstudio/bin/lms"} if lms.exists() else None


def detect_llamacpp() -> dict | None:
    candidates = sorted(Path.home().glob("llama.cpp/**/llama-server"), key=lambda p: p.stat().st_mtime, reverse=True)
    candidates += [Path(p) for p in ("/usr/local/bin/llama-server", "/usr/bin/llama-server") if Path(p).exists()]
    bins = [p for p in candidates if p.is_file() and ".lmstudio" not in str(p)]
    if not bins:
        return None
    binp = bins[0]
    libdirs = [str(binp.parent)] + [str(p) for p in binp.parent.parent.glob("cudart-*") if p.is_dir()]
    models = next((str(p) for p in (Path.home() / "models", Path.home() / "llama.cpp" / "models") if p.is_dir()),
                  "~/models")
    return {"enabled": True, "server_bin": str(binp), "ld_library_path": libdirs, "models_dir": models,
            "port": 1236}


def lmstudio_profiles() -> dict:
    lms = Path.home() / ".lmstudio" / "bin" / "lms"
    def lms_json(cmd: str) -> list:
        try:
            return json.loads(subprocess.run([str(lms), cmd, "--json"], capture_output=True, text=True,
                                             timeout=60).stdout or "[]")
        except (OSError, ValueError, subprocess.SubprocessError):
            return []

    out = {}
    # a model that is loaded right now keeps its identifier and settings, so the agent adopts it without a reload
    for m in lms_json("ps"):
        if m.get("type") == "llm" and m.get("identifier"):
            out[m["identifier"]] = {"backend": "lmstudio", "model": m["modelKey"],
                                    "context": m.get("contextLength") or 8192, "parallel": m.get("parallel") or 1}
    loaded = {p["model"] for p in out.values()}
    for m in lms_json("ls"):
        if m.get("type") != "llm" or m["modelKey"] in loaded:
            continue
        out[slug(m["modelKey"].split("/")[-1])] = {
            "backend": "lmstudio", "model": m["modelKey"],
            "context": min(m.get("maxContextLength") or 8192, 32768), "parallel": 1}
    return out


def init_node(force: bool = False) -> Path:
    path = CONFIG_DIR / "node.yaml"
    if path.exists() and not force:
        return path
    backends = {}
    if (lm := detect_lmstudio()):
        backends["lmstudio"] = lm
    if (lc := detect_llamacpp()):
        backends["llamacpp"] = lc
    profiles = lmstudio_profiles() if "lmstudio" in backends else {}
    if "llamacpp" in backends:
        models_dir = Path(backends["llamacpp"]["models_dir"]).expanduser()
        for g in sorted(models_dir.glob("*.gguf")) if models_dir.is_dir() else []:
            if "mmproj" in g.name.lower():
                continue
            name = slug(g.stem)
            profiles.setdefault(name if name not in profiles else f"{name}-llamacpp", {
                "backend": "llamacpp", "model": g.name, "context": 32768, "parallel": 1,
                "args": ["--flash-attn", "on", "--cache-type-k", "q8_0", "--cache-type-v", "q8_0"]})
    cfg = {
        "name": socket.gethostname(),
        "port": 8765,
        "mdns": True,
        "default_profile": None,
        "watchdog": {"interval": 30, "warn_available_mb": 1500, "warn_rss_mb": 4000, "max_restarts_per_hour": 6},
        "backends": backends,
        "profiles": profiles,
    }
    save_yaml(path, cfg)
    return path


def init_controller(force: bool = False) -> Path:
    path = CONFIG_DIR / "controller.yaml"
    if path.exists() and not force:
        return path
    save_yaml(path, {
        "port": 8780,
        "litellm_url": "http://127.0.0.1:4000",
        "public_url": None,
        "poll_interval": 10,
        "discovery": {"mdns": True, "scan": True, "scan_interval": 300, "subnets": "auto", "static_nodes": []},
    })
    return path


def set_public_url(url: str) -> None:
    path = CONFIG_DIR / "controller.yaml"
    cfg = load_yaml(path)
    cfg["public_url"] = url
    save_yaml(path, cfg)
