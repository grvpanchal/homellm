"""Write starter config files by looking at what is on this machine, and upgrade older ones."""

from __future__ import annotations

import re
import shutil
import socket
from pathlib import Path
from typing import Any

from . import engine
from .common import CONFIG_DIR, DEFAULT_CONTEXT, MODELS_DIR, STATE_DIR, load_yaml, save_yaml

CONFIG_VERSION = 2
LMSTUDIO_MODELS = MODELS_DIR
DEFAULT_ARGS = ["--flash-attn", "on", "--cache-type-k", "q8_0", "--cache-type-v", "q8_0"]


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9.-]+", "-", s.lower()).strip("-")[:48]


def _models_dir() -> str:
    LMSTUDIO_MODELS.mkdir(parents=True, exist_ok=True)
    return str(LMSTUDIO_MODELS)


def _extra_dirs(models_dir: str) -> list[str]:
    """Folders with GGUFs from other tools, so nothing has to be downloaded twice."""
    return [str(p) for p in (LMSTUDIO_MODELS, Path.home() / "models", Path.home() / "llama.cpp" / "models")
            if p.is_dir() and p.resolve() != Path(models_dir).expanduser().resolve()]


def _first_parts(root: Path) -> list[Path]:
    return [g for g in sorted(root.rglob("*.gguf"))
            if "mmproj" not in g.name.lower() and "/MTP/" not in str(g) and not re.search(r"-000(0[2-9]|[1-9]\d)-of-",
                                                                                           g.name)]


def _mmproj_for(gguf: Path) -> Path | None:
    mm = sorted(gguf.parent.glob("*mmproj*.gguf"), key=lambda p: ("bf16" in p.name.lower(), p.name))
    return mm[0] if mm and gguf.parent != Path.home() / "models" else None


def init_node(force: bool = False) -> Path:
    path = CONFIG_DIR / "node.yaml"
    if path.exists() and not force:
        for note in migrate_node():
            print(note)
        return path
    models_dir = _models_dir()
    lc: dict[str, Any] = {"enabled": True, "models_dir": models_dir, "port": 1236}
    if extra := _extra_dirs(models_dir):
        lc["extra_models_dirs"] = extra
    profiles: dict[str, Any] = {}
    for root in [Path(models_dir), *map(Path, extra)]:
        for g in _first_parts(root) if root.is_dir() else []:
            name = slug(re.sub(r"-00001-of-\d{5}$", "", g.stem))
            if name in profiles:
                continue
            mm = _mmproj_for(g)
            profiles[name] = {
                "backend": "llamacpp",
                "model": str(g.relative_to(models_dir)) if root == Path(models_dir) else str(g),
                "context": DEFAULT_CONTEXT, "parallel": 1,
                "args": (["--mmproj", str(mm)] if mm else []) + DEFAULT_ARGS}
    cfg = {
        "config_version": CONFIG_VERSION,
        "name": socket.gethostname(),
        "port": 8765,
        "mdns": True,
        "default_profile": None,
        "watchdog": {"interval": 30, "warn_available_mb": 1500, "warn_rss_mb": 4000, "max_restarts_per_hour": 6},
        "backends": {"llamacpp": lc},
        "profiles": profiles,
    }
    save_yaml(path, cfg)
    return path


# ---- upgrade: LM Studio -> homellm's own llama-server ------------------------------------------------------------
def find_lmstudio_gguf(model_key: str, root: Path = LMSTUDIO_MODELS) -> Path | None:
    """LM Studio keys (google/gemma-4-26b-a4b) don't map 1:1 to folders (lmstudio-community/gemma-4-26B-A4B-it-GGUF);
    match the key's last part against folder and file names."""
    if not root.is_dir():
        return None
    want = re.sub(r"[-_.]?gguf$", "", model_key.split("/")[-1].lower())
    hits = []
    for g in _first_parts(root):
        hay = f"{g.parent.name} {g.stem}".lower()
        if want in hay:
            hits.append(g)
    exact = [g for g in hits if model_key.lower() in str(g).lower()]
    hits = exact or hits
    # prefer a 4-bit quant: it is what LM Studio downloads by default
    hits.sort(key=lambda g: ("q4_k_m" not in g.name.lower(), g.stat().st_size))
    return hits[0] if hits else None


def _resolve(model: str, models_dir: Path) -> Path:
    p = Path(model).expanduser()
    return (p if p.is_absolute() else models_dir / p).resolve()


def _use_lmstudio_dir(cfg: dict[str, Any], lm_root: Path) -> list[str]:
    """Point models_dir at LM Studio's folder when the configured one is gone (e.g. ~/models was deleted)."""
    lc = (cfg.get("backends") or {}).get("llamacpp")
    if not lc or not lc.get("models_dir"):
        return []
    old = Path(lc["models_dir"]).expanduser()
    if old.is_dir() or old == lm_root:
        return []
    lm_root.mkdir(parents=True, exist_ok=True)
    lc["models_dir"] = str(lm_root)
    lc["extra_models_dirs"] = [d for d in lc.get("extra_models_dirs", [])
                               if Path(d).expanduser() != lm_root and Path(d).expanduser().is_dir()]
    if not lc["extra_models_dirs"]:
        lc.pop("extra_models_dirs")
    return [f"models_dir: {old} is gone, using {lm_root}"]


def migrate_node(config_dir: Path | None = None, state_dir: Path | None = None,
                 lm_root: Path = LMSTUDIO_MODELS) -> list[str]:
    """Upgrade node.yaml / profiles.yaml in place (backups kept as *.bak-v1):

    - drop the LM Studio backend; turn each LM Studio profile into a llama.cpp one that runs the same GGUF from
      ~/.lmstudio/models. If an equivalent llama.cpp profile already exists, it takes over the LM Studio profile's
      name (the model id clients use) and the duplicate goes away;
    - use homellm's own llama-server when it is installed.
    """
    cdir, sdir = config_dir or CONFIG_DIR, state_dir or STATE_DIR
    path, ppath, spath = cdir / "node.yaml", cdir / "profiles.yaml", sdir / "node-state.yaml"
    if not path.exists():
        return []
    cfg = load_yaml(path)
    notes = _use_lmstudio_dir(cfg, lm_root)
    if notes:
        save_yaml(path, cfg)
    if int(cfg.get("config_version", 1)) >= CONFIG_VERSION:
        return notes
    extra, state = load_yaml(ppath), load_yaml(spath)
    backends = cfg.setdefault("backends", {})
    lc = backends.get("llamacpp") or {}
    lc.setdefault("enabled", True)
    lc.setdefault("models_dir", _models_dir())
    lc.setdefault("port", 1236)
    if backends.pop("lmstudio", None) is not None:
        notes.append("removed the LM Studio backend")
    if lm_root.is_dir() and str(lm_root) not in lc.get("extra_models_dirs", []):
        lc["extra_models_dirs"] = [*lc.get("extra_models_dirs", []), str(lm_root)]
    if lc.get("server_bin") and engine.current():
        notes.append(f"llama-server: now homellm's own copy (was {lc['server_bin']})")
        lc.pop("server_bin")
        lc.pop("ld_library_path", None)
    backends["llamacpp"] = lc
    models_dir = Path(lc["models_dir"]).expanduser()

    stores = {"node.yaml": cfg.setdefault("profiles", {}), "profiles.yaml": extra}

    def llamacpp_twin(gguf: Path) -> tuple[str, str] | None:
        for sname, profs in stores.items():
            for pname, p in profs.items():
                if p.get("backend") == "llamacpp" and _resolve(str(p.get("model", "")), models_dir) == gguf.resolve():
                    return sname, pname
        return None

    for profs in stores.values():
        for name, p in list(profs.items()):
            if p.get("backend") != "lmstudio":
                continue
            gguf = find_lmstudio_gguf(str(p.get("model", "")), lm_root)
            if not gguf:
                del profs[name]
                notes.append(f"profile {name}: LM Studio model {p.get('model')} not found on disk, removed; "
                             "download it from the catalog")
                if state.get("active") == name:
                    state["active"] = None
                continue
            twin = llamacpp_twin(gguf)
            if twin:
                tstore, tname = twin
                new = dict(stores[tstore].pop(tname))
                if state.get("active") == tname:
                    state["active"] = name
                notes.append(f"profile {name}: now llama.cpp, using the settings of {tname} (merged into it)")
            else:
                mm = _mmproj_for(gguf)
                new = {"backend": "llamacpp", "model": str(gguf), "context": p.get("context", DEFAULT_CONTEXT),
                       "parallel": p.get("parallel", 1),
                       "args": (["--mmproj", str(mm)] if mm else []) + ["--flash-attn", "on"]}
                notes.append(f"profile {name}: now llama.cpp, running {gguf}")
            profs[name] = new

    cfg["config_version"] = CONFIG_VERSION
    for p, data in ((path, cfg), (ppath, extra)):
        if p.exists():
            shutil.copy2(p, p.with_name(p.name + ".bak-v1"))
            save_yaml(p, data)
    if spath.exists():
        save_yaml(spath, state)
    return notes


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
