"""homellm's own copy of llama.cpp's llama-server, installed from the official ggml-org release builds.

Layout (per machine, no root):

    ~/.local/share/homellm/llama.cpp/
        b12170-cuda-12/           llama-server + its libs (+ CUDA runtime libs for CUDA builds)
        current -> b12170-cuda-12

A new version is unpacked next to the old one, checked with `llama-server --version`, and only then does
`current` move. A running llama-server keeps the version it started with until its next (re)load.
"""

from __future__ import annotations

import functools
import json
import os
import platform
import re
import shutil
import subprocess
import tarfile
import time
from pathlib import Path
from typing import Any

import httpx

from .hf import Progress, download_file

REPO = "ggml-org/llama.cpp"
API = f"https://api.github.com/repos/{REPO}"
SHARE_DIR = Path(os.environ.get("HOMELLM_SHARE_DIR", Path.home() / ".local" / "share" / "homellm"))
ENGINE_DIR = SHARE_DIR / "llama.cpp"
CURRENT = ENGINE_DIR / "current"
MANIFEST = "homellm-engine.json"
KEEP_VERSIONS = 2

FLAVORS = ("cuda-12", "cuda-13", "rocm", "vulkan", "cpu")


class EngineError(RuntimeError):
    pass


# ---- picking a build ---------------------------------------------------------------------------------------------
def _arch() -> str:
    m = platform.machine().lower()
    return "arm64" if m in ("aarch64", "arm64") else "x64"


def nvidia_compute_caps() -> list[float]:
    if not shutil.which("nvidia-smi"):
        return []
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10).stdout
        return [float(x) for x in out.split() if re.fullmatch(r"\d+\.\d+", x)]
    except (OSError, subprocess.SubprocessError, ValueError):
        return []


def amd_gpu() -> bool:
    for dev in Path("/sys/class/drm").glob("card[0-9]*/device"):
        try:
            if (dev / "vendor").read_text().strip() == "0x1002" and (dev / "mem_info_vram_total").exists():
                return True
        except OSError:
            continue
    return False


def has_vulkan() -> bool:
    return any(Path(p).exists() for p in ("/usr/lib/x86_64-linux-gnu/libvulkan.so.1", "/usr/lib/libvulkan.so.1",
                                          "/usr/lib64/libvulkan.so.1", "/usr/lib/aarch64-linux-gnu/libvulkan.so.1"))


def detect_flavor() -> str:
    """Best build for this machine. CUDA 13 dropped Pascal/Volta (P40, P100, V100), so they get CUDA 12."""
    caps = nvidia_compute_caps()
    if caps:
        if _arch() == "arm64":
            return "cuda-13"
        return "cuda-13" if min(caps) >= 7.5 else "cuda-12"
    if amd_gpu():
        return "rocm" if Path("/opt/rocm").exists() else "vulkan"
    return "vulkan" if has_vulkan() else "cpu"


@functools.lru_cache(maxsize=1)
def detect_flavor_cached() -> str:
    return detect_flavor()


def asset_patterns(flavor: str, tag: str, arch: str | None = None) -> tuple[re.Pattern, re.Pattern | None]:
    """Regexes for the main archive and (CUDA only) the CUDA runtime archive of a release."""
    arch = arch or _arch()
    t = re.escape(tag)
    if flavor.startswith("cuda-"):
        major = flavor.split("-")[1]
        main = re.compile(rf"llama-{t}-bin-ubuntu-cuda-{major}\.[\d.]+-{arch}\.tar\.gz")
        rt = re.compile(rf"cudart-llama-{t}-bin-ubuntu-cuda-{major}\.[\d.]+-{arch}\.tar\.gz")
        return main, rt
    if flavor == "rocm":
        return re.compile(rf"llama-{t}-bin-ubuntu-rocm-[\d.]+-{arch}\.tar\.gz"), None
    if flavor == "vulkan":
        return re.compile(rf"llama-{t}-bin-ubuntu-vulkan-{arch}\.tar\.gz"), None
    if flavor == "cpu":
        return re.compile(rf"llama-{t}-bin-ubuntu-{arch}\.tar\.gz"), None
    raise EngineError(f"unknown flavor {flavor!r}; one of {', '.join(FLAVORS)}")


def pick_assets(release: dict[str, Any], flavor: str, arch: str | None = None) -> list[dict[str, Any]] | None:
    main_re, rt_re = asset_patterns(flavor, release["tag_name"], arch)
    assets = release.get("assets") or []
    main = next((a for a in assets if main_re.fullmatch(a["name"])), None)
    if not main:
        return None
    out = [main]
    if rt_re:
        rt = next((a for a in assets if rt_re.fullmatch(a["name"])), None)
        if not rt:
            return None
        out.append(rt)
    return out


def _gh(client: httpx.Client, path: str) -> Any:
    hdrs = {"Accept": "application/vnd.github+json"}
    if os.environ.get("GITHUB_TOKEN"):
        hdrs["Authorization"] = f"Bearer {os.environ['GITHUB_TOKEN']}"
    r = client.get(f"{API}{path}", headers=hdrs)
    if r.status_code == 403 and "rate limit" in r.text.lower():
        raise EngineError("GitHub API rate limit hit; wait an hour or set GITHUB_TOKEN")
    if r.status_code == 404:
        raise EngineError(f"llama.cpp release not found: {path}")
    r.raise_for_status()
    return r.json()


def find_release(flavor: str, tag: str | None = None, client: httpx.Client | None = None) -> tuple[dict, list]:
    """The given tag, or the newest release that has a build for this flavor (llama.cpp ships many a day)."""
    c = client or httpx.Client(timeout=30, follow_redirects=True)
    try:
        releases = [_gh(c, f"/releases/tags/{tag}")] if tag else _gh(c, "/releases?per_page=10")
    finally:
        if client is None:
            c.close()
    for rel in releases:
        if rel.get("draft") or not re.fullmatch(r"b\d+", rel.get("tag_name", "")):
            continue
        assets = pick_assets(rel, flavor)
        if assets:
            return rel, assets
    raise EngineError(f"no {flavor} build for {_arch()} in {'release ' + tag if tag else 'recent releases'}")


# ---- installing ----------------------------------------------------------------------------------------------------
def safe_extract(archive: Path, dest: Path) -> None:
    """Unpack a release tarball into dest, dropping its top folder. Refuses paths and links that escape dest."""
    dest.mkdir(parents=True, exist_ok=True)
    root = dest.resolve()
    with tarfile.open(archive) as tar:
        for m in tar.getmembers():
            parts = Path(m.name).parts
            rel = Path(*parts[1:]) if len(parts) > 1 else None
            if rel is None or m.isdir():
                continue
            target = (dest / rel).resolve()
            if not str(target).startswith(str(root) + os.sep) or rel.is_absolute() or ".." in rel.parts:
                raise EngineError(f"unsafe path in {archive.name}: {m.name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            if m.issym():
                link = Path(m.linkname)
                if link.is_absolute() or not str((target.parent / link).resolve()).startswith(str(root) + os.sep):
                    raise EngineError(f"unsafe link in {archive.name}: {m.name} -> {m.linkname}")
                if target.is_symlink() or target.exists():
                    target.unlink()
                target.symlink_to(link)
            elif m.isfile():
                src = tar.extractfile(m)
                if src is None:
                    continue
                with src, target.open("wb") as f:
                    shutil.copyfileobj(src, f, 1 << 20)
                target.chmod(0o755 if m.mode & 0o111 else 0o644)
            # hard links, devices, fifos: not in llama.cpp archives, skipped


def server_version(bin_dir: Path) -> str:
    """`llama-server --version` with the bundled libs; raises if the build cannot run on this machine."""
    exe = bin_dir / "llama-server"
    if not exe.exists():
        raise EngineError(f"no llama-server in {bin_dir}")
    env = {**os.environ, "LD_LIBRARY_PATH": str(bin_dir)}
    try:
        r = subprocess.run([str(exe), "--version"], capture_output=True, text=True, timeout=60, env=env)
    except (OSError, subprocess.SubprocessError) as e:
        raise EngineError(f"llama-server does not run: {e}") from e
    text = (r.stdout + r.stderr).strip()
    m = re.search(r"version: .*", text)
    if r.returncode != 0 or not m:
        raise EngineError(f"llama-server --version failed: {text[-400:]}")
    return m.group(0)


def current() -> dict[str, Any] | None:
    """The installed engine `current` points at, or None."""
    if not CURRENT.exists():
        return None
    real = CURRENT.resolve()
    try:
        info = json.loads((real / MANIFEST).read_text())
    except (OSError, ValueError):
        info = {}
    return {**info, "dir": str(real), "server_bin": str(real / "llama-server")}


def installed() -> list[dict[str, Any]]:
    out = []
    for d in sorted(ENGINE_DIR.glob("b*-*")):
        try:
            out.append({**json.loads((d / MANIFEST).read_text()), "dir": str(d)})
        except (OSError, ValueError):
            continue
    return out


def _activate(d: Path) -> None:
    tmp = ENGINE_DIR / ".current.tmp"
    if tmp.is_symlink() or tmp.exists():
        tmp.unlink()
    tmp.symlink_to(d.name)
    tmp.replace(CURRENT)


def _prune(keep: Path) -> None:
    dirs = sorted((d for d in ENGINE_DIR.glob("b*-*") if d.is_dir() and d != keep),
                  key=lambda d: d.stat().st_mtime, reverse=True)
    for d in dirs[KEEP_VERSIONS - 1:]:
        shutil.rmtree(d, ignore_errors=True)


def install(flavor: str | None = None, tag: str | None = None, force: bool = False,
            progress: Progress | None = None, log=print) -> dict[str, Any]:
    """Install (or switch to) a llama.cpp release build and point `current` at it."""
    flavor = flavor or detect_flavor()
    if flavor not in FLAVORS:
        raise EngineError(f"unknown flavor {flavor!r}; one of {', '.join(FLAVORS)}")
    with httpx.Client(timeout=httpx.Timeout(60, read=300), follow_redirects=True) as c:
        rel, assets = find_release(flavor, tag, c)
        tag = rel["tag_name"]
        d = ENGINE_DIR / f"{tag}-{flavor}"
        if (d / MANIFEST).exists() and not force:
            log(f"{tag} ({flavor}) already installed")
            _activate(d)
            return current() or {}
        log(f"installing llama.cpp {tag} ({flavor}): {', '.join(a['name'] for a in assets)}")
        dl = ENGINE_DIR / "dl"
        staging = ENGINE_DIR / f".staging-{tag}-{flavor}"
        shutil.rmtree(staging, ignore_errors=True)
        try:
            for a in assets:
                archive = download_file(a["browser_download_url"], dl / a["name"], a.get("size") or 0,
                                        (a.get("digest") or "").removeprefix("sha256:") or None, progress, c)
                safe_extract(archive, staging)
            version = server_version(staging)
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise
    (staging / MANIFEST).write_text(json.dumps({"tag": tag, "flavor": flavor, "version": version,
                                                "assets": [a["name"] for a in assets],
                                                "installed_at": int(time.time())}, indent=2))
    shutil.rmtree(d, ignore_errors=True)
    staging.rename(d)
    _activate(d)
    for a in assets:  # archives are big (CUDA runtime ~600 MB); the unpacked copy is all we need
        (dl / a["name"]).unlink(missing_ok=True)
    _prune(d)
    log(f"llama-server {tag} ready: {version}")
    return current() or {}


def latest_tag(flavor: str | None = None) -> str:
    rel, _ = find_release(flavor or detect_flavor())
    return rel["tag_name"]
