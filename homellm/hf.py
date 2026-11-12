"""Hugging Face GGUF models: the curated catalog, listing a repo's quants, and resumable verified downloads.

Used by the node agent (downloads), the controller (browsing for the dashboard) and the CLI (`homellm pull`).
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx

from .common import CONFIG_DIR, DEFAULT_CONTEXT, load_yaml

HF = "https://huggingface.co"
BUNDLED_CATALOG = Path(__file__).parent / "catalog.yaml"
USER_CATALOG = CONFIG_DIR / "catalog.yaml"

SPLIT_RE = re.compile(r"-(\d{5})-of-(\d{5})$")
DEFAULT_ARGS = ["--flash-attn", "on", "--cache-type-k", "q8_0", "--cache-type-v", "q8_0"]


class HFError(RuntimeError):
    pass


def hf_headers(token: str | None = None) -> dict[str, str]:
    token = token or os.environ.get("HF_TOKEN")
    return {"Authorization": f"Bearer {token}"} if token else {}


# ---- catalog ---------------------------------------------------------------------------------------------------
def load_catalog() -> list[dict[str, Any]]:
    """Bundled catalog, with entries from ~/.config/homellm/catalog.yaml added or overriding by name."""
    entries: dict[str, dict[str, Any]] = {}
    for path in (BUNDLED_CATALOG, USER_CATALOG):
        for e in load_yaml(path).get("models") or []:
            if e.get("name") and e.get("repo"):
                entries[e["name"]] = {**entries.get(e["name"], {}), **e}
    return list(entries.values())


def catalog_entry(name: str) -> dict[str, Any] | None:
    return next((e for e in load_catalog() if e["name"] == name), None)


# ---- repo listing ----------------------------------------------------------------------------------------------
def list_repo(repo: str, revision: str = "main", client: httpx.Client | None = None) -> list[dict[str, Any]]:
    """All files in a model repo: [{path, size, sha256}]."""
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", repo):
        raise HFError(f"not a Hugging Face repo id: {repo!r} (expected owner/name)")
    c = client or httpx.Client(timeout=30, follow_redirects=True)
    try:
        r = c.get(f"{HF}/api/models/{repo}/tree/{revision}", params={"recursive": "true"}, headers=hf_headers())
    finally:
        if client is None:
            c.close()
    if r.status_code in (401, 403):
        raise HFError(f"{repo} is gated or private: accept its license on huggingface.co and set HF_TOKEN")
    if r.status_code == 404:
        raise HFError(f"{repo} not found on Hugging Face")
    r.raise_for_status()
    return [
        {"path": f["path"], "size": f.get("lfs", {}).get("size") or f.get("size") or 0,
         "sha256": (f.get("lfs") or {}).get("oid")}
        for f in r.json() if f.get("type") == "file"
    ]


def _stem(path: str) -> str:
    return SPLIT_RE.sub("", Path(path).name[: -len(".gguf")])


def is_mmproj(path: str) -> bool:
    return "mmproj" in Path(path).name.lower()


def group_ggufs(files: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group a repo's .gguf files into loadable units: split parts belong together, mmproj files stand alone.

    Speculative-decoding helpers (MTP/, eagle3-) are left out: they are not models you serve on their own.
    """
    groups: dict[str, dict[str, Any]] = {}
    for f in files:
        p = f["path"]
        if not p.endswith(".gguf") or p.startswith("MTP/") or Path(p).name.startswith(("eagle", "mtp-")):
            continue
        key = str(Path(p).parent / _stem(p))
        g = groups.setdefault(key, {"name": _stem(p), "files": [], "size": 0, "mmproj": is_mmproj(p)})
        g["files"].append(f)
        g["size"] += f["size"]
    out = []
    for g in groups.values():
        g["files"].sort(key=lambda f: f["path"])
        g["size_gb"] = round(g["size"] / 1e9, 2)
        g["paths"] = [f["path"] for f in g["files"]]
        out.append(g)
    return sorted(out, key=lambda g: (g["mmproj"], g["size"]))


def find_quant(groups: list[dict[str, Any]], quant: str) -> dict[str, Any]:
    """Pick the model group whose name ends in `quant` (e.g. Q4_K_M). `Q4_K_M` does not match `UD-Q4_K_M`."""
    q = quant.lower()
    hits = []
    for g in groups:
        name = g["name"].lower()
        if g["mmproj"] or not re.search(rf"(^|[-_.]){re.escape(q)}$", name):
            continue
        if not q.startswith("ud-") and name.endswith("ud-" + q):
            continue
        hits.append(g)
    if not hits:
        have = ", ".join(g["name"] for g in groups if not g["mmproj"])
        raise HFError(f"no {quant} file; available: {have}")
    return hits[0]


def find_mmproj(groups: list[dict[str, Any]], prefer: str | None = None) -> dict[str, Any] | None:
    mm = [g for g in groups if g["mmproj"]]
    if prefer:
        mm.sort(key=lambda g: prefer.lower() not in g["name"].lower())
    else:  # F16 runs everywhere; BF16 is slow on older GPUs
        mm.sort(key=lambda g: ("f16" not in g["name"].lower() or "bf16" in g["name"].lower(), g["size"]))
    return mm[0] if mm else None


def parse_hf_url(url: str) -> tuple[str, str, str] | None:
    """https://huggingface.co/owner/repo/{resolve,blob}/rev/path/file.gguf -> (repo, revision, path)."""
    m = re.match(r"https?://(?:www\.)?(?:huggingface\.co|hf\.co)/([\w.-]+/[\w.-]+)/(?:resolve|blob)/([^/]+)/(.+?)"
                 r"(?:\?.*)?$", url)
    return (m.group(1), m.group(2), m.group(3)) if m else None


# ---- planning ----------------------------------------------------------------------------------------------------
def plan(repo: str, quant: str | None = None, files: list[str] | None = None, mmproj: bool | str = False,
         revision: str = "main", client: httpx.Client | None = None) -> dict[str, Any]:
    """Decide what to download: the model files (all split parts) and optionally a projector for vision."""
    groups = group_ggufs(list_repo(repo, revision, client))
    if files:
        wanted = set(files)
        model = next((g for g in groups if wanted & set(g["paths"]) and not g["mmproj"]), None)
        if not model:
            raise HFError(f"{', '.join(files)} not found in {repo}")
    else:
        model = find_quant(groups, quant or "Q4_K_M")
    proj = None
    if mmproj:
        proj = find_mmproj(groups, mmproj if isinstance(mmproj, str) else None)
    return {"repo": repo, "revision": revision, "model": model, "mmproj": proj,
            "size": model["size"] + (proj["size"] if proj else 0)}


def local_dir(models_dir: Path, repo: str) -> Path:
    """LM Studio's layout, <publisher>/<repo>/, so LM Studio sees downloads too (and unsloth's identically named
    mmproj-F16.gguf files don't collide)."""
    owner, name = repo.split("/")[:2]
    return models_dir / owner / name


def profile_for(plan_: dict[str, Any], models_dir: Path, entry: dict[str, Any] | None = None) -> tuple[str, dict]:
    """A ready-to-load profile for a downloaded plan, using the catalog entry's settings when there is one."""
    entry = entry or {}
    d = local_dir(models_dir, plan_["repo"])
    first = d / plan_["model"]["paths"][0]
    model = str(first.relative_to(models_dir))
    args = [str(a) for a in entry.get("args", DEFAULT_ARGS)]
    if plan_["mmproj"]:
        args = ["--mmproj", str(d / plan_["mmproj"]["paths"][0]), *args]
    name = entry.get("name") or re.sub(r"[^a-z0-9.-]+", "-", plan_["model"]["name"].lower()).strip("-")[:48]
    profile = {"backend": "llamacpp", "model": model, "context": int(entry.get("context", DEFAULT_CONTEXT)),
               "parallel": int(entry.get("parallel", 1)), "args": args,
               "source": f"hf://{plan_['repo']}/{plan_['model']['paths'][0]}"}
    return name, profile


# ---- downloading -------------------------------------------------------------------------------------------------
Progress = Callable[[int, int, str], None]


def download_file(url: str, dest: Path, size: int = 0, sha256: str | None = None, progress: Progress | None = None,
                  client: httpx.Client | None = None, headers: dict[str, str] | None = None,
                  should_stop: Callable[[], bool] | None = None) -> Path:
    """Download to dest via dest.part, resuming a partial file, and check size and sha256 before renaming."""
    if dest.exists() and (not size or dest.stat().st_size == size):
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    have = part.stat().st_size if part.exists() else 0
    if size and have > size:
        part.unlink()
        have = 0
    need = max(size - have, 0)
    free = shutil.disk_usage(dest.parent).free
    if need and free < need + (1 << 30):
        raise HFError(f"not enough disk space in {dest.parent}: need {need / 1e9:.1f} GB, "
                      f"{free / 1e9:.1f} GB free")
    hasher = hashlib.sha256() if sha256 else None
    if hasher and have:
        with part.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 24), b""):
                hasher.update(chunk)
    c = client or httpx.Client(timeout=httpx.Timeout(60, read=300), follow_redirects=True)
    hdrs = dict(headers or {})
    if have:
        hdrs["Range"] = f"bytes={have}-"
    try:
        with c.stream("GET", url, headers=hdrs) as r:
            if r.status_code == 416 and size and have == size:
                pass  # already complete
            else:
                if r.status_code in (401, 403):
                    raise HFError(f"access denied for {url}: gated model? set HF_TOKEN")
                r.raise_for_status()
                if have and r.status_code != 206:  # server ignored Range: start over
                    have = 0
                    hasher = hashlib.sha256() if sha256 else None
                total = size or have + int(r.headers.get("content-length") or 0)
                with part.open("ab" if have else "wb") as f:
                    done = have
                    for chunk in r.iter_bytes(1 << 20):
                        if should_stop and should_stop():
                            raise HFError("cancelled")
                        f.write(chunk)
                        if hasher:
                            hasher.update(chunk)
                        done += len(chunk)
                        if progress:
                            progress(done, total, dest.name)
    finally:
        if client is None:
            c.close()
    got = part.stat().st_size
    if size and got != size:
        raise HFError(f"{dest.name}: got {got} bytes, expected {size}; run the download again to resume")
    if hasher and hasher.hexdigest() != sha256:
        part.unlink()
        raise HFError(f"{dest.name}: sha256 mismatch, deleted the partial file")
    part.replace(dest)
    return dest


def fetch_plan(plan_: dict[str, Any], models_dir: Path, progress: Progress | None = None,
               client: httpx.Client | None = None, should_stop: Callable[[], bool] | None = None) -> list[Path]:
    """Download every file of a plan into models_dir/<repo name>/, reporting combined progress."""
    files = list(plan_["model"]["files"]) + (list(plan_["mmproj"]["files"]) if plan_["mmproj"] else [])
    total = sum(f["size"] for f in files)
    base = 0
    out = []
    d = local_dir(models_dir, plan_["repo"])
    for f in files:
        dest = d / f["path"]
        url = f"{HF}/{plan_['repo']}/resolve/{plan_['revision']}/{f['path']}"

        def step(done: int, _t: int, name: str, base: int = base) -> None:
            if progress:
                progress(base + done, total, name)

        out.append(download_file(url, dest, f["size"], f.get("sha256"), step, client, hf_headers(), should_stop))
        base += f["size"]
        if progress:
            progress(base, total, dest.name)
    return out
