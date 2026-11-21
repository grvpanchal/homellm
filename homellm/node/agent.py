"""Node agent: runs on every machine with a GPU.

- Keeps the *active* profile loaded (watchdog), restarting the engine if it dies.
- Watches memory and the kernel OOM killer, alerts before things fall over.
- Exposes a small authenticated REST API for the controller.
- Advertises itself on the LAN over mDNS (_homellm._tcp).
"""

from __future__ import annotations

import collections
import itertools
import logging
import socket
import subprocess
import threading
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel
from zeroconf import IPVersion, ServiceInfo
from zeroconf.asyncio import AsyncZeroconf

from .. import NODE_PORT, SERVICE_TYPE, __version__, engine, hf, initcfg
from ..common import (
    DEFAULT_CONTEXT,
    CONFIG_DIR,
    STATE_DIR,
    bearer_guard,
    gpu_processes,
    gpu_stats,
    load_yaml,
    memory_stats,
    node_id,
    notify,
    primary_ip,
    save_yaml,
)
from .backends import Backend, BackendError, build_backends

log = logging.getLogger("homellm.node")

CONFIG_PATH = CONFIG_DIR / "node.yaml"
PROFILES_PATH = CONFIG_DIR / "profiles.yaml"
STATE_PATH = STATE_DIR / "node-state.yaml"


class Node:
    def __init__(self, token: str):
        self.token = token
        try:
            migration = initcfg.migrate_node(CONFIG_PATH.parent, STATE_PATH.parent)
        except Exception as e:  # a bad upgrade must not keep the agent from starting
            migration = [f"config upgrade failed: {e}"]
        self.cfg = load_yaml(CONFIG_PATH)
        self.name = self.cfg.get("name") or socket.gethostname()
        self.id = node_id()
        self.backends: dict[str, Backend] = build_backends(self.cfg)
        wd = self.cfg.get("watchdog") or {}
        self.interval = int(wd.get("interval", 30))
        self.warn_avail_mb = int(wd.get("warn_available_mb", 1500))
        self.warn_rss_mb = int(wd.get("warn_rss_mb", 4000))
        self.max_restarts = int(wd.get("max_restarts_per_hour", 6))
        state = load_yaml(STATE_PATH)
        self.active: str | None = state.get("active", self.cfg.get("default_profile"))
        # only an explicit "stop" unloads models; a fresh node with no choice yet leaves the GPU alone
        self.stop_requested: bool = bool(state.get("stopped", False))
        self.state = "idle"  # idle | loading | ready | error | stopped
        self.last_error: str | None = None
        self.events: collections.deque = collections.deque(maxlen=200)
        self.restarts: collections.deque = collections.deque(maxlen=50)
        self.lock = threading.Lock()
        self.wake = threading.Event()
        self.started = time.time()
        self.jobs: dict[str, dict[str, Any]] = {}
        self._last_oom_check = time.time()
        self._alerted: dict[str, float] = {}
        self._job_ids = itertools.count(1)
        self.engine_busy = threading.Lock()
        for note in migration:
            self.event("info", f"config upgrade: {note}")

    # ---- profiles -------------------------------------------------------------------------------------------
    def profiles(self) -> dict[str, dict[str, Any]]:
        merged = dict(self.cfg.get("profiles") or {})
        merged.update(load_yaml(PROFILES_PATH))
        return merged

    def save_profile(self, name: str, profile: dict[str, Any]) -> None:
        extra = load_yaml(PROFILES_PATH)
        extra[name] = profile
        save_yaml(PROFILES_PATH, extra)

    def delete_profile(self, name: str) -> None:
        extra = load_yaml(PROFILES_PATH)
        if name not in extra:
            raise HTTPException(400, "only profiles created from the dashboard can be deleted; edit node.yaml")
        del extra[name]
        save_yaml(PROFILES_PATH, extra)

    def new_job(self, kind: str, source: str) -> str:
        job_id = f"{kind}-{int(time.time())}-{next(self._job_ids)}"
        # keep the job list short: finished jobs older than the 10 newest go away
        done = [j for j, v in self.jobs.items() if v["state"] != "running"]
        for j in done[:-10]:
            del self.jobs[j]
        self.jobs[job_id] = {"kind": kind, "source": source, "state": "running", "output": "", "started": time.time()}
        return job_id

    # ---- events / alerts ------------------------------------------------------------------------------------
    def event(self, level: str, msg: str) -> None:
        self.events.append({"ts": time.time(), "level": level, "msg": msg})
        getattr(log, "warning" if level == "alert" else "info")(msg)
        if level == "alert":
            last = self._alerted.get(msg, 0)
            if time.time() - last > 600:
                self._alerted[msg] = time.time()
                notify("homellm", f"{self.name}: {msg}")

    # ---- desired state --------------------------------------------------------------------------------------
    def set_active(self, profile: str | None) -> None:
        if profile is not None and profile not in self.profiles():
            raise HTTPException(404, f"unknown profile {profile}")
        with self.lock:
            self.active = profile
            self.stop_requested = profile is None
            save_yaml(STATE_PATH, {"active": profile, "stopped": self.stop_requested})
            self.state = "loading" if profile else "stopped"
            self.last_error = None
            self.restarts.clear()
        self.event("info", f"active profile set to {profile}")
        self.wake.set()

    # ---- watchdog -------------------------------------------------------------------------------------------
    def watchdog_loop(self) -> None:
        self.event("info", f"watchdog started (interval {self.interval}s)")
        while True:
            try:
                self.tick()
            except Exception as e:  # never let the watchdog die
                self.last_error = str(e)
                self.event("alert", f"watchdog error: {e}")
            self.wake.wait(self.interval)
            self.wake.clear()

    def tick(self) -> None:
        self.check_oom()
        self.check_memory()
        with self.lock:
            active = self.active
        if not active and not self.stop_requested:
            active = self.adopt_loaded()
        if not active:
            if self.stop_requested:
                for b in self.backends.values():
                    if b.available() and self._backend_busy(b):
                        b.stop()
                self.state = "stopped"
            else:
                self.state = "idle"
            return
        profile = self.profiles().get(active)
        if not profile:
            self.state, self.last_error = "error", f"profile {active} no longer exists"
            return
        backend = self.backends.get(profile["backend"])
        if not backend:
            self.state, self.last_error = "error", f"backend {profile['backend']} not enabled on this node"
            return
        if not backend.available():
            self.state, self.last_error = "error", backend.unavailable_reason()
            return
        # only one engine holds the GPU at a time
        for name, other in self.backends.items():
            if name != profile["backend"] and other.available() and self._backend_busy(other):
                self.event("info", f"stopping {name} to free the GPU")
                other.stop()
        if backend.is_ready(active, profile):
            if self.state != "ready":
                self.event("info", f"{active} ready on {backend.name}")
            self.state, self.last_error = "ready", None
            self.check_engine(backend)
            return
        now = time.time()
        recent = [t for t in self.restarts if now - t < 3600]
        if len(recent) >= self.max_restarts:
            self.state = "error"
            self.last_error = f"gave up: {len(recent)} restarts in the last hour"
            self.event("alert", f"{active}: {self.last_error}; switch profile from the dashboard to retry")
            return
        if self.state == "ready":
            self.event("alert", f"{active} went down, restarting")
        else:
            self.event("info", f"loading {active} on {backend.name}")
        self.restarts.append(now)
        self.state = "loading"
        try:
            backend.start(active, profile)
            if backend.is_ready(active, profile) or self._wait_ready(backend, active, profile):
                self.state, self.last_error = "ready", None
                self.event("info", f"{active} ready on {backend.name}")
            else:
                self.state, self.last_error = "error", "engine started but model never became ready"
                self.event("alert", f"{active}: {self.last_error}")
        except (BackendError, subprocess.SubprocessError, OSError) as e:
            self.state, self.last_error = "error", str(e)
            self.event("alert", f"{active} failed to start: {e}")
        if self.state == "error":
            hogs = self.foreign_gpu_users()
            if hogs:
                self.last_error = f"{self.last_error}; GPU memory held by {hogs}"
                self.event("alert", f"GPU memory held by {hogs}")

    def adopt_loaded(self) -> str | None:
        """If a profile's model is already being served (the agent restarted), adopt it instead of reloading."""
        for name, profile in self.profiles().items():
            backend = self.backends.get(profile.get("backend"))
            if backend and backend.available() and backend.is_ready(name, profile):
                with self.lock:
                    self.active = name
                    save_yaml(STATE_PATH, {"active": name, "stopped": False})
                self.event("info", f"adopted already-loaded {name} on {backend.name}")
                return name
        return None

    def _wait_ready(self, backend: Backend, name: str, profile: dict, timeout: int = 600) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.active != name:
                return False
            if backend.is_ready(name, profile):
                return True
            time.sleep(3)
        return False

    def foreign_gpu_users(self) -> str:
        """Other programs holding VRAM (a leftover LM Studio, a desktop app), which make loads fail."""
        ours = {pid for b in self.backends.values() if (pid := b.server_pid())}
        hogs = []
        for p in gpu_processes():
            if p["pid"] in ours or p["used_mb"] < 512:
                continue
            lmstudio = ".lmstudio" in p["name"] or "LM-Studio" in p["name"]
            what = "LM Studio (quit it, or `lms unload --all`)" if lmstudio else p["name"].rsplit("/", 1)[-1]
            hogs.append(f"{what} pid {p['pid']} {p['used_mb']} MB")
        return ", ".join(hogs)

    @staticmethod
    def _backend_busy(b: Backend) -> bool:
        return b.server_pid() is not None

    def check_engine(self, backend: Backend) -> None:
        h = backend.health()
        for w in h.get("warnings", []):
            self.event("alert", w)
        if (h.get("rss_mb") or 0) > self.warn_rss_mb:
            self.event("alert", f"{backend.name} engine using {h['rss_mb']} MB RAM")

    def check_memory(self) -> None:
        m = memory_stats()
        if m["available_mb"] < self.warn_avail_mb:
            self.event("alert", f"low memory: {m['available_mb']} MB available")

    def check_oom(self) -> None:
        since = int(self._last_oom_check)
        self._last_oom_check = time.time()
        try:
            out = subprocess.run(["journalctl", "-k", "--since", f"@{since}", "--no-pager", "-o", "cat"],
                                 capture_output=True, text=True, timeout=10).stdout
        except (subprocess.SubprocessError, OSError):
            return
        for line in out.splitlines():
            if "Killed process" in line and "llama-server" in line:
                self.event("alert", f"OOM killer: {line.strip()}")

    # ---- status ---------------------------------------------------------------------------------------------
    def status(self) -> dict[str, Any]:
        ip = self.cfg.get("advertise_ip") or primary_ip()
        profiles = self.profiles()
        active = self.active
        served = None
        if active and active in profiles and self.state == "ready":
            backend = self.backends.get(profiles[active]["backend"])
            if backend:
                served = {"model": active, "api_base": f"http://{ip}:{backend.port()}/v1",
                          "backend": backend.name, "health": backend.health()}
        return {
            "id": self.id, "name": self.name, "version": __version__, "ip": ip,
            "uptime_s": int(time.time() - self.started),
            "active": active, "state": self.state, "last_error": self.last_error,
            "served": served,
            "backends": {n: {"available": b.available(), "port": b.port(), "engine": b.engine_info(),
                             **({} if b.available() else {"reason": b.unavailable_reason()})}
                         for n, b in self.backends.items()},
            "engine_flavor": engine.detect_flavor_cached(),
            "profiles": profiles,
            "memory": memory_stats(), "gpus": gpu_stats(),
            "events": list(self.events)[-30:],
            "jobs": {k: {kk: vv for kk, vv in v.items() if kk != "cancel"} for k, v in list(self.jobs.items())},
        }


class ActivateBody(BaseModel):
    profile: str | None


class ProfileBody(BaseModel):
    name: str
    backend: str = "llamacpp"
    model: str
    context: int | None = None
    parallel: int | None = None
    args: list[str] = []


class DownloadBody(BaseModel):
    """What to fetch. One of:
    - catalog: a name from the model catalog (repo, quant and profile settings come from it)
    - repo (+ quant or files): any Hugging Face GGUF repo
    - source: a huggingface.co file URL, or any direct .gguf URL
    """
    catalog: str | None = None
    repo: str | None = None
    quant: str | None = None
    files: list[str] | None = None
    mmproj: bool | str | None = None
    revision: str = "main"
    source: str | None = None
    filename: str | None = None
    make_profile: bool = True
    profile_name: str | None = None
    backend: str = "llamacpp"  # accepted for old clients; llamacpp is the only engine


class EngineBody(BaseModel):
    version: str | None = None  # e.g. b12170; newest when empty
    flavor: str | None = None  # cuda-12 | cuda-13 | rocm | vulkan | cpu; detected when empty
    force: bool = False
    restart: bool = True  # reload the active model on the new version


def create_app(token: str) -> FastAPI:
    node = Node(token)
    auth = Depends(bearer_guard(token))
    zc_holder: dict[str, Any] = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        threading.Thread(target=node.watchdog_loop, daemon=True, name="watchdog").start()
        if node.cfg.get("mdns", True):
            try:
                zc_holder["zc"] = await advertise(node)
            except Exception as e:  # mDNS is a convenience, the subnet scan still finds us
                node.event("alert", f"mDNS advertise failed: {e}")
        yield
        if "zc" in zc_holder:
            await zc_holder["zc"].async_unregister_all_services()
            await zc_holder["zc"].async_close()

    app = FastAPI(title="homellm node", version=__version__, lifespan=lifespan)

    @app.get("/health")
    def health():
        return {"service": "homellm-node", "id": node.id, "name": node.name, "version": __version__}

    @app.get("/status", dependencies=[auth])
    def status():
        return node.status()

    @app.get("/models", dependencies=[auth])
    def models():
        out = []
        for b in node.backends.values():
            if b.available():
                out += b.list_models()
        return {"models": out, "profiles": node.profiles()}

    @app.post("/activate", dependencies=[auth])
    def activate(body: ActivateBody):
        node.set_active(body.profile)
        return {"ok": True, "active": node.active}

    @app.post("/profiles", dependencies=[auth])
    def add_profile(body: ProfileBody):
        if body.backend not in node.backends:
            raise HTTPException(400, f"backend {body.backend} not enabled on this node")
        profile = {k: v for k, v in body.model_dump().items() if k != "name" and v not in (None, [])}
        node.save_profile(body.name, profile)
        node.event("info", f"profile {body.name} saved")
        return {"ok": True}

    @app.delete("/profiles/{name}", dependencies=[auth])
    def delete_profile(name: str):
        if name == node.active:
            raise HTTPException(400, "stop or switch away from this profile first")
        node.delete_profile(name)
        return {"ok": True}

    @app.post("/download", dependencies=[auth])
    def download(body: DownloadBody):
        backend = node.backends.get("llamacpp")
        if not backend:
            raise HTTPException(400, "llamacpp backend not enabled on this node")
        if not (body.catalog or body.repo or body.source):
            raise HTTPException(400, "give catalog, repo or source")
        if body.catalog and not hf.catalog_entry(body.catalog):
            raise HTTPException(404, f"{body.catalog} is not in the catalog")
        label = body.catalog or body.repo or body.source
        job_id = node.new_job("download", label)
        threading.Thread(target=run_download, args=(node, backend, body, job_id), daemon=True).start()
        return {"ok": True, "job": job_id}

    @app.delete("/jobs/{job_id}", dependencies=[auth])
    def cancel_job(job_id: str):
        job = node.jobs.get(job_id)
        if not job:
            raise HTTPException(404, "no such job")
        if job["state"] == "running":
            job["cancel"] = True
        else:
            del node.jobs[job_id]
        return {"ok": True}

    @app.get("/catalog", dependencies=[auth])
    def catalog():
        return {"models": hf.load_catalog()}

    @app.get("/engine", dependencies=[auth])
    def engine_status():
        return {"current": engine.current(), "installed": engine.installed(), "flavor": engine.detect_flavor()}

    @app.post("/engine/install", dependencies=[auth])
    def engine_install(body: EngineBody):
        if node.engine_busy.locked():
            raise HTTPException(409, "an engine install is already running")
        job_id = node.new_job("engine", f"llama.cpp {body.version or 'latest'} {body.flavor or ''}".strip())
        threading.Thread(target=run_engine_install, args=(node, body, job_id), daemon=True).start()
        return {"ok": True, "job": job_id}

    @app.get("/logs", dependencies=[auth])
    def logs(lines: int = 100):
        out = {}
        for n, b in node.backends.items():
            if hasattr(b, "logs"):
                out[n] = b.logs(lines)
        return out

    return app


def _progress(job: dict[str, Any]):
    last = [0.0]

    def cb(done: int, total: int, name: str) -> None:
        now = time.time()
        if now - last[0] < 1 and done < total:
            return
        last[0] = now
        job["done_bytes"], job["total_bytes"] = done, total
        pct = f" ({done / total * 100:.0f}%)" if total else ""
        size = f"{done / 1e9:.2f} / {total / 1e9:.2f} GB{pct}" if total else f"{done / 1e9:.2f} GB"
        job["output"] = f"{name}: {size}"
    return cb


def run_download(node: Node, backend: Backend, body: DownloadBody, job_id: str) -> None:
    job = node.jobs[job_id]
    models_dir: Any = backend.models_dir  # type: ignore[attr-defined]
    stop = lambda: bool(job.get("cancel"))  # noqa: E731
    try:
        entry = hf.catalog_entry(body.catalog) if body.catalog else None
        repo, quant, files, revision = body.repo, body.quant, body.files, body.revision
        mmproj = body.mmproj
        if entry:
            repo, quant = entry["repo"], quant or entry.get("quant")
            mmproj = entry.get("mmproj", False) if mmproj is None else mmproj
        elif body.source and (parsed := hf.parse_hf_url(body.source)):
            repo, revision, path = parsed
            files = [path]
        if repo:
            job["output"] = f"looking up {repo}"
            plan = hf.plan(repo, quant, files, bool(mmproj) and mmproj, revision)
            job["source"] = f"{repo} · {plan['model']['name']}"
            job["total_bytes"] = plan["size"]
            hf.fetch_plan(plan, models_dir, _progress(job), should_stop=stop)
            if body.make_profile:
                name, profile = hf.profile_for(plan, models_dir, entry)
                name = body.profile_name or name
                node.save_profile(name, profile)
                job["profile"] = name
                node.event("info", f"profile {name} created")
        else:  # plain URL to a .gguf
            src = body.source or ""
            name = body.filename or src.rstrip("/").split("/")[-1].split("?")[0]
            if not name.endswith(".gguf") or "/" in name:
                raise ValueError("a direct download needs a URL ending in a .gguf file name")
            dest = models_dir / "downloads" / name
            hf.download_file(src, dest, progress=_progress(job), should_stop=stop)
            if body.make_profile:
                pname = body.profile_name or initcfg.slug(dest.stem)
                node.save_profile(pname, {"backend": "llamacpp", "model": str(dest.relative_to(models_dir)),
                                          "context": DEFAULT_CONTEXT, "parallel": 1, "args": list(hf.DEFAULT_ARGS),
                                          "source": src})
                job["profile"] = pname
        job["state"] = "done"
        job["output"] = (job.get("output", "") + "\ndone" + (f" · profile {job['profile']}" if job.get("profile")
                                                              else "")).strip()
        node.event("info", f"download {job['source']}: done")
    except Exception as e:
        job["state"], job["output"] = ("cancelled" if job.get("cancel") else "failed"), str(e)
        node.event("alert", f"download {job['source']} {job['state']}: {e}")


def run_engine_install(node: Node, body: EngineBody, job_id: str) -> None:
    job = node.jobs[job_id]
    lines: collections.deque = collections.deque(maxlen=8)

    def log(msg: str) -> None:
        lines.append(msg)
        job["output"] = "\n".join(lines)

    with node.engine_busy:
        try:
            before = (engine.current() or {}).get("dir")
            cur = engine.install(body.flavor, body.version, body.force, _progress(job), log)
            job["source"] = f"llama.cpp {cur.get('tag')} ({cur.get('flavor')})"
            job["state"] = "done"
            node.event("info", f"engine: {job['source']} installed")
            # rebuild so a node that had no engine yet picks it up
            node.backends = build_backends(node.cfg)
            if body.restart and cur.get("dir") != before:
                profile = node.profiles().get(node.active or "")
                b = node.backends.get((profile or {}).get("backend", ""))
                if profile and b and getattr(b, "managed", False) and b.server_pid():
                    node.event("info", f"restarting {node.active} on the new llama-server")
                    b.stop()
                    with node.lock:
                        node.restarts.clear()
                    node.wake.set()
        except Exception as e:
            job["state"], job["output"] = "failed", f"{job.get('output', '')}\n{e}".strip()
            node.event("alert", f"engine install failed: {e}")


async def advertise(node: Node) -> AsyncZeroconf:
    ip = node.cfg.get("advertise_ip") or primary_ip()
    port = int(node.cfg.get("port", NODE_PORT))
    info = ServiceInfo(
        SERVICE_TYPE,
        f"{node.name}-{node.id[:6]}.{SERVICE_TYPE}",
        addresses=[socket.inet_aton(ip)],
        port=port,
        properties={"id": node.id, "name": node.name, "version": __version__},
        server=f"{socket.gethostname()}.local.",
    )
    zc = AsyncZeroconf(ip_version=IPVersion.V4Only)
    await zc.async_register_service(info, allow_name_change=True)
    node.event("info", f"advertising {SERVICE_TYPE} on {ip}:{port}")
    return zc


def run(token: str) -> None:
    import uvicorn

    cfg = load_yaml(CONFIG_PATH)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    uvicorn.run(create_app(token), host=cfg.get("listen", "0.0.0.0"), port=int(cfg.get("port", NODE_PORT)),
                log_level="warning")
