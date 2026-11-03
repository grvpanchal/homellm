"""Node agent: runs on every machine with a GPU.

- Keeps the *active* profile loaded (watchdog), restarting the engine if it dies.
- Watches memory and the kernel OOM killer, alerts before things fall over.
- Exposes a small authenticated REST API for the controller.
- Advertises itself on the LAN over mDNS (_homellm._tcp).
"""

from __future__ import annotations

import collections
import logging
import socket
import subprocess
import threading
import time
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel
from zeroconf import IPVersion, ServiceInfo
from zeroconf.asyncio import AsyncZeroconf

from .. import NODE_PORT, SERVICE_TYPE, __version__
from ..common import (
    CONFIG_DIR,
    STATE_DIR,
    bearer_guard,
    gpu_stats,
    load_yaml,
    memory_stats,
    node_id,
    notify,
    primary_ip,
    save_yaml,
)
from .backends import Backend, BackendError, LMStudioBackend, build_backends

log = logging.getLogger("homellm.node")

CONFIG_PATH = CONFIG_DIR / "node.yaml"
PROFILES_PATH = CONFIG_DIR / "profiles.yaml"
STATE_PATH = STATE_DIR / "node-state.yaml"


class Node:
    def __init__(self, token: str):
        self.token = token
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
        if not backend or not backend.available():
            self.state, self.last_error = "error", f"backend {profile['backend']} not available on this node"
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

    def adopt_loaded(self) -> str | None:
        """If a profile's model is already loaded (e.g. by hand in LM Studio), take it over instead of reloading."""
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

    @staticmethod
    def _backend_busy(b: Backend) -> bool:
        if isinstance(b, LMStudioBackend):
            return bool(b.loaded())
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
            if "Killed process" in line and any(k in line for k in ("llama-server", "lm-studio", "llmworker")):
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
            "backends": {n: {"available": b.available(), "port": b.port()} for n, b in self.backends.items()},
            "profiles": profiles,
            "memory": memory_stats(), "gpus": gpu_stats(),
            "events": list(self.events)[-30:],
            "jobs": self.jobs,
        }


class ActivateBody(BaseModel):
    profile: str | None


class ProfileBody(BaseModel):
    name: str
    backend: str
    model: str
    context: int | None = None
    parallel: int | None = None
    args: list[str] = []


class DownloadBody(BaseModel):
    backend: str = "lmstudio"
    source: str  # lmstudio: model name / HF URL; llamacpp: direct .gguf URL
    filename: str | None = None


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
        backend = node.backends.get(body.backend)
        if not backend or not backend.available():
            raise HTTPException(400, f"backend {body.backend} not available")
        job_id = f"dl-{int(time.time())}"
        node.jobs[job_id] = {"source": body.source, "state": "running", "output": ""}
        threading.Thread(target=run_download, args=(node, backend, body, job_id), daemon=True).start()
        return {"ok": True, "job": job_id}

    @app.get("/logs", dependencies=[auth])
    def logs(lines: int = 100):
        out = {}
        for n, b in node.backends.items():
            if hasattr(b, "logs"):
                out[n] = b.logs(lines)
        return out

    return app


def run_download(node: Node, backend: Backend, body: DownloadBody, job_id: str) -> None:
    job = node.jobs[job_id]
    try:
        if body.backend == "lmstudio":
            proc = backend.download(body.source)  # type: ignore[attr-defined]
            tail: collections.deque = collections.deque(maxlen=20)
            for line in proc.stdout:  # type: ignore[union-attr]
                tail.append(line.rstrip()[-200:])
                job["output"] = "\n".join(tail)
            proc.wait()
            job["state"] = "done" if proc.returncode == 0 else "failed"
        else:
            name = body.filename or body.source.rstrip("/").split("/")[-1].split("?")[0]
            if not name.endswith(".gguf"):
                raise ValueError("llamacpp downloads need a direct .gguf URL")
            dest = backend.models_dir / name  # type: ignore[attr-defined]
            tmp = dest.with_suffix(".part")
            with httpx.stream("GET", body.source, follow_redirects=True, timeout=60) as r:
                r.raise_for_status()
                total = int(r.headers.get("content-length", 0))
                done = 0
                with tmp.open("wb") as f:
                    for chunk in r.iter_bytes(1 << 20):
                        f.write(chunk)
                        done += len(chunk)
                        job["output"] = f"{done / 1e9:.2f} / {total / 1e9:.2f} GB" if total else f"{done / 1e9:.2f} GB"
            tmp.rename(dest)
            job["state"] = "done"
        node.event("info", f"download {body.source}: {job['state']}")
    except Exception as e:
        job["state"], job["output"] = "failed", str(e)
        node.event("alert", f"download {body.source} failed: {e}")


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
