"""Controller: runs on the main server next to LiteLLM.

- Discovers node agents on the LAN (mDNS + subnet sweep + static list).
- Polls each node and mirrors whatever it serves into LiteLLM.
- Serves the management dashboard and proxies model-switch / profile / download actions to nodes.
"""

from __future__ import annotations

import asyncio
import collections
import ipaddress
import json
import logging
import os
import subprocess
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from .. import CONTROLLER_PORT, __version__
from ..common import CONFIG_DIR, STATE_DIR, bearer_guard, load_yaml, save_yaml
from . import discovery
from .litellm_sync import LiteLLM

log = logging.getLogger("homellm.controller")

CONFIG_PATH = CONFIG_DIR / "controller.yaml"
NODES_PATH = STATE_DIR / "nodes.yaml"
STATIC = Path(__file__).parent / "static"


class Controller:
    def __init__(self, cluster_token: str, master_key: str):
        self.cfg = load_yaml(CONFIG_PATH)
        disc = self.cfg.get("discovery") or {}
        self.mdns = disc.get("mdns", True)
        self.scan_enabled = disc.get("scan", True)
        self.scan_interval = int(disc.get("scan_interval", 300))
        subnets = disc.get("subnets", "auto")
        self.subnets = discovery.local_subnets() if subnets == "auto" else [
            ipaddress.ip_network(s, strict=False) for s in subnets]
        self.static_nodes: list[str] = list(disc.get("static_nodes") or [])
        self.poll_interval = int(self.cfg.get("poll_interval", 10))
        self.cluster_token = cluster_token
        self.litellm = LiteLLM(self.cfg.get("litellm_url", "http://127.0.0.1:4000"), master_key)
        self.public_url = self.cfg.get("public_url")
        self.nodes: dict[str, dict[str, Any]] = {}  # node id -> record
        for rec in (load_yaml(NODES_PATH).get("nodes") or []):
            self.nodes[rec["id"]] = {**rec, "online": False, "status": None, "misses": 0}
        self.events: collections.deque = collections.deque(maxlen=300)
        self.litellm_ok = False
        self.last_sync: dict[str, Any] = {}
        self.last_scan: float | None = None
        self.client = httpx.AsyncClient(headers={"Authorization": f"Bearer {cluster_token}"})

    def event(self, level: str, msg: str) -> None:
        self.events.append({"ts": time.time(), "level": level, "msg": msg})
        getattr(log, "warning" if level == "alert" else "info")(msg)

    def persist(self) -> None:
        save_yaml(NODES_PATH, {"nodes": [
            {"id": n["id"], "name": n.get("name"), "url": n["url"], "source": n.get("source")}
            for n in self.nodes.values()]})

    # ---- discovery ------------------------------------------------------------------------------------------
    async def found(self, url: str, source: str) -> None:
        info = await discovery.probe(self.client, url)
        if not info:
            return
        nid = info["id"]
        rec = self.nodes.get(nid)
        if rec is None:
            self.nodes[nid] = {"id": nid, "name": info["name"], "url": url, "source": source, "online": False,
                               "status": None, "misses": 0}
            self.event("info", f"discovered node {info['name']} at {url} via {source}")
            self.persist()
            await self.poll_one(self.nodes[nid])
        elif rec["url"] != url:
            self.event("info", f"node {info['name']} moved {rec['url']} -> {url}")
            rec["url"], rec["source"] = url, source
            self.persist()

    async def scan_loop(self) -> None:
        while True:
            await self.scan_now()
            await asyncio.sleep(self.scan_interval)

    async def scan_now(self) -> int:
        for url in self.static_nodes:
            await self.found(url.rstrip("/"), "static")
        hits = 0
        if self.scan_enabled and self.subnets:
            hits = await discovery.scan(self.subnets, self.found)
        self.last_scan = time.time()
        return hits

    # ---- polling + sync -------------------------------------------------------------------------------------
    async def poll_one(self, rec: dict[str, Any]) -> None:
        try:
            r = await self.client.get(f"{rec['url']}/status", timeout=8)
            if r.status_code in (401, 403):
                raise RuntimeError("cluster token rejected - node has a different HOMELLM_CLUSTER_TOKEN")
            r.raise_for_status()
            st = r.json()
            if not rec["online"]:
                self.event("info", f"node {st['name']} online")
            prev = (rec.get("status") or {}).get("state")
            if prev and prev != st["state"]:
                self.event("alert" if st["state"] == "error" else "info",
                           f"{st['name']}: {prev} -> {st['state']}" + (f" ({st['last_error']})"
                                                                         if st.get("last_error") else ""))
            rec.update(status=st, online=True, misses=0, name=st["name"], last_seen=time.time(), error=None)
        except Exception as e:
            rec["misses"] = rec.get("misses", 0) + 1
            rec["error"] = str(e) or type(e).__name__
            if rec["online"] and rec["misses"] >= 3:
                rec["online"] = False
                self.event("alert", f"node {rec.get('name')} offline: {rec['error']}")

    async def poll_loop(self) -> None:
        while True:
            await asyncio.gather(*(self.poll_one(r) for r in list(self.nodes.values())))
            self.litellm_ok = await self.litellm.healthy(self.client)
            if self.litellm_ok:
                try:
                    added, deleted = await self.litellm.sync(self.client, list(self.nodes.values()))
                    for d in added:
                        self.event("info", f"LiteLLM: registered {d}")
                    for d in deleted:
                        self.event("info", f"LiteLLM: removed {d}")
                    self.last_sync = {"ts": time.time(), "ok": True}
                except httpx.HTTPError as e:
                    self.last_sync = {"ts": time.time(), "ok": False, "error": str(e)}
            await asyncio.sleep(self.poll_interval)

    def node(self, nid: str) -> dict[str, Any]:
        rec = self.nodes.get(nid)
        if not rec:
            raise HTTPException(404, "unknown node")
        return rec

    async def node_call(self, nid: str, method: str, path: str, body: Any = None) -> Any:
        rec = self.node(nid)
        try:
            r = await self.client.request(method, f"{rec['url']}{path}", json=body, timeout=30)
        except httpx.HTTPError as e:
            raise HTTPException(502, f"node unreachable: {e}") from e
        if r.status_code >= 400:
            raise HTTPException(r.status_code, r.json().get("detail", r.text) if r.headers.get(
                "content-type", "").startswith("application/json") else r.text)
        asyncio.ensure_future(self.poll_one(rec))
        return r.json()


def tunnel_status() -> dict[str, Any]:
    try:
        r = subprocess.run(["systemctl", "--user", "is-active", "homellm-tunnel"], capture_output=True, text=True,
                           timeout=5)
        return {"active": r.stdout.strip() == "active", "state": r.stdout.strip()}
    except (subprocess.SubprocessError, OSError):
        return {"active": False, "state": "unknown"}


class UrlBody(BaseModel):
    url: str


class ActivateBody(BaseModel):
    profile: str | None


class TestBody(BaseModel):
    model: str
    prompt: str = "Write a 150-word story about a lighthouse keeper."
    max_tokens: int = 2048


def create_app() -> FastAPI:
    ctl = Controller(os.environ["HOMELLM_CLUSTER_TOKEN"], os.environ["LITELLM_MASTER_KEY"])
    auth = Depends(bearer_guard(os.environ["HOMELLM_ADMIN_TOKEN"]))
    tasks: list[asyncio.Task] = []
    browser = discovery.MDNSBrowser(ctl.found)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        ctl.event("info", f"controller {__version__} started; subnets {[str(s) for s in ctl.subnets]}")
        if ctl.mdns:
            try:
                await browser.start()
            except Exception as e:
                ctl.event("alert", f"mDNS browse failed: {e}")
        tasks.append(asyncio.create_task(ctl.scan_loop()))
        tasks.append(asyncio.create_task(ctl.poll_loop()))
        yield
        for t in tasks:
            t.cancel()
        await browser.stop()

    app = FastAPI(title="homellm controller", version=__version__, lifespan=lifespan)

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/healthz")
    def healthz():
        return {"ok": True, "version": __version__}

    @app.get("/api/overview", dependencies=[auth])
    def overview():
        nodes = []
        for rec in ctl.nodes.values():
            nodes.append({k: rec.get(k) for k in ("id", "name", "url", "source", "online", "error", "last_seen",
                                                   "status")})
        return {
            "version": __version__,
            "litellm": {"ok": ctl.litellm_ok, "url": ctl.litellm.url, "last_sync": ctl.last_sync},
            "tunnel": {**tunnel_status(), "public_url": ctl.public_url},
            "last_scan": ctl.last_scan, "subnets": [str(s) for s in ctl.subnets],
            "nodes": nodes, "events": list(ctl.events)[-80:],
        }

    @app.post("/api/scan", dependencies=[auth])
    async def scan():
        return {"found": await ctl.scan_now()}

    @app.post("/api/nodes", dependencies=[auth])
    async def add_node(body: UrlBody):
        url = body.url.rstrip("/")
        if "://" not in url:
            url = f"http://{url}"
        if url.count(":") == 1:
            url += ":8765"
        if not await discovery.probe(ctl.client, url):
            raise HTTPException(400, f"no homellm node answering at {url}")
        if url not in ctl.static_nodes:
            ctl.static_nodes.append(url)
            cfg = load_yaml(CONFIG_PATH)
            cfg.setdefault("discovery", {}).setdefault("static_nodes", [])
            if url not in cfg["discovery"]["static_nodes"]:
                cfg["discovery"]["static_nodes"].append(url)
                save_yaml(CONFIG_PATH, cfg)
        await ctl.found(url, "static")
        return {"ok": True}

    @app.delete("/api/nodes/{nid}", dependencies=[auth])
    def forget_node(nid: str):
        rec = ctl.node(nid)
        del ctl.nodes[nid]
        ctl.persist()
        ctl.event("info", f"forgot node {rec.get('name')}")
        return {"ok": True}

    @app.post("/api/nodes/{nid}/activate", dependencies=[auth])
    async def activate(nid: str, body: ActivateBody):
        return await ctl.node_call(nid, "POST", "/activate", body.model_dump())

    @app.get("/api/nodes/{nid}/models", dependencies=[auth])
    async def models(nid: str):
        return await ctl.node_call(nid, "GET", "/models")

    @app.post("/api/nodes/{nid}/profiles", dependencies=[auth])
    async def add_profile(nid: str, body: dict):
        return await ctl.node_call(nid, "POST", "/profiles", body)

    @app.delete("/api/nodes/{nid}/profiles/{name}", dependencies=[auth])
    async def delete_profile(nid: str, name: str):
        return await ctl.node_call(nid, "DELETE", f"/profiles/{name}")

    @app.post("/api/nodes/{nid}/download", dependencies=[auth])
    async def download(nid: str, body: dict):
        return await ctl.node_call(nid, "POST", "/download", body)

    @app.get("/api/nodes/{nid}/logs", dependencies=[auth])
    async def logs(nid: str):
        return await ctl.node_call(nid, "GET", "/logs")

    @app.post("/api/test", dependencies=[auth])
    async def test(body: TestBody):
        """Send one streamed request through LiteLLM and report speed."""
        payload = {"model": body.model, "messages": [{"role": "user", "content": body.prompt}],
                   "max_tokens": body.max_tokens, "stream": True, "stream_options": {"include_usage": True}}
        t0 = time.perf_counter()
        first = last = None
        chunks, usage, text = 0, None, ""
        try:
            async with ctl.client.stream("POST", f"{ctl.litellm.url}/v1/chat/completions", json=payload,
                                         headers=ctl.litellm.headers, timeout=300) as r:
                if r.status_code >= 400:
                    raise HTTPException(r.status_code, (await r.aread()).decode()[:500])
                async for line in r.aiter_lines():
                    if not line.startswith("data:") or line.endswith("[DONE]"):
                        continue
                    try:
                        j = json.loads(line[5:])
                    except ValueError:
                        continue
                    usage = j.get("usage") or usage
                    delta = (j.get("choices") or [{}])[0].get("delta") or {}
                    piece = (delta.get("content") or "") + (delta.get("reasoning_content") or "")
                    if piece:
                        now = time.perf_counter()
                        first = first or now
                        last = now
                        chunks += 1
                        text += delta.get("content") or ""
        except httpx.HTTPError as e:
            raise HTTPException(502, f"LiteLLM request failed: {e}") from e
        tokens = (usage or {}).get("completion_tokens") or chunks
        gen = (last - first) if first and last else 0
        return {
            "tokens_per_second": round((tokens - 1) / gen, 1) if gen > 0 and tokens > 1 else None,
            "ttft_s": round(first - t0, 2) if first else None,
            "completion_tokens": tokens, "total_s": round(time.perf_counter() - t0, 1),
            "answer": text[:600],
        }

    return app


def run() -> None:
    import uvicorn

    cfg = load_yaml(CONFIG_PATH)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    uvicorn.run(create_app(), host=cfg.get("listen", "0.0.0.0"), port=int(cfg.get("port", CONTROLLER_PORT)),
                log_level="warning")
