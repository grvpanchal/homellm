"""Find node agents on the LAN: mDNS first, a /24 sweep as fallback (mDNS is often blocked by firewalls)."""

from __future__ import annotations

import asyncio
import ipaddress
import logging
import socket
from collections.abc import Awaitable, Callable

import httpx
import psutil
from zeroconf import IPVersion, ServiceStateChange
from zeroconf.asyncio import AsyncServiceBrowser, AsyncServiceInfo, AsyncZeroconf

from .. import NODE_PORT, SERVICE_TYPE

log = logging.getLogger("homellm.controller")

Found = Callable[[str, str], Awaitable[None]]  # (base_url, source)

SKIP_IFACES = ("lo", "docker", "br-", "veth", "virbr", "tun", "tailscale", "cni", "flannel")


def local_subnets() -> list[ipaddress.IPv4Network]:
    nets = []
    for iface, addrs in psutil.net_if_addrs().items():
        if iface.startswith(SKIP_IFACES):
            continue
        for a in addrs:
            if a.family == socket.AF_INET and a.address and not a.address.startswith("169.254."):
                net = ipaddress.ip_network(f"{a.address}/24", strict=False)
                if net.is_private and net not in nets:
                    nets.append(net)
    return nets


async def probe(client: httpx.AsyncClient, base: str) -> dict | None:
    try:
        r = await client.get(f"{base}/health", timeout=1.5)
        if r.status_code == 200 and r.json().get("service") == "homellm-node":
            return r.json()
    except (httpx.HTTPError, ValueError):
        pass
    return None


async def scan(subnets: list[ipaddress.IPv4Network], found: Found, port: int = NODE_PORT) -> int:
    sem = asyncio.Semaphore(96)
    hits = 0

    async with httpx.AsyncClient() as client:
        async def one(ip: str) -> None:
            nonlocal hits
            async with sem:
                base = f"http://{ip}:{port}"
                if await probe(client, base):
                    hits += 1
                    await found(base, "scan")

        await asyncio.gather(*(one(str(ip)) for net in subnets for ip in net.hosts()))
    return hits


class MDNSBrowser:
    def __init__(self, found: Found):
        self.found = found
        self.zc: AsyncZeroconf | None = None
        self.browser: AsyncServiceBrowser | None = None

    async def start(self) -> None:
        self.zc = AsyncZeroconf(ip_version=IPVersion.V4Only)
        self.browser = AsyncServiceBrowser(self.zc.zeroconf, SERVICE_TYPE, handlers=[self._on_change])

    def _on_change(self, zeroconf, service_type: str, name: str, state_change: ServiceStateChange) -> None:
        if state_change in (ServiceStateChange.Added, ServiceStateChange.Updated):
            asyncio.ensure_future(self._resolve(service_type, name))

    async def _resolve(self, service_type: str, name: str) -> None:
        info = AsyncServiceInfo(service_type, name)
        if not await info.async_request(self.zc.zeroconf, 3000):
            return
        for addr in info.parsed_addresses(IPVersion.V4Only):
            await self.found(f"http://{addr}:{info.port}", "mdns")
            return

    async def stop(self) -> None:
        if self.browser:
            await self.browser.async_cancel()
        if self.zc:
            await self.zc.async_close()
