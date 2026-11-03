"""Keep LiteLLM's model list in sync with what the nodes are actually serving.

Every ready node contributes one deployment. Deployments created here carry an id prefixed with
`homellm-`; anything else in LiteLLM (cloud models you added by hand) is left alone. When the same
profile name is served by several nodes, LiteLLM load-balances across them.
"""

from __future__ import annotations

import re
from typing import Any

import httpx

PREFIX = "homellm-"


def deployment_id(node_id: str, model: str) -> str:
    return PREFIX + re.sub(r"[^a-zA-Z0-9_.-]", "_", f"{node_id}-{model}")


def desired_deployments(nodes: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out = {}
    for n in nodes:
        st = n.get("status") or {}
        served = st.get("served")
        if not n.get("online") or not served:
            continue
        dep_id = deployment_id(st["id"], served["model"])
        out[dep_id] = {
            "model_name": served["model"],
            "litellm_params": {
                "model": f"openai/{served['model']}",
                "api_base": served["api_base"],
                "api_key": "homellm",
            },
            "model_info": {"id": dep_id, "homellm_node": st.get("name"), "homellm_backend": served.get("backend")},
        }
    return out


def plan(desired: dict[str, dict], current: list[dict]) -> tuple[list[dict], list[str]]:
    """Return (deployments to add, ids to delete). A changed api_base or upstream model is delete + add."""
    have = {}
    for d in current:
        dep_id = (d.get("model_info") or {}).get("id", "")
        if dep_id.startswith(PREFIX):
            have[dep_id] = d
    to_add, to_delete = [], []
    for dep_id, want in desired.items():
        cur = have.get(dep_id)
        if cur is None:
            to_add.append(want)
            continue
        cp = cur.get("litellm_params") or {}
        wp = want["litellm_params"]
        if cp.get("api_base") != wp["api_base"] or cp.get("model") != wp["model"] \
                or cur.get("model_name") != want["model_name"]:
            to_delete.append(dep_id)
            to_add.append(want)
    to_delete += [dep_id for dep_id in have if dep_id not in desired]
    return to_add, to_delete


class LiteLLM:
    def __init__(self, url: str, master_key: str):
        self.url = url.rstrip("/")
        self.headers = {"Authorization": f"Bearer {master_key}"}

    async def healthy(self, client: httpx.AsyncClient) -> bool:
        try:
            r = await client.get(f"{self.url}/health/liveliness", timeout=5)
            return r.status_code == 200
        except httpx.HTTPError:
            return False

    async def models(self, client: httpx.AsyncClient) -> list[dict]:
        r = await client.get(f"{self.url}/model/info", headers=self.headers, timeout=10)
        r.raise_for_status()
        return r.json().get("data", [])

    async def sync(self, client: httpx.AsyncClient, nodes: list[dict]) -> tuple[list[str], list[str]]:
        to_add, to_delete = plan(desired_deployments(nodes), await self.models(client))
        for dep_id in to_delete:
            r = await client.post(f"{self.url}/model/delete", headers=self.headers, json={"id": dep_id}, timeout=10)
            r.raise_for_status()
        for dep in to_add:
            r = await client.post(f"{self.url}/model/new", headers=self.headers, json=dep, timeout=10)
            r.raise_for_status()
        return [d["model_info"]["id"] for d in to_add], to_delete
