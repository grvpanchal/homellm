"""homellm command line.

    homellm node                 run the node agent (systemd runs this)
    homellm controller           run the controller + dashboard
    homellm status               nodes and what they serve
    homellm switch NODE PROFILE  load a profile on a node ("none" to stop)
    homellm profiles NODE        list profiles and models on disk
    homellm init node|controller write starter config from what is installed
    homellm set-public-url URL   record the tunnel URL shown in the dashboard
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import httpx

from . import CONTROLLER_PORT
from .common import CONFIG_DIR


def _env(key: str) -> str:
    val = os.environ.get(key)
    if val:
        return val
    for f in ("secrets.env", "node.env"):
        p = CONFIG_DIR / f
        if p.exists():
            for line in p.read_text().splitlines():
                if line.startswith(f"{key}="):
                    return line.split("=", 1)[1].strip()
    sys.exit(f"{key} not set (expected in environment or {CONFIG_DIR}/secrets.env)")


def _api(method: str, path: str, **kw):
    base = os.environ.get("HOMELLM_CONTROLLER", f"http://127.0.0.1:{CONTROLLER_PORT}")
    r = httpx.request(method, base + path, headers={"Authorization": f"Bearer {_env('HOMELLM_ADMIN_TOKEN')}"},
                      timeout=60, **kw)
    if r.status_code >= 400:
        sys.exit(f"error {r.status_code}: {r.text}")
    return r.json()


def _find_node(nodes: list[dict], key: str) -> dict:
    for n in nodes:
        if key in (n["id"], n.get("name")) or n["id"].startswith(key):
            return n
    sys.exit(f"no node matching {key!r}; try `homellm status`")


def cmd_status(_args) -> None:
    ov = _api("GET", "/api/overview")
    lt, tn = ov["litellm"], ov["tunnel"]
    print(f"LiteLLM: {'up' if lt['ok'] else 'DOWN'} ({lt['url']})   "
          f"tunnel: {tn['state']} {tn.get('public_url') or ''}")
    print(f"{'NODE':<16}{'STATE':<10}{'PROFILE':<28}{'VRAM':<16}{'RAM free':<10}URL")
    for n in ov["nodes"]:
        st = n.get("status") or {}
        gpu = ", ".join(f"{g['vram_used_mb'] // 1024}/{g['vram_total_mb'] // 1024}G" for g in st.get("gpus", [])
                        if g["vram_total_mb"] > 4096)
        state = st.get("state", "?") if n["online"] else "offline"
        mem = f"{st.get('memory', {}).get('available_mb', 0) // 1024}G" if st else ""
        print(f"{n.get('name', '?'):<16}{state:<10}{st.get('active') or '-':<28}{gpu or '-':<16}{mem:<10}{n['url']}")


def cmd_switch(args) -> None:
    node = _find_node(_api("GET", "/api/overview")["nodes"], args.node)
    profile = None if args.profile.lower() in ("none", "off", "stop") else args.profile
    _api("POST", f"/api/nodes/{node['id']}/activate", json={"profile": profile})
    if not profile or args.no_wait:
        print("ok")
        return
    print(f"loading {profile} on {node['name']}", end="", flush=True)
    for _ in range(300):
        time.sleep(3)
        st = _find_node(_api("GET", "/api/overview")["nodes"], node["id"]).get("status") or {}
        if st.get("active") == profile and st.get("state") in ("ready", "error"):
            print(f"\n{st['state']}" + (f": {st['last_error']}" if st.get("last_error") else ""))
            return
        print(".", end="", flush=True)
    print("\nstill loading; check `homellm status`")


def cmd_profiles(args) -> None:
    node = _find_node(_api("GET", "/api/overview")["nodes"], args.node)
    data = _api("GET", f"/api/nodes/{node['id']}/models")
    print("profiles:")
    for name, p in data["profiles"].items():
        print(f"  {name:<28}{p['backend']:<10}{p['model']}  ctx={p.get('context', '-')}")
    print("models on disk:")
    for m in data["models"]:
        print(f"  {m['backend']:<10}{m['model']:<60}{m.get('size_gb', '?')} GB")


def main() -> None:
    ap = argparse.ArgumentParser(prog="homellm")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("node", help="run the node agent")
    sub.add_parser("controller", help="run the controller and dashboard")
    sub.add_parser("status", help="show nodes")
    sw = sub.add_parser("switch", help="load a profile on a node")
    sw.add_argument("node")
    sw.add_argument("profile")
    sw.add_argument("--no-wait", action="store_true")
    pr = sub.add_parser("profiles", help="list profiles and models on a node")
    pr.add_argument("node")
    ini = sub.add_parser("init", help="write starter config")
    ini.add_argument("what", choices=["node", "controller"])
    ini.add_argument("--force", action="store_true")
    pu = sub.add_parser("set-public-url", help="record the tunnel URL")
    pu.add_argument("url")
    args = ap.parse_args()

    if args.cmd == "node":
        from .node.agent import run
        run(_env("HOMELLM_CLUSTER_TOKEN"))
    elif args.cmd == "controller":
        for k in ("HOMELLM_CLUSTER_TOKEN", "HOMELLM_ADMIN_TOKEN", "LITELLM_MASTER_KEY"):
            os.environ.setdefault(k, _env(k))
        from .controller.app import run
        run()
    elif args.cmd == "init":
        from . import initcfg
        print((initcfg.init_node if args.what == "node" else initcfg.init_controller)(args.force))
    elif args.cmd == "set-public-url":
        from . import initcfg
        initcfg.set_public_url(args.url)
    else:
        {"status": cmd_status, "switch": cmd_switch, "profiles": cmd_profiles}[args.cmd](args)


if __name__ == "__main__":
    main()
