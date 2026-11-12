"""homellm command line.

    homellm node                 run the node agent (systemd runs this)
    homellm controller           run the controller + dashboard
    homellm status               nodes and what they serve
    homellm switch NODE PROFILE  load a profile on a node ("none" to stop)
    homellm profiles NODE        list profiles and models on disk
    homellm init node|controller write starter config from what is installed (and upgrade old config)
    homellm engine [status|install|update] [--version bNNNN] [--flavor cuda-12|cuda-13|rocm|vulkan|cpu]
                                 homellm's own llama-server on this machine
    homellm catalog              Hugging Face models ready to download
    homellm pull NAME|REPO [--quant Q] [--node NODE] [--list]
                                 download a catalog model or any HF GGUF repo (here, or on NODE via the controller)
    homellm set-public-url URL   record the tunnel URL shown in the dashboard
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import httpx

from . import CONTROLLER_PORT
from .common import CONFIG_DIR, MODELS_DIR


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


def cmd_engine(args) -> None:
    from . import engine
    if args.action == "status":
        cur = engine.current()
        print(f"detected build : {engine.detect_flavor()}")
        if not cur:
            print("installed      : none  (run `homellm engine install`)")
            return
        print(f"current        : {cur.get('tag')} ({cur.get('flavor')})  {cur['dir']}")
        print(f"               : {cur.get('version', '')}")
        others = [i for i in engine.installed() if i["dir"] != cur["dir"]]
        for i in others:
            print(f"also installed : {i.get('tag')} ({i.get('flavor')})")
        try:
            latest = engine.latest_tag(cur.get("flavor"))
            print(f"latest         : {latest}" + ("" if latest == cur.get("tag") else "  -> `homellm engine update`"))
        except Exception as e:
            print(f"latest         : unknown ({e})")
        return
    flavor = args.flavor or (None if args.action == "install" else (engine.current() or {}).get("flavor"))
    try:
        engine.install(flavor, args.version, args.force, _bar())
    except engine.EngineError as e:
        sys.exit(str(e))
    print("\nthe node agent uses it on the next model load (Switch, or `homellm switch NODE PROFILE`)")


def _bar():
    last = [0.0]

    def cb(done: int, total: int, name: str) -> None:
        if time.time() - last[0] < 0.5 and done < total:
            return
        last[0] = time.time()
        pct = done / total * 100 if total else 0
        print(f"\r  {name[:50]:<50} {done / 1e9:6.2f}/{total / 1e9:.2f} GB {pct:5.1f}%", end="", flush=True)
    return cb


def cmd_catalog(_args) -> None:
    from . import hf
    print(f"{'NAME':<26}{'VRAM':>6}  {'QUANT':<12}{'REPO':<52}NOTES")
    for e in hf.load_catalog():
        vram = f"{e['vram_gb']}G" if e.get("vram_gb") else "?"
        print(f"{e['name']:<26}{vram:>6}  {e.get('quant', 'Q4_K_M'):<12}{e['repo']:<52}{e.get('notes', '')}")
    print(f"\nadd your own in {hf.USER_CATALOG}")


def cmd_pull(args) -> None:
    from . import hf
    entry = hf.catalog_entry(args.model)
    repo = entry["repo"] if entry else args.model
    if args.list:
        for g in hf.group_ggufs(hf.list_repo(repo)):
            kind = "mmproj" if g["mmproj"] else "model "
            print(f"  {kind} {g['size_gb']:7.2f} GB  {g['name']}" + (f"  ({len(g['paths'])} parts)"
                                                                       if len(g["paths"]) > 1 else ""))
        return
    if args.node:
        node = _find_node(_api("GET", "/api/overview")["nodes"], args.node)
        body = {"catalog": args.model} if entry else {"repo": repo}
        if args.quant:
            body["quant"] = args.quant
        if args.mmproj:
            body["mmproj"] = True
        r = _api("POST", f"/api/nodes/{node['id']}/download", json=body)
        print(f"download started on {node['name']} (job {r['job']}); progress: dashboard -> Models")
        return
    from .common import load_yaml, save_yaml
    cfg = load_yaml(CONFIG_DIR / "node.yaml")
    lc = (cfg.get("backends") or {}).get("llamacpp") or {}
    models_dir = Path(os.path.expanduser(lc.get("models_dir", str(MODELS_DIR))))
    mmproj = args.mmproj or (entry or {}).get("mmproj", False)
    try:
        plan = hf.plan(repo, args.quant or (entry or {}).get("quant"), mmproj=mmproj)
    except hf.HFError as e:
        sys.exit(str(e))
    print(f"{repo}: {', '.join(plan['model']['paths'])}" + (f" + {plan['mmproj']['paths'][0]}" if plan["mmproj"]
                                                            else "") + f"  ({plan['size'] / 1e9:.1f} GB)")
    try:
        hf.fetch_plan(plan, models_dir, _bar())
    except hf.HFError as e:
        sys.exit(f"\n{e}")
    name, profile = hf.profile_for(plan, models_dir, entry)
    extra_path = CONFIG_DIR / "profiles.yaml"
    extra = load_yaml(extra_path)
    extra[name] = profile
    save_yaml(extra_path, extra)
    print(f"\nprofile {name} saved; load it with the dashboard or `homellm switch <this node> {name}`")


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
    en = sub.add_parser("engine", help="homellm's own llama-server on this machine")
    en.add_argument("action", nargs="?", default="status", choices=["status", "install", "update"])
    en.add_argument("--version", help="llama.cpp release tag, e.g. b12170 (default: newest)")
    en.add_argument("--flavor", choices=["cuda-12", "cuda-13", "rocm", "vulkan", "cpu"],
                    help="build to install (default: detected from the GPU)")
    en.add_argument("--force", action="store_true", help="reinstall even if that version is present")
    sub.add_parser("catalog", help="Hugging Face models ready to download")
    pl = sub.add_parser("pull", help="download a catalog model or a Hugging Face GGUF repo")
    pl.add_argument("model", help="catalog name (see `homellm catalog`) or Hugging Face repo owner/name")
    pl.add_argument("--quant", help="quant to pick, e.g. Q4_K_M, UD-Q4_K_XL, Q8_0")
    pl.add_argument("--mmproj", action="store_true", help="also fetch the vision projector")
    pl.add_argument("--node", help="download on this node (via the controller) instead of here")
    pl.add_argument("--list", action="store_true", help="only list the repo's GGUF files")
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
        {"status": cmd_status, "switch": cmd_switch, "profiles": cmd_profiles, "engine": cmd_engine,
         "catalog": cmd_catalog, "pull": cmd_pull}[args.cmd](args)


if __name__ == "__main__":
    main()
