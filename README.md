# homellm

Run your own LLMs on the GPUs you have at home, across several machines, behind one OpenAI-compatible API.

```
                       internet
                          │  https://llm.example.com  (Cloudflare named tunnel)
┌─────────────────────────┼──────────────────────────── main server ─┐
│                    ┌────▼─────┐   model list   ┌────────────────┐   │
│   clients ───────▶ │ LiteLLM  │ ◀───────────── │   controller   │   │
│   (LAN :4000)      │ :4000/ui │    sync        │ dashboard :8780│   │
│                    └────┬─────┘                └───────┬────────┘   │
│                         │ routes requests              │ discovers   │
│                    ┌────▼─────┐                        │ (mDNS +     │
│                    │node agent│◀───────────────────────┤  LAN sweep) │
│                    │ + engine │                        │             │
└────────────────────┴──────────┴────────────────────────┼─────────────┘
                                                         │
             ┌───────────── LAN machine ──┐   ┌──────────▼── LAN machine ─┐
             │ node agent :8765           │   │ node agent :8765          │
             │ LM Studio / llama.cpp      │   │ LM Studio / llama.cpp     │
             └────────────────────────────┘   └───────────────────────────┘
```

| Piece | What it does |
|---|---|
| **Node agent** (`homellm node`, every GPU machine) | Keeps the chosen model loaded and restarts the engine if it dies (watchdog). Watches RAM and the kernel OOM killer and sends desktop alerts. Switches models on request. Advertises itself on the LAN over mDNS. |
| **Engines** | **LM Studio** (driven through `lms`) or upstream **llama.cpp** `llama-server` (run as its own systemd unit). Both start with the RAM prompt cache off (`LLAMA_ARG_CACHE_RAM=0`), which otherwise grows until a small box runs out of memory. |
| **Controller** (`homellm controller`, main server) | Finds nodes (mDNS, a /24 sweep, or addresses you add), polls them, and registers whatever each one serves in LiteLLM. Two nodes serving the same profile name are load-balanced. Serves the dashboard. |
| **LiteLLM** | The real [LiteLLM](https://github.com/BerriAI/litellm) proxy with Postgres: API keys, budgets, usage, logs and its admin UI. Cloud models you add there sit next to your local ones. |
| **Tunnel** | A Cloudflare named tunnel puts LiteLLM on your own domain over HTTPS, with no ports opened on your router. |

## Install

Linux with systemd. No Docker. Everything runs as user services except Postgres.

### Main server

```bash
git clone https://github.com/grvpanchal/homellm && cd homellm
./scripts/install-server.sh                 # node agent + LiteLLM + controller, generates secrets
sudo bash scripts/server-root-setup.sh      # Postgres for LiteLLM, LAN firewall rules, linger
systemctl --user restart homellm-litellm
```

Then open the dashboard at `http://<server-ip>:8780` and sign in with `HOMELLM_ADMIN_TOKEN` from `~/.config/homellm/secrets.env`.
The LiteLLM UI is at `http://<server-ip>:4000/ui` (user `admin`, password `LITELLM_MASTER_KEY`).

### Other machines on the LAN

```bash
git clone https://github.com/grvpanchal/homellm && cd homellm
./scripts/install-node.sh --token <HOMELLM_CLUSTER_TOKEN from the server's secrets.env>
sudo loginctl enable-linger $USER           # keep running when logged out
sudo ufw allow from <server-ip> to any port 8765,1234,1236 proto tcp   # if ufw is on
```

The controller finds the new node within a few minutes. Press **Scan LAN** to look now, or **Add node** to enter an address by hand when mDNS and the sweep can't reach it.

For LM Studio nodes, turn on **Serve on local network** in LM Studio's developer settings so the server can reach it.

### Access from anywhere

```bash
./scripts/setup-tunnel.sh llm.example.com
```

This needs a domain on your Cloudflare account. The first run opens a browser for `cloudflared tunnel login`. Afterwards:

```bash
curl https://llm.example.com/v1/chat/completions \
  -H "Authorization: Bearer <a key made in the LiteLLM UI>" \
  -H "Content-Type: application/json" \
  -d '{"model": "gemma-4-26b-a4b", "messages": [{"role": "user", "content": "hi"}]}'
```

Only LiteLLM goes through the tunnel, and every request needs a key. To expose the dashboard too, pass `--dashboard dash.example.com`, and put it behind Cloudflare Access.

## Models and profiles

A **profile** is a named, ready-to-run model setup on a node. Its name is the model id clients ask for.

```yaml
# ~/.config/homellm/node.yaml (created by the installer from what it finds)
profiles:
  gemma-4-26b-a4b:
    backend: lmstudio
    model: google/gemma-4-26b-a4b     # `lms ls` key
    context: 50000
    parallel: 2
  gemma-4-26b-qat-262k:
    backend: llamacpp
    model: gemma-4-26B_q4_0-it.gguf   # file in models_dir
    context: 262144
    parallel: 2
    args: [--flash-attn, "on", --cache-type-k, q8_0, --cache-type-v, q8_0, --no-kv-unified]
```

There are three ways to switch models:
- **Dashboard:** pick a profile on the node card and press **Switch**.
- **Terminal:** `homellm switch <node> <profile>`, or `homellm switch <node> none` to unload.
- **Add a new one:** in **Models** you can download a model (an LM Studio name, a Hugging Face URL, or a direct `.gguf` link for llama.cpp), turn any model on disk into a profile, then load it.

Each node runs one model at a time, so the engine has the GPU to itself. When the model changes, LiteLLM's model list updates within one poll (about 10 seconds).

## Watchdog

The node agent checks every 30 seconds (`watchdog` in `node.yaml`):

- **Engine down or model unloaded:** it reloads the model. If that fails more than `max_restarts_per_hour` times, it stops retrying and the node shows **error** until you switch profiles.
- **OOM killer hit the engine:** you get an alert.
- **Free RAM below `warn_available_mb`, or the engine above `warn_rss_mb`:** you get an alert.
- **LM Studio started without `LLAMA_ARG_CACHE_RAM=0`:** you get an alert, because RAM will climb.

Alerts go to the dashboard's event list and to the desktop through `notify-send`. Everything restarts through systemd (`Restart=always`).

## Commands

```bash
homellm status                     # nodes, state, VRAM, RAM
homellm profiles <node>            # profiles and models on disk
homellm switch <node> <profile>    # load and wait until ready
journalctl --user -u homellm-node -f
journalctl --user -u homellm-controller -f
journalctl --user -u homellm-litellm -f
```

## Ports

| Port | Service | Who needs it |
|---|---|---|
| 4000 | LiteLLM API and UI | LAN clients (and the tunnel, locally) |
| 8780 | Dashboard | you, on the LAN |
| 8765 | Node agent | the main server |
| 1234 / 1236 | LM Studio / llama-server | the main server (LiteLLM calls them directly) |
| 5353/udp | mDNS | discovery |

## Security

- **Secrets:** they live in `~/.config/homellm/secrets.env` (mode 600) and are never committed.
- **Node agents:** they only answer requests that carry the cluster token. `/health` is the exception and reveals nothing beyond the node's name.
- **Engines:** LM Studio and llama-server have no authentication of their own. Allow their ports only from the server's IP.
- **Internet exposure:** only LiteLLM is exposed, and it requires keys.

## Development

```bash
uv venv && uv pip install -e '.[dev]'
.venv/bin/pytest -q && .venv/bin/ruff check homellm tests
```

## License

MIT
