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
             │ llama-server (own copy)    │   │ llama-server (own copy)   │
             └────────────────────────────┘   └───────────────────────────┘
```

| Piece | What it does |
|---|---|
| **Node agent** (`homellm node`, every GPU machine) | Keeps the chosen model loaded and restarts the engine if it dies (watchdog). Watches RAM and the kernel OOM killer and sends desktop alerts. Switches models on request. Advertises itself on the LAN over mDNS. |
| **Engine** | homellm's own copy of upstream **llama.cpp** `llama-server`, installed from the official release builds into `~/.local/share/homellm/llama.cpp` (CUDA, ROCm, Vulkan or CPU, picked from the GPU). It runs as its own systemd unit with the RAM prompt cache off (`--cache-ram 0`), which otherwise grows until a small box runs out of memory. No LM Studio needed. |
| **Model catalog** | A curated list of GGUF models on Hugging Face ([`homellm/catalog.yaml`](homellm/catalog.yaml)), downloadable on any node from the dashboard or `homellm pull`, plus any other Hugging Face GGUF repo. |
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
sudo ufw allow from <server-ip> to any port 8765,1236 proto tcp   # if ufw is on
```

The installer downloads the llama.cpp build that matches the machine: CUDA 12 for Pascal/Volta cards (P40, P100,
V100; CUDA 13 dropped them), CUDA 13 for newer NVIDIA cards, ROCm or Vulkan for AMD, and CPU otherwise. Override with
`--flavor cuda-12|cuda-13|rocm|vulkan|cpu`, pin a release with `--engine-version b12170`, or skip it with
`--no-engine`. For gated Hugging Face models (Llama, some Gemma), pass `--hf-token <token>`.

The controller finds the new node within a few minutes. Press **Scan LAN** to look now, or **Add node** to enter an address by hand when mDNS and the sweep can't reach it.

### Coming from LM Studio

Models live in `~/.lmstudio/models`, shared with LM Studio, but homellm serves them with its own llama-server, so
the installer unloads and quits LM Studio and turns off its autostart. With linger on, the node agent starts at boot
and reloads the last active model.

Re-running `install-node.sh` (or just restarting the agent) upgrades `node.yaml` once, keeping a `.bak-v1` copy:
LM Studio profiles become llama.cpp profiles that run the same GGUF files from `~/.lmstudio/models` (nothing is
downloaded again), under the same names, so clients keep working. Quit LM Studio afterwards: if it still has a
model loaded it holds the VRAM, and the node shows which process is in the way.

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
backends:
  llamacpp:
    models_dir: ~/.lmstudio/models    # LM Studio's folder; downloads go to <publisher>/<repo>/ like LM Studio's
    port: 1236
    # server_bin: /path/to/llama-server   # only to use your own build instead of homellm's
profiles:
  gemma-4-26b-a4b:
    backend: llamacpp
    model: unsloth/gemma-4-26B-A4B-it-GGUF/gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf   # relative to models_dir, or absolute
    context: 65536                    # the default for new profiles
    parallel: 2
    args: [--mmproj, ~/.lmstudio/models/unsloth/gemma-4-26B-A4B-it-GGUF/mmproj-F16.gguf, --flash-attn, "on"]
  gemma-4-26b-qat-262k:
    backend: llamacpp
    model: gemma-4-26B_q4_0-it.gguf
    context: 262144
    parallel: 2
    args: [--flash-attn, "on", --cache-type-k, q8_0, --cache-type-v, q8_0, --no-kv-unified]
```

There are three ways to switch models:
- **Dashboard:** pick a profile on the node card and press **Switch**.
- **Terminal:** `homellm switch <node> <profile>`, or `homellm switch <node> none` to unload.
- **Add a new one:** in **Models**, download from the catalog, list any Hugging Face repo and pick a quant, or paste a
  `.gguf` URL. The download creates a ready-to-load profile. Any GGUF already on disk can also become a profile.

### Downloading models

```bash
homellm catalog                               # the curated list, with rough VRAM needs
homellm pull qwen3-14b                        # download here and create the profile
homellm pull qwen3-14b --node ui-pc           # download on another node (through the controller)
homellm pull unsloth/Qwen3-8B-GGUF --list     # any HF repo: show its quants
homellm pull unsloth/Qwen3-8B-GGUF --quant Q8_0
```

Downloads resume where they stopped, check free disk space first, and verify each file's SHA-256 against Hugging
Face. Split models (`-00001-of-00003.gguf`) are fetched whole. Add your own catalog entries in
`~/.config/homellm/catalog.yaml` (same format as [`homellm/catalog.yaml`](homellm/catalog.yaml)); on the main
server they show up in the dashboard for every node.

### The llama-server engine

```bash
homellm engine            # installed version, detected build, newest release
homellm engine update     # newest llama.cpp release, same build flavor
homellm engine install --version b12170 --flavor cuda-12
```

The dashboard's **Update engine** button does the same on a node and reloads its model on the new version. Each
version is unpacked next to the previous one and checked with `llama-server --version` before `current` moves, so
a broken build never replaces a working one. The last two versions are kept.

Each node runs one model at a time, so the engine has the GPU to itself. When the model changes, LiteLLM's model list updates within one poll (about 10 seconds).

## Watchdog

The node agent checks every 30 seconds (`watchdog` in `node.yaml`):

- **Engine down or model unloaded:** it reloads the model. If that fails more than `max_restarts_per_hour` times, it stops retrying and the node shows **error** until you switch profiles.
- **OOM killer hit the engine:** you get an alert.
- **Free RAM below `warn_available_mb`, or the engine above `warn_rss_mb`:** you get an alert.
- **Another program holds the VRAM** (a leftover LM Studio, a game): the failed load names the process.

Alerts go to the dashboard's event list and to the desktop through `notify-send`. Everything restarts through systemd (`Restart=always`).

## GPU fan control

Passive datacenter GPUs (Tesla P100/P40) need add-on fans. [`fan-control/`](fan-control/README.md) is a separate
root service that drives them from GPU temperature, as quietly as possible, with fail-safes: missing sensors set
every fan to full speed, and stopping the service hands the fans back to the BIOS. Each host has its own config in
`fan-control/hosts/`. Install with `sudo fan-control/install.sh`.

## Commands

```bash
homellm status                     # nodes, state, VRAM, RAM
homellm profiles <node>            # profiles and models on disk
homellm switch <node> <profile>    # load and wait until ready
homellm catalog / pull <name>      # download models from Hugging Face
homellm engine [update]            # homellm's llama-server on this machine
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
| 1236 | llama-server | the main server (LiteLLM calls it directly) |
| 5353/udp | mDNS | discovery |

## Security

- **Secrets:** they live in `~/.config/homellm/secrets.env` (mode 600) and are never committed.
- **Node agents:** they only answer requests that carry the cluster token. `/health` is the exception and reveals nothing beyond the node's name.
- **Engine:** llama-server has no authentication of its own. Allow port 1236 only from the server's IP. It comes from
  the official `ggml-org/llama.cpp` GitHub releases, with the archive's SHA-256 checked when GitHub publishes one.
- **Internet exposure:** only LiteLLM is exposed, and it requires keys.

## Development

```bash
uv venv && uv pip install -e '.[dev]'
.venv/bin/pytest -q && .venv/bin/ruff check homellm tests
```

## License

MIT
