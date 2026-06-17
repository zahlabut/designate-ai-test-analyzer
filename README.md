# Designate AI Test Analyzer

Autonomous diagnostic agent for OpenStack Designate Tempest failures. It reads test source, runs `stestr`, pulls `journalctl` logs, and uses an LLM to correlate everything into a root-cause verdict.

**Stack:** CrewAI + Ollama (`llama3.2:1b` via Podman) + Rich CLI

---

## Tool flow

When you run `main.py`, the CLI prints this pipeline before test selection:

```
  Setup          Stage 1              Stage 2              Stage 3
  ─────          ───────              ───────              ───────
  Ollama ✓   →   Analyze logic   →   Run test        →   Root cause
  tempest ✓      (read_source)       (stestr)            (journalctl)
                   cyan · LLM          yellow              magenta · LLM
                                        │
                                        ├─ PASS  → done
                                        ├─ SKIP  → done
                                        └─ FAIL  → Stage 3
```

| Phase | What it does |
|-------|--------------|
| **Setup** | Verify Ollama and `tempest.conf`; list Ollama models (pick one if several); discover tests with `stestr list` (optional grep); pick one by index |
| **Stage 1** | Loads the test method **and helper methods it calls** from source, then Ollama explains the full end-to-end flow (API, DNS checks, propagation) |
| **Stage 2** | Runs `stestr run --serial` against DevStack; full output saved under `/opt/stack/agent_runs/run_<timestamp>/` |
| **Stage 3** | **FAIL only** — builds a **log evidence report** (Tempest traceback + run log + each Designate service separately); services with no errors are labeled explicitly; Ollama verdict is grounded in that report |

**Stage 2 outcomes:** PASS or SKIP → done · FAIL → Stage 3

---

## VM hardware requirements

This setup runs **DevStack** and **Ollama in Podman** (`llama3.2:1b`) on the same VM.

### Recommended VM

| Resource | Size | Why |
|----------|------|-----|
| **RAM** | **16 GiB** | DevStack + Ollama (`llama3.2:1b`) + headroom |
| **vCPU** | **4** | DevStack services; LLM inference is slow on 2 |
| **Disk** | **80 GiB** | DevStack growth, one model (~1.3 GiB), logs/artifacts |
| **Swap** | **8 GiB** | Buffer when RAM spikes during Tempest or model load |

**Not recommended:** 8 GiB RAM — DevStack alone can use most of it; Ollama will fail to load the model.

---

## DevStack setup (full walkthrough)

These steps were validated on Ubuntu Noble DevStack (`devstack-noble-new`), using the **DevStack Tempest venv**. Activate it once per SSH session before `stestr`, `pip install`, or `main.py` — your shell prompt should show `(venv)`.

### 1. Activate the DevStack venv

```bash
ssh stack@<VM_IP>    # password: stack

source /opt/stack/data/venv/bin/activate
cd /opt/stack/tempest
stestr list | grep designate | head
```

The analyzer reads all settings from **`conf.ini`** (paths, Ollama URL, optional model). It also creates `/etc/tempest/tempest.conf` → DevStack config on startup if missing (requires `sudo`).

### 2. Clone and install Python dependencies

Stay in the same session with the venv active:

```bash
cd /opt/stack
git clone https://github.com/zahlabut/designate-ai-test-analyzer.git
cd designate-ai-test-analyzer

pip install -r requirements.txt
```

Review **`conf.ini`** in this directory — defaults match a standard DevStack VM; change paths or Ollama URL if yours differ.

Do **not** use system Python or `--break-system-packages` — install into the DevStack venv.

### 3. Fix Tempest / `stestr list` (if discovery fails)

If `stestr list` fails with an OpenSSL error such as `AttributeError: module 'lib' has no attribute 'GEN_EMAIL'`, upgrade the crypto stack **in the venv**:

```bash
source /opt/stack/data/venv/bin/activate
pip install --upgrade pyOpenSSL cryptography
cd /opt/stack/tempest
stestr list | grep designate | head
```

### 4. Install and run Ollama (Podman)

Ollama provides the local LLM. Run it in a Podman container on the DevStack VM.

Install Podman:

```bash
sudo apt install podman
```

On Ubuntu Noble, Podman does **not** resolve short image names like `ollama/ollama`. Use the full registry path:

```bash
podman volume create ollama-data

podman run -d \
  --name ollama \
  -p 127.0.0.1:11434:11434 \
  -v ollama-data:/root/.ollama \
  docker.io/ollama/ollama
```

Optional — allow short image names in Podman:

```bash
sudo tee /etc/containers/registries.conf.d/docker.conf <<'EOF'
unqualified-search-registries = ["docker.io"]
EOF
```

#### Pull the model

```bash
podman exec -it ollama ollama pull llama3.2:1b
```

Verify:

```bash
curl http://127.0.0.1:11434/api/tags
```

You should see JSON listing at least one model.

### 5. Troubleshooting Ollama

#### `model requires more system memory than is available`

The VM is too small for DevStack + `llama3.2:1b`. Resize to **16 GiB RAM** (see [hardware requirements](#vm-hardware-requirements)).

#### `bind: address already in use` on port 11434

Port 11434 is already taken — often a leftover `ollama` container. Remove it and start fresh:

```bash
podman rm -f ollama
podman run -d \
  --name ollama \
  -p 127.0.0.1:11434:11434 \
  -v ollama-data:/root/.ollama \
  docker.io/ollama/ollama
```

If you cannot use port 11434, map a different host port:

```bash
podman rm -f ollama
podman run -d \
  --name ollama \
  -p 127.0.0.1:11435:11434 \
  -v ollama-data:/root/.ollama \
  docker.io/ollama/ollama

podman exec -it ollama ollama pull llama3.2:1b
```

Then set `base_url = http://127.0.0.1:11435` in `conf.ini`.

#### `container state improper` on `podman exec`

The container failed to start (often due to a port conflict). Run `podman ps -a`, remove it with `podman rm -f ollama`, fix the underlying issue, then start again.

### 6. Run the analyzer

With the DevStack venv still active:

```bash
source /opt/stack/data/venv/bin/activate
cd /opt/stack/designate-ai-test-analyzer
python3 main.py
```

On startup you should see the **tool flow** diagram, then Ollama model selection (if several models are installed).

The script reads **`conf.ini`**, checks Ollama, and lists pulled models before AI stages.

#### Configuration

All settings are in **`conf.ini`** next to `main.py`. Each option has a comment explaining what it does. Main sections:

| Section | Keys | Purpose |
|---------|------|---------|
| `[ollama]` | `base_url`, `model` | Ollama HTTP URL; optional fixed model (empty = pick at startup) |
| `[tempest]` | `tempest_path`, `config`, `stestr_bin`, … | Paths for stestr and tempest.conf |
| `[paths]` | `agent_runs_dir` | Where run logs and artifacts are saved |
| `[crewai]` | `tracing_enabled`, `api_key` | CrewAI internal settings (usually leave as-is) |

Example — Ollama on a remote host: edit `conf.ini`:

```ini
[ollama]
base_url = http://10.9.95.131:11434
model = ollama/llama3.2:1b
```

---

## Prerequisites summary

| Requirement | Notes |
|-------------|-------|
| VM sizing | **16 GiB RAM**, 4 vCPU, 80 GiB disk — see [hardware requirements](#vm-hardware-requirements) |
| DevStack with Designate | DNS enabled; api, central, producer, worker, and mdns running |
| DevStack venv | `source /opt/stack/data/venv/bin/activate` before `main.py` |
| `conf.ini` | Edit paths/Ollama URL for your VM (shipped with defaults for DevStack) |
| `designate-tempest-plugin` | Installed in Tempest environment |
| `stestr list` working | See crypto fix in step 3 if discovery fails |
| Ollama | Podman container with `llama3.2:1b` on the DevStack VM |
| `sudo` | Required for `journalctl` log extraction in Stage 3 |

---

## Usage

```bash
source /opt/stack/data/venv/bin/activate
cd /opt/stack/designate-ai-test-analyzer
python3 main.py
```

1. **Grep tests** — enter a filter (e.g. `recordset`, `multipool`) or press ENTER for all designate tests.
2. **Select test** — enter the index number from the list (required).
3. **Stages** — see [Tool flow](#tool-flow) for the full pipeline (Setup → Stage 1 → Stage 2 → Stage 3 on failure).

If the test is **skipped** (e.g. missing test data file), Stage 2 prints the skip reason and Stage 3 is not run.

---

## Project structure

| Path | Purpose |
|------|---------|
| `conf.ini` | All tool settings (Ollama, Tempest paths, artifact directory) |
| `main.py` | Agent definition, tools, and stage orchestration |
| `requirements.txt` | Python dependencies (`crewai`, `rich`) |
| `/opt/stack/agent_runs/run_<timestamp>/` | Per-run artifacts: `tempest_run.log`, `log_evidence.txt`, `designate_logs/*.log` |
