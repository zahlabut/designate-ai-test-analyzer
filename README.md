# Designate AI Test Analyzer

Autonomous diagnostic agent for OpenStack Designate Tempest failures. It reads test source, runs `stestr`, pulls `journalctl` logs, and uses an LLM to correlate everything into a root-cause verdict.

**Stack:** CrewAI + Ollama (Llama 3.1) + Rich CLI

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
| **Setup** | Verify Ollama and `tempest.conf`; discover tests with `stestr list` (optional grep); pick one by index |
| **Stage 1** | Loads the test method **and helper methods it calls** from source, then Ollama explains the full end-to-end flow (API, DNS checks, propagation) |
| **Stage 2** | Runs `stestr run --serial` against DevStack; full output saved under `/opt/stack/agent_runs/run_<timestamp>/` |
| **Stage 3** | **FAIL only** — builds a **log evidence report** (Tempest traceback + run log + each Designate service separately); services with no errors are labeled explicitly; Ollama verdict is grounded in that report |

**Stage 2 outcomes:** PASS or SKIP → done · FAIL → Stage 3

---

## VM hardware requirements (DevStack + Ollama on same host)

DevStack and Ollama compete for the same RAM. An **8 GiB / 2 vCPU** VM can run Tempest, but will fail to load `llama3.1` locally (Ollama reported ~4.8 GiB needed while only ~1.3 GiB was free with DevStack running).

### Recommended VM sizes

| Scenario | RAM | vCPU | Disk | Swap |
|----------|-----|------|------|------|
| DevStack only, Ollama on **another host** | 12–16 GiB (16,384 MiB) | 4 | 60–80 GiB | 4–8 GiB |
| DevStack + **small local model** (`llama3.2:1b`, `qwen2.5:0.5b`) | **16 GiB** (16,384 MiB) | 4 | 80 GiB | 8 GiB |
| DevStack + **`llama3.1` locally** | 24–32 GiB | 8 | 100 GiB | 8–16 GiB |

**Minimum not recommended:** 8 GiB RAM / 2 vCPU / 40 GiB disk — workable for DevStack alone, not for local LLM inference.

### RAM budget (rough)

```
DevStack (Designate, Neutron, MySQL, Nova, …)  →  6–8 GiB
Ollama daemon                                   →  ~0.5 GiB
Model at runtime:
  qwen2.5:0.5b / tinyllama                       →  ~0.5–1 GiB
  llama3.2:1b                                    →  ~1.5–2 GiB
  llama3.1                                       →  ~5 GiB
Headroom (Tempest runs, OS cache)               →  2–4 GiB
```

**Rule of thumb:** `RAM ≈ 8 GiB + model size + 2–4 GiB headroom`

### Disk

| Component | Space |
|-----------|-------|
| DevStack base (OS, packages, logs) | ~25–35 GiB |
| OpenStack images/volumes over time | +10–20 GiB |
| Ollama models | 1–5 GiB per model |
| Tempest / plugin / agent artifacts | ~2–5 GiB |

Plan **80 GiB** minimum when keeping models on the same VM.

### Check if your VM is big enough

```bash
free -h
grep -E 'MemTotal|MemAvailable|SwapTotal' /proc/meminfo
nproc
df -h /
ps aux --sort=-%mem | head -10
```

| Reading | Concern |
|---------|---------|
| `available` < 2 GiB | Too tight for any local model |
| `SwapTotal` = 0 | No buffer when RAM fills — add swap |
| Disk > 85% full | Model pulls may fail |
| Only 2 vCPU | LLM inference will be slow |

### Model choice vs available RAM

| Model | Approx. RAM to load | Fits on 16 GiB DevStack VM? |
|-------|-------------------|----------------------------|
| `qwen2.5:0.5b` | ~0.5–1 GiB | Yes (best margin) |
| `llama3.2:1b` | ~1.5–2 GiB | Yes |
| `llama3.1` | ~5 GiB | No — use remote Ollama or 32 GiB VM |

Set the model via environment variable:

```bash
export OLLAMA_MODEL=ollama/llama3.2:1b
```

---

## DevStack setup (full walkthrough)

These steps were validated on Ubuntu Noble DevStack (`devstack-noble-new`), using the **DevStack Tempest venv**. Activate it once per SSH session before `stestr`, `pip install`, or `main.py` — your shell prompt should show `(venv)`.

### 1. Activate the DevStack venv

```bash
ssh stack@<VM_IP>    # password: stack

source /opt/stack/data/venv/bin/activate
export TEMPEST_CONFIG=/opt/stack/tempest/etc/tempest.conf
cd /opt/stack/tempest
stestr list | grep designate | head
```

The analyzer uses `stestr` internally for discovery (`stestr list`) and execution (`stestr run --serial`) in Stage 2 — you do not need to run tests manually or escape `[id-...]` brackets yourself.

DevStack writes credentials to `/opt/stack/tempest/etc/tempest.conf`. Without `TEMPEST_CONFIG`, `stestr` defaults to missing `/etc/tempest/tempest.conf` and fails with `Password is not defined`.

One-time fix (optional — makes manual runs work without the export):

```bash
sudo mkdir -p /etc/tempest
sudo ln -sf /opt/stack/tempest/etc/tempest.conf /etc/tempest/tempest.conf
```

### 2. Clone and install Python dependencies

Stay in the same session with the venv active:

```bash
cd /opt/stack
git clone https://github.com/zahlabut/designate-ai-test-analyzer.git
cd designate-ai-test-analyzer

pip install -r requirements.txt
```

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

#### Pull a model

Choose a model that fits your VM RAM (see [hardware requirements](#vm-hardware-requirements-devstack--ollama-on-same-host)). On a **16 GiB** DevStack VM, prefer a small model:

```bash
podman exec -it ollama ollama pull llama3.2:1b
```

Verify:

```bash
curl http://127.0.0.1:11434/api/tags
```

You should see JSON listing at least one model.

```bash
export OLLAMA_MODEL=ollama/llama3.2:1b
```

### 5. Troubleshooting Ollama

#### `model requires more system memory than is available`

The VM is too small for the chosen model with DevStack running. Either:

- Switch to a smaller model: `export OLLAMA_MODEL=ollama/llama3.2:1b`
- Run Ollama on a remote host with more RAM (see [hardware requirements](#vm-hardware-requirements-devstack--ollama-on-same-host))
- Resize the VM to 16+ GiB RAM

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

podman exec -it ollama ollama pull llama3.1
export OLLAMA_BASE_URL=http://127.0.0.1:11435
```

#### `container state improper` on `podman exec`

The container failed to start (often due to a port conflict). Run `podman ps -a`, remove it with `podman rm -f ollama`, fix the underlying issue, then start again.

#### Smaller model (limited RAM)

```bash
podman exec -it ollama ollama pull llama3.2:1b
export OLLAMA_MODEL=ollama/llama3.2:1b
```

### 6. Run the analyzer

With the DevStack venv still active:

```bash
source /opt/stack/data/venv/bin/activate
export TEMPEST_CONFIG=/opt/stack/tempest/etc/tempest.conf
cd /opt/stack/designate-ai-test-analyzer
python3 main.py
```

On startup you should see the **tool flow** diagram (above), then:

```
LLM: ollama/llama3.1 @ http://127.0.0.1:11434
```

The script checks Ollama connectivity before starting AI stages. If it fails, fix Ollama first (steps 4–5).

#### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | Ollama HTTP endpoint (no `/v1` suffix) |
| `OLLAMA_MODEL` | `ollama/llama3.1` | Model name passed to CrewAI |

Example — Ollama on a remote host:

```bash
source /opt/stack/data/venv/bin/activate
export OLLAMA_BASE_URL=http://10.9.95.131:11434
export OLLAMA_MODEL=ollama/llama3.1
cd /opt/stack/designate-ai-test-analyzer
python3 main.py
```

---

## Prerequisites summary

| Requirement | Notes |
|-------------|-------|
| VM sizing | See [hardware requirements](#vm-hardware-requirements-devstack--ollama-on-same-host) — **16 GiB RAM** minimum for local small model |
| DevStack with Designate | DNS enabled; api, central, producer, worker, and mdns running |
| DevStack venv | `source /opt/stack/data/venv/bin/activate` before `stestr` and `main.py` |
| `designate-tempest-plugin` | Installed in Tempest environment |
| `stestr list` working | See crypto fix in step 3 if discovery fails |
| Ollama | Podman container on the DevStack VM (or remote instance for larger models) |
| `sudo` | Required for `journalctl` log extraction in Stage 3 |

---

## Usage

```bash
source /opt/stack/data/venv/bin/activate
export TEMPEST_CONFIG=/opt/stack/tempest/etc/tempest.conf
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
| `main.py` | Agent definition, tools, and stage orchestration |
| `requirements.txt` | Python dependencies (`crewai`, `rich`) |
| `/opt/stack/agent_runs/run_<timestamp>/` | Per-run artifacts: `tempest_run.log`, `log_evidence.txt`, `designate_logs/*.log` |
