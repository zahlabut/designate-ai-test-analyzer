# Designate AI Test Analyzer

Autonomous diagnostic agent for OpenStack Designate Tempest failures. It reads test source, runs `stestr`, pulls `journalctl` logs, and uses an LLM to correlate everything into a root-cause verdict.

**Stack:** CrewAI + Ollama (Llama 3.1) + Rich CLI

---

## DevStack setup (full walkthrough)

These steps were validated on Ubuntu Noble DevStack (`devstack-noble`), running directly from system Python (no Tempest venv).

### 1. Clone and install Python dependencies

```bash
cd /opt/stack
git clone https://github.com/zahlabut/designate-ai-test-analyzer.git
cd designate-ai-test-analyzer

python3 -m pip install -r requirements.txt --break-system-packages
```

On Noble, `--break-system-packages` is usually required when installing into the system Python alongside apt packages.

### 2. Fix Tempest / `stestr list` (if discovery fails)

If `stestr list` fails with an OpenSSL error such as `AttributeError: module 'lib' has no attribute 'GEN_EMAIL'`, upgrade the crypto stack for the same `python3` that runs Tempest:

```bash
python3 -m pip install --upgrade pyOpenSSL cryptography --break-system-packages
```

Verify Tempest discovery works:

```bash
cd /opt/stack/tempest
stestr list | grep designate | head
```

### 3. Install Podman

```bash
sudo apt install podman
```

### 4. Run Ollama locally via Podman

On Ubuntu Noble, Podman does **not** resolve short image names like `ollama/ollama`. Use the full registry path:

```bash
podman volume create ollama-data

podman run -d \
  --name ollama \
  -p 127.0.0.1:11434:11434 \
  -v ollama-data:/root/.ollama \
  docker.io/ollama/ollama
```

Pull a model into the container:

```bash
podman exec -it ollama ollama pull llama3.1
```

Verify Ollama is responding:

```bash
curl http://127.0.0.1:11434/api/tags
```

You should see JSON listing at least one model.

#### Optional: allow short image names in Podman

```bash
sudo tee /etc/containers/registries.conf.d/docker.conf <<'EOF'
unqualified-search-registries = ["docker.io"]
EOF
```

After this, `ollama/ollama` works without the `docker.io/` prefix.

### 5. Troubleshooting Ollama / Podman

#### `bind: address already in use` on port 11434

Something is already listening on 11434. Check whether Ollama is already running:

```bash
ss -tlnp | grep 11434
curl -s http://127.0.0.1:11434/api/tags
podman ps -a
```

- **If `curl` returns model JSON** — Ollama is already up. Skip `podman run` and go to step 6.
- **If you need a fresh Podman container** — remove the failed one, free the port, then re-run:

```bash
podman rm -f ollama
# stop whatever holds 11434 (see ss/lsof output), then:
podman run -d \
  --name ollama \
  -p 127.0.0.1:11434:11434 \
  -v ollama-data:/root/.ollama \
  docker.io/ollama/ollama
```

- **If you cannot free port 11434** — run Ollama on a different host port:

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

```bash
cd /opt/stack/designate-ai-test-analyzer
python3 main.py
```

On startup you should see:

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
export OLLAMA_BASE_URL=http://10.9.95.131:11434
export OLLAMA_MODEL=ollama/llama3.1
python3 main.py
```

---

## Prerequisites summary

| Requirement | Notes |
|-------------|-------|
| DevStack with Designate | DNS enabled, central + worker running |
| `designate-tempest-plugin` | Installed in Tempest environment |
| `stestr list` working | See crypto fix in step 2 if discovery fails |
| Ollama | Local Podman container or remote instance |
| `sudo` | Required for `journalctl` log extraction in Stage 3 |

---

## Usage

```bash
python3 main.py
```

1. **Grep tests** — enter a filter (e.g. `recordset`, `multipool`) or press ENTER for all designate tests.
2. **Select test** — enter the index number, or press ENTER for the default multipool test.
3. **Autonomous stages:**
   - **Stage 0: Context priming** — AI internalizes Designate Central/Worker architecture.
   - **Stage 1: Logic discovery** — AI reads test source and explains intent step by step.
   - **Stage 2: Execution** — runs `stestr run <test>`, saves output to `/opt/stack/agent_runs/run_<timestamp>/`.
   - **Stage 3: Root cause** (on failure only) — fetches `journalctl` logs and produces a technical verdict.

---

## Project structure

| Path | Purpose |
|------|---------|
| `main.py` | Agent definition, tools, and stage orchestration |
| `requirements.txt` | Python dependencies (`crewai`, `rich`) |
| `/opt/stack/agent_runs/` | Runtime artifacts (tempest logs, designate logs) per run |
