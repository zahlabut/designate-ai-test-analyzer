# Designate AI Test Analyzer

Autonomous diagnostic agent for OpenStack Designate Tempest failures. It reads test source, runs `stestr`, pulls `journalctl` logs, and uses an LLM to correlate everything into a root-cause verdict.

**Stack:** CrewAI + Ollama (Podman) + Rich CLI

---

## Tool flow

```
  Setup          Stage 1              Stage 2              Stage 3
  ─────          ───────              ───────              ───────
  Ollama ✓   →   Analyze logic   →   Run test        →   Root cause
  tempest ✓      (test source)       (stestr)            (journalctl)
                   cyan · LLM          yellow              magenta · LLM
                                        │
                                        ├─ PASS  → done
                                        ├─ SKIP  → done
                                        └─ FAIL  → Stage 3
```

| Phase | What it does |
|-------|--------------|
| **Setup** | Check Ollama + `tempest.conf`; pick LLM model; grep/list tests; select one |
| **Stage 1** | Load test + helper source code; Ollama explains the end-to-end flow |
| **Stage 2** | Run `stestr run --serial`; save output under `/opt/stack/agent_runs/` |
| **Stage 3** | **FAIL only** — log evidence report per service + Ollama root-cause verdict |

---

## VM requirements

DevStack + Ollama (Podman) on the same VM:

| Resource | Size |
|----------|------|
| RAM | **16 GiB** |
| vCPU | **4** |
| Disk | **80 GiB** |
| Swap | **8 GiB** (recommended) |

8 GiB RAM is not enough — Ollama will fail to load a model with DevStack running.

---

## Setup

Validated on Ubuntu Noble DevStack. Use the **DevStack venv** (`source /opt/stack/data/venv/bin/activate`).

### 1. Clone and install

```bash
cd /opt/stack
git clone https://github.com/zahlabut/designate-ai-test-analyzer.git
cd designate-ai-test-analyzer
pip install -r requirements.txt
```

Edit **`conf.ini`** if your paths differ (defaults match a standard DevStack VM). All settings and comments are in that file.

### 2. Ollama (Podman)

```bash
sudo apt install podman
podman volume create ollama-data

podman run -d \
  --name ollama \
  -p 127.0.0.1:11434:11434 \
  -v ollama-data:/root/.ollama \
  docker.io/ollama/ollama

podman exec -it ollama ollama pull llama3.2:1b
curl http://127.0.0.1:11434/api/tags
```

On Noble use the full image path `docker.io/ollama/ollama` (short names may not resolve).

### 3. Run

```bash
source /opt/stack/data/venv/bin/activate
cd /opt/stack/designate-ai-test-analyzer
python3 main.py
```

The tool creates `/etc/tempest/tempest.conf` → DevStack config if missing (`sudo` required). At startup: pick an Ollama model (if several exist), grep tests, select by index, then stages run automatically.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `stestr list` OpenSSL / `GEN_EMAIL` error | `pip install --upgrade pyOpenSSL cryptography` in the venv |
| Ollama out of memory | Resize VM to 16 GiB RAM |
| Port 11434 in use | `podman rm -f ollama` and re-run the container |
| Alternate Ollama port | Map e.g. `-p 127.0.0.1:11435:11434`, set `base_url` in `conf.ini` |
| `container state improper` | `podman ps -a`, `podman rm -f ollama`, fix port conflict, start again |

---

## Project files

| Path | Purpose |
|------|---------|
| `conf.ini` | All tool settings (Ollama, Tempest paths, artifact directory) |
| `main.py` | Stage orchestration and CLI |
| `requirements.txt` | Python dependencies |
| `/opt/stack/agent_runs/run_<timestamp>/` | `tempest_run.log`, `log_evidence.txt`, `designate_logs/*.log` |
