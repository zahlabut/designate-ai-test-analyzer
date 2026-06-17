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
| **Summary** | Closing recap: test, brief intent, PASS/FAIL, root cause if failed |

### Example output

Rendered in the browser via **GitHub Pages**. If links return **404**, enable Pages once — see [Enable GitHub Pages](#enable-github-pages) (takes ~1 minute).

| Outcome | Example |
|---------|---------|
| **PASS** — Stage 3 skipped | [example-pass.html](https://zahlabut.github.io/designate-ai-test-analyzer/example-pass.html) — `test_update_records_propagated_to_backends_07_MX_under_APEX` |
| **FAIL** — Stage 3 root-cause analysis | [example-fail.html](https://zahlabut.github.io/designate-ai-test-analyzer/example-fail.html) — `test_update_records_propagated_to_backends_14_NAPTR_Record` (wrong nameserver port in `tempest.conf`) |

All examples: [zahlabut.github.io/designate-ai-test-analyzer/](https://zahlabut.github.io/designate-ai-test-analyzer/)

---

## Enable GitHub Pages

GitHub does **not** publish the site automatically. Until Pages is enabled, every link above returns 404 even though `docs/` is already in the repo.

**Recommended (no workflow needed):**

1. Open **[Settings → Pages](https://github.com/zahlabut/designate-ai-test-analyzer/settings/pages)** for this repo.
2. **Build and deployment → Source:** select **Deploy from a branch**.
3. **Branch:** `master` · **Folder:** `/docs` → click **Save**.
4. Wait 1–2 minutes. Refresh [the site](https://zahlabut.github.io/designate-ai-test-analyzer/) — you should see the examples index.

**Alternative (GitHub Actions):** push `.github/workflows/pages.yml`, then set Source to **GitHub Actions** on the same Settings → Pages screen. The workflow publishes `docs/` on every push to `master`.

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
source /opt/stack/data/venv/bin/activate
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

#### Which model to pull?

**`llama3.2:1b` is a small model** (~1.3 GiB download, ~1.5–2 GiB RAM while running). It is the recommended default for a **16 GiB** DevStack VM.

| Model | Size | RAM (approx.) | Fits 16 GiB DevStack VM? |
|-------|------|---------------|----------------------------|
| `qwen2.5:0.5b` | Tiny | ~0.5–1 GiB | Yes — most headroom |
| **`llama3.2:1b`** | **Small** | **~1.5–2 GiB** | **Yes — recommended** |
| `llama3.2:3b` | Medium | ~2–3 GiB | Tight — may OOM under load |
| `llama3.1` | Large | ~5 GiB | No — use 32 GiB VM or remote Ollama |

Pull one (or several — the tool lets you pick at startup):

```bash
# Recommended for 16 GiB VM
podman exec -it ollama ollama pull llama3.2:1b

# Smaller / faster (less accurate)
podman exec -it ollama ollama pull qwen2.5:0.5b

# Larger VM only (32 GiB+ RAM)
podman exec -it ollama ollama pull llama3.1
```

Optional: pin a model in `conf.ini` so startup skips the picker:

```ini
[ollama]
model = ollama/llama3.2:1b
```

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
| `docs/` | GitHub Pages site — rendered example output (`index.html`, `example-pass.html`, `example-fail.html`) |
| `.github/workflows/pages.yml` | Publishes `docs/` to GitHub Pages on push to `master` |
| `/opt/stack/agent_runs/run_<timestamp>/` | `tempest_run.log`, `log_evidence.txt`, `designate_logs/*.log` |
