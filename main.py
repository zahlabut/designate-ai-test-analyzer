"""
Designate E2E Test Investigator
================================

This script helps debug failing OpenStack Designate Tempest tests on a DevStack VM.

High-level flow:
  1. Setup  — check Ollama + Tempest config, list tests, user picks one
  2. Stage 1 — read test source code, LLM explains what the test does
  3. Stage 2 — run the test with stestr (no LLM)
  4. Stage 3 — on FAIL only: collect logs, LLM suggests root cause

Run: python3 main.py  (settings in conf.ini next to this file)
"""

# --- IMPORTS ---
# Standard library: run shell commands, read files, talk to Ollama over HTTP, etc.
import subprocess
import os
import sys
import re
import json
import importlib.util
import urllib.error
import urllib.request
from configparser import ConfigParser
from datetime import datetime

# CrewAI: framework that sends tasks to the local LLM (Ollama) and optional tools.
from crewai import Agent, Task, Crew

# Rich: pretty coloured panels and text in the terminal (better than plain print).
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

# Single shared console instance used everywhere for output.
console = Console()

# --- CONFIGURATION (conf.ini) ---
# All user-facing settings live in conf.ini. CrewAI still needs a few internal
# os.environ values at runtime; those are set from conf.ini, not by the user.


CONF_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "conf.ini")


def load_settings() -> ConfigParser:
    """
    Why: One config file is easier to document and edit than many env exports.
    What: Reads conf.ini from the same directory as main.py; exits with a clear
          error if the file is missing.
    """
    if not os.path.isfile(CONF_FILE):
        console.print(Panel(
            Text(
                f"Configuration file not found:\n  {CONF_FILE}\n\n"
                "Copy conf.ini from the repository and edit paths for your VM."
            ),
            title="Startup error",
            border_style="red",
        ))
        sys.exit(1)
    parser = ConfigParser()
    parser.read(CONF_FILE)
    return parser


def apply_crewai_from_config(cfg: ConfigParser, base_url: str) -> None:
    """
    Why: CrewAI reads OpenAI-compatible settings from os.environ internally.
    What: Sets those env vars from conf.ini so the user never exports them manually.
    """
    os.environ["OPENAI_API_BASE"] = f"{base_url.rstrip('/')}/v1"
    os.environ["OPENAI_API_KEY"] = cfg.get("crewai", "api_key", fallback="ollama")
    tracing = cfg.get("crewai", "tracing_enabled", fallback="false").strip().lower()
    os.environ["CREWAI_TRACING_ENABLED"] = "true" if tracing in ("1", "true", "yes") else "false"


def init_globals_from_config(cfg: ConfigParser) -> None:
    """Load module-level settings from conf.ini into global variables."""
    global OLLAMA_BASE, CONFIGURED_OLLAMA_MODEL
    global TEMPEST_PATH, DEVSTACK_TEMPEST_CONF, SYSTEM_TEMPEST_CONF
    global TEMPEST_CONFIG, STESTR_BIN, BASE_HISTORY_DIR

    OLLAMA_BASE = cfg.get("ollama", "base_url", fallback="http://127.0.0.1:11434").rstrip("/")
    CONFIGURED_OLLAMA_MODEL = cfg.get("ollama", "model", fallback="").strip()
    apply_crewai_from_config(cfg, OLLAMA_BASE)

    TEMPEST_PATH = cfg.get("tempest", "tempest_path", fallback="/opt/stack/tempest")
    DEVSTACK_TEMPEST_CONF = cfg.get("tempest", "devstack_conf", fallback="/opt/stack/tempest/etc/tempest.conf")
    SYSTEM_TEMPEST_CONF = cfg.get("tempest", "system_conf", fallback="/etc/tempest/tempest.conf")
    TEMPEST_CONFIG = cfg.get("tempest", "config", fallback=DEVSTACK_TEMPEST_CONF)
    STESTR_BIN = cfg.get("tempest", "stestr_bin", fallback="/opt/stack/data/venv/bin/stestr")
    BASE_HISTORY_DIR = cfg.get("paths", "agent_runs_dir", fallback="/opt/stack/agent_runs")


# --- OLLAMA / LLM SETUP ---
# Functions below connect to Ollama (Podman), let the user pick a model,
# and wire the chosen model into CrewAI.

def ollama_api_model_name(configured_model: str) -> str:
    """
    Why: We store models as 'ollama/llama3.1:latest' but Ollama API wants 'llama3.1:latest'.
    What: Strips the 'ollama/' prefix for API calls and matching.
    """
    return configured_model.removeprefix("ollama/")


def crewai_model_name(ollama_tag: str) -> str:
    """
    Why: CrewAI expects model names in 'ollama/<name>' form.
    What: Adds the 'ollama/' prefix if it is missing.
    """
    return f"ollama/{ollama_tag}" if not ollama_tag.startswith("ollama/") else ollama_tag


def fetch_ollama_models(base_url: str) -> tuple[list[str], str | None]:
    """
    Why: We need to know which LLM models are actually installed in Ollama.
    What: Calls Ollama's /api/tags endpoint and returns a sorted list of model names,
          or an error message if Ollama is not reachable.
    """
    url = f"{base_url.rstrip('/')}/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            if resp.status != 200:
                return [], f"Ollama returned HTTP {resp.status} from {url}"
            data = json.loads(resp.read().decode())
    except urllib.error.URLError as e:
        return [], (
            f"Cannot reach Ollama at {base_url} ({e.reason}).\n"
            "Start the Podman container or edit base_url in conf.ini."
        )
    except Exception as e:
        return [], f"Cannot reach Ollama at {base_url}: {e}"

    models = sorted({m.get("name", "") for m in data.get("models", []) if m.get("name")})
    return models, None


def resolve_model_in_list(configured_model: str, available: list[str]) -> str | None:
    """
    Why: conf.ini may specify ollama/llama3.1 but Ollama lists llama3.1:latest.
    What: Finds the best matching name from the available list, or None if not found.
    """
    requested = ollama_api_model_name(configured_model)
    if requested in available:
        return requested
    matches = [
        name for name in available
        if name == requested or name.startswith(f"{requested}:") or name.split(":")[0] == requested
    ]
    return matches[0] if matches else None


def set_active_ollama_model(ollama_tag: str) -> str:
    """
    Why: The chosen model must be applied before creating the CrewAI agent.
    What: Saves the model globally, updates OPENAI_MODEL_NAME, and resets the agent
          so it is rebuilt with the correct LLM on the next AI stage.
    """
    global OLLAMA_MODEL
    OLLAMA_MODEL = crewai_model_name(ollama_tag)
    os.environ["OPENAI_MODEL_NAME"] = OLLAMA_MODEL
    reset_analyst()
    return OLLAMA_MODEL


def prompt_model_selection(models: list[str]) -> str:
    """
    Why: When several models are installed, the user must pick which one to use.
    What: Shows a numbered list and loops until a valid index is entered.
    """
    while True:
        choice = input(f"\nSelect Ollama model [0-{len(models) - 1}]: ").strip()
        if not choice.isdigit():
            console.print("[yellow]Enter a number from the list.[/yellow]")
            continue
        idx = int(choice)
        if 0 <= idx < len(models):
            return models[idx]
        console.print(f"[yellow]Invalid index — use 0 to {len(models) - 1}.[/yellow]")


def setup_ollama_model(base_url: str, preferred_model: str = "") -> tuple[bool, str | None]:
    """
    Why: Startup must confirm Ollama works and a model is selected before any AI stage.
    What: Lists models from Ollama; auto-picks if only one; prompts if several;
          honours conf.ini [ollama] model when set. Returns (ok, error_message).
    """
    models, err = fetch_ollama_models(base_url)
    if err:
        return False, err
    if not models:
        return False, (
            "No models found in Ollama.\n"
            "Pull one into the Podman container, e.g.:\n"
            "  podman exec -it ollama ollama pull llama3.2:1b"
        )

    selected = None
    if preferred_model:
        selected = resolve_model_in_list(preferred_model, models)
        if selected:
            console.print(f"[dim]Using conf.ini model: {crewai_model_name(selected)}[/dim]")
        else:
            console.print(
                f"[yellow]conf.ini model={preferred_model} not found — choose from list.[/yellow]"
            )

    if not selected:
        if len(models) == 1:
            selected = models[0]
            console.print(f"[dim]Using Ollama model: {selected}[/dim]")
        else:
            console.print()
            console.print("[bold]Available Ollama models[/bold]")
            for i, name in enumerate(models):
                console.print(Text.assemble((f"{i:>3}  ", "dim cyan"), (name, "white")))
            selected = prompt_model_selection(models)

    set_active_ollama_model(selected)
    return True, None


# --- GLOBAL PATHS AND CONSTANTS ---
# Populated from conf.ini at import time (see init_globals_from_config).

OLLAMA_BASE = ""
OLLAMA_MODEL = ""  # Filled in by setup_ollama_model() before any CrewAI call
CONFIGURED_OLLAMA_MODEL = ""  # From conf.ini [ollama] model (optional)
TEMPEST_PATH = ""
DEVSTACK_TEMPEST_CONF = ""
SYSTEM_TEMPEST_CONF = ""
TEMPEST_CONFIG = ""
STESTR_BIN = ""
BASE_HISTORY_DIR = ""

init_globals_from_config(load_settings())

# Fallback Designate systemd units if auto-discovery finds nothing
DESIGNATE_SERVICES = (
    "designate-api",
    "designate-central",
    "designate-producer",
    "designate-worker",
    "designate-mdns",
)

# Background text given to the LLM so it understands the DevStack / Designate environment.
SYSTEM_CONTEXT = """
Environment: OpenStack DevStack (All-in-one).
Service: Designate (DNS-as-a-Service) with DNS enabled.
Testing Tool: Tempest with designate-tempest-plugin.
Architecture:
- API: REST interface for clients.
- Central: Logic/DB/Pool coordination.
- Producer: Periodic tasks and zone transfers.
- Worker: Backend sync (BIND9/PowerDNS).
- mDNS: Multicast DNS integration.
"""


# --- TEST SOURCE READING (Stage 1) ---
# These functions locate test Python files on disk and pull out the test method
# plus helper methods it calls — without relying on the LLM to find files itself.

# unittest helpers we skip when following self.other_method() calls
SKIP_CALLEES = frozenset({
    "assertEqual", "assertTrue", "assertFalse", "assertIn", "assertNotIn",
    "assertRaises", "assertIsNone", "assertIsNotNone", "addCleanup", "id",
})

# Pattern to spot error-like lines in logs (ERROR, timeout, traceback, etc.)
LOG_ERROR_RE = re.compile(
    r"(?i)(error|exception|traceback|critical|fatal|failed|failure|timeout|refused|denied)"
)


def resolve_test_module(test_path: str) -> tuple[str, str, str, str]:
    """
    Why: Tempest test ids look like long dotted names; we need the actual .py file.
    What: Parses the test id into module, class, method, and filesystem path.
    """
    clean_path = test_path.split("[")[0]
    parts = clean_path.split(".")
    method_name = parts[-1]
    class_name = parts[-2]
    module_path = ".".join(parts[:-2])
    spec = importlib.util.find_spec(module_path)
    if not spec or not spec.origin:
        raise FileNotFoundError(f"Could not locate module for {module_path}")
    return module_path, class_name, method_name, spec.origin


def methods_defined_in_file(content: str) -> set[str]:
    """
    Why: We only want to load helper methods that exist in the same test file.
    What: Returns all function names defined in that Python file.
    """
    return set(re.findall(r"^\s+def (\w+)\(", content, re.MULTILINE))


def extract_method_source(content: str, method_name: str) -> str | None:
    """
    Why: We need the raw Python source of one test or helper method.
    What: Uses a regex to cut out everything from 'def method_name' to the next 'def'.
    """
    pattern = re.compile(
        rf"^\s+def {re.escape(method_name)}\(.*?(?=^\s+def |\nclass |\Z)",
        re.DOTALL | re.MULTILINE,
    )
    match = pattern.search(content)
    return match.group(0).strip() if match else None


def called_methods_from_source(method_source: str) -> list[str]:
    """
    Why: Tests often call helpers like self._test_update_records(...).
    What: Finds all self.something( calls in a method body.
    """
    return sorted(set(re.findall(r"self\.(\w+)\(", method_source)))


def load_test_source_bundle(test_path: str, max_helpers: int = 12) -> str:
    """
    Why: Stage 1 needs the full story — test method plus helpers — not just one function.
    What: Loads the test method and recursively loads helper methods it calls,
          returns one big text block for the LLM to analyse.
    """
    _, _, method_name, source_file = resolve_test_module(test_path)
    with open(source_file, encoding="utf-8") as f:
        content = f.read()

    defined = methods_defined_in_file(content)
    sections: list[str] = []
    seen: set[str] = set()
    queue = [method_name]

    while queue and len(seen) < max_helpers + 1:
        name = queue.pop(0)
        if name in seen:
            continue
        src = extract_method_source(content, name)
        if not src:
            continue
        seen.add(name)
        label = "Test method" if name == method_name else "Helper"
        sections.append(f"# {label}: {name}\n{src}")

        for callee in called_methods_from_source(src):
            if callee in defined and callee not in seen and callee not in SKIP_CALLEES:
                queue.append(callee)

    if not sections:
        return f"Error: could not extract source for {method_name} from {source_file}"
    return f"# Source file: {source_file}\n\n" + "\n\n".join(sections)


# --- LOG COLLECTION AND ANALYSIS (Stage 3) ---
# Pull journal logs from each Designate service, find error lines,
# and build a structured report before asking the LLM for a verdict.


def fetch_unit_log(unit: str, since: str) -> str:
    """
    Why: Each Designate service logs to systemd journal separately.
    What: Runs journalctl for one unit (e.g. designate-worker) since the test started.
    """
    cmd = f"sudo journalctl -u {unit} --since '{since}' --no-pager"
    try:
        return subprocess.check_output(cmd, shell=True).decode("utf-8").strip()
    except subprocess.CalledProcessError:
        return ""


def fetch_designate_logs_by_service(since: str) -> dict[str, str]:
    """
    Why: Stage 3 needs logs from every Designate service, not merged into one blob.
    What: Discovers all designate systemd units and fetches each log into a dict.
    """
    logs = {}
    for unit in designate_journal_units():
        service = unit.removeprefix("devstack@").removesuffix(".service")
        logs[service] = fetch_unit_log(unit, since)
    return logs


def extract_error_excerpts(log: str, context: int = 1, max_excerpts: int = 15) -> list[str]:
    """
    Why: Full logs are too long; we want lines that look like errors plus nearby context.
    What: Scans log lines for error keywords and returns short excerpts.
    """
    if not log:
        return []
    lines = log.splitlines()
    excerpts: list[str] = []
    seen_ranges: set[tuple[int, int]] = set()

    for i, line in enumerate(lines):
        if not LOG_ERROR_RE.search(line):
            continue
        start = max(0, i - context)
        end = min(len(lines), i + context + 1)
        if (start, end) in seen_ranges:
            continue
        seen_ranges.add((start, end))
        excerpts.append("\n".join(lines[start:end]))
        if len(excerpts) >= max_excerpts:
            break
    return excerpts


def summarize_log_section(title: str, log: str) -> str:
    """
    Why: The evidence report must say clearly if a service had errors or was clean.
    What: Formats one log section with a header, error excerpts, or 'no errors detected'.
    """
    lines = log.splitlines() if log else []
    header = f"=== {title} ==="
    if not lines:
        return f"{header}\nNo log entries in this window.\n"

    excerpts = extract_error_excerpts(log)
    if not excerpts:
        return (
            f"{header}\n"
            f"{len(lines)} log line(s) in window — no errors detected.\n"
            f"Last line: {lines[-1][:200]}\n"
        )

    body = "\n---\n".join(excerpts)
    if len(body) > 2500:
        body = body[:2500] + "\n… (truncated)"
    return (
        f"{header}\n"
        f"{len(lines)} log line(s); {len(excerpts)} error excerpt(s):\n"
        f"{body}\n"
    )


def read_tempest_dns_config() -> str:
    """
    Why: DNS propagation failures often come from wrong nameserver ports in tempest.conf.
    What: Reads [dns] and [designate] nameserver lines from tempest.conf for the report.
    """
    if not os.path.isfile(TEMPEST_CONFIG):
        return ""
    hits: list[str] = []
    section = ""
    with open(TEMPEST_CONFIG, encoding="utf-8", errors="replace") as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith("[") and stripped.endswith("]"):
                section = stripped
            if section in ("[dns]", "[designate]") and "nameserver" in stripped.lower():
                hits.append(f"{section} {stripped}")
    return "\n".join(hits)


def build_log_evidence_report(
    tempest_log_path: str,
    tempest_trace: str,
    service_logs: dict[str, str],
) -> str:
    """
    Why: The LLM needs a structured summary, not raw megabytes of logs.
    What: Combines tempest.conf DNS settings, traceback, tempest log, and each
          Designate service into one evidence report string.
    """
    sections = []

    dns_cfg = read_tempest_dns_config()
    if dns_cfg:
        sections.append(f"=== Tempest DNS config ===\n{dns_cfg}\n")

    sections.append(summarize_log_section("Tempest failure", tempest_trace or "No traceback captured."))

    if os.path.isfile(tempest_log_path):
        with open(tempest_log_path, encoding="utf-8", errors="replace") as f:
            tempest_full = f.read()
        sections.append(summarize_log_section("Tempest run log", tempest_full))

    for service in sorted(service_logs):
        sections.append(summarize_log_section(f"designate-{service}", service_logs[service]))

    return "\n".join(sections)


def save_service_logs(run_dir: str, service_logs: dict[str, str]) -> None:
    """
    Why: Users may want to inspect full raw logs after the run.
    What: Writes each service log to run_dir/designate_logs/<service>.log on disk.
    """
    logs_dir = os.path.join(run_dir, "designate_logs")
    os.makedirs(logs_dir, exist_ok=True)
    for service, log in service_logs.items():
        path = os.path.join(logs_dir, f"{service}.log")
        with open(path, "w", encoding="utf-8") as f:
            f.write(log or "")


# --- LEGACY SOURCE READER (unused) ---
# Stage 1 uses load_test_source_bundle() instead. Kept as reference for reading
# a single test method from disk without the LLM.


def read_source_file(test_path: str) -> str:
    """Read one test method from the plugin (not used by CrewAI anymore)."""
    try:
        clean_path = test_path.split('[')[0]
        parts = clean_path.split('.')
        method_name = parts[-1]
        module_path = ".".join(parts[:-2])
        spec = importlib.util.find_spec(module_path)
        if spec and spec.origin:
            with open(spec.origin, 'r') as f:
                content = f.read()
            # Extract method body
            pattern = re.compile(rf"def {method_name}.*?(?=\n\s+def|\Z)", re.DOTALL)
            match = pattern.search(content)
            return match.group(0) if match else "Error: Could not find method body in file."
        return f"Error: Could not locate file for {module_path}"
    except Exception as e:
        return f"Error reading code: {str(e)}"


# --- CREWAI AGENT ---
# The "analyst" is the AI persona that explains tests and diagnoses failures.
# We create it lazily (after model selection) because it locks in the LLM at creation time.

_analyst: Agent | None = None


def reset_analyst() -> None:
    """
    Why: Changing the Ollama model requires a fresh agent bound to the new model.
    What: Clears the cached agent so get_analyst() builds a new one.
    """
    global _analyst
    _analyst = None


def get_analyst() -> Agent:
    """
    Why: The agent must be created only after the user picks an Ollama model.
    What: Builds the Designate Expert agent once, reuses it for Stage 1 and Stage 3.
          No tools — Stage 1 source is pre-loaded; Stage 3 uses log evidence only.
    """
    global _analyst
    if _analyst is None:
        _analyst = Agent(
            role='Designate Expert Architect',
            goal='Explain and troubleshoot Designate E2E tests with clear, accurate technical summaries.',
            backstory=f'You are an expert troubleshooter in this environment: {SYSTEM_CONTEXT}.',
            tools=[],
            verbose=False,
            allow_delegation=False,
        )
    return _analyst


# --- TERMINAL UI HELPERS ---
# Pretty-print stage headers, result panels, and safe text for Rich.


def rich_text(value: str) -> Text:
    """
    Why: Test names contain [id-uuid] which Rich would treat as formatting markup.
    What: Wraps plain text so brackets display correctly in the terminal.
    """
    return Text(value)


def test_method_name(test_path: str) -> str:
    """
    Why: Full test ids are very long; panels need a short display name.
    What: Returns just the method name (e.g. test_update_records_propagated_to_backends_01_A).
    """
    return test_path.split("[")[0].rsplit(".", 1)[-1]


def print_stage_header(number: int, title: str, style: str, description: str, meta: list[str] | None = None):
    """
    Why: Each stage should be visually distinct and explain what will happen.
    What: Prints a coloured Rich panel with stage number, description, and optional details.
    """
    body = Text(description, style=style)
    if meta:
        body.append("\n")
        for line in meta:
            body.append(line + "\n", style="dim")
    console.print()
    console.print(Panel(body, title=f"[bold {style}]Stage {number} · {title}[/bold {style}]", border_style=style))


def print_result_panel(title: str, body: str, style: str):
    """
    Why: LLM output and log summaries are easier to read in bordered panels.
    What: Prints a Rich panel with a title and coloured border.
    """
    console.print(Panel(rich_text(body), title=title, border_style=style))


def run_crew_task(task: Task) -> str:
    """
    Why: Stage 1 and Stage 3 both send a Task to the same analyst agent.
    What: Runs one CrewAI task and returns the LLM response as a string.
    """
    return str(Crew(agents=[get_analyst()], tasks=[task], verbose=False).kickoff())


# --- TEMPEST / STESTR HELPERS ---
# Configure tempest.conf, discover tests, run stestr, and interpret PASS/FAIL/SKIP.


def tempest_env() -> dict[str, str]:
    """
    Why: stestr subprocess needs TEMPEST_CONFIG and the activated venv PATH.
    What: Copies the shell environment and sets TEMPEST_CONFIG from conf.ini.
    """
    env = os.environ.copy()
    env["TEMPEST_CONFIG"] = TEMPEST_CONFIG
    return env


def ensure_tempest_conf_symlink() -> tuple[bool, str, bool]:
    """
    Why: stestr defaults to /etc/tempest/tempest.conf but DevStack puts config elsewhere.
    What: Creates /etc/tempest/tempest.conf → DevStack path if missing (needs sudo).
          Returns (ok, path_or_error, created_new_symlink).
    """
    if os.path.isfile(SYSTEM_TEMPEST_CONF):
        return True, SYSTEM_TEMPEST_CONF, False
    if not os.path.isfile(DEVSTACK_TEMPEST_CONF):
        return False, f"DevStack tempest.conf not found at {DEVSTACK_TEMPEST_CONF}", False
    try:
        subprocess.run(
            ["sudo", "mkdir", "-p", "/etc/tempest"],
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            ["sudo", "ln", "-sf", DEVSTACK_TEMPEST_CONF, SYSTEM_TEMPEST_CONF],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        err = (e.stderr or e.stdout or str(e)).strip()
        return False, f"Could not create {SYSTEM_TEMPEST_CONF} symlink: {err}", False
    if os.path.isfile(SYSTEM_TEMPEST_CONF):
        return True, SYSTEM_TEMPEST_CONF, True
    return False, f"Symlink created but {SYSTEM_TEMPEST_CONF} is still missing", False


def verify_tempest_config() -> tuple[bool, str]:
    """
    Why: Startup must fail fast if Tempest credentials file is missing.
    What: Checks DevStack tempest.conf exists and ensures the /etc symlink if needed.
    """
    if not os.path.isfile(TEMPEST_CONFIG) and not os.path.isfile(DEVSTACK_TEMPEST_CONF):
        return False, (
            f"Tempest config not found at {TEMPEST_CONFIG}.\n"
            f"Expected DevStack config at {DEVSTACK_TEMPEST_CONF}."
        )

    ok, path_or_err, created = ensure_tempest_conf_symlink()
    if not ok and not os.path.isfile(TEMPEST_CONFIG):
        return False, path_or_err

    config = TEMPEST_CONFIG if os.path.isfile(TEMPEST_CONFIG) else path_or_err
    if created:
        return True, f"{config} (created symlink {SYSTEM_TEMPEST_CONF})"
    return True, config


def stestr_run_filter(test_id: str) -> str:
    """
    Why: stestr treats test filters as regex; [id-uuid] brackets break the match.
    What: Escapes special characters so the exact test id is run safely.
    """
    return re.escape(test_id)


def parse_tempest_result(output: str, returncode: int) -> tuple[str, str]:
    """
    Why: Stage 2 must branch correctly on PASS, FAIL, SKIP, or NOT_RUN.
    What: Parses stestr stdout for counts and traceback; returns (status, detail text).
    """
    skip_match = re.search(r"SKIPPED:\s*(.+)", output)
    invalid_regex = re.search(r"Invalid regex:\s*(.+?)\s*provided in filters", output)
    ran_match = re.search(r"Ran:\s*(\d+)\s*tests", output)
    ran = int(ran_match.group(1)) if ran_match else None

    totals = {
        name: int(m.group(1))
        for name, m in (
            ("failed", re.search(r"Failed:\s*(\d+)", output)),
            ("skipped", re.search(r"Skipped:\s*(\d+)", output)),
            ("passed", re.search(r"Passed:\s*(\d+)", output)),
        )
        if m
    }

    failed = totals.get("failed", 0)
    skipped = totals.get("skipped", 0)
    passed = totals.get("passed", 0)
    skip_reason = skip_match.group(1).strip() if skip_match else "unknown skip reason"

    if invalid_regex:
        return "NOT_RUN", (
            f"stestr could not match the test (invalid regex filter).\n"
            f"{invalid_regex.group(1).strip()}\n"
            "Square brackets in [id-...] must be escaped for stestr — this is fixed in main.py."
        )

    if ran == 0 and failed == 0 and passed == 0 and skipped == 0:
        detail = output.strip()[-2000:] or f"stestr exited with code {returncode}"
        if "No config file found" in output:
            detail = (
                "stestr was not run from /opt/stack/tempest (missing .stestr.conf).\n"
                + detail
            )
        return "NOT_RUN", detail

    if skipped > 0 and failed == 0 and passed == 0:
        return "SKIP", skip_reason

    if failed > 0:
        if "Captured traceback:" in output:
            detail = output.split("Captured traceback:")[-1].split("Captured pythonlogging:")[0].strip()
        else:
            detail = output[-4000:].strip() or f"stestr exited with code {returncode}"
        return "FAIL", detail

    if passed > 0:
        return "PASS", ""

    if skip_match:
        return "SKIP", skip_reason

    if returncode != 0:
        return "FAIL", output[-4000:].strip() or f"stestr exited with code {returncode}"

    return "PASS", ""


def designate_journal_units() -> list[str]:
    """
    Why: We need the real systemd unit names for each Designate service on this VM.
    What: Lists devstack@designate*.service units, or falls back to DESIGNATE_SERVICES.
    """
    try:
        out = subprocess.check_output(
            [
                "systemctl", "list-units", "--all",
                "--no-legend", "--no-pager",
                "devstack@designate*.service",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        units = [line.split()[0] for line in out.splitlines() if line.strip()]
        if units:
            return sorted(set(units))
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.SubprocessError):
        pass
    return [f"devstack@{name}.service" for name in DESIGNATE_SERVICES]


def designate_service_names(units: list[str]) -> str:
    """
    Why: Stage headers should show human-readable service names, not full unit paths.
    What: Turns 'devstack@designate-worker.service' into 'designate-worker' list text.
    """
    return ", ".join(u.removeprefix("devstack@").removesuffix(".service") for u in units)


# --- STAGE RUNNERS ---
# Stage 1 = explain test, Stage 2 = run stestr, Stage 3 = diagnose failure.


def run_stage_logic_discovery(test_path):
    """
    Why: Before running a test, we want to know what it is supposed to verify.
    What: Loads test source code, asks Ollama to explain the flow, prints result panel.
    """
    try:
        source_bundle = load_test_source_bundle(test_path)
    except FileNotFoundError as e:
        print_result_panel("Stage 1 error", str(e), "red")
        return str(e)

    print_stage_header(
        1, "Analyze test logic", "cyan",
        "Loads the test method and helper methods it calls from source, "
        "then uses Ollama to explain the full end-to-end flow.",
        [llm_stage_line(), f"Test: {test_path}"],
    )

    task = Task(
        description=(
            f"The complete Python source for this Tempest test is already provided below "
            f"(test method + helpers). Do NOT call tools. Do NOT output JSON.\n\n"
            f"{source_bundle}\n\n"
            "Write a step-by-step explanation of what this test does end-to-end:\n"
            "- Setup (zones, recordsets, API calls)\n"
            "- DNS / propagation checks (dig, nameservers, ports)\n"
            "- Waits, assertions, and expected Designate behavior\n"
            "Use only the source above. Plain English prose — no JSON, no tool syntax."
        ),
        expected_output="Complete step-by-step breakdown of the test flow in plain prose.",
        agent=get_analyst(),
    )
    result = run_crew_task(task)
    print_result_panel(f"Test intent · {test_method_name(test_path)}", result, "cyan")
    return result


def run_stage_execution(test_path):
    """
    Why: We must actually run the Tempest test to see if it passes or fails.
    What: Runs stestr --serial, saves output to agent_runs/, prints PASS/FAIL/SKIP panel.
          Returns status, failure detail, start time, and artifact directory.
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(BASE_HISTORY_DIR, f"run_{ts}")
    os.makedirs(run_dir, exist_ok=True)

    output_log = os.path.join(run_dir, "tempest_run.log")
    start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    run_filter = stestr_run_filter(test_path)

    print_stage_header(
        2, "Run Tempest test", "yellow",
        "Executes the selected test with stestr against DevStack (no LLM). "
        "Uses tempest.conf credentials and saves full output to agent_runs/.",
        [
            f"TEMPEST_CONFIG={TEMPEST_CONFIG}",
            f"cwd={TEMPEST_PATH}",
            f"stestr run --serial {run_filter}",
        ],
    )

    console.print("[yellow]Running…[/yellow] ", end="")
    process = subprocess.Popen(
        [STESTR_BIN, "run", "--serial", run_filter],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=TEMPEST_PATH,
        env=tempest_env(),
        text=True,
        bufsize=1,
    )

    full_output = []
    with open(output_log, "w") as f:
        for line in process.stdout:
            f.write(line)
            full_output.append(line)
            if "..." in line:
                sys.stdout.write(".")
                sys.stdout.flush()

    process.wait()
    console.print()  # newline after progress dots
    output_str = "".join(full_output)
    status, detail = parse_tempest_result(output_str, process.returncode)

    if status == "PASS":
        print_result_panel(
            "Execution result · PASS",
            f"Tempest reported success.\n\nLog: {output_log}\nArtifacts: {run_dir}",
            "green",
        )
    elif status == "SKIP":
        print_result_panel(
            "Execution result · SKIPPED",
            f"{detail}\n\nLog: {output_log}\nArtifacts: {run_dir}",
            "yellow",
        )
    elif status == "NOT_RUN":
        print_result_panel(
            "Execution result · NOT RUN",
            f"{detail}\n\nLog: {output_log}\nArtifacts: {run_dir}",
            "red",
        )
    else:
        summary = detail[:1500] + ("…" if len(detail) > 1500 else "")
        print_result_panel(
            "Execution result · FAIL",
            f"{summary}\n\nFull traceback in log: {output_log}\nArtifacts: {run_dir}",
            "red",
        )

    return status, detail, start_time, run_dir


def run_stage_root_cause(logic, trace, start_time, run_dir):
    """
    Why: On failure, engineers need logs correlated with test intent and traceback.
    What: Collects per-service logs, builds evidence report, asks Ollama for root cause.
          Only runs when Stage 2 status is FAIL.
    """
    tempest_log_path = os.path.join(run_dir, "tempest_run.log")
    units = designate_journal_units()
    service_list = designate_service_names(units)

    print_stage_header(
        3, "Root-cause analysis", "magenta",
        "Collects Tempest output and per-service Designate journal logs, "
        "builds a structured evidence report, then uses Ollama for a verdict.",
        [llm_stage_line(), f"Logs since: {start_time}", f"Services: {service_list}"],
    )

    service_logs = fetch_designate_logs_by_service(start_time)
    save_service_logs(run_dir, service_logs)

    evidence = build_log_evidence_report(tempest_log_path, trace, service_logs)
    evidence_path = os.path.join(run_dir, "log_evidence.txt")
    with open(evidence_path, "w", encoding="utf-8") as f:
        f.write(evidence)

    # Show deterministic report first (truncated for terminal if very long)
    display = evidence if len(evidence) <= 12000 else evidence[:12000] + "\n… (see log_evidence.txt for full report)"
    print_result_panel("Log evidence report", display, "magenta")

    task = Task(
        description=(
            f"FINAL INVESTIGATION — base your answer ONLY on the evidence below.\n"
            "Do not invent errors, services, or log lines that are not present.\n\n"
            f"TEST INTENT:\n{logic}\n\n"
            f"LOG EVIDENCE (Tempest + each Designate service):\n{evidence[:14000]}\n\n"
            "Summarize:\n"
            "1. What failed in Tempest (client-side vs backend)\n"
            "2. Which Designate service(s) show errors, if any\n"
            "3. Services with 'no errors detected' — state that explicitly\n"
            "4. Most likely root cause linking test intent, traceback, and log evidence\n"
            "Plain prose only — no JSON, no tool-call syntax."
        ),
        expected_output="Root cause verdict grounded in the log evidence.",
        agent=get_analyst(),
    )
    result = run_crew_task(task)
    print_result_panel("Root cause verdict", result, "magenta")
    console.print(rich_text(f"Full evidence: {evidence_path}"), style="dim")
    return result


# --- CLI ENTRY POINT ---
# Functions for test discovery, user prompts, and the main script flow.


def llm_stage_line() -> str:
    """
    Why: Stage headers show which Ollama model and URL are in use.
    What: Returns one line like 'Ollama (ollama/llama3.1:latest) @ http://127.0.0.1:11434'.
    """
    return f"Ollama ({OLLAMA_MODEL}) @ {OLLAMA_BASE}"


def get_full_test_list(grep_str=None):
    """
    Why: User picks a test from a filtered list, not by typing the full id.
    What: Runs 'stestr list', keeps designate tests, optionally filters by grep string.
          Returns (tests, error_message).
    """
    list_result = subprocess.run(
        [STESTR_BIN, "list"],
        cwd=TEMPEST_PATH,
        env=tempest_env(),
        capture_output=True,
        text=True,
    )
    if list_result.returncode != 0:
        err = (list_result.stderr or list_result.stdout or "stestr list failed").strip()
        return [], err

    tests = [
        line.strip() for line in list_result.stdout.splitlines()
        if line.strip() and "designate" in line and "test_" in line
    ]
    if grep_str:
        tests = [t for t in tests if grep_str.lower() in t.lower()]
    return tests, None


def prompt_test_selection(tests):
    """
    Why: User must explicitly choose a test — there is no default selection.
    What: Loops until a valid index into the displayed test list is entered.
    """
    while True:
        choice = input(f"\nSelect test number [0-{len(tests) - 1}]: ").strip()
        if not choice.isdigit():
            console.print("[yellow]Enter a number from the list.[/yellow]")
            continue
        idx = int(choice)
        if 0 <= idx < len(tests):
            return tests[idx]
        console.print(f"[yellow]Invalid index — use 0 to {len(tests) - 1}.[/yellow]")


def print_tool_flow():
    """
    Why: New users should see the pipeline before anything runs.
    What: Prints the ASCII Setup → Stage 1 → Stage 2 → Stage 3 diagram at startup.
    """
    flow = Text.assemble(
        ("  Setup          Stage 1              Stage 2              Stage 3\n", "bold"),
        ("  ─────          ───────              ───────              ───────\n", "dim"),
        ("  Ollama ✓", "dim"),
        ("   →   ", "dim"),
        ("Analyze logic", "cyan"),
        ("   →   ", "dim"),
        ("Run test", "yellow"),
        ("   →   ", "dim"),
        ("Root cause\n", "magenta"),
        ("  tempest ✓", "dim"),
        ("      ", ""),
        ("(read_source)", "dim"),
        ("      ", ""),
        ("(stestr)", "dim"),
        ("            ", ""),
        ("(journalctl)\n", "dim"),
        ("                   ", ""),
        ("cyan · LLM", "cyan"),
        ("          ", ""),
        ("yellow", "yellow"),
        ("              ", ""),
        ("magenta · LLM\n", "magenta"),
        ("                                        │\n", "dim"),
        ("                                        ├─ ", "dim"),
        ("PASS", "green"),
        ("  → done\n", "dim"),
        ("                                        ├─ ", "dim"),
        ("SKIP", "yellow"),
        ("  → done\n", "dim"),
        ("                                        └─ ", "dim"),
        ("FAIL", "red"),
        ("  → Stage 3\n", "dim"),
    )
    console.print()
    console.print(flow)


if __name__ == "__main__":
    # Script entry point: runs only when you execute 'python3 main.py' directly.

    # 1. Banner and pipeline diagram
    console.print(Panel(
        Text.from_markup(
            "[bold green]Designate E2E Test Investigator[/bold green]\n"
            "Select a Tempest test → analyze intent → run → diagnose failures"
        ),
        border_style="green",
    ))
    print_tool_flow()

    # 2. Ollama: connect, list models, user picks one (or auto if only one)
    llm_ok, llm_error = setup_ollama_model(OLLAMA_BASE, CONFIGURED_OLLAMA_MODEL)
    if not llm_ok:
        print_result_panel("Startup error", llm_error, "red")
        sys.exit(1)
    console.print(rich_text(f"LLM: {OLLAMA_MODEL} @ {OLLAMA_BASE}"), justify="left")

    # 3. Tempest: verify config and create /etc/tempest symlink if needed
    cfg_ok, cfg_msg = verify_tempest_config()
    if not cfg_ok:
        print_result_panel("Startup error", cfg_msg, "red")
        sys.exit(1)
    console.print(rich_text(f"Tempest config: {cfg_msg}"), style="dim")

    # 4. Discover tests (optional grep) and let user pick one
    grep_query = input("Grep tests (e.g. 'multipool') or ENTER for all: ").strip()
    tests, stestr_error = get_full_test_list(grep_query)

    if stestr_error:
        print_result_panel("Test discovery failed", stestr_error, "red")
        sys.exit(1)

    if not tests:
        label = f"'{grep_query}'" if grep_query else "designate tests"
        console.print(rich_text(f"No tests found matching {label}."), style="red")
        sys.exit(1)

    console.print()
    console.print("[bold]Available tests[/bold] [dim](grep filter applied)[/dim]" if grep_query else "[bold]Available tests[/bold]")
    for i, t in enumerate(tests):
        console.print(Text.assemble((f"{i:>3}  ", "dim cyan"), (t, "white")))

    target_test = prompt_test_selection(tests)
    print_result_panel("Selected test", target_test, "blue")

    # 5. Run the three stages (Stage 3 only if Stage 2 fails)
    logic_summary = run_stage_logic_discovery(target_test)
    status, detail, start_time, run_dir = run_stage_execution(target_test)

    if status == "FAIL":
        run_stage_root_cause(logic_summary, detail, start_time, run_dir)
    elif status == "PASS":
        console.print("\n[bold green]Done — test passed, no further analysis needed.[/bold green]")
    else:
        console.print(
            "\n[dim]Done — test did not run to completion; fix the issue above and re-run.[/dim]"
        )