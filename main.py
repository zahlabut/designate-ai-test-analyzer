import subprocess
import os
import sys
import re
import json
import importlib.util
import urllib.error
import urllib.request
from datetime import datetime
from crewai import Agent, Task, Crew
from crewai.tools import tool
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

console = Console()

# --- CONFIGURATION ---

def configure_ollama_env() -> str:
    """Set Ollama API endpoint for CrewAI. Model is chosen later in setup_ollama_model()."""
    base = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
    os.environ.setdefault("OPENAI_API_BASE", f"{base}/v1")
    os.environ.setdefault("OPENAI_API_KEY", "ollama")
    return base


def ollama_api_model_name(configured_model: str) -> str:
    """Model id Ollama expects (CrewAI strips the ollama/ prefix)."""
    return configured_model.removeprefix("ollama/")


def crewai_model_name(ollama_tag: str) -> str:
    return f"ollama/{ollama_tag}" if not ollama_tag.startswith("ollama/") else ollama_tag


def fetch_ollama_models(base_url: str) -> tuple[list[str], str | None]:
    """Return (model_names, error_message)."""
    url = f"{base_url.rstrip('/')}/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            if resp.status != 200:
                return [], f"Ollama returned HTTP {resp.status} from {url}"
            data = json.loads(resp.read().decode())
    except urllib.error.URLError as e:
        return [], (
            f"Cannot reach Ollama at {base_url} ({e.reason}).\n"
            "Start the Podman container or set OLLAMA_BASE_URL, e.g.:\n"
            "  export OLLAMA_BASE_URL=http://127.0.0.1:11434"
        )
    except Exception as e:
        return [], f"Cannot reach Ollama at {base_url}: {e}"

    models = sorted({m.get("name", "") for m in data.get("models", []) if m.get("name")})
    return models, None


def resolve_model_in_list(configured_model: str, available: list[str]) -> str | None:
    requested = ollama_api_model_name(configured_model)
    if requested in available:
        return requested
    matches = [
        name for name in available
        if name == requested or name.startswith(f"{requested}:") or name.split(":")[0] == requested
    ]
    return matches[0] if matches else None


def set_active_ollama_model(ollama_tag: str) -> str:
    global OLLAMA_MODEL
    OLLAMA_MODEL = crewai_model_name(ollama_tag)
    os.environ["OPENAI_MODEL_NAME"] = OLLAMA_MODEL
    reset_analyst()
    return OLLAMA_MODEL


def prompt_model_selection(models: list[str]) -> str:
    while True:
        choice = input(f"\nSelect Ollama model [0-{len(models) - 1}]: ").strip()
        if not choice.isdigit():
            console.print("[yellow]Enter a number from the list.[/yellow]")
            continue
        idx = int(choice)
        if 0 <= idx < len(models):
            return models[idx]
        console.print(f"[yellow]Invalid index — use 0 to {len(models) - 1}.[/yellow]")


def setup_ollama_model(base_url: str) -> tuple[bool, str | None]:
    """Connect to Ollama, list models, and pick one. Returns (ok, error_message)."""
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
    if "OLLAMA_MODEL" in os.environ:
        selected = resolve_model_in_list(os.environ["OLLAMA_MODEL"], models)
        if selected:
            console.print(f"[dim]Using OLLAMA_MODEL: {crewai_model_name(selected)}[/dim]")
        else:
            console.print(
                f"[yellow]OLLAMA_MODEL={os.environ['OLLAMA_MODEL']} not found — choose from list.[/yellow]"
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


OLLAMA_BASE = configure_ollama_env()
OLLAMA_MODEL = ""  # populated by setup_ollama_model() before any CrewAI call
os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")

TEMPEST_PATH = "/opt/stack/tempest"
DEVSTACK_TEMPEST_CONF = "/opt/stack/tempest/etc/tempest.conf"
SYSTEM_TEMPEST_CONF = "/etc/tempest/tempest.conf"
TEMPEST_CONFIG = os.environ.get("TEMPEST_CONFIG", DEVSTACK_TEMPEST_CONF)
STESTR_BIN = os.environ.get("STESTR", "/opt/stack/data/venv/bin/stestr")
BASE_HISTORY_DIR = "/opt/stack/agent_runs"

# DevStack designate units (fallback when systemctl discovery finds none)
DESIGNATE_SERVICES = (
    "designate-api",
    "designate-central",
    "designate-producer",
    "designate-worker",
    "designate-mdns",
)

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


# --- SOURCE & LOG ANALYSIS ---

SKIP_CALLEES = frozenset({
    "assertEqual", "assertTrue", "assertFalse", "assertIn", "assertNotIn",
    "assertRaises", "assertIsNone", "assertIsNotNone", "addCleanup", "id",
})

LOG_ERROR_RE = re.compile(
    r"(?i)(error|exception|traceback|critical|fatal|failed|failure|timeout|refused|denied)"
)


def resolve_test_module(test_path: str) -> tuple[str, str, str, str]:
    """Return (module_path, class_name, method_name, source_file)."""
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
    return set(re.findall(r"^\s+def (\w+)\(", content, re.MULTILINE))


def extract_method_source(content: str, method_name: str) -> str | None:
    pattern = re.compile(
        rf"^\s+def {re.escape(method_name)}\(.*?(?=^\s+def |\nclass |\Z)",
        re.DOTALL | re.MULTILINE,
    )
    match = pattern.search(content)
    return match.group(0).strip() if match else None


def called_methods_from_source(method_source: str) -> list[str]:
    return sorted(set(re.findall(r"self\.(\w+)\(", method_source)))


def load_test_source_bundle(test_path: str, max_helpers: int = 12) -> str:
    """Load the test method plus helper methods it calls (recursively)."""
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


def fetch_unit_log(unit: str, since: str) -> str:
    cmd = f"sudo journalctl -u {unit} --since '{since}' --no-pager"
    try:
        return subprocess.check_output(cmd, shell=True).decode("utf-8").strip()
    except subprocess.CalledProcessError:
        return ""


def fetch_designate_logs_by_service(since: str) -> dict[str, str]:
    logs = {}
    for unit in designate_journal_units():
        service = unit.removeprefix("devstack@").removesuffix(".service")
        logs[service] = fetch_unit_log(unit, since)
    return logs


def extract_error_excerpts(log: str, context: int = 1, max_excerpts: int = 15) -> list[str]:
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
    """Extract nameserver settings from tempest.conf for failure context."""
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
    logs_dir = os.path.join(run_dir, "designate_logs")
    os.makedirs(logs_dir, exist_ok=True)
    for service, log in service_logs.items():
        path = os.path.join(logs_dir, f"{service}.log")
        with open(path, "w", encoding="utf-8") as f:
            f.write(log or "")

@tool("read_source")
def read_source(test_path: str):
    """
    Reads the Python source code for a test method.
    Input should be the full python path (e.g. module.ClassName.method_name).
    """
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


# --- AGENT ---

_analyst: Agent | None = None


def reset_analyst() -> None:
    global _analyst
    _analyst = None


def get_analyst() -> Agent:
    """Build CrewAI agent after Ollama model is selected (Agent caches LLM at creation)."""
    global _analyst
    if _analyst is None:
        _analyst = Agent(
            role='Designate Expert Architect',
            goal='Explain and troubleshoot Designate E2E tests with clear, accurate technical summaries.',
            backstory=f'You are an expert troubleshooter in this environment: {SYSTEM_CONTEXT}.',
            tools=[read_source],
            verbose=False,
            allow_delegation=False,
        )
    return _analyst


# --- STAGE HANDLERS ---

def rich_text(value: str) -> Text:
    """Plain text safe for Rich (avoids [id-...] being parsed as markup)."""
    return Text(value)


def test_method_name(test_path: str) -> str:
    return test_path.split("[")[0].rsplit(".", 1)[-1]


def print_stage_header(number: int, title: str, style: str, description: str, meta: list[str] | None = None):
    body = Text(description, style=style)
    if meta:
        body.append("\n")
        for line in meta:
            body.append(line + "\n", style="dim")
    console.print()
    console.print(Panel(body, title=f"[bold {style}]Stage {number} · {title}[/bold {style}]", border_style=style))


def print_result_panel(title: str, body: str, style: str):
    console.print(Panel(rich_text(body), title=title, border_style=style))


def run_crew_task(task: Task) -> str:
    return str(Crew(agents=[get_analyst()], tasks=[task], verbose=False).kickoff())


def tempest_env() -> dict[str, str]:
    env = os.environ.copy()
    env["TEMPEST_CONFIG"] = TEMPEST_CONFIG
    return env


def ensure_tempest_conf_symlink() -> tuple[bool, str, bool]:
    """Create /etc/tempest/tempest.conf → DevStack path when missing. Returns (ok, path_or_error, created)."""
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
    """stestr run filters are regex; escape [id-...] and other special chars."""
    return re.escape(test_id)


def parse_tempest_result(output: str, returncode: int) -> tuple[str, str]:
    """Return (status, detail). status is PASS, FAIL, SKIP, or NOT_RUN."""
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
    """Return devstack@designate*.service units on this host."""
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
    return ", ".join(u.removeprefix("devstack@").removesuffix(".service") for u in units)


def run_stage_logic_discovery(test_path):
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
            f"Analyze this Tempest test source (test method + helpers it calls):\n\n"
            f"{source_bundle}\n\n"
            "Explain step-by-step what the test does end-to-end. Include:\n"
            "- Setup (zones, recordsets, API calls)\n"
            "- DNS / propagation checks (dig, nameservers, ports)\n"
            "- Waits, assertions, and expected Designate behavior\n"
            "Base the narrative ONLY on the source above — do not guess or invent steps. "
            "Plain prose only — no JSON, no tool-call syntax."
        ),
        expected_output="Complete step-by-step breakdown of the test flow.",
        agent=get_analyst(),
    )
    result = run_crew_task(task)
    print_result_panel(f"Test intent · {test_method_name(test_path)}", result, "cyan")
    return result


def run_stage_execution(test_path):
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


# --- MAIN ---

def llm_stage_line() -> str:
    return f"Ollama ({OLLAMA_MODEL}) @ {OLLAMA_BASE}"


def get_full_test_list(grep_str=None):
    """Return (tests, error). error is set when stestr discovery fails."""
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
    """Require the user to pick a test index from the displayed list."""
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
    """Show the pipeline diagram below the title banner."""
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
    console.print(Panel(
        Text.from_markup(
            "[bold green]Designate E2E Test Investigator[/bold green]\n"
            "Select a Tempest test → analyze intent → run → diagnose failures"
        ),
        border_style="green",
    ))
    print_tool_flow()

    llm_ok, llm_error = setup_ollama_model(OLLAMA_BASE)
    if not llm_ok:
        print_result_panel("Startup error", llm_error, "red")
        sys.exit(1)
    console.print(rich_text(f"LLM: {OLLAMA_MODEL} @ {OLLAMA_BASE}"), justify="left")

    cfg_ok, cfg_msg = verify_tempest_config()
    if not cfg_ok:
        print_result_panel("Startup error", cfg_msg, "red")
        sys.exit(1)
    console.print(rich_text(f"Tempest config: {cfg_msg}"), style="dim")

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