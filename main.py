import subprocess
import os
import sys
import re
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

def configure_llm():
    """Configure CrewAI to use a local/remote Ollama OpenAI-compatible endpoint."""
    base = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
    model = os.environ.get("OLLAMA_MODEL", "ollama/llama3.1")
    os.environ.setdefault("OPENAI_API_BASE", f"{base}/v1")
    os.environ.setdefault("OPENAI_MODEL_NAME", model)
    os.environ.setdefault("OPENAI_API_KEY", "ollama")
    return base, os.environ["OPENAI_MODEL_NAME"]


def verify_llm_connection(base_url):
    """Return (ok, error_message)."""
    url = f"{base_url.rstrip('/')}/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            if resp.status != 200:
                return False, f"Ollama returned HTTP {resp.status} from {url}"
    except urllib.error.URLError as e:
        return False, (
            f"Cannot reach Ollama at {base_url} ({e.reason}).\n"
            "Start Ollama on this host or set OLLAMA_BASE_URL to the machine running it, e.g.:\n"
            "  export OLLAMA_BASE_URL=http://10.9.95.131:11434"
        )
    except Exception as e:
        return False, f"Cannot reach Ollama at {base_url}: {e}"
    return True, None


OLLAMA_BASE, OLLAMA_MODEL = configure_llm()
os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")

TEMPEST_PATH = "/opt/stack/tempest"
TEMPEST_CONFIG = os.environ.get("TEMPEST_CONFIG", "/opt/stack/tempest/etc/tempest.conf")
STESTR_BIN = os.environ.get("STESTR", "/opt/stack/data/venv/bin/stestr")
BASE_HISTORY_DIR = "/opt/stack/agent_runs"

SYSTEM_CONTEXT = """
Environment: OpenStack DevStack (All-in-one).
Service: Designate (DNS-as-a-Service) with DNS enabled.
Testing Tool: Tempest with designate-tempest-plugin.
Architecture:
- Central: Logic/DB/Pool coordination.
- Worker: Backend sync (BIND9/PowerDNS).
"""


# --- TOOLS ---

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

analyst = Agent(
    role='Designate Expert Architect',
    goal='Explain and troubleshoot Designate E2E tests with clear, accurate technical summaries.',
    backstory=f'You are an expert troubleshooter in this environment: {SYSTEM_CONTEXT}.',
    tools=[read_source],
    verbose=False,
    allow_delegation=False
)


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
    return str(Crew(agents=[analyst], tasks=[task], verbose=False).kickoff())


def tempest_env() -> dict[str, str]:
    env = os.environ.copy()
    env["TEMPEST_CONFIG"] = TEMPEST_CONFIG
    return env


def verify_tempest_config() -> tuple[bool, str]:
    if os.path.isfile(TEMPEST_CONFIG):
        return True, TEMPEST_CONFIG
    fallback = "/etc/tempest/tempest.conf"
    if os.path.isfile(fallback):
        return True, fallback
    return False, (
        f"Tempest config not found at {TEMPEST_CONFIG}.\n"
        "DevStack creates it at /opt/stack/tempest/etc/tempest.conf — set:\n"
        "  export TEMPEST_CONFIG=/opt/stack/tempest/etc/tempest.conf\n"
        "Or symlink: sudo ln -sf /opt/stack/tempest/etc/tempest.conf /etc/tempest/tempest.conf"
    )


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


def run_stage_logic_discovery(test_path):
    print_stage_header(
        1, "Analyze test logic", "cyan",
        "Reads the Tempest test source with read_source, then uses Ollama to explain "
        "what the test does and what Designate behavior it expects.",
        [llm_stage_line(), f"Test: {test_path}"],
    )
    task = Task(
        description=(
            f"Use read_source on '{test_path}', then explain step-by-step what the test does. "
            "Write a technical narrative only — no JSON, no tool-call syntax."
        ),
        expected_output="Step-by-step breakdown of the test's Python logic.",
        agent=analyst
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
    log_file = os.path.join(run_dir, "designate_services.log")

    print_stage_header(
        3, "Root-cause analysis", "magenta",
        "Pulls designate-central and designate-worker journal logs from the test window, "
        "then uses Ollama to correlate logs with the test intent and failure traceback.",
        [llm_stage_line(), f"Logs since: {start_time}"],
    )

    cmd = (
        "sudo journalctl -u devstack@designate-central.service "
        "-u devstack@designate-worker.service "
        f"--since '{start_time}' --no-pager"
    )
    try:
        logs = subprocess.check_output(cmd, shell=True).decode('utf-8')
        with open(log_file, "w") as f:
            f.write(logs)
    except:
        logs = "No backend logs found."

    task = Task(
        description=(
            f"FINAL INVESTIGATION:\n\n"
            f"TEST INTENT:\n{logic}\n\n"
            f"FAILURE / TRACEBACK:\n{trace}\n\n"
            f"DESIGNATE LOGS:\n{logs[-12000:]}\n\n"
            "Explain why the backend failed. Name the service (Central/Worker) and the error. "
            "Plain prose only — no JSON, no tool-call syntax."
        ),
        expected_output="Root cause verdict in plain language.",
        agent=analyst
    )
    result = run_crew_task(task)
    print_result_panel("Root cause verdict", result, "magenta")
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


if __name__ == "__main__":
    console.print(Panel(
        Text.from_markup(
            "[bold green]Designate E2E Test Investigator[/bold green]\n"
            "Select a Tempest test → analyze intent → run → diagnose failures"
        ),
        border_style="green",
    ))
    console.print(rich_text(f"LLM: {OLLAMA_MODEL} @ {OLLAMA_BASE}"), justify="left")

    llm_ok, llm_error = verify_llm_connection(OLLAMA_BASE)
    if not llm_ok:
        print_result_panel("Startup error", llm_error, "red")
        sys.exit(1)

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