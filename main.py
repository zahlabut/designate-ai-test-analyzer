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
    goal='Explain and troubleshoot Designate E2E tests stage by stage with maximum detail.',
    backstory=f'You are an expert troubleshooter in this environment: {SYSTEM_CONTEXT}. Your priority is providing clear, human-readable technical summaries of each stage.',
    tools=[read_source],
    verbose=True,  # Keeps the internal tool usage visible in console if needed
    allow_delegation=False
)


# --- STAGE HANDLERS ---

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


def llm_stage_line() -> str:
    return f"Ollama ({OLLAMA_MODEL}) via CrewAI @ {OLLAMA_BASE}"


def run_stage_logic_discovery(test_path):
    console.print(f"\n[bold blue]🔎 STAGE 1: ANALYZING TEST LOGIC...[/bold blue]")
    console.print(f"[dim]{llm_stage_line()}[/dim]")
    task = Task(
        description=(
            f"1. Use 'read_source' for '{test_path}'.\n"
            f"2. Analyze the code.\n"
            f"3. Output a clear, detailed explanation of what the test is doing step-by-step. "
            "Do not output JSON. Output a technical narrative."
        ),
        expected_output="A step-by-step technical breakdown of the test's Python logic.",
        agent=analyst
    )
    result = Crew(agents=[analyst], tasks=[task]).kickoff()

    # We display the result of the AI's analysis, not the tool call itself
    console.print(Panel(
        Text(str(result), style="cyan"),
        title=f"PROBED LOGIC: {test_path.split('.')[-1]}",
        border_style="cyan"
    ))
    return str(result)


def run_stage_execution(test_path):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(BASE_HISTORY_DIR, f"run_{ts}")
    os.makedirs(run_dir, exist_ok=True)

    output_log = os.path.join(run_dir, "tempest_run.log")
    start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    console.print(f"\n[bold yellow]🚀 STAGE 2: EXECUTING FRESH TEST RUN...[/bold yellow]")
    console.print(f"[dim]TEMPEST_CONFIG={TEMPEST_CONFIG}[/dim]")
    console.print(f"[dim]cwd={TEMPEST_PATH}[/dim]")
    console.print(f"[dim]Test id: {test_path}[/dim]")

    run_filter = stestr_run_filter(test_path)
    console.print(f"[dim]stestr run --serial {run_filter}[/dim]")

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
    output_str = "".join(full_output)
    status, detail = parse_tempest_result(output_str, process.returncode)

    if status == "PASS":
        console.print(f"\n[bold green]Result: PASS[/bold green]")
    elif status == "SKIP":
        console.print(f"\n[bold yellow]Result: SKIPPED[/bold yellow]")
        console.print(Panel(Text(detail, style="yellow"), title="Skip reason", border_style="yellow"))
    elif status == "NOT_RUN":
        console.print(f"\n[bold red]Result: NOT RUN[/bold red]")
        console.print(Panel(Text(detail, style="red"), title="Test did not execute", border_style="red"))
    else:
        console.print(f"\n[bold red]Result: FAIL[/bold red]")

    console.print(f"[dim]Artifacts folder: {run_dir}[/dim]")
    console.print(f"[dim]Full log: {output_log}[/dim]")

    return status, detail, start_time, run_dir


def run_stage_root_cause(logic, trace, start_time, run_dir):
    console.print(f"\n[bold magenta]🔍 STAGE 3: GATHERING LOGS & CORRELATING...[/bold magenta]")
    console.print(f"[dim]{llm_stage_line()}[/dim]")

    log_file = os.path.join(run_dir, "designate_services.log")
    cmd = f"sudo journalctl -u devstack@designate-central.service -u devstack@designate-worker.service --since '{start_time}' --no-pager"
    try:
        logs = subprocess.check_output(cmd, shell=True).decode('utf-8')
        with open(log_file, "w") as f:
            f.write(logs)
    except:
        logs = "No backend logs found."

    task = Task(
        description=(
            f"FINAL INVESTIGATION:\n\n"
            f"THE TEST INTENT (from Stage 1):\n{logic}\n\n"
            f"THE TRACEBACK (from Stage 2):\n{trace}\n\n"
            f"DESIGNATE LOGS:\n{logs[-12000:]}\n\n"
            "Explain exactly why the backend failed to meet the test expectations. Pinpoint the specific service (Central/Worker) and the error."
        ),
        expected_output="Comprehensive Root Cause Verdict.",
        agent=analyst
    )
    result = Crew(agents=[analyst], tasks=[task]).kickoff()
    return str(result)


# --- MAIN ---

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
    console.print(Panel("[bold green]DESIGNATE E2E SYSTEM-AWARE INVESTIGATOR[/bold green]"))
    console.print(f"[dim]LLM: {OLLAMA_MODEL} @ {OLLAMA_BASE}[/dim]")

    llm_ok, llm_error = verify_llm_connection(OLLAMA_BASE)
    if not llm_ok:
        console.print("[red]Ollama is not reachable — AI stages cannot run:[/red]")
        console.print(Panel(Text(llm_error, style="red"), border_style="red"))
        sys.exit(1)

    cfg_ok, cfg_msg = verify_tempest_config()
    if not cfg_ok:
        console.print("[red]Tempest config missing — stestr run will fail without credentials:[/red]")
        console.print(Panel(Text(cfg_msg, style="red"), border_style="red"))
        sys.exit(1)
    console.print(f"[dim]Tempest config: {cfg_msg}[/dim]")

    grep_query = input("Grep tests (e.g. 'multipool') or ENTER for all: ").strip()
    tests, stestr_error = get_full_test_list(grep_query)

    if stestr_error:
        console.print("[red]stestr list failed — Tempest test discovery is broken:[/red]")
        console.print(Panel(Text(stestr_error, style="red"), border_style="red"))
        sys.exit(1)

    if not tests:
        label = f"'{grep_query}'" if grep_query else "designate tests"
        console.print(f"[red]No tests found matching {label}.[/red]")
        sys.exit(1)

    for i, t in enumerate(tests):
        print(f"[{i}] {t}")

    target_test = prompt_test_selection(tests)
    console.print(f"\n[bold]Selected:[/bold] {target_test}")

    logic_summary = run_stage_logic_discovery(target_test)
    status, detail, start_time, run_dir = run_stage_execution(target_test)

    if status == "SKIP":
        console.print(
            "\n[dim]Test was skipped — no backend failure to investigate. "
            "Fix the skip reason above and re-run.[/dim]"
        )
    elif status == "NOT_RUN":
        console.print(
            "\n[dim]Test never executed — no Designate logs to analyze. "
            "Fix the stestr error above and re-run.[/dim]"
        )
    elif status == "FAIL":
        final_verdict = run_stage_root_cause(logic_summary, detail, start_time, run_dir)
        console.print("\n")
        console.print(
            Panel(Text(str(final_verdict), style="green"), title="✅ FINAL ROOT CAUSE VERDICT", border_style="green"))
    else:
        console.print("\n[bold green]Success: test passed.[/bold green]")