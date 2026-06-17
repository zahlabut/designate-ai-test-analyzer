import subprocess
import os
import sys
import re
import time
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

def run_stage_logic_discovery(test_path):
    console.print(f"\n[bold blue]🔎 STAGE 1: ANALYZING TEST LOGIC...[/bold blue]")
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

    clean_test = test_path.split('[')[0]
    output_log = os.path.join(run_dir, "tempest_run.log")
    start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    console.print(f"\n[bold yellow]🚀 STAGE 2: EXECUTING FRESH TEST RUN...[/bold yellow]")

    process = subprocess.Popen(f"stestr run '{clean_test}'", shell=True,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               cwd=TEMPEST_PATH, text=True, bufsize=1)

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
    status = "PASS" if process.returncode == 0 else "FAIL"

    trace = "No failure."
    if status == "FAIL" and "Captured traceback:" in output_str:
        trace = output_str.split("Captured traceback:")[-1].split("Captured pythonlogging:")[0].strip()

    color = "green" if status == "PASS" else "red"
    console.print(f"\n[bold {color}]Result: {status}[/bold {color}]")
    console.print(f"[dim]Artifacts folder: {run_dir}[/dim]")

    return status, trace, start_time, run_dir


def run_stage_root_cause(logic, trace, start_time, run_dir):
    console.print(f"\n[bold magenta]🔍 STAGE 3: GATHERING LOGS & CORRELATING...[/bold magenta]")

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
        ["stestr", "list"],
        cwd=TEMPEST_PATH,
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
    status, traceback, start_time, run_dir = run_stage_execution(target_test)

    if status == "FAIL":
        final_verdict = run_stage_root_cause(logic_summary, traceback, start_time, run_dir)
        console.print("\n")
        console.print(
            Panel(Text(str(final_verdict), style="green"), title="✅ FINAL ROOT CAUSE VERDICT", border_style="green"))
    else:
        console.print("\n[bold green]Success: Backend matches code intent.[/bold green]")