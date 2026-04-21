import subprocess
import os
import sys
import re
import time
import importlib.util
from datetime import datetime
from crewai import Agent, Task, Crew
from crewai.tools import tool
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

console = Console()

# --- CONFIGURATION ---
os.environ["OPENAI_API_BASE"] = "http://10.9.95.131:11434/v1"
os.environ["OPENAI_MODEL_NAME"] = "ollama/llama3.1"
os.environ["OPENAI_API_KEY"] = "ollama"

TEMPEST_PATH = "/opt/stack/tempest"
BASE_HISTORY_DIR = "/opt/stack/agent_runs"
DEFAULT_TEST_PATH = "designate_tempest_plugin.tests.scenario.v2.test_designate_multipool.DesignateMultiPoolTest.test_move_zone_to_another_pool"

SYSTEM_CONTEXT = """
Environment: OpenStack DevStack (All-in-one).
Service: Designate (DNS-as-a-Service) with DNS enabled.
Testing Tool: Tempest with designate-tempest-plugin.
Architecture: 
- Central: Logic/DB/Pool coordination.
- Worker: Backend sync (BIND9/PowerDNS).
Target: Troubleshooting pool transitions and zone sync failures.
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

def run_stage_context_priming():
    console.print(f"\n[bold green]🌱 STAGE 0: PRIMING SYSTEM CONTEXT...[/bold green]")
    task = Task(
        description=f"Internalize this architecture: {SYSTEM_CONTEXT}. Briefly explain your understanding of how a Zone Move affects the Pool ID in the backend.",
        expected_output="A summary of the Designate pool sync architecture.",
        agent=analyst
    )
    result = Crew(agents=[analyst], tasks=[task]).kickoff()
    # CrewAI 0.28+ returns a CrewOutput object; use str() to get the result
    console.print(Panel(Text(str(result), style="italic green"), title="AI SYSTEM UNDERSTANDING", border_style="green"))


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
    try:
        cmd = "stestr list | grep designate | grep test_"
        if grep_str:
            cmd += f" | grep -i '{grep_str}'"
        out = subprocess.check_output(cmd, shell=True, cwd=TEMPEST_PATH, stderr=subprocess.DEVNULL).decode('utf-8')
        return [line.strip() for line in out.split('\n') if line.strip()]
    except:
        return []


if __name__ == "__main__":
    console.print(Panel("[bold green]DESIGNATE E2E SYSTEM-AWARE INVESTIGATOR[/bold green]"))

    grep_query = input("Grep tests (e.g. 'multipool') or ENTER for all: ").strip()
    tests = get_full_test_list(grep_query)

    if not tests:
        console.print(f"[red]No tests found matching '{grep_query}'.[/red]")
        sys.exit(1)

    for i, t in enumerate(tests):
        print(f"[{i}] {t}")

    choice = input(f"\nSelect test number (ENTER for default Multipool): ").strip()
    target_test = tests[int(choice)] if choice.isdigit() and int(choice) < len(tests) else DEFAULT_TEST_PATH

    # STAGES
    run_stage_context_priming()
    logic_summary = run_stage_logic_discovery(target_test)
    status, traceback, start_time, run_dir = run_stage_execution(target_test)

    if status == "FAIL":
        final_verdict = run_stage_root_cause(logic_summary, traceback, start_time, run_dir)
        console.print("\n")
        console.print(
            Panel(Text(str(final_verdict), style="green"), title="✅ FINAL ROOT CAUSE VERDICT", border_style="green"))
    else:
        console.print("\n[bold green]Success: Backend matches code intent.[/bold green]")