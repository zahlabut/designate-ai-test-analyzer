cat <<EOF > README.md
# 🚀 Designate AI Test Analyzer

**Designate AI Test Analyzer** is an autonomous diagnostic agent built to transition OpenStack Designate development into an **Agentic SDLC**. By offloading routine test execution, source code interpretation, and log correlation to an AI collaborator, engineers move directly from "Test Failure" to "Root Cause" without manual intervention.

---

## 🧠 AI Framework & Modules

This agent utilizes a sophisticated AI stack to provide deep system insights:

* **CrewAI**: The primary orchestration framework. It manages the agent's lifecycle, transitioning through stages of investigation and ensuring tasks are completed in a logical sequence.
* **Ollama (Llama 3.1)**: The core reasoning engine. It processes complex tracebacks and multi-service logs to find patterns that a simple grep would miss.
* **LangChain Tools**: Integrated into the agent to provide secure, structured access to the filesystem and shell (reading source code, executing \`stestr\`, and pulling \`journalctl\`).
* **Rich**: Powering the CLI interface to provide real-time, stage-by-stage visual updates of the AI's internal reasoning.

---

## 🛠 Prerequisites

### 1. Remote AI Backend (Ollama)
The agent requires a remote Ollama instance. Deploy it on your remote host using the following commands:

**Deploy via Podman:**
\`\`\`bash
podman volume create ollama-data

podman run -d \\
  --name ollama \\
  -v ollama-data:/root/.ollama \\
  -p 0.0.0.0:11434:11434 \\
  ollama/ollama
\`\`\`

**Pull and Initialize the Model:**
\`\`\`bash
podman exec -it ollama ollama run llama3.1
\`\`\`

### 2. DevStack Environment
* **Service**: Designate (DNS enabled) must be active on the host.
* **Plugin**: \`designate-tempest-plugin\` must be installed in your tempest environment.
* **Permissions**: The user running the script must have \`sudo\` privileges to access \`journalctl\` for log extraction.

### 3. Python Dependencies
\`\`\`bash
pip install crewai rich
\`\`\`

---

## 📖 Usage

### Launch the Agent
\`\`\`bash
python3 main.py
\`\`\`

### Workflow Steps
1. **Grep Tests**: Enter a search string (e.g., \`multipool\` or \`quota\`) to filter the test suite dynamically.
2. **Select Test**: Type the index number of the specific test you wish to investigate.
3. **The Autonomous Flow**:
    * **Stage 0: Context Priming** - Primes the AI on Designate's Central/Worker architecture and DevStack specifics.
    * **Stage 1: Logic Discovery** - The AI reads the Python source code using \`read_source\` to determine the test's intent and success conditions.
    * **Stage 2: Execution** - Triggers a fresh \`stestr\` run and captures the live console output and traceback.
    * **Stage 3: Root Cause Analysis** - If a failure occurs, the AI autonomously fetches backend logs via \`journalctl\` and correlates them with the traceback to provide a final **Technical Verdict**.

---

## 📂 Project Structure

* \`main.py\`: The core logic containing the agent definition, tools, and autonomous stages.
* \`requirements.txt\`: Python dependency list for easy environment setup.
* \`agent_runs/\`: Local directory created at runtime where the agent stores logs and artifacts for every execution.

---

> **Mission Alignment**: This project operationalizes our **AI-first strategy** across Product Engineering by empowering engineers to use AI agents to handle routine execution, increasing our collective impact.
EOF