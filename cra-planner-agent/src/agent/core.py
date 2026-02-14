import os
import logging
import threading
import platform
import json
import asyncio
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# --- MONKEYPATCH for PyArrow/Datasets incompatibility ---
# This fixes "AttributeError: module 'pyarrow' has no attribute 'PyExtensionType'"
try:
    import pyarrow
    if not hasattr(pyarrow, 'PyExtensionType'):
        pyarrow.PyExtensionType = pyarrow.ExtensionType
except ImportError:
    pass
# --------------------------------------------------------

from langchain_openai import AzureChatOpenAI
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_core.tools import Tool
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatResult
from langchain_core.prompts import PromptTemplate

# Import tools and helpers
from .tools import (
    _get_http_client, FormattedOutputHandler,
    create_structured_tools
)
# Note: Other tools (ReadFile, docker_image_search, etc.) are currently defined in planner_agent.py 
# but are simple wrappers. For this refactor, I will define them here 
# or import if I moved them. I only moved a few complex ones to tools.py.
# To make this fully functional, I need to make sure ALL tools are available.
# I will quickly re-implement the standard file tools here using the path helpers from tools.py,
# or assume they will be moved. To be safe/clean, I will implement them here or in tools.py.
# Since I only moved `search_web` and `errors` to tools.py, I should probably add the file tools to tools.py
# later or define them inline here if they are simple. 
# Better: I will implement the missing standard tools in tools.py in a follow-up or right now?
# I'll stick to what I moved to tools.py so far, but I missed the simple file tools!
# I should have moved 'read_local_file', 'list_directory' etc. to tools.py.
# I will patch tools.py first!

logger = logging.getLogger(__name__)

_api_semaphore = threading.Semaphore(2)

class GPT5NanoWrapper(BaseChatModel):
    """Wrapper for gpt-5-nano that strips unsupported parameters and rate-limits API calls."""
    llm: Any  # Allow Runnable / ChatModel for bind_tools compatibility
    class Config:
        arbitrary_types_allowed = True

    def invoke(self, input, config=None, **kwargs):
        """Rate-limited invoke for compatibility with bound tools."""
        # Remove unsupported parameters
        kwargs.pop('stop', None)
        with _api_semaphore:
            return self.llm.invoke(input, config=config, **kwargs)

    async def ainvoke(self, input, config=None, **kwargs):
        """Rate-limited async invoke for compatibility with bound tools."""
        # Remove unsupported parameters
        kwargs.pop('stop', None)
        with _api_semaphore:
            return await self.llm.ainvoke(input, config=config, **kwargs)

    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, **kwargs: Any) -> ChatResult:
        """Fallback for LangChain versions that call _generate directly."""
        kwargs.pop('stop', None)
        with _api_semaphore:
            # Ensure the bound object has _generate before calling
            if hasattr(self.llm, '_generate'):
                return self.llm._generate(messages, **kwargs)
            else:
                # Fallback to invoke if _generate is not available
                result = self.llm.invoke(messages, **kwargs)
                # Convert to ChatResult if needed
                from langchain_core.outputs import ChatGeneration
                return ChatResult(generations=[ChatGeneration(message=result)])

    @property
    def _llm_type(self) -> str:
        return "gpt-5-nano-wrapper"

    def bind_tools(self, tools, **kwargs):
        """Bind tools and wrap the result to maintain rate limiting."""
        bound = self.llm.bind_tools(tools, **kwargs)
        return GPT5NanoWrapper(llm=bound)

def _get_host_platform() -> tuple:
    """Detect host architecture for Docker compatibility checks."""
    system = platform.system().lower()
    machine = platform.machine().lower()
    
    if system == 'darwin':
        if machine == 'arm64':
            return 'linux/arm64', 'ARM64 (Apple Silicon)'
        return 'linux/amd64', 'AMD64 (Intel Mac)'
    elif system == 'linux':
        if machine in ['aarch64', 'base', 'arm64']:
             return 'linux/arm64', 'ARM64 (Linux)'
        return 'linux/amd64', 'AMD64 (Linux)'
    elif system == 'windows':
        return 'linux/amd64', 'AMD64 (Windows)'
        
    return 'linux/amd64', 'AMD64 (Unknown)'

def create_learner_agent(
    max_iterations: int = 50,
    verbose: bool = True,
    repository_path: str = None,
    repo_name: str = None,
    detected_language: str = None,
    extra_tools: List[Tool] = None
):
    """
    Create a LEARNER agent.
    
    This agent is NOT scripted. It receives a Goal and Context, and uses tools to achieve it.
    """
    load_dotenv()
    
    # 1. Setup Path Context
    # REMOVED: _set_repository_base_path(repository_path) - context is now bound to tools
    pass
    
    # 2. Config
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")

    if not all([api_key, endpoint, deployment]):
        raise ValueError("Missing required environment variables.")

    # 3. LLM
    base_llm = AzureChatOpenAI(
        azure_deployment=deployment,
        api_key=api_key,
        azure_endpoint=endpoint,
        api_version=api_version,
        max_retries=5,
        timeout=120,
        http_client=_get_http_client(),
    )
    llm = GPT5NanoWrapper(llm=base_llm)

    # 4. Tools - Use Structured Tools with Pydantic schemas
    # 4. Tools - Use Structured Tools with Pydantic schemas bound to this repo
    tools_list = create_structured_tools(repository_path)

    if extra_tools:
        tools_list.extend(extra_tools)
    
    # 5. Prompt - Phase A: Containerization only
    _, host_arch_name = _get_host_platform()
    
    template = """You are an AUTONOMOUS DevOps engineer creating Dockerfiles.
HOST ARCHITECTURE: {host_arch_name}

TOOLS:
{tools}

TOOL INPUT CONTRACT (MUST FOLLOW):
- WriteToFile:
  Action Input: {{"file_path":"Dockerfile","content":"<full file contents>"}}
  Also used for: .dockerignore
- ReadLocalFile:
  Action Input: {{"file_path":"package.json"}}
- ListDirectory:
  Action Input: {{"directory":"."}}
- VerifyBuild:
  Action Input: "" (empty string)

IMPORTANT:
- Action Input must be exactly a JSON object for that tool (or "" for VerifyBuild). No prose.
- You may emit at most one tool call per message.

═══════════════════════════════════════════════════════════════════════════════
WORKFLOW:
═══════════════════════════════════════════════════════════════════════════════

1. ListDirectory to see project structure
2. ReadLocalFile for build files (package.json, requirements.txt, pom.xml, etc.)
3. Identify language, framework, and dependencies
4. WriteToFile to create Dockerfile:
   - Install ALL system packages, runtime, and build tools (including git, make, gcc if needed)
   - Copy source with: COPY . /app
   - Set WORKDIR /app
   - Use ENV DEBIAN_FRONTEND=noninteractive
   - Add -y flags to all apt-get calls
   - Include dev/test dependencies too (they will be needed later for test suite)
   - DO NOT set NODE_ENV=production or equivalent — tests need dev dependencies
   - DO NOT add any RUN or CMD instruction that executes tests (no RUN npm test, no RUN pytest, etc.)
   - Tests are run via a SEPARATE run_tests.sh script, never inside the Dockerfile
5. WriteToFile to create .dockerignore
   - DO NOT exclude: tests/, test/, spec/, __tests__/, run_tests.sh, Makefile
6. VerifyBuild — builds the Docker image

IF BUILD FAILS:
  - Use SearchDockerError with keywords from the error
  - Fix the Dockerfile
  - VerifyBuild again
  - Repeat until build succeeds

═══════════════════════════════════════════════════════════════════════════════
RULES:
═══════════════════════════════════════════════════════════════════════════════

1. You MUST call VerifyBuild before your Final Answer
2. You CANNOT give Final Answer unless VerifyBuild returned status="success" or status="incomplete"
3. When build fails, use SearchDockerError to find solutions
4. Use ENV DEBIAN_FRONTEND=noninteractive and -y on all apt-get
5. Install build tools (git, make, gcc, etc.) that the project may need

COMMON FIX PATTERNS:
- "npm ERR! could not find package-lock.json" → Use "npm install" not "npm ci"
- "pip: No module named X" → RUN apt-get install -y python3-dev gcc
- "exec format error" → Use --platform=linux/amd64 in FROM
- "COPY failed: file not found" → Check with ListDirectory, fix paths
- "[Y/n]" hanging → ENV DEBIAN_FRONTEND=noninteractive + -y

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

CRITICAL OUTPUT RULES:
1. You can ONLY output "Thought:", "Action:", "Action Input:", or "Final Answer:"
2. You must NEVER write "Observation:" - the system adds that automatically
3. Each response must contain EXACTLY ONE action
4. Stop immediately after writing "Action Input:"

When VerifyBuild returns status="success" or status="incomplete", respond:
Thought: VerifyBuild confirmed the build. Final Answer: Successfully created Dockerfile for [repo-name]. Build verified.

Question: {input}
Thought:{agent_scratchpad}"""

    prompt = PromptTemplate.from_template(template).partial(
        host_arch_name=host_arch_name
    )
    
    # ReAct Agent
    agent = create_react_agent(llm, tools_list, prompt)

    def _handle_parsing_error(error) -> str:
        """Guide the agent back to proper format on parse errors."""
        return (
            "FORMAT ERROR: Your output was not in the correct format. "
            "You MUST use exactly this format:\n"
            "Thought: <your reasoning>\n"
            "Action: <tool name>\n"
            "Action Input: <JSON input or empty string>\n\n"
            "OR to finish:\n"
            "Thought: I now know the final answer\n"
            "Final Answer: <your answer>\n\n"
            "Try again with the correct format."
        )

    executor = AgentExecutor(
        agent=agent,
        tools=tools_list,
        verbose=verbose,
        handle_parsing_errors=_handle_parsing_error,
        max_iterations=max_iterations,
        return_intermediate_steps=True
    )

    # Return executor, handler, and base_llm for guidelines generation
    return executor, FormattedOutputHandler(), base_llm


def create_test_agent(
    max_iterations: int = 50,
    verbose: bool = True,
    repository_path: str = None,
    repo_name: str = None,
    detected_language: str = None,
    extra_tools: List[Tool] = None
):
    """
    Create a TEST AGENT (Phase B).
    
    This agent discovers the test command, creates run_tests.sh, and verifies
    that the test suite passes inside the already-built Docker container.
    """
    load_dotenv()
    
    # Config
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")

    if not all([api_key, endpoint, deployment]):
        raise ValueError("Missing required environment variables.")

    # LLM
    base_llm = AzureChatOpenAI(
        azure_deployment=deployment,
        api_key=api_key,
        azure_endpoint=endpoint,
        api_version=api_version,
        max_retries=5,
        timeout=120,
        http_client=_get_http_client(),
    )
    llm = GPT5NanoWrapper(llm=base_llm)

    # Tools
    tools_list = create_structured_tools(repository_path)
    if extra_tools:
        tools_list.extend(extra_tools)
    
    _, host_arch_name = _get_host_platform()
    
    template = """You are an AUTONOMOUS DevOps engineer. A Dockerfile already exists for this repo.
Your job: discover the test command, create run_tests.sh, and verify that tests pass.
HOST ARCHITECTURE: {host_arch_name}

TOOLS:
{tools}

TOOL INPUT CONTRACT:
- WriteToFile:
  Action Input: {{"file_path":"run_tests.sh","content":"<script contents>"}}
  Also used for: Dockerfile (if you need to add missing deps)
- ReadLocalFile:
  Action Input: {{"file_path":"Dockerfile"}}
- ListDirectory:
  Action Input: {{"directory":"."}}
- VerifyBuild:
  Action Input: "" (empty string)
- DiagnoseTestFailure:
  Action Input: {{"test_output":"<from VerifyBuild>","dockerfile_content":"<from ReadLocalFile>","run_tests_content":"<from ReadLocalFile>"}}
- RunInContainer:
  Action Input: {{"command":"pip install pytest && pytest --co -q"}}

IMPORTANT:
- Action Input must be exactly a JSON object for that tool (or "" for VerifyBuild).
- You may emit at most one tool call per message.

═══════════════════════════════════════════════════════════════════════════════
WORKFLOW:
═══════════════════════════════════════════════════════════════════════════════

STEP 1 — DISCOVER TEST COMMAND (check in this order):
  a. CI workflows: ReadLocalFile(".github/workflows/<file>.yml") — look for 'run:' steps
     with pytest/npm test/mvn test/make check/go test/cargo test etc.
  b. Makefile: look for 'test:' or 'check:' or 'ci:' targets
  c. package.json → "scripts.test"; tox.ini → [commands]; pyproject.toml → [tool.pytest]
  d. README.md / CONTRIBUTING.md — testing instructions
  e. Fallback heuristics:
     Python → python -m pytest; Node.js → npm test; Java/Maven → mvn test
     Go → go test ./...; Rust → cargo test; C/C++ → make check or ctest

STEP 2 — CREATE run_tests.sh:
  - Start with: #!/bin/bash\nset -eo pipefail
  - Install test-only dependencies (pip install pytest, npm install, etc.)
  - Handle submodules: git submodule update --init --recursive (if .gitmodules exists)
  - Run the discovered test command
  - DO NOT use 'tee' or write to files - VerifyBuild captures all output automatically

STEP 3 — VERIFY:
  - VerifyBuild rebuilds the image and runs: docker run <image> bash /app/run_tests.sh

STEP 4 — IF TESTS FAIL (MANDATORY):
  a. Call DiagnoseTestFailure with the test_output + ReadLocalFile('Dockerfile') + ReadLocalFile('run_tests.sh')
  b. Apply the suggested fix (update Dockerfile or run_tests.sh)
  c. VerifyBuild again
  d. Repeat until status="success"

You can also use RunInContainer to run quick diagnostic commands inside the built image
before committing to a full VerifyBuild cycle.

═══════════════════════════════════════════════════════════════════════════════
TEST FAILURE FIX RULES:
═══════════════════════════════════════════════════════════════════════════════

ALLOWED fixes:
  ✓ Install missing dependency (in run_tests.sh or Dockerfile)
  ✓ Add missing ENV variable (DATABASE_URL=sqlite:///test.db, etc.)
  ✓ Fix WORKDIR or cd path in run_tests.sh
  ✓ Initialize git submodules
  ✓ Skip network/integration tests: pytest -k "not network"

NEVER do these:
  ✗ Overwrite or delete test files
  ✗ Replace test command with echo or true
  ✗ Add --collect-only or --ignore flags that skip all tests

═══════════════════════════════════════════════════════════════════════════════

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

CRITICAL OUTPUT RULES:
1. You can ONLY output "Thought:", "Action:", "Action Input:", or "Final Answer:"
2. You must NEVER write "Observation:"
3. Each response must contain EXACTLY ONE action
4. Stop immediately after writing "Action Input:"

CRITICAL: When VerifyBuild returns status="success", you MUST IMMEDIATELY give your Final Answer.
Do NOT call any more tools. Do NOT try to read test_results.txt (it's in an ephemeral container).
The test output is already in the VerifyBuild response.

When VerifyBuild returns status="success":
Thought: Tests passed! Final Answer: Successfully created run_tests.sh. Test suite passes.

Question: {input}
Thought:{agent_scratchpad}"""

    prompt = PromptTemplate.from_template(template).partial(
        host_arch_name=host_arch_name
    )
    
    agent = create_react_agent(llm, tools_list, prompt)

    def _handle_parsing_error(error) -> str:
        """Guide the agent back to proper format on parse errors."""
        return (
            "FORMAT ERROR: Your output was not in the correct format. "
            "You MUST use exactly this format:\n"
            "Thought: <your reasoning>\n"
            "Action: <tool name>\n"
            "Action Input: <JSON input or empty string>\n\n"
            "OR to finish:\n"
            "Thought: I now know the final answer\n"
            "Final Answer: <your answer>\n\n"
            "Try again with the correct format."
        )

    executor = AgentExecutor(
        agent=agent,
        tools=tools_list,
        verbose=verbose,
        handle_parsing_errors=_handle_parsing_error,
        max_iterations=max_iterations,
        return_intermediate_steps=True
    )

    return executor, FormattedOutputHandler(), base_llm
