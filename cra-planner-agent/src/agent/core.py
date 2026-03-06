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
_async_api_semaphore = asyncio.Semaphore(2)

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
        async with _async_api_semaphore:
            return await self.llm.ainvoke(input, config=config, **kwargs)

    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, **kwargs: Any) -> ChatResult:
        """Fallback for LangChain versions that call _generate directly."""
        kwargs.pop('stop', None)
        _stop_sequences = stop  # Save for post-hoc truncation
        with _api_semaphore:
            # Ensure the bound object has _generate before calling
            if hasattr(self.llm, '_generate'):
                chat_result = self.llm._generate(messages, **kwargs)
            else:
                # Fallback to invoke if _generate is not available
                raw = self.llm.invoke(messages, **kwargs)
                # Convert to ChatResult if needed
                from langchain_core.outputs import ChatGeneration
                chat_result = ChatResult(generations=[ChatGeneration(message=raw)])
        # Post-hoc truncation at stop sequences so ReAct parsing works
        # even though gpt-5-nano doesn't support native stop sequences.
        if _stop_sequences and chat_result.generations:
            for gen in chat_result.generations:
                text = gen.message.content
                if not isinstance(text, str):
                    continue
                earliest = len(text)
                for seq in _stop_sequences:
                    idx = text.find(seq)
                    if idx != -1 and idx < earliest:
                        earliest = idx
                if earliest < len(text):
                    # Strip trailing whitespace so the ReAct parser doesn't
                    # trip on stray newlines before the stop token boundary.
                    gen.message.content = text[:earliest].rstrip()
        return chat_result

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
        if machine in ['aarch64', 'arm64']:
             return 'linux/arm64', 'ARM64 (Linux)'
        return 'linux/amd64', 'AMD64 (Linux)'
    elif system == 'windows':
        return 'linux/amd64', 'AMD64 (Windows)'
        
    return 'linux/amd64', 'AMD64 (Unknown)'

def create_learner_agent(
    max_iterations: int = 15,
    verbose: bool = True,
    repository_path: str = None,
    extra_tools: List[Tool] = None
):
    """
    Create a LEARNER agent.

    This agent is NOT scripted. It receives a Goal and Context, and uses tools to achieve it.
    """
    load_dotenv()
    
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
    
    # 5. Prompt - Enhanced with Research-Fix Workflow
    _, host_arch_name = _get_host_platform()
    
    template = """You are an AUTONOMOUS DevOps engineer creating Dockerfiles.
HOST ARCHITECTURE: {host_arch_name}

TOOLS:
{tools}

TOOL INPUT CONTRACT (MUST FOLLOW):
- WriteToFile:
  Action Input: {{"file_path":"Dockerfile","content":"<full file contents>"}}
- ReadLocalFile:
  Action Input: {{"file_path":"package.json"}}
- ListDirectory:
  Action Input: {{"directory":"."}}
- VerifyBuild:
  Action Input: "" (empty string)
- SearchDockerError:
  Action Input: {{"error_keywords":"<error>", "agent_context":"<what you tried/observed>"}}

IMPORTANT:
- Action Input must be exactly a JSON object for that tool (or "" for VerifyBuild). No prose.
- You may emit at most one tool call per message.

═══════════════════════════════════════════════════════════════════════════════
CRITICAL WORKFLOW - YOU MUST FOLLOW THIS EXACTLY:
═══════════════════════════════════════════════════════════════════════════════

PHASE 1 - ANALYZE:
  1. ListDirectory to see project structure
  2. ReadLocalFile to check package.json, requirements.txt, pom.xml, etc.
  3. Identify language, framework, and dependencies

PHASE 2 - CREATE:
  4. WriteToFile to create Dockerfile
  5. WriteToFile to create .dockerignore

PHASE 3 - VERIFY (MANDATORY):
  6. VerifyBuild to test the Dockerfile

PHASE 4 - IF BUILD FAILS (MANDATORY LOOP):
  7. Read the error message carefully
  8. IMMEDIATELY use SearchDockerError(error_keywords="...", agent_context="...") to get a fix
  9. Do NOT guess or try to fix it yourself without searching first
  10. Apply the fix from the AI analysis
  11. VerifyBuild again

═══════════════════════════════════════════════════════════════════════════════
ABSOLUTE RULES:
═══════════════════════════════════════════════════════════════════════════════

1. You MUST call VerifyBuild at least once before your Final Answer
2. You CANNOT give a Final Answer if VerifyBuild has not returned "SUCCESS"
3. When VerifyBuild fails, you MUST use SearchDockerError IMMEDIATELY. Do not think "I can fix this" - you must get external validation first.
4. You MUST fix the Dockerfile and VerifyBuild again after researching
5. Your Final Answer can ONLY report success if the last VerifyBuild passed

═══════════════════════════════════════════════════════════════════════════════
COMMON FIX PATTERNS (Learn from these):
═══════════════════════════════════════════════════════════════════════════════

Error: "npm ERR! could not find package-lock.json"
Fix: Use "npm install" instead of "npm ci", or "COPY package*.json ./"

Error: "pip: No module named X" or missing system dependencies
Fix: Install build deps: RUN apt-get update && apt-get install -y python3-dev gcc

Error: "exec format error" or platform mismatch
Fix: Use --platform=linux/amd64 in FROM statement

Error: "COPY failed: file not found"
Fix: Check actual file names with ListDirectory, fix COPY paths

Error: "manifest unknown" or "not found" or "pull access denied"
Fix: The base image tag does not exist. WriteToFile will block invalid tags and show
     available alternatives — pick a valid tag from that list and rewrite the FROM line.

═══════════════════════════════════════════════════════════════════════════════
MULTI-MODULE JAVA PROJECTS (Maven/Gradle):
═══════════════════════════════════════════════════════════════════════════════

When the classification shows is_monorepo=true with Maven/Gradle:
1. Use FindFiles or ListDirectory to map the module structure (find pom.xml files)
2. The ROOT pom.xml is usually an aggregator — it orchestrates, it does NOT produce a JAR
3. Build with: mvn package -DskipTests -pl <target-module> -am
   - -pl <module>: build specific module
   - -am: also make dependencies (resolves SNAPSHOT versions)
4. NEVER try to mvn install first then build separately — SNAPSHOT jars only exist in reactor
5. For the Dockerfile, copy the ENTIRE project, build with mvn, then copy out the target JAR

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
2. You must NEVER write "Observation:" - the system adds that automatically after your Action runs
3. Each response must contain EXACTLY ONE action - never chain multiple actions together
4. Stop immediately after writing "Action Input:" - do not continue writing

FINAL ANSWER FORMAT:
When VerifyBuild returns SUCCESS in the Observation, your NEXT response must be:

Thought: VerifyBuild returned SUCCESS.
Final Answer: Successfully created Dockerfile for [repo-name]. Build verified and smoke test passed.

Keep it to ONE sentence. DO NOT write lengthy documentation after success.

IMPORTANT:
1. To create a file, you MUST use the WriteToFile tool.
2. You MUST verify your work with VerifyBuild before finishing.
3. If VerifyBuild fails, you MUST research and fix it.
4. Your Final Answer should be CONCISE (1 sentence) after VerifyBuild SUCCESS.
5. Every Action MUST be immediately followed by "Action Input:" on the next line
6. You can only output ONE action per response, then you must wait for the system to provide the Observation

FORMAT EXAMPLE (follow this EXACTLY):
Thought: I need to check the project structure
Action: ListDirectory
Action Input: {{"directory":"."}}

(then STOP and wait for Observation)

Begin!

Question: {input}
Thought:{agent_scratchpad}"""

    prompt = PromptTemplate.from_template(template).partial(
        host_arch_name=host_arch_name
    )
    
    # ReAct Agent
    agent = create_react_agent(llm, tools_list, prompt)
    
    executor = AgentExecutor(
        agent=agent,
        tools=tools_list,
        verbose=verbose,
        handle_parsing_errors=True,
        max_iterations=max_iterations,
        return_intermediate_steps=True
    )

    # Return executor, handler, and base_llm for guidelines generation
    return executor, FormattedOutputHandler(), base_llm
