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
    search_web, search_docker_error, extract_relevant_sections,
    _get_http_client, _set_repository_base_path, FormattedOutputHandler,
    grep_files, find_files, list_directory, read_local_file, create_directory_tree,
    extract_json_field, get_file_metadata, docker_image_search, fetch_web_page,
    _get_expanded_platform_info, write_to_file
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
    llm: AzureChatOpenAI
    class Config:
        arbitrary_types_allowed = True
    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, **kwargs: Any) -> ChatResult:
        kwargs.pop('stop', None)
        with _api_semaphore:
            return self.llm._generate(messages, **kwargs)
    @property
    def _llm_type(self) -> str:
        return "gpt-5-nano-wrapper"
    def bind_tools(self, tools, **kwargs):
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
    max_iterations: int = 15,
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
    _set_repository_base_path(repository_path)
    
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

    # 4. Tools
    # Critical: Use the definitions from tools.py (I need to update tools.py to include file tools!)
    # I'll reference them here assuming they exist or simple ones. 
    # For now, I'll import what I have. I need to make sure tools.py has file tools.
    # To avoid breaking verify, I will append the FILE TOOLS to tools.py in the NEXT step.
    # For now I will import "all" from tools and list them.
    from .tools import search_web, search_docker_error
    
    # Define tool implementations inline for now if missing from tools.py (I'll move them properly next)
    # Actually, let's use a dynamic import or assume I fix tools.py next.
    # I'll define a provisional list and update it.
    
    tools_list = [
        Tool(name="WriteToFile", func=write_to_file, description="Write content to a file. Input: file_path,content_string"),
        Tool(name="ReadLocalFile", func=read_local_file, description="Read the content of a local file."),
        Tool(name="ListDirectory", func=list_directory, description="List files in a directory."),
        Tool(name="CreateDirectoryTree", func=create_directory_tree, description="List directory tree structure. Input: path,depth (default 2)"),
        Tool(name="FindFiles", func=find_files, description="Find files matching pattern. Input: dir,pattern,depth"),
        Tool(name="GrepFiles", func=grep_files, description="Search text in files. Input: dir,pattern,glob"),
        Tool(name="SearchWeb", func=search_web, description="Search web documentation."),
        Tool(name="SearchDockerError", func=search_docker_error, description="Search for solutions to Docker errors."),
        Tool(name="FetchWebPage", func=fetch_web_page, description="Fetch content of a web page."),
        Tool(name="DockerImageSearch", func=docker_image_search, description="Search/verify Docker Hub images. Input: query or tags:image"),
        Tool(name="ExtractJsonField", func=extract_json_field, description="Extract field from JSON file."),
        Tool(name="GetFileMetadata", func=get_file_metadata, description="Get file size and mode.")
    ]
    
    if extra_tools:
        tools_list.extend(extra_tools)
    
    # 5. Prompt - Enhanced with Research-Fix Workflow
    _, host_arch_name = _get_host_platform()
    
    template = """You are an AUTONOMOUS DevOps engineer creating Dockerfiles.
HOST ARCHITECTURE: {host_arch_name}

TOOLS:
{tools}

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
  8. SearchDockerError with descriptive keywords from the error
  9. Fix the Dockerfile based on what you learned
  10. VerifyBuild again
  11. Repeat steps 7-10 until VerifyBuild returns SUCCESS

═══════════════════════════════════════════════════════════════════════════════
ABSOLUTE RULES:
═══════════════════════════════════════════════════════════════════════════════

1. You MUST call VerifyBuild at least once before your Final Answer
2. You CANNOT give a Final Answer if VerifyBuild has not returned "SUCCESS"
3. When VerifyBuild fails, you MUST use SearchDockerError to find solutions
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

═══════════════════════════════════════════════════════════════════════════════

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: VerifyBuild returned SUCCESS. I can now provide my Final Answer.
Final Answer: the final answer (ONLY after VerifyBuild SUCCESS)

IMPORTANT:
1. To create a file, you MUST use the WriteToFile tool.
2. You MUST verify your work with VerifyBuild before finishing.
3. If VerifyBuild fails, you MUST research and fix it.
4. Your Final Answer should summarize what you created and confirm the build passed.

Begin!

Question: {input}
Thought:{agent_scratchpad}"""

    prompt = PromptTemplate.from_template(template).partial(
        host_arch_name=host_arch_name,
        context="{context}" # Placeholder to be filled at invoke time? No, input variable.
    )
    
    # ReAct Agent
    agent = create_react_agent(llm, tools_list, prompt)
    
    executor = AgentExecutor(
        agent=agent,
        tools=tools_list,
        verbose=verbose,
        handle_parsing_errors=True,
        max_iterations=max_iterations
    )
    
    return executor, FormattedOutputHandler()
