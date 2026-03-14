import os
import logging
import threading
import platform
import json
import asyncio
import time
import random
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

logger = logging.getLogger(__name__)

_api_semaphore = threading.Semaphore(2)
_async_api_semaphore = asyncio.Semaphore(2)

def _is_rate_limit_error(e: Exception) -> bool:
    """Check if an exception is a rate-limit (429) error."""
    return (
        'RateLimitError' in type(e).__name__
        or '429' in str(e)
        or 'RateLimitReached' in str(e)
    )


class GPT5NanoWrapper(BaseChatModel):
    """Wrapper for gpt-5-nano that strips unsupported parameters and rate-limits API calls."""
    llm: Any  # Allow Runnable / ChatModel for bind_tools compatibility
    class Config:
        arbitrary_types_allowed = True

    def _backoff_retry(self, fn, max_attempts=5, base_delay=2.0):
        """Execute fn with exponential backoff on rate limit errors."""
        for attempt in range(max_attempts):
            try:
                return fn()
            except Exception as e:
                if _is_rate_limit_error(e) and attempt < max_attempts - 1:
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    logger.warning(f"[RateLimit] Attempt {attempt+1}/{max_attempts}, backing off {delay:.1f}s")
                    time.sleep(delay)
                else:
                    raise

    async def _async_backoff_retry(self, fn, max_attempts=5, base_delay=2.0):
        """Async version of _backoff_retry."""
        for attempt in range(max_attempts):
            try:
                return await fn()
            except Exception as e:
                if _is_rate_limit_error(e) and attempt < max_attempts - 1:
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    logger.warning(f"[RateLimit] Attempt {attempt+1}/{max_attempts}, backing off {delay:.1f}s")
                    await asyncio.sleep(delay)
                else:
                    raise

    def invoke(self, input, config=None, **kwargs):
        """Rate-limited invoke with exponential backoff on 429 errors."""
        kwargs.pop('stop', None)
        def _call():
            with _api_semaphore:
                return self.llm.invoke(input, config=config, **kwargs)
        return self._backoff_retry(_call)

    async def ainvoke(self, input, config=None, **kwargs):
        """Rate-limited async invoke with exponential backoff on 429 errors."""
        kwargs.pop('stop', None)
        async def _call():
            async with _async_api_semaphore:
                return await self.llm.ainvoke(input, config=config, **kwargs)
        return await self._async_backoff_retry(_call)

    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, **kwargs: Any) -> ChatResult:
        """Fallback for LangChain versions that call _generate directly."""
        kwargs.pop('stop', None)
        _stop_sequences = stop  # Save for post-hoc truncation

        def _call():
            with _api_semaphore:
                if hasattr(self.llm, '_generate'):
                    return self.llm._generate(messages, **kwargs)
                else:
                    raw = self.llm.invoke(messages, **kwargs)
                    from langchain_core.outputs import ChatGeneration
                    return ChatResult(generations=[ChatGeneration(message=raw)])

        chat_result = self._backoff_retry(_call)

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

WORKFLOW:
1. ANALYZE: ListDirectory + ReadLocalFile to understand the project structure, language, and dependencies.
2. CREATE: WriteToFile to create Dockerfile and .dockerignore (only after step 1).
3. VERIFY: VerifyBuild to test the Dockerfile.
4. IF FAILS: Call SearchDockerError with the error, apply its fix, then VerifyBuild again.

RULES:
1. You MUST call VerifyBuild. Final Answer may ONLY report success if the last VerifyBuild returned SUCCESS.
2. When VerifyBuild fails, call SearchDockerError IMMEDIATELY before any WriteToFile. Apply its advice exactly.
3. Complete step 1 (ANALYZE) before writing a Dockerfile. Confirm all COPY source paths via ListDirectory.
4. Do not repeat the same SearchDockerError query. If advice fails, expand context before searching again.
5. Do not change the base image tag unless SearchDockerError or VerifyBuild explicitly indicates a tag/platform issue.

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

OUTPUT RULES:
1. You can ONLY output "Thought:", "Action:", "Action Input:", or "Final Answer:"
2. You must NEVER write "Observation:" - the system adds that automatically
3. Each response must contain EXACTLY ONE action - never chain multiple actions
4. Stop immediately after "Action Input:" - do not continue writing
5. After VerifyBuild SUCCESS, give a ONE sentence Final Answer

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
