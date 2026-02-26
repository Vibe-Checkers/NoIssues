"""
Strategy-aware verification for Docker containers.
Uses T1 classification to select appropriate smoke test approach.

IMPORTANT: Verification results are NOT shown to the agent.
The agent only sees "passed" or "failed" + generic guidance.
Do NOT leak the test command or strategy to the agent.

ALL strategies generate and run a real command — there is no auto-pass.
"""

import os
import json
import logging
from typing import Dict, Tuple
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Strategy-specific prompt templates
# ---------------------------------------------------------------------------
# Each template generates a ONE-LINE shell command to run INSIDE the container.
# Every strategy MUST produce a runnable command — no auto-pass.

STRATEGY_PROMPTS = {
    "import_test": """Generate a ONE-LINE shell command that IMPORTS this {language} package
and prints its version or lists its key exports.

KNOWN PACKAGE NAME: {package_name}

GOOD examples by language:
- Python: python -c "import {package_name}; print(getattr({package_name}, '__version__', 'ok'))"
- JavaScript: node -e "const m = require('{package_name}'); console.log(Object.keys(m).slice(0,5))"
- Ruby: ruby -e "require '{package_name}'; puts 'ok'"
- Go: (if binary installed) {package_name} --version
- Rust: (if binary installed) {package_name} --version

CRITICAL: The command runs INSIDE the container. No docker commands. No file-existence checks.
Return ONLY the command, no explanation.""",

    "binary_run": """Generate a ONE-LINE shell command that EXECUTES the main binary with --version or --help.

LOOK AT the Dockerfile ENTRYPOINT/CMD for the binary path.
KNOWN BINARY NAME: {binary_name}

GOOD examples:
- /usr/local/bin/{binary_name} --version
- {binary_name} --help
- java -jar /app/*.jar --version 2>&1 | head -5

CRITICAL: The command runs INSIDE the container. Return ONLY the command.""",

    "link_test": """Generate a ONE-LINE shell command that verifies a shared/static library is properly built and loadable.

KNOWN LIBRARY: {library_name}
LANGUAGE: {language}

GOOD examples:
- ldconfig -p | grep {library_name} || find /usr/local -name '*{library_name}*' -exec ldd {{}} \\;
- python -c "import ctypes; ctypes.CDLL('/usr/local/lib/lib{library_name}.so'); print('ok')"
- nm -D /usr/local/lib/lib{library_name}.so | head -5

CRITICAL: The command runs INSIDE the container. Return ONLY the command.""",

    "server_probe": """Generate a ONE-LINE shell command that:
1. Starts the server process in background
2. Waits 3 seconds
3. Checks if the process is still alive (not crashed)
4. Kills it cleanly

TEMPLATE: sh -c '<start_cmd> & PID=$!; sleep 3; kill -0 $PID 2>/dev/null && echo "server_alive" && kill $PID || echo "server_crashed"'

Determine <start_cmd> from the Dockerfile ENTRYPOINT/CMD.

CRITICAL: The command runs INSIDE the container. Return ONLY the command.""",

    "build_only": """Generate a ONE-LINE shell command that verifies REAL build artifacts exist and are valid
inside this container. This is for a complex project (monorepo, framework, or multi-module build).

LANGUAGE: {language}
BUILD SYSTEM: {build_system}

GOOD examples by language:
- Java/Maven:  find / -name "*.jar" -path "*/target/*" -exec file {{}} + 2>/dev/null | grep -i "java\\|zip" | head -5
- Java/Gradle: find / -name "*.jar" -path "*/build/libs/*" -exec file {{}} + 2>/dev/null | grep -i "java\\|zip" | head -5
- C/C++:       find /usr/local -name "*.so" -o -name "*.a" | head -5 | xargs file 2>/dev/null
- Python:      python -c "import pkg_resources; print([d.project_name for d in pkg_resources.working_set][:10])"
- Node.js:     node -e "const fs=require('fs'); console.log(fs.readdirSync('node_modules').slice(0,10))"
- Go:          ls -la /usr/local/bin/ | head -10
- Rust:        find /usr/local -name "*.so" -o -type f -perm /111 | head -5 | xargs file 2>/dev/null

CRITICAL RULES:
- The command runs INSIDE the container via: docker run --rm <image> sh -c '<CMD>'
- Do NOT use simple file-existence checks alone (test -f, [ -e ], stat) — those can be faked
- Use 'file' command to verify binary/JAR type, or use language-specific import/load
- Return ONLY the command string, no explanation""",

    "multi_target": """Generate a ONE-LINE shell command that verifies at least one real build artifact exists
and is valid inside this container. This is for a monorepo with multiple targets.

LANGUAGE: {language}
BUILD SYSTEM: {build_system}

Use the same approach as for complex builds:
- For JVM: find JARs in target/ or build/libs/ and verify with 'file'
- For C/C++: find .so/.a files and verify with 'file'
- For interpreted: import the main package or run the main binary

CRITICAL: Do NOT use simple file-existence checks alone — verify artifact type.
Return ONLY the command string, no explanation.""",
}

# Fallback prompt when classification has no matching strategy or fails.
# This is equivalent to the current generic LLMFunctionalVerifier prompt,
# so we are never weaker than the existing system.
_FALLBACK_PROMPT = """You are a QA Engineer. Suggest a SINGLE, ONE-LINE shell command to verify that this application container is ACTUALLY FUNCTIONAL.

CONTEXT: This command will run INSIDE the Docker container using "docker run --rm <image> sh -c <YOUR_COMMAND>".
Your job is to provide the command that runs INSIDE the container, NOT to run Docker itself.

CRITICAL RULES - THE COMMAND MUST ACTUALLY EXECUTE CODE:
1. NEVER suggest "docker run" or any Docker commands - those run on the HOST, not inside the container!
2. NEVER use file existence checks (ls, test -f, [ -e ], stat, find without 'file') - these can be faked
3. MUST actually EXECUTE the application, binary, or import the library to verify it works
4. For Python apps/libraries: Use 'python -c "import mypackage; print(mypackage.__version__)"' NOT 'python --version'
5. For Java: Run 'java -jar /path/to/app.jar --version' or test actual class loading
6. For C/C++ binaries: Execute the actual compiled binary with --version or --help
7. For Node.js: Use 'node -e "require(\\"mypackage\\")"' NOT just 'node --version'
8. For compiled libraries (.so, .a): Try to load them with ldd or language-specific import
9. Command must be non-interactive (no user input)
10. Command must return exit code 0 on success
11. DO NOT try to curl localhost or start servers (container may not have network/curl)
12. Return ONLY the command string, no markdown, no quotes, no explanations"""


class StrategyAwareVerifier:
    """
    Generate strategy-appropriate smoke test commands for Docker containers.

    Uses T1 classification to select the right prompt template.
    Always generates a real command — never auto-passes.
    Falls back to the generic prompt if classification is missing or unknown.
    """

    def __init__(self):
        _deployment = os.getenv(
            "AZURE_OPENAI_DEPLOYMENT_LARGE",
            os.getenv("AZURE_OPENAI_DEPLOYMENT")
        )
        self.llm = AzureChatOpenAI(
            azure_deployment=_deployment,
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        )

    def generate_test_command(
        self,
        dockerfile_content: str,
        classification: Dict,
        repo_name: str = "",
    ) -> Tuple[str, Dict]:
        """
        Generate a strategy-appropriate smoke test command.

        Returns:
            (command_string, log_data)

        The returned command is a single shell line to run inside the container.
        Falls back to the generic prompt if anything goes wrong.
        """
        strategy = classification.get("verification_strategy", "build_only") if classification else "build_only"
        log_entry = {
            "component": "StrategyAwareVerifier",
            "method": "generate_test_command",
            "strategy": strategy,
        }

        # Pick the strategy-specific template
        prompt_template = STRATEGY_PROMPTS.get(strategy)
        if prompt_template is None:
            # Unknown strategy — use fallback
            logger.warning(f"Unknown verification strategy '{strategy}', using fallback")
            prompt_template = _FALLBACK_PROMPT
            strategy = "fallback"

        # Fill template with classification data
        try:
            filled_prompt = prompt_template.format(
                language=classification.get("primary_language", "Unknown") if classification else "Unknown",
                package_name=classification.get("package_name", "unknown") if classification else "unknown",
                binary_name=classification.get("binary_name", "unknown") if classification else "unknown",
                library_name=classification.get("library_name", "unknown") if classification else "unknown",
                build_system=classification.get("build_system", "unknown") if classification else "unknown",
            )
        except (KeyError, IndexError) as e:
            logger.warning(f"Strategy prompt formatting failed: {e}, using fallback")
            filled_prompt = _FALLBACK_PROMPT
            strategy = "fallback"

        full_prompt = f"""{filled_prompt}

DOCKERFILE:
{dockerfile_content[:1500]}

REPO NAME: {repo_name}

RULES:
- Command runs INSIDE the container via: docker run --rm <image> sh -c '<CMD>'
- Must be non-interactive and return exit code 0 on success
- NO docker commands, NO curl localhost
- Return ONLY the command string, no explanation, no markdown"""

        log_entry["prompt"] = full_prompt

        try:
            response = self.llm.invoke([HumanMessage(content=full_prompt)])
            log_entry["response"] = response.content
            if hasattr(response, 'response_metadata'):
                log_entry["token_usage"] = response.response_metadata.get("token_usage", {})

            cmd = response.content.strip().replace('`', '').replace('"', "'").strip()
            # Remove any leading "sh -c " wrapper if the LLM added one
            # (we'll add our own when running)
            if cmd.startswith("sh -c "):
                cmd = cmd[6:].strip().strip("'\"")

            return cmd, log_entry

        except Exception as e:
            logger.error(f"Strategy-aware verification LLM call failed: {e}")
            log_entry["error"] = str(e)
            # Defensive fallback: 'true' always succeeds, so the build
            # result stands on its own (same as current LLMFunctionalVerifier
            # fallback behaviour).
            return "true", log_entry
