#!/usr/bin/env python3
"""
Run Agent Script
Full workflow script that clones GitHub repositories and performs analysis.
Use this for complete repository analysis tasks.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

from planner_agent import create_planner_agent


def clone_repository(repo_url: str, target_dir: str = "./temp") -> str:
    """
    Clone a GitHub repository to a local directory.

    Args:
        repo_url: GitHub repository URL
        target_dir: Directory to clone repositories into

    Returns:
        Path to the cloned repository
    """
    print(f"\n[CLONE] Cloning repository: {repo_url}")

    # Create target directory if it doesn't exist
    Path(target_dir).mkdir(parents=True, exist_ok=True)

    # Extract repo name from URL
    repo_name = repo_url.rstrip('/').split('/')[-1].replace('.git', '')
    clone_path = os.path.join(target_dir, repo_name)

    # Remove existing clone if it exists
    if os.path.exists(clone_path):
        print(f"[WARNING] Repository already exists at {clone_path}")
        response = input("Remove and re-clone? (y/N): ").strip().lower()
        if response == 'y':
            print(f"[DELETE] Removing existing repository...")
            shutil.rmtree(clone_path)
        else:
            print(f"[OK] Using existing repository at {clone_path}")
            return clone_path

    # Clone the repository
    try:
        print(f"[CLONING] Cloning to {clone_path}...")
        result = subprocess.run(
            ["git", "clone", repo_url, clone_path],
            capture_output=True,
            text=True,
            check=True
        )
        print(f"[SUCCESS] Repository cloned successfully!")
        return clone_path

    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Error cloning repository: {e.stderr}")
        sys.exit(1)


def analyze_repository(agent, repo_path: str, repo_name: str, callback_handler):
    """
    Run analysis queries on a cloned repository.

    Args:
        agent: The planner agent instance
        repo_path: Path to the cloned repository
        repo_name: Name of the repository
        callback_handler: FormattedOutputHandler instance for token tracking
    """
    print(f"\n{'='*70}")
    print(f"Analyzing Repository: {repo_name}")
    print('='*70)

    # Define analysis queries using discovery-based approach with relative paths
    queries = [
        "Analyze the repository. First show me the directory tree structure (depth 2), then identify what type of project this is by finding and examining configuration files.",

        "Based on what you discovered in the previous step, find all build-related configuration files and extract the key information like dependencies and build scripts.",

        "Based on everything you've learned so far, read the README file and extract installation/build instructions. Also search for any 'install' or 'build' commands mentioned in configuration files or scripts.",

        "Now create a comprehensive step-by-step build instruction document that includes: 1) Project type and language, 2) Prerequisites, 3) Installation commands, 4) Build commands, 5) How to run/test. Use all the information you've gathered in previous steps."
    ]

    # Initialize conversation history to maintain context across queries
    chat_history = []
    final_instructions = None
    total_tokens = {"input": 0, "output": 0, "total": 0}
    tool_usage = {}  # Track how many times each tool is used

    for i, query in enumerate(queries, 1):
        print(f"\n{'-'*70}")
        print(f"Analysis Step {i}/{len(queries)}")
        print(f"{'-'*70}")
        print(f"Query: {query}\n")

        try:
            # Invoke agent with accumulated chat history for context continuity
            result = agent.invoke({
                "input": query,
                "chat_history": chat_history
            })

            output = result['output']
            print(f"\n[RESULT]\n{output}\n")

            # Track tool usage if intermediate steps are available
            if 'intermediate_steps' in result:
                for action, _ in result['intermediate_steps']:
                    tool_name = action.tool
                    tool_usage[tool_name] = tool_usage.get(tool_name, 0) + 1

            # Track token usage if available in result - try multiple possible keys
            usage_found = False

            # Try different possible locations for usage data
            for key in ['usage_metadata', 'usage', 'token_usage', 'llm_output']:
                if key in result and result[key]:
                    usage = result[key]
                    if isinstance(usage, dict):
                        # Try different token key names
                        input_tokens = usage.get('input_tokens') or usage.get('prompt_tokens') or usage.get('total_input_tokens', 0)
                        output_tokens = usage.get('output_tokens') or usage.get('completion_tokens') or usage.get('total_output_tokens', 0)
                        total = usage.get('total_tokens', input_tokens + output_tokens)

                        if input_tokens or output_tokens:
                            total_tokens["input"] += input_tokens
                            total_tokens["output"] += output_tokens
                            total_tokens["total"] += total
                            usage_found = True
                            break

            # Debug: print available keys on first iteration if no usage found
            if i == 1 and not usage_found:
                print(f"[DEBUG] Result keys available: {list(result.keys())}")

            # Save the final step output (build instructions)
            if i == len(queries):
                final_instructions = output

            # Add this interaction to chat history for next query
            chat_history.extend([
                {"role": "user", "content": query},
                {"role": "assistant", "content": output}
            ])

        except Exception as e:
            print(f"\n[ERROR] {e}\n")

    # Save final build instructions to markdown file
    if final_instructions:
        output_file = f"{repo_name}_BUILD_INSTRUCTIONS.md"
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(f"# Build Instructions: {repo_name}\n\n")
                f.write(f"*Generated by Planner Agent*\n\n")
                f.write("---\n\n")
                f.write(final_instructions)

            # Get absolute path for clearer output
            abs_path = os.path.abspath(output_file)
            print(f"\n{'='*70}")
            print("Build Instructions Saved")
            print('='*70)
            print(f"File: {output_file}")
            print(f"Path: {abs_path}")
            print('='*70)
        except Exception as e:
            print(f"\n[ERROR] Failed to save build instructions: {e}")
    else:
        print(f"\n[WARNING] No final instructions to save (all queries may have failed)")

    # Print tool usage report
    print(f"\n{'='*70}")
    print("Tool Usage Report")
    print('='*70)
    if tool_usage:
        # Sort by usage count (descending)
        sorted_tools = sorted(tool_usage.items(), key=lambda x: x[1], reverse=True)
        total_tool_calls = sum(tool_usage.values())

        for tool_name, count in sorted_tools:
            percentage = (count / total_tool_calls * 100) if total_tool_calls > 0 else 0
            print(f"{tool_name:20s} : {count:3d} calls ({percentage:5.1f}%)")

        print(f"{'-'*70}")
        print(f"{'Total tool calls':20s} : {total_tool_calls:3d}")
    else:
        print("[WARNING] Tool usage information not available")
    print('='*70)

    # Print token usage summary
    print(f"\n{'='*70}")
    print("Token Usage Summary")
    print('='*70)

    # Get token usage from callback handler
    if callback_handler and callback_handler.token_usage["total"] > 0:
        usage = callback_handler.token_usage
        print(f"Input tokens:  {usage['input']:,}")
        print(f"Output tokens: {usage['output']:,}")
        print(f"Total tokens:  {usage['total']:,}")
    elif total_tokens["total"] > 0:
        print(f"Input tokens:  {total_tokens['input']:,}")
        print(f"Output tokens: {total_tokens['output']:,}")
        print(f"Total tokens:  {total_tokens['total']:,}")
    else:
        print("[WARNING] Token usage information not available")
    print('='*70)


def main():
    """Main function to run repository analysis workflow."""
    print("="*70)
    print("Planner Agent - Repository Analysis Workflow")
    print("="*70)

    # Check if repository URL is provided
    if len(sys.argv) < 2:
        print("\nUsage: python run_agent.py <github_repo_url>")
        print("\nExample:")
        print("  python run_agent.py https://github.com/psf/requests")
        print("  python run_agent.py https://github.com/microsoft/playwright")
        sys.exit(1)

    repo_url = sys.argv[1]

    # Validate GitHub URL
    if "github.com" not in repo_url:
        print(f"\n[ERROR] Invalid GitHub URL: {repo_url}")
        print("Please provide a valid GitHub repository URL")
        sys.exit(1)

    try:
        # Step 1: Clone the repository
        print("\n" + "="*70)
        print("Step 1: Cloning Repository")
        print("="*70)
        repo_path = clone_repository(repo_url)
        repo_name = os.path.basename(repo_path)

        # Step 2: Initialize the agent
        print("\n" + "="*70)
        print("Step 2: Initializing Agent")
        print("="*70)
        agent, callback_handler = create_planner_agent(max_iterations=25, verbose=True, repository_path=repo_path)
        print("\n[OK] Agent initialized successfully!")

        # Step 3: Analyze the repository
        print("\n" + "="*70)
        print("Step 3: Running Analysis")
        print("="*70)
        analyze_repository(agent, repo_path, repo_name, callback_handler)

        # Summary
        print("\n" + "="*70)
        print("Analysis Complete!")
        print("="*70)
        print(f"\n[OK] Repository: {repo_name}")
        print(f"[OK] Location: {repo_path}")
        print(f"\nYou can now manually explore the repository at: {repo_path}")
        print("="*70)

    except KeyboardInterrupt:
        print("\n\n[WARNING] Analysis interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n[ERROR] {e}")
        print("\nPlease ensure:")
        print("1. Git is installed and accessible")
        print("2. You have a .env file with AZURE_OPENAI_* variables")
        print("3. Your Azure OpenAI API key is valid")
        print("4. You have internet connectivity")
        sys.exit(1)


if __name__ == "__main__":
    main()
