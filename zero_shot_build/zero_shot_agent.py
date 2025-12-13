#!/usr/bin/env python3
"""
Zero-Shot Build Agent
Generated Dockerfiles based ONLY on README.md and a System Prompt.
Then attempts to build the image immediately.
"""

import os
import sys
import subprocess
import shutil
import time
import stat
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# Load environment variables
root_dir = Path(__file__).parent.parent
env_path = root_dir / "cra-planner-agent" / ".env"

if not env_path.exists():
    # Fallback to current directory or parent
    env_path = root_dir / ".env"

if env_path.exists():
    load_dotenv(env_path)
else:
    print(f"[WARNING] .env not found at {env_path}")

def setup_llm():
    """Initialize Azure OpenAI LLM."""
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")

    if not all([api_key, endpoint, deployment]):
        print("Error: Missing Azure OpenAI credentials in .env file")
        sys.exit(1)

    return AzureChatOpenAI(
        azure_deployment=deployment,
        api_key=api_key,
        azure_endpoint=endpoint,
        api_version=api_version
    )

def clone_repository(repo_url, target_dir="./temp_repo"):
    """Clone repository to a temporary directory."""
    print(f"\n[CLONE] Cloning {repo_url}...")
    
    if os.path.exists(target_dir):
        def remove_readonly(func, path, excinfo):
            os.chmod(path, stat.S_IWRITE)
            func(path)
            
        shutil.rmtree(target_dir, onexc=remove_readonly)
        
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", repo_url, target_dir],
            check=True,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace'
        )
        print(f"[OK] Cloned to {target_dir}")
        return Path(target_dir)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed to clone repository: {e.stderr}")
        sys.exit(1)

def get_readme_content(repo_path):
    """Find and read the README file."""
    for file in os.listdir(repo_path):
        if file.lower().startswith("readme"):
            print(f"[READ] Found README: {file}")
            try:
                with open(repo_path / file, 'r', encoding='utf-8', errors='ignore') as f:
                    return f.read()
            except Exception as e:
                print(f"[ERROR] Could not read README: {e}")
                sys.exit(1)
    
    print("[WARNING] No README found in repository.")
    return "No README file found in this repository."

def generate_dockerfile(llm, readme_content):
    """Generate Dockerfile using LLM."""
    print("\n[GENERATE] Generating Dockerfile using Zero-Shot prompting...")
    
    system_prompt = """You are an expert DevOps engineer. Your task is to create a production-ready Dockerfile for a project based ONLY on its README content.

RULES:
1. USE BEST PRACTICES: Multi-stage builds, specific versions, non-root users if possible.
2. INFER DEPENDENCIES: Look for language (Python, Node, Go, etc.) and package managers (pip, npm, go mod, etc.) in the text.
3. OUTPUT FORMAT: Return ONLY the Dockerfile content. No markdown code blocks, no explanations, no "Here is the file". Just the raw Dockerfile content.
4. If the README is missing or empty, try to create a generic Dockerfile for the most likely language if detectable, or a safe default.
"""

    user_message = f"Here is the README content of the project:\n\n{readme_content}\n\nGenerate the Dockerfile now."
    
    try:
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message)
        ])
        
        content = response.content.strip()
        
        # Strip markdown code blocks if the LLM ignores the rule
        if content.startswith("```"):
            lines = content.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            content = "\n".join(lines)
            
        print(f"[OK] Dockerfile generated ({len(content)} chars)")
        return content
    except Exception as e:
        print(f"[ERROR] LLM generation failed: {e}")
        sys.exit(1)

def build_docker_image(repo_path, dockerfile_content, image_tag="zero-shot-build:latest"):
    """Attempt to build the Docker image."""
    print("\n[BUILD] Attempting to build Docker image...")
    
    dockerfile_path = repo_path / "Dockerfile"
    with open(dockerfile_path, 'w', encoding='utf-8') as f:
        f.write(dockerfile_content)
    
    try:
        # Check docker availability
        subprocess.run(["docker", "--version"], check=True, capture_output=True, encoding='utf-8', errors='replace')
        
        start_time = time.time()
        result = subprocess.run(
            ["docker", "build", "-t", image_tag, "."],
            cwd=repo_path,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace'
        )
        duration = time.time() - start_time
        
        if result.returncode == 0:
            print(f"[SUCCESS] Docker build completed in {duration:.2f}s")
            print(f"[INFO] Image tag: {image_tag}")
            return True, result.stdout
        else:
            print(f"[FAILURE] Docker build failed (Exit Code: {result.returncode})")
            print(f"[ERROR LOG] Last 10 lines of output:")
            print("\n".join(result.stderr.splitlines()[-10:]))
            return False, result.stderr
            
    except FileNotFoundError:
        print("[ERROR] Docker not found. Is it installed and in your PATH?")
        return False, "Docker not found"
    except Exception as e:
        print(f"[ERROR] Build exception: {e}")
        return False, str(e)

def main():
    if len(sys.argv) < 2:
        print("Usage: python zero_shot_agent.py <repo_url>")
        sys.exit(1)
        
    repo_url = sys.argv[1]
    repo_name = repo_url.rstrip('/').split('/')[-1].replace('.git', '')
    
    # Create working directory
    script_dir = Path(__file__).parent
    work_dir = script_dir / "temp" / repo_name
    work_dir.mkdir(parents=True, exist_ok=True)
    
    # Execution Flow
    repo_path = clone_repository(repo_url, target_dir=work_dir)
    readme_content = get_readme_content(repo_path)
    
    llm = setup_llm()
    dockerfile_content = generate_dockerfile(llm, readme_content)
    
    success, log = build_docker_image(repo_path, dockerfile_content, image_tag=f"zero-shot-{repo_name.lower()}:latest")
    
    # Save results including the Dockerfile and logs
    with open(script_dir / f"{repo_name}_Dockerfile", 'w', encoding='utf-8') as f:
        f.write(dockerfile_content)
        
    print(f"\n[INFO] Saved Dockerfile to {repo_name}_Dockerfile")
    
    if not success:
        with open(script_dir / f"{repo_name}_build_error.log", 'w', encoding='utf-8') as f:
            f.write(log)
        print(f"[INFO] Saved error log to {repo_name}_build_error.log")
        
    # Cleanup
    if os.getenv("KEEP_TEMP", "false").lower() != "true":
         try:
            def remove_readonly(func, path, excinfo):
                os.chmod(path, stat.S_IWRITE)
                func(path)
            shutil.rmtree(work_dir, onexc=remove_readonly)
            print("[CLEANUP] Temporary directory removed")
         except Exception as e:
            print(f"[WARNING] Cleanup failed: {e}")

if __name__ == "__main__":
    main()
