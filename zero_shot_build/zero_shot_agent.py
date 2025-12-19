import os
import sys
import json
import subprocess
import argparse
import shutil
from typing import Optional, List, Dict
from litellm import completion
import litellm
from dotenv import load_dotenv

class DockerBuildTester:
    def __init__(self, repo_path: str):
        self.repo_path = repo_path

    def run_build(self, dockerfile_path: Optional[str] = None) -> Dict:
        print(f"--- Starting Docker Build in {self.repo_path} ---")
        try:
            # Use provided dockerfile_path or default to repo_path/Dockerfile
            if not dockerfile_path:
                dockerfile_path = os.path.join(self.repo_path, "Dockerfile")
            
            if not os.path.exists(dockerfile_path):
                return {"success": False, "error": f"Dockerfile not found at {dockerfile_path}"}

            # Define image name (sanitized)
            image_name = "zero_shot_build_test"
            
            # Run docker build. We use --no-cache to ensure a clean build.
            # We use -f to specify the generated Dockerfile location.
            cmd = ["docker", "build", "--no-cache", "-t", image_name, "-f", dockerfile_path, "."]
            
            process = subprocess.Popen(
                cmd,
                cwd=self.repo_path,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True
            )

            full_output = []
            while True:
                line = process.stdout.readline()
                if not line:
                    break
                print(line.strip())
                full_output.append(line)

            process.wait()
            success = process.returncode == 0
            
            result = {
                "success": success,
                "exit_code": process.returncode,
                "output": "".join(full_output)
            }
            
            with open(os.path.join(self.repo_path, "build.log"), "w") as f:
                f.write(result["output"])
                
            return result
        except Exception as e:
            return {"success": False, "error": str(e)}

class BaselineAgent:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.one_shot_example = {
            "readme": "# Sample Python App\nA simple Flask application that serves a hello world message.",
            "tree": ".\n├── app.py\n├── requirements.txt\n└── README.md",
            "dockerfile": "FROM python:3.9-slim\nWORKDIR /app\nCOPY requirements.txt .\nRUN pip install --no-cache-dir -r requirements.txt\nCOPY . .\nEXPOSE 5000\nCMD [\"python\", \"app.py\"]"
        }

    def get_repo_root(self) -> str:
        return subprocess.check_output(['git', 'rev-parse', '--show-toplevel'], text=True).strip()

    def get_readme(self, repo_path: str) -> str:
        readme_path = os.path.join(repo_path, "README.md")
        if os.path.exists(readme_path):
            with open(readme_path, "r", encoding="utf-8") as f:
                return f.read()
        return "No README.md found."

    def get_file_tree(self, repo_path: str, max_depth: int = 3) -> str:
        tree = []
        for root, dirs, files in os.walk(repo_path):
            dirs[:] = [d for d in dirs if d not in ['.git', '__pycache__', 'node_modules', '.next', 'venv', 'target', 'build']]
            level = root.replace(repo_path, '').count(os.sep)
            if level >= max_depth:
                continue
            indent = ' ' * 4 * level
            tree.append(f"{indent}{os.path.basename(root) or '.'}/")
            sub_indent = ' ' * 4 * (level + 1)
            for f in files:
                if f not in ['Dockerfile', 'build.log', 'docker_build_results.json']:
                    tree.append(f"{sub_indent}{f}")
        return "\n".join(tree)

    def _construct_prompt(self, mode: str, readme: str, tree: str) -> List[Dict]:
        is_one_shot = "one" in mode
        include_readme = "readme" in mode or "combined" in mode
        include_tree = "tree" in mode or "combined" in mode
        
        system_msg = "You are an expert DevOps engineer specializing in Docker containerization."
        content = []

        if is_one_shot:
            content.append("### EXAMPLE TASK ###")
            example_input = []
            if include_readme: example_input.append(f"README:\n{self.one_shot_example['readme']}")
            if include_tree: example_input.append(f"FILE TREE:\n{self.one_shot_example['tree']}")
            content.append("\n".join(example_input))
            content.append(f"### EXAMPLE OUTPUT DOCKERFILE ###\n{self.one_shot_example['dockerfile']}")
            content.append("\n--- END OF EXAMPLE ---\n")

        content.append("### ACTUAL TASK ###")
        actual_input = []
        if include_readme: actual_input.append(f"PROJECT README:\n{readme}")
        if include_tree: actual_input.append(f"PROJECT FILE TREE:\n{tree}")
        content.append("\n".join(actual_input))
        
        content.append("\nINSTRUCTIONS: Analyze the project requirements above. First, think step by step about the appropriate base image, necessary dependencies, and the build/run commands. Then, provide the final Dockerfile content starting with '```dockerfile'.")
        
        return [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": "\n".join(content)}
        ]

    def generate_dockerfile(self, mode: str, repo_path: str) -> str:
        readme = self.get_readme(repo_path)
        tree = self.get_file_tree(repo_path)
        
        messages = self._construct_prompt(mode, readme, tree)
        
        print(f"--- Generating Dockerfile using mode: {mode} ---")
        response = completion(
            model=self.model_name,
            messages=messages,
        )
        
        full_response = response.choices[0].message.content
        
        if "```dockerfile" in full_response:
            dockerfile_content = full_response.split("```dockerfile")[1].split("```")[0].strip()
        elif "```" in full_response:
            dockerfile_content = full_response.split("```")[1].split("```")[0].strip()
        else:
            dockerfile_content = full_response.strip()
            
        return dockerfile_content

    def save_dockerfile(self, content: str, repo_path: str):
        with open(os.path.join(repo_path, "Dockerfile"), "w") as f:
            f.write(content)

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cwd = os.getcwd()
    potential_envs = [
        os.path.join(script_dir, ".env"), 
        os.path.join(cwd, ".env"),        
        os.path.join(script_dir, "..", ".env"), 
        os.path.join(script_dir, "..", "cra-planner-agent", ".env") 
    ]
    for env_path in potential_envs:
        if os.path.exists(env_path):
            load_dotenv(env_path)
            print(f"ℹ️ Loaded environment from: {os.path.abspath(env_path)}")
            break

    if os.getenv("AZURE_OPENAI_ENDPOINT") and not os.getenv("AZURE_OPENAI_API_BASE"):
        os.environ["AZURE_OPENAI_API_BASE"] = os.getenv("AZURE_OPENAI_ENDPOINT")
    
    default_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
    default_model = f"azure/{default_deployment}" if default_deployment else "azure/gpt-4o"
    
    parser = argparse.ArgumentParser(description="Run Baseline Docker Build Agent")
    parser.add_argument("--mode", choices=["readme_zero", "readme_one", "tree_zero", "tree_one", "combined_zero", "combined_one"], required=True)
    parser.add_argument("--model", default=default_model)
    parser.add_argument("--repo-path", help="Path to the repository to analyze.")
    parser.add_argument("--output-dir", help="Directory to save the generated Dockerfile and build results.")
    args = parser.parse_args()

    if args.model.startswith("azure/"):
        base = os.getenv("AZURE_OPENAI_API_BASE")
        key = os.getenv("AZURE_OPENAI_API_KEY")
        if not base or not key:
            print(f"❌ Error: Missing Azure environment variables.")
            sys.exit(1)
        litellm.api_base = base
        litellm.api_key = key

    agent = BaselineAgent(model_name=args.model)
    repo_path = args.repo_path if args.repo_path else agent.get_repo_root()
    
    if not os.path.isdir(repo_path):
        print(f"❌ Error: {repo_path} is not a valid directory.")
        sys.exit(1)

    # Determine where to save results
    output_dir = args.output_dir if args.output_dir else repo_path
    os.makedirs(output_dir, exist_ok=True)

    # 1. Generate
    dockerfile = agent.generate_dockerfile(args.mode, repo_path)
    agent.save_dockerfile(dockerfile, output_dir)
    
    # 2. Build & Test
    # We still need to build IN the repo_path, but we can save logs to output_dir
    tester = DockerBuildTester(repo_path)
    generated_dockerfile_path = os.path.join(output_dir, "Dockerfile")
    result = tester.run_build(dockerfile_path=generated_dockerfile_path)
    
    # 3. Report Results
    result_path = os.path.join(output_dir, "docker_build_results.json")
    with open(result_path, "w") as f:
        summary = {k: v for k, v in result.items() if k != "output"}
        summary["mode"] = args.mode
        summary["model"] = args.model
        summary["repo_path"] = repo_path
        json.dump(summary, f, indent=4)
    
    # Save build log to output_dir specifically
    with open(os.path.join(output_dir, "build.log"), "w") as f:
        f.write(result.get("output", ""))
    
    if result["success"]:
        print(f"\n✅ Dockerfile generated and built successfully! (Results in {output_dir})")
    else:
        print(f"\n❌ Docker build failed. (Logs in {output_dir})")

if __name__ == "__main__":
    main()
