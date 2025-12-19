import subprocess
import os
import json
import time
import sys
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

def run_single_test(mode, entry, downloads_dir, results_root, agent_script):
    """
    Runs a single agent test for a specific mode and repository.
    """
    # Handle URLs vs local paths
    if entry.startswith("http"):
        repo_name = entry.split("/")[-1].replace(".git", "")
        repo_path = os.path.join(downloads_dir, repo_name)
        if not os.path.exists(repo_path):
            # Use a lock-like check to avoid multiple clones of the same repo
            # In a simple script, we just attempt it. Git will handle concurrency mostly.
            subprocess.run(["git", "clone", "--depth", "1", entry, repo_path], capture_output=True)
    else:
        repo_path = entry
        repo_name = os.path.basename(entry)

    # Mode-specific output directory
    output_dir = os.path.join(results_root, mode, repo_name)
    os.makedirs(output_dir, exist_ok=True)

    try:
        cmd = [
            sys.executable, agent_script, 
            "--mode", mode, 
            "--repo-path", repo_path,
            "--output-dir", output_dir
        ]
        
        start_time = time.time()
        # Increase timeout for complex builds
        process = subprocess.run(cmd, capture_output=True, text=True, timeout=600) 
        duration = time.time() - start_time
        
        success = False
        error_msg = ""
        
        if process.returncode != 0:
            error_msg = f"Agent Crash: {process.stderr[-200:].strip()}"
        else:
            result_json_path = os.path.join(output_dir, "docker_build_results.json")
            if os.path.exists(result_json_path):
                with open(result_json_path, "r") as rj:
                    data = json.load(rj)
                    success = data.get("success", False)
                    error_msg = data.get("error", "Unknown error")
            else:
                error_msg = "No results.json produced"

        return {
            "mode": mode,
            "repo": entry,
            "repo_name": repo_name,
            "success": success,
            "duration": duration,
            "error": error_msg
        }
                
    except Exception as e:
        return {
            "mode": mode,
            "repo": entry,
            "repo_name": repo_name,
            "success": False,
            "duration": 0,
            "error": str(e)
        }

def run_benchmark():
    modes = [
        "readme_zero", "readme_one", 
        "tree_zero", "tree_one", 
        "combined_zero", "combined_one"
    ]
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_file = os.path.join(script_dir, "repositories.txt")
    
    if not os.path.exists(repo_file):
        repo_file = os.path.join(script_dir, "..", "repositories.txt")
        
    if not os.path.exists(repo_file):
        print(f"[!] File not found: repositories.txt")
        return

    with open(repo_file, "r") as f:
        repo_entries = [line.strip() for line in f if line.strip()]

    # Directories
    downloads_dir = os.path.join(script_dir, "downloads")
    results_root = os.path.join(script_dir, "results")
    agent_script = os.path.abspath("zero_shot_agent.py")
    
    os.makedirs(downloads_dir, exist_ok=True)
    os.makedirs(results_root, exist_ok=True)

    print(f"Starting Parallel Benchmark...")
    print(f"Modes: {len(modes)}")
    print(f"Repos: {len(repo_entries)}")
    print(f"Total Tasks: {len(modes) * len(repo_entries)}")
    print(f"Results will be saved to: {results_root}")
    print("-" * 40)

    all_results = []
    
    # We use ThreadPoolExecutor because the tasks involve LLM calls (network) 
    # and Docker builds (subprocesses). 
    # MAX_WORKERS set to 4 to avoid overwhelming the system with concurrent Docker builds.
    MAX_WORKERS = 4 
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []
        for mode in modes:
            for entry in repo_entries:
                futures.append(executor.submit(run_single_test, mode, entry, downloads_dir, results_root, agent_script))

        completed = 0
        total = len(futures)
        
        for future in as_completed(futures):
            res = future.result()
            all_results.append(res)
            completed += 1
            
            status = "[OK]" if res["success"] else "[FAIL]"
            print(f"[{completed}/{total}] {status} Mode: {res['mode']} | Repo: {res['repo_name']} ({res['duration']:.1f}s)")
            if not res["success"]:
                print(f"      Error: {res['error']}")

    # Process and Save Report
    report_data = {mode: {"success": 0, "total": 0, "details": []} for mode in modes}
    for res in all_results:
        mode = res["mode"]
        report_data[mode]["total"] += 1
        if res["success"]:
            report_data[mode]["success"] += 1
        report_data[mode]["details"].append(res)

    report_name = f"benchmark_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    with open(os.path.join(script_dir, report_name), "w", encoding="utf-8") as rf:
        rf.write("# Multi-Mode Dockerization Benchmark Results\n\n")
        rf.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        rf.write(f"Total Tasks: {len(all_results)}\n\n")
        
        rf.write("## Summary Table\n\n")
        rf.write("| Prompting Mode | Success Rate | Count | Avg Duration |\n")
        rf.write("| :--- | :--- | :--- | :--- |\n")
        for mode in modes:
            data = report_data[mode]
            success_rate = (data["success"] / data["total"] * 100) if data["total"] > 0 else 0
            avg_dur = sum(d["duration"] for d in data["details"]) / data["total"] if data["total"] > 0 else 0
            rf.write(f"| `{mode}` | {success_rate:.1f}% | {data['success']}/{data['total']} | {avg_dur:.1f}s |\n")
        
        rf.write("\n## Breakdown by Mode\n")
        for mode in modes:
            rf.write(f"\n### {mode}\n")
            rf.write("| Repository | Status | Duration | Error |\n")
            rf.write("| :--- | :--- | :--- | :--- |\n")
            # Sort details by repo name
            sorted_details = sorted(report_data[mode]["details"], key=lambda x: x["repo_name"])
            for detail in sorted_details:
                status = "PASS" if detail["success"] else "FAIL"
                rf.write(f"| {detail['repo_name']} | {status} | {detail['duration']:.1f}s | {detail['error']} |\n")

    # Save raw JSON for analysis
    with open(os.path.join(script_dir, "benchmark_results_full.json"), "w") as jf:
        json.dump(all_results, jf, indent=4)

    print("-" * 40)
    print(f"Report generated: {report_name}")
    print(f"Full results JSON: benchmark_results_full.json")

if __name__ == "__main__":
    run_benchmark()
