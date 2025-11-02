#!/usr/bin/env python3
"""
Example usage of the Planner Agent
Demonstrates how to use the planner to generate installation guides
"""

import os
import sys
from planner_agent import PlannerAgent, InstallationGuide

def example_usage():
    """
    Example showing how to use the Planner Agent programmatically
    """
    print("="*70)
    print("Planner Agent - Example Usage")
    print("="*70)
    
    # Check for API key (OpenAI or Azure OpenAI)
    has_azure = os.getenv("AZURE_OPENAI_API_KEY")
    has_openai = os.getenv("OPENAI_API_KEY")
    
    if not (has_azure or has_openai):
        print("\n[!] Note: This example requires either OpenAI or Azure OpenAI credentials.")
        print("For OpenAI: export OPENAI_API_KEY='your-key'")
        print("For Azure OpenAI: export AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_DEPLOYMENT")
        print("\nWithout the API key, you can still see the code structure,")
        print("but the LLM-based guide generation won't work.")
        return
    
    try:
        # Initialize planner
        print("\n[-] Initializing Planner Agent...")
        if has_azure:
            print("[*] Using Azure OpenAI")
            planner = PlannerAgent(use_azure=True, temperature=0.3)
        else:
            print("[*] Using standard OpenAI")
            planner = PlannerAgent(model_name="gpt-3.5-turbo", temperature=0.3)
        print("[+] Planner initialized")
        
        # Example repositories to analyze
        examples = [
            {
                "name": "Python Requests Library",
                "url": "https://github.com/psf/requests",
                "language": "Python"
            },
            {
                "name": "Flask Web Framework",
                "url": "https://github.com/pallets/flask",
                "language": "Python"
            },
            {
                "name": "Express.js",
                "url": "https://github.com/expressjs/express",
                "language": "JavaScript"
            }
        ]
        
        # Analyze first example
        example = examples[0]
        print(f"\n[?] Analyzing: {example['name']}")
        print(f"   URL: {example['url']}")
        
        # Generate installation guide
        print("\n[>] Generating installation guide...")
        guide = planner.generate_installation_guide(example['url'])
        
        # Display results
        print("\n" + "="*70)
        print("INSTALLATION GUIDE")
        print("="*70)
        print(f"\nProject: {guide.project_info.repo_owner}/{guide.project_info.repo_name}")
        print(f"Language: {guide.project_info.language}")
        print(f"Stars: {guide.project_info.stars}")
        print(f"Generated at: {guide.generated_at}")
        
        print("\n" + "-"*70)
        print("PREREQUISITES")
        print("-"*70)
        for i, prereq in enumerate(guide.prerequisites[:5], 1):
            print(f"{i}. {prereq}")
        
        print("\n" + "-"*70)
        print("INSTALLATION STEPS")
        print("-"*70)
        for i, step in enumerate(guide.installation_steps[:5], 1):
            print(f"\n{i}. {step['step']}")
            if step.get('command'):
                print(f"   Command: {step['command']}")
        
        print("\n" + "-"*70)
        print("BUILD COMMANDS")
        print("-"*70)
        for i, cmd in enumerate(guide.build_commands[:5], 1):
            print(f"{i}. {cmd}")
        
        if guide.verification_steps:
            print("\n" + "-"*70)
            print("VERIFICATION")
            print("-"*70)
            for i, verify in enumerate(guide.verification_steps[:3], 1):
                print(f"{i}. {verify}")
        
        if guide.troubleshooting:
            print("\n" + "-"*70)
            print("TROUBLESHOOTING")
            print("-"*70)
            for i, trouble in enumerate(guide.troubleshooting[:3], 1):
                print(f"{i}. {trouble}")
        
        print("\n" + "="*70)
        print("[+] Example completed successfully!")
        print("="*70)
        
        # Show how to use in a Builder Agent
        print("\n[*] Next Steps: Using the guide with Builder Agent")
        print("The InstallationGuide object can be passed to a Builder Agent")
        print("which would execute these steps automatically.")
        
    except ValueError as e:
        print(f"\n[!] Configuration error: {e}")
    except Exception as e:
        print(f"\n[!] Error: {e}")
        import traceback
        traceback.print_exc()

def demo_api_calls():
    """
    Demonstrate the API calls without requiring OpenAI key
    """
    print("\n" + "="*70)
    print("Demonstration: GitHub API Integration")
    print("="*70)
    
    import requests
    
    # This doesn't require an API key
    test_url = "https://github.com/psf/requests"
    
    try:
        # Parse repository
        parts = test_url.replace("https://github.com/", "").split("/")
        owner, repo = parts[0], parts[1]
        
        print(f"\nFetching info for: {owner}/{repo}")
        
        # Fetch from GitHub API
        api_url = f"https://api.github.com/repos/{owner}/{repo}"
        response = requests.get(api_url)
        response.raise_for_status()
        data = response.json()
        
        print("[+] Repository found")
        print(f"  Language: {data.get('language', 'Unknown')}")
        print(f"  Stars: {data.get('stargazers_count', 0):,}")
        print(f"  Description: {data.get('description', 'No description')}")
        
        # Fetch README
        print(f"\nFetching README...")
        readme_url = f"https://api.github.com/repos/{owner}/{repo}/readme"
        readme_response = requests.get(readme_url)
        if readme_response.status_code == 200:
            print("[+] README found")
        else:
            print("[i] No README found")
        
        print("\n" + "="*70)
        print("[+] API demonstration completed")
        print("="*70)
        print("\nThis shows how the Planner Agent fetches repository information")
        print("before generating the installation guide.")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # Run API demo (no API key needed)
    demo_api_calls()
    
    # Run full example (requires OpenAI API key)
    print("\n" + "="*70)
    print("Running full example (requires OpenAI API key)")
    print("="*70)
    example_usage()

