#!/usr/bin/env python3
"""
Simple test script for the Planner Agent
"""

import os
import sys
from planner_agent import PlannerAgent, InstallationGuide

def test_planner():
    """Test the planner agent with a simple repository"""
    
    print("="*70)
    print("Testing Planner Agent")
    print("="*70)
    
    # Check for API key (OpenAI or Azure OpenAI)
    has_azure = os.getenv("AZURE_OPENAI_API_KEY")
    has_openai = os.getenv("OPENAI_API_KEY")
    
    if not (has_azure or has_openai):
        print("ERROR: No API credentials found")
        print("For OpenAI: export OPENAI_API_KEY='your-key'")
        print("For Azure OpenAI: export AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_DEPLOYMENT")
        return False
    
    try:
        # Initialize planner
        print("\n1. Initializing planner agent...")
        if has_azure:
            print("   [*] Using Azure OpenAI")
            planner = PlannerAgent(use_azure=True)
        else:
            print("   [*] Using standard OpenAI")
            planner = PlannerAgent()
        print("   [+] Planner initialized successfully")
        
        # Test with a simple Python repository
        test_repo = "https://github.com/psf/requests"
        print(f"\n2. Analyzing repository: {test_repo}")
        
        # Generate installation guide
        guide = planner.generate_installation_guide(test_repo)
        print("   [+] Installation guide generated")
        
        # Display results
        print("\n" + "="*70)
        print("GENERATED INSTALLATION GUIDE")
        print("="*70)
        
        print(f"\nProject: {guide.project_info.repo_owner}/{guide.project_info.repo_name}")
        print(f"Language: {guide.project_info.language}")
        print(f"Stars: {guide.project_info.stars}")
        
        print("\nPrerequisites:")
        for i, prereq in enumerate(guide.prerequisites[:5], 1):
            print(f"  {i}. {prereq}")
        
        print("\nInstallation Steps:")
        for i, step in enumerate(guide.installation_steps[:5], 1):
            print(f"  {i}. {step['step']}")
            if step.get('command'):
                print(f"     Command: {step['command']}")
        
        print("\nBuild Commands:")
        for i, cmd in enumerate(guide.build_commands[:5], 1):
            print(f"  {i}. {cmd}")
        
        print("\nVerification Steps:")
        for i, verify in enumerate(guide.verification_steps[:3], 1):
            print(f"  {i}. {verify}")
        
        if guide.troubleshooting:
            print("\nTroubleshooting:")
            for i, trouble in enumerate(guide.troubleshooting[:3], 1):
                print(f"  {i}. {trouble}")
        
        print("\n" + "="*70)
        print("[+] Test completed successfully!")
        print("="*70)
        
        return True
        
    except Exception as e:
        print(f"\n[!] Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_planner()
    sys.exit(0 if success else 1)

