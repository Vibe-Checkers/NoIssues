#!/usr/bin/env python3
"""
Test Agent Script
Simple test script for testing the planner agent with basic queries.
Use this for quick testing and validation.
"""

from planner_agent import create_planner_agent


def main():
    """Run simple test queries on the agent."""
    print("="*70)
    print("Planner Agent - Simple Test Script")
    print("="*70)
    print("\nInitializing agent...\n")

    try:
        # Create the agent
        agent = create_planner_agent(max_iterations=10, verbose=True)
        print("[OK] Agent initialized successfully!\n")

        # Test queries demonstrating new capabilities
        queries = [
            "Show me a directory tree of the current directory with depth 2",
            "Find all Python files in the current directory",
            "Search for the word 'agent' in all Python files"
        ]

        print(f"Running {len(queries)} test queries...\n")

        for i, query in enumerate(queries, 1):
            print(f"\n{'='*70}")
            print(f"Query {i}/{len(queries)}: {query}")
            print('='*70)

            try:
                result = agent.invoke({"input": query})
                print(f"\n[RESULT]\n{result['output']}\n")
            except Exception as e:
                print(f"\n[ERROR] Error processing query: {e}\n")

        print("="*70)
        print("All test queries completed!")
        print("="*70)

    except Exception as e:
        print(f"\n[ERROR] Error initializing agent: {e}")
        print("\nPlease ensure:")
        print("1. You have a .env file with AZURE_OPENAI_* variables")
        print("2. Your Azure OpenAI API key is valid")
        print("3. The deployment name and endpoint are correct")


if __name__ == "__main__":
    main()
