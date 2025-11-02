#!/usr/bin/env python3
"""
Simple test script for Azure OpenAI endpoint
Tests basic connectivity and chat completion
"""

import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_azure_openai():
    """Test Azure OpenAI endpoint with a simple chat completion request."""

    # Get configuration from environment
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")

    print("Configuration:")
    print(f"  Endpoint: {endpoint}")
    print(f"  Deployment: {deployment}")
    print(f"  API Key: {'*' * 10}{api_key[-4:] if api_key else 'NOT SET'}")
    print()

    if not all([api_key, endpoint, deployment]):
        print("Error: Missing required environment variables")
        print("Please ensure AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, and AZURE_OPENAI_DEPLOYMENT are set")
        return

    # Construct the full URL
    # Azure OpenAI format: {endpoint}openai/deployments/{deployment}/chat/completions?api-version=2024-02-15-preview
    url = f"{endpoint.rstrip('/')}/openai/deployments/{deployment}/chat/completions?api-version=2024-02-15-preview"

    print(f"Testing URL: {url}")
    print()

    # Prepare headers
    headers = {
        "Content-Type": "application/json",
        "api-key": api_key
    }

    # Prepare request body
    data = {
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": "Say 'Hello! The connection is working.' if you can read this."
            }
        ],
        "max_completion_tokens": 100
    }

    try:
        print("Sending request...")
        response = requests.post(url, headers=headers, json=data, timeout=30)

        print(f"Status Code: {response.status_code}")
        print()

        if response.status_code == 200:
            result = response.json()
            print("SUCCESS!")
            print()
            print("Response:")
            print(f"  Model: {result.get('model', 'N/A')}")
            message_content = result['choices'][0]['message']['content']
            print(f"  Message: {message_content if message_content else '(empty response)'}")
            print(f"  Finish reason: {result['choices'][0].get('finish_reason', 'N/A')}")
            print(f"  Tokens used: {result.get('usage', {}).get('total_tokens', 'N/A')}")
            print()
            print("Full response data:")
            import json
            print(json.dumps(result, indent=2))
        else:
            print("FAILED!")
            print()
            print("Error Response:")
            try:
                error_data = response.json()
                print(f"  {error_data}")
            except:
                print(f"  {response.text}")

    except requests.exceptions.Timeout:
        print("ERROR: Request timed out")
    except requests.exceptions.RequestException as e:
        print(f"ERROR: Request failed - {e}")
    except Exception as e:
        print(f"ERROR: Unexpected error - {e}")

if __name__ == "__main__":
    print("=" * 60)
    print("Azure OpenAI Endpoint Test")
    print("=" * 60)
    print()

    test_azure_openai()

    print()
    print("=" * 60)
