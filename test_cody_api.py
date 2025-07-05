#!/usr/bin/env python3
"""Test Cody API endpoints to understand the service better."""

import httpx
import json

# API configuration
API_KEY = "sgp_e3cb05880bc40429_a4cdf38ba7b8aa431ddd3591655307dd67a7fb42"
BASE_URL = "https://sourcegraph.com"

def test_list_models():
    """Test the list models endpoint."""
    url = f"{BASE_URL}/.api/llm/models"
    headers = {
        "Authorization": f"token {API_KEY}",
        "X-Requested-With": "openhands 0.1.0",
        "Content-Type": "application/json"
    }
    
    try:
        response = httpx.get(url, headers=headers)
        response.raise_for_status()
        
        models = response.json()
        print("=== Available Models ===")
        print(json.dumps(models, indent=2))
        
        # Extract model IDs
        if 'data' in models:
            print("\n=== Model IDs ===")
            for model in models['data']:
                print(f"- {model['id']}")
                
    except httpx.HTTPStatusError as e:
        print(f"HTTP Error {e.response.status_code}: {e.response.text}")
    except Exception as e:
        print(f"Error: {e}")

def test_simple_completion():
    """Test a simple chat completion."""
    url = f"{BASE_URL}/.api/llm/chat/completions"
    headers = {
        "Authorization": f"token {API_KEY}",
        "X-Requested-With": "openhands 0.1.0",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": "anthropic::2023-06-01::claude-3.5-sonnet",
        "messages": [
            {"role": "user", "content": "Say hello in JSON format"}
        ],
        "max_tokens": 100
    }
    
    try:
        response = httpx.post(url, headers=headers, json=data)
        response.raise_for_status()
        
        result = response.json()
        print("\n=== Chat Completion Response ===")
        print(json.dumps(result, indent=2))
        
    except httpx.HTTPStatusError as e:
        print(f"HTTP Error {e.response.status_code}: {e.response.text}")
    except Exception as e:
        print(f"Error: {e}")

def test_function_calling():
    """Test function calling support."""
    url = f"{BASE_URL}/.api/llm/chat/completions"
    headers = {
        "Authorization": f"token {API_KEY}",
        "X-Requested-With": "openhands 0.1.0",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": "anthropic::2023-06-01::claude-3.5-sonnet",
        "messages": [
            {"role": "user", "content": "What's the weather in Tokyo?"}
        ],
        "tools": [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city name"
                        }
                    },
                    "required": ["location"]
                }
            }
        }],
        "max_tokens": 200
    }
    
    try:
        response = httpx.post(url, headers=headers, json=data)
        response.raise_for_status()
        
        result = response.json()
        print("\n=== Function Calling Response ===")
        print(json.dumps(result, indent=2))
        
    except httpx.HTTPStatusError as e:
        print(f"HTTP Error {e.response.status_code}: {e.response.text}")
        print(f"Response headers: {dict(e.response.headers)}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    print("Testing Cody API...")
    test_list_models()
    test_simple_completion()
    test_function_calling()