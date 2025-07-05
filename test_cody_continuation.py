#!/usr/bin/env python3
"""Test script for Cody provider continuation feature."""

import os
import json
import litellm
from openhands.core.config import LLMConfig
from openhands.llm import LLM
from openhands.core.logger import openhands_logger as logger

# Enable debug logging
logger.setLevel("DEBUG")

def test_continuation():
    """Test the continuation feature with a low max_tokens."""
    
    # No configuration needed - continuation is always enabled
    
    # Configure LLM
    config = LLMConfig(
        model="cody/anthropic::2024-10-22::claude-3-5-sonnet-latest",
        api_key=os.getenv("LLM_API_KEY", "test-key"),
        base_url=os.getenv("LLM_BASE_URL", "https://sourcegraph.com"),
        max_output_tokens=50,  # Very low to trigger truncation
    )
    
    llm = LLM(config)
    
    # Test 1: Simple text completion
    print("\n=== Test 1: Simple Text Completion ===")
    messages = [
        {"role": "user", "content": "Write a Python function to calculate factorial of a number with detailed comments."}
    ]
    
    response = llm.completion(messages=messages)
    print(f"Response finish_reason: {response.choices[0].finish_reason}")
    print(f"Response content length: {len(response.choices[0].message.content or '')}")
    print(f"Response content:\n{response.choices[0].message.content}")
    
    # Test 2: Function calling
    print("\n\n=== Test 2: Function Calling ===")
    messages = [
        {"role": "user", "content": "Create a bash script that lists all Python files in the current directory and counts them."}
    ]
    
    tools = [{
        "type": "function",
        "function": {
            "name": "execute_bash",
            "description": "Execute a bash command",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The bash command to execute"
                    }
                },
                "required": ["command"]
            }
        }
    }]
    
    response = llm.completion(messages=messages, tools=tools)
    print(f"Response finish_reason: {response.choices[0].finish_reason}")
    
    if hasattr(response.choices[0].message, 'tool_calls') and response.choices[0].message.tool_calls:
        print(f"Tool calls found: {len(response.choices[0].message.tool_calls)}")
        for i, tool_call in enumerate(response.choices[0].message.tool_calls):
            print(f"\nTool call {i + 1}:")
            print(f"  Function: {tool_call.function.name}")
            try:
                args = json.loads(tool_call.function.arguments)
                print(f"  Arguments: {json.dumps(args, indent=2)}")
            except json.JSONDecodeError as e:
                print(f"  Arguments (raw): {tool_call.function.arguments}")
                print(f"  JSON Error: {e}")
    else:
        print("No tool calls in response")
        if response.choices[0].message.content:
            print(f"Response content:\n{response.choices[0].message.content}")


if __name__ == "__main__":
    test_continuation()