#!/usr/bin/env python3
"""Test script for Cody provider continuation feature."""

import logging
import os
from openhands.llm.cody_provider import CodyLLM
from litellm.utils import ModelResponse

# Set up logging to see debug messages
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def test_continuation():
    """Test if continuation works when response is truncated."""
    
    # Initialize Cody provider
    cody = CodyLLM()
    
    # Prepare test parameters
    model = "cody/claude-3-haiku"
    messages = [{
        "role": "user",
        "content": "Write a very long story about a programmer who discovers a magical computer. Include lots of details and make it at least 2000 words long."
    }]
    
    api_base = os.getenv("LLM_BASE_URL", "https://sourcegraph.com")
    api_key = os.getenv("LLM_API_KEY", "test-key")
    
    # Create model response object
    model_response = ModelResponse()
    
    # Set max_tokens to a small value to trigger truncation
    optional_params = {
        "max_tokens": 100,  # Very small to ensure truncation
        "auto_continue": True,  # Enable continuation
        "max_continuations": 3
    }
    
    print(f"Testing Cody continuation with max_tokens={optional_params['max_tokens']}")
    print(f"API Base: {api_base}")
    print(f"Model: {model}")
    print("-" * 50)
    
    try:
        # Make the request
        response = cody.completion(
            model=model,
            messages=messages,
            api_base=api_base,
            custom_prompt_dict={},
            model_response=model_response,
            print_verbose=print,
            encoding=None,
            api_key=api_key,
            logging_obj=None,
            optional_params=optional_params,
            headers={}
        )
        
        print("\nResponse received!")
        print(f"Finish reason: {response.choices[0].finish_reason}")
        print(f"Content length: {len(response.choices[0].message.content)} chars")
        print(f"First 100 chars: {response.choices[0].message.content[:100]}...")
        print(f"Last 100 chars: ...{response.choices[0].message.content[-100:]}")
        
        if hasattr(response, 'usage') and response.usage:
            print(f"\nToken usage:")
            print(f"  Prompt tokens: {response.usage.get('prompt_tokens', 'N/A')}")
            print(f"  Completion tokens: {response.usage.get('completion_tokens', 'N/A')}")
            print(f"  Total tokens: {response.usage.get('total_tokens', 'N/A')}")
        
    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_continuation()