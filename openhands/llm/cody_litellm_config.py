"""
Configuration for integrating Cody provider with LiteLLM.
"""

import litellm
from openhands.llm.cody_provider import register_cody_provider

# Common Cody model configurations
CODY_MODELS = {
    # Anthropic models
    "cody/anthropic::2023-06-01::claude-3-opus": {
        "max_tokens": 4096,
        "max_input_tokens": 200000,
        "max_output_tokens": 4096,
    },
    "cody/anthropic::2023-06-01::claude-3.5-sonnet": {
        "max_tokens": 8192,
        "max_input_tokens": 200000,
        "max_output_tokens": 8192,
    },
    "cody/anthropic::2023-06-01::claude-3-haiku": {
        "max_tokens": 4096,
        "max_input_tokens": 200000,
        "max_output_tokens": 4096,
    },
    # OpenAI models
    "cody/openai::v1::gpt-4": {
        "max_tokens": 8192,
        "max_input_tokens": 128000,
        "max_output_tokens": 8192,
    },
    "cody/openai::v1::gpt-3.5-turbo": {
        "max_tokens": 4096,
        "max_input_tokens": 16385,
        "max_output_tokens": 4096,
    },
    # Generic fallback
    "cody/*": {
        "max_tokens": 4096,
        "max_input_tokens": 128000,
        "max_output_tokens": 4096,
    }
}


def initialize_cody_provider():
    """Initialize and register the Cody provider with LiteLLM."""
    
    # Register the Cody provider
    register_cody_provider()
    
    # Register Cody models with LiteLLM model cost map
    if not hasattr(litellm, 'model_cost'):
        litellm.model_cost = {}
    
    for model_name, config in CODY_MODELS.items():
        litellm.model_cost[model_name] = {
            "max_tokens": config.get("max_tokens", 4096),
            "max_input_tokens": config.get("max_input_tokens", 128000),
            "max_output_tokens": config.get("max_output_tokens", 4096),
            "input_cost_per_token": 0.0,
            "output_cost_per_token": 0.0,
            "litellm_provider": "cody",
            "mode": "chat"
        }
    
    # Initialize model alias map if it doesn't exist
    if not hasattr(litellm, 'model_alias_map'):
        litellm.model_alias_map = {}
        
    # Set up simple model aliases
    litellm.model_alias_map.update({
        "cody-opus": "cody/anthropic::2023-06-01::claude-3-opus",
        "cody-sonnet": "cody/anthropic::2023-06-01::claude-3.5-sonnet",
        "cody-haiku": "cody/anthropic::2023-06-01::claude-3-haiku",
        "cody-gpt4": "cody/openai::v1::gpt-4",
        "cody-gpt35": "cody/openai::v1::gpt-3.5-turbo",
    })
    
    return True


# Auto-initialize when imported
initialize_cody_provider()