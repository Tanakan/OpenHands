"""
Sourcegraph Cody LLM provider integration.

This module provides configuration and setup for using Cody with OpenHands.
"""

from typing import Dict, Any

from openhands.llm.cody_provider import register_cody_provider


# Available Cody models (based on API response)
CODY_MODELS = [
    # Claude Sonnet 4 (Latest)
    'cody/anthropic::2024-10-22::claude-sonnet-4-latest',  # Claude Sonnet 4
    'cody/anthropic::2024-10-22::claude-sonnet-4-thinking-latest',  # Claude Sonnet 4 with Thinking (Pro tier)
    
    # Claude 3.7 Sonnet
    'cody/anthropic::2024-10-22::claude-3-7-sonnet-latest',  # Claude 3.7 Sonnet
    'cody/anthropic::2024-10-22::claude-3-7-sonnet-extended-thinking',  # Claude 3.7 Sonnet with Thinking (Pro tier)
    
    # Claude 3.5 Sonnet (Note: Different API versions)
    'cody/anthropic::2023-06-01::claude-3.5-sonnet',  # Claude 3.5 Sonnet
    'cody/anthropic::2023-06-01::claude-3-5-sonnet-20240620',  # Claude 3.5 Sonnet (specific version)
    
    # Claude 3.5 Haiku
    'cody/anthropic::2024-10-22::claude-3-5-haiku-latest',  # Claude 3.5 Haiku
    'cody/anthropic::2023-06-01::claude-3-haiku',  # Claude 3 Haiku
    
    # Claude 3 Opus
    'cody/anthropic::2023-06-01::claude-3-opus',  # Claude 3 Opus
    
    # Claude 3 Sonnet
    'cody/anthropic::2023-06-01::claude-3-sonnet',  # Claude 3 Sonnet
    
    # Claude 2
    'cody/anthropic::2023-01-01::claude-2.1',  # Claude 2.1
    'cody/anthropic::2023-01-01::claude-2.0',  # Claude 2.0
    
    # Google Gemini models
    'cody/google::v1::gemini-2.0-flash',  # Gemini 2.0 Flash
    'cody/google::v1::gemini-2.0-flash-lite',  # Gemini 2.0 Flash-Lite
    'cody/google::v1::gemini-2.0-flash-exp',  # Gemini 2.0 Flash Experimental
    'cody/google::v1::gemini-2.5-flash-preview-04-17',  # Gemini 2.5 Flash Preview
    'cody/google::v1::gemini-1.5-pro',  # Gemini 1.5 Pro
    'cody/google::v1::gemini-1.5-pro-002',  # Gemini 1.5 Pro v002
    'cody/google::v1::gemini-1.5-flash',  # Gemini 1.5 Flash
    'cody/google::v1::gemini-1.5-flash-002',  # Gemini 1.5 Flash v002
    'cody/google::v1::gemini-2.0-pro-exp-02-05',  # Gemini 2.0 Pro Experimental
    'cody/google::v1::gemini-2.5-pro-preview-03-25',  # Gemini 2.5 Pro Preview (Pro tier)
    
    # OpenAI models
    'cody/openai::2024-02-01::gpt-4o',  # GPT-4o
    'cody/openai::2024-02-01::gpt-4.1',  # GPT-4.1
    'cody/openai::2024-02-01::gpt-4o-mini',  # GPT-4o Mini
    'cody/openai::2024-02-01::gpt-4.1-mini',  # GPT-4.1 Mini
    'cody/openai::2024-02-01::gpt-4.1-nano',  # GPT-4.1 Nano
    'cody/openai::2024-02-01::gpt-4-turbo',  # GPT-4 Turbo
    'cody/openai::2024-02-01::gpt-3.5-turbo',  # GPT-3.5 Turbo
    'cody/openai::2024-02-01::o1',  # OpenAI o1
    'cody/openai::2024-02-01::o3',  # OpenAI o3
    'cody/openai::2024-02-01::o3-mini-medium',  # OpenAI o3 Mini Medium
    'cody/openai::2024-02-01::o4-mini',  # OpenAI o4 Mini (Pro tier)
    
    # Mistral models
    'cody/mistral::v1::mixtral-8x7b-instruct',  # Mixtral 8x7B Instruct
    'cody/mistral::v1::mixtral-8x22b-instruct',  # Mixtral 8x22B Instruct
    
    # Fireworks models
    'cody/fireworks::v1::deepseek-v3',  # Deepseek v3
    'cody/fireworks::v1::deepseek-coder-v2-lite-base',  # Deepseek Coder v2 Lite
    'cody/fireworks::v1::starcoder',  # StarCoder
]


def is_cody_model(model_name: str) -> bool:
    """Check if a model name is a Cody model."""
    return model_name.startswith('cody/')


def get_cody_config(base_config: Dict[str, Any]) -> Dict[str, Any]:
    """Get configuration for Cody provider."""
    from openhands.core.logger import openhands_logger as logger
    
    model = base_config.get('model', '')
    api_key = base_config.get('api_key')
    base_url = base_config.get('base_url', '')
    
    # Debug logging
    logger.debug(f'[Cody] get_cody_config called with:')
    logger.debug(f'[Cody]   model: {model}')
    logger.debug(f'[Cody]   base_url: {base_url!r}')
    logger.debug(f'[Cody]   api_key present: {bool(api_key)}')
    
    # Handle SecretStr objects
    if hasattr(api_key, 'get_secret_value'):
        api_key_value = api_key.get_secret_value()
    else:
        api_key_value = str(api_key) if api_key else ''
    
    # Ensure the provider is registered
    register_cody_provider()
    
    # Determine api_base value
    api_base = base_url if base_url else None
    logger.debug(f'[Cody] Resolved api_base: {api_base!r}')
    
    # Return configuration for LiteLLM
    config = {
        'model': model,  # Keep the full model name with cody/ prefix
        'api_key': api_key_value,
        'api_base': api_base,  # Pass None if empty to let provider validate
        'custom_llm_provider': 'cody',  # Use our custom provider
    }
    
    logger.debug(f'[Cody] Returning config with api_base={config["api_base"]!r}')
    return config