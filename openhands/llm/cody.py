"""
Sourcegraph Cody LLM provider integration.

This module provides configuration and setup for using Cody with OpenHands.
"""

from typing import Dict, Any

from openhands.llm.cody_provider import register_cody_provider


# Available Cody models
CODY_MODELS = [
    # Claude Sonnet 4 (Latest)
    'cody/anthropic::2024-10-22::claude-sonnet-4-latest',  # Claude Sonnet 4
    'cody/anthropic::2024-10-22::claude-sonnet-4-thinking-latest',  # Claude Sonnet 4 with Thinking (Pro tier)
    
    # Claude 3.7 Sonnet
    'cody/anthropic::2024-10-22::claude-3-7-sonnet-latest',  # Claude 3.7 Sonnet
    'cody/anthropic::2024-10-22::claude-3-7-sonnet-extended-thinking',  # Claude 3.7 Sonnet with Thinking (Pro tier)
    
    # Claude 3.5 Sonnet
    'cody/anthropic::2024-10-22::claude-3-5-sonnet-latest',  # Claude 3.5 Sonnet
    
    # Claude 3.5 Haiku (Speed optimized)
    'cody/anthropic::2024-10-22::claude-3-5-haiku-latest',  # Claude 3.5 Haiku
    
    # Claude 3 Opus (Pro tier, deprecated)
    'cody/anthropic::2023-06-01::claude-3-opus',  # Claude 3 Opus (deprecated)
    
    # Google Gemini models
    'cody/google::v1::gemini-2.0-flash',  # Gemini 2.0 Flash
    'cody/google::v1::gemini-2.0-flash-lite',  # Gemini 2.0 Flash-Lite
    'cody/google::v1::gemini-2.5-flash-preview-04-17',  # Gemini 2.5 Flash Preview
    'cody/google::v1::gemini-1.5-pro-002',  # Gemini 1.5 Pro
    'cody/google::v1::gemini-2.5-pro-preview-03-25',  # Gemini 2.5 Pro Preview (Pro tier)
    
    # OpenAI models
    'cody/openai::2024-08-01::chatgpt-4o-latest',  # GPT-4o Latest
    'cody/openai::2024-08-06::gpt-4o-mini',  # GPT-4o Mini
    'cody/openai::v1::o1',  # OpenAI o1
    'cody/openai::v1::o1-preview',  # OpenAI o1 Preview
    'cody/openai::v1::o1-mini',  # OpenAI o1 Mini
    'cody/openai::v1::o4-mini',  # OpenAI o4 Mini (Pro tier)
    'cody/openai::v1::o4',  # OpenAI o4 (Pro tier)
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