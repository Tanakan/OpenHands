# Cody Provider Integration with LiteLLM

This document describes how the Sourcegraph Cody provider has been integrated with LiteLLM in OpenHands.

## Overview

The Cody provider is now fully integrated with LiteLLM, allowing you to use Sourcegraph Cody's API in the same way you would use OpenAI, Anthropic, or any other LiteLLM-supported provider.

## Key Features

1. **Standard LiteLLM Interface**: Use `litellm.completion()` with Cody models
2. **Automatic Registration**: The provider is automatically registered when OpenHands is imported
3. **Model Aliases**: Convenient short aliases for common models
4. **SSL Bypass Option**: For corporate environments (opt-in via environment variable)
5. **Automatic Continuation**: Handles truncated responses automatically
6. **Full Async Support**: Both sync and async operations are supported

## Configuration

### Environment Variables

- `LLM_BASE_URL`: Your Sourcegraph instance URL
- `LLM_API_KEY`: Your Cody API token
- `CODY_SSL_BYPASS`: Set to "true" to bypass SSL verification (optional)
- `CODY_CLIENT_NAME`: Client identifier (defaults to "openhands")
- `CODY_DEFAULT_MAX_TOKENS`: Default max tokens (defaults to 512)

### Model Names

Full model names follow the pattern: `cody/${ProviderID}::${APIVersionID}::${ModelID}`

Examples:
- `cody/anthropic::2023-06-01::claude-3-opus`
- `cody/anthropic::2023-06-01::claude-3.5-sonnet`
- `cody/openai::v1::gpt-4`

### Model Aliases

For convenience, short aliases are available:
- `cody-opus` → `cody/anthropic::2023-06-01::claude-3-opus`
- `cody-sonnet` → `cody/anthropic::2023-06-01::claude-3.5-sonnet`
- `cody-haiku` → `cody/anthropic::2023-06-01::claude-3-haiku`
- `cody-gpt4` → `cody/openai::v1::gpt-4`
- `cody-gpt35` → `cody/openai::v1::gpt-3.5-turbo`
- `cody-gemini` → `cody/google::v1::gemini-pro`

## Usage Examples

### Basic Usage with LiteLLM

```python
import litellm

response = litellm.completion(
    model="cody/anthropic::2023-06-01::claude-3.5-sonnet",
    messages=[
        {"role": "user", "content": "Hello, how are you?"}
    ],
    api_base="https://your-sourcegraph.com",
    api_key="your-cody-token"
)
```

### Using with OpenHands LLM Class

```python
from openhands.core.config import LLMConfig
from openhands.llm import LLM

config = LLMConfig(
    model="cody-sonnet",  # Using alias
    api_key="your-cody-token",
    base_url="https://your-sourcegraph.com",
    max_tokens=1024
)

llm = LLM(config)
response = llm.completion(messages=[...])
```

## Implementation Details

### Files Modified/Created

1. **`openhands/llm/cody_provider.py`**: 
   - Updated to follow LiteLLM conventions
   - Added environment variable support
   - Improved SSL handling (opt-in)

2. **`openhands/llm/cody_litellm_config.py`**: 
   - New file for model registration and configuration
   - Defines model metadata and aliases

3. **`openhands/llm/__init__.py`**: 
   - Auto-imports Cody provider registration

### Key Changes

1. **Provider Registration**: Now uses standard LiteLLM `custom_provider_map`
2. **Model Cost Map**: Registered in `litellm.model_cost` for consistency
3. **Environment Variables**: Configurable behavior without code changes
4. **Auto-Continuation**: Seamlessly handles `finish_reason="length"`

## Automatic Response Continuation

The Cody provider automatically handles truncated responses when `finish_reason="length"`:

1. Detects truncated responses
2. Sends continuation requests with appropriate prompts
3. Concatenates responses transparently
4. No limit on continuation attempts
5. Can be disabled with `auto_continue=False`

This ensures complete responses even with low `max_tokens` settings.

## Testing

Run the integration test:
```bash
python test_cody_litellm_integration.py
```

Run the usage examples:
```bash
export LLM_BASE_URL='https://your-sourcegraph.com'
export LLM_API_KEY='your-cody-token'
python example_cody_litellm_usage.py
```

## Benefits

1. **Unified Interface**: Use Cody just like any other LiteLLM provider
2. **Consistency**: Same configuration and usage patterns as other providers
3. **Flexibility**: Easy to switch between providers
4. **Maintainability**: Leverages LiteLLM's infrastructure
5. **Feature Parity**: Supports all LiteLLM features (retries, callbacks, etc.)