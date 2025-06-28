# Cody Provider Setup Guide for OpenHands

## Overview
The Cody provider has been successfully integrated into OpenHands. This guide explains how to configure it.

## Setup Steps

1. **Start OpenHands CLI**
   ```bash
   poetry run openhands
   ```

2. **Provider Selection**
   - Use arrow keys to navigate to "Select another provider"
   - Press Enter
   - Type `cody` when prompted for provider name
   - Press Enter

3. **Model Selection**
   - When prompted, enter the Cody model name:
     ```
     anthropic::2024-10-22::claude-sonnet-4-latest
     ```
   - Press Enter

4. **Base URL Configuration** (NEW - Cody specific)
   - You will be prompted for "Enter Cody Base URL"
   - Enter: `https://sourcegraph.com`
   - Press Enter

5. **API Key**
   - Enter your Cody API key
   - Press Enter

6. **Save Settings**
   - Select "Yes, save" to save the configuration

## Available Cody Models

The following models are available:
- `anthropic::2024-10-22::claude-sonnet-4-latest` (Claude Sonnet 4)
- `anthropic::2024-10-22::claude-sonnet-4-thinking-latest` (Pro tier)
- `anthropic::2024-10-22::claude-3-7-sonnet-latest`
- `anthropic::2024-10-22::claude-3-5-sonnet-latest`
- `google::v1::gemini-2.0-flash`
- `openai::2024-08-01::chatgpt-4o-latest`
- And more...

## Environment Variables (Alternative Setup)

You can also set these environment variables before running OpenHands:
```bash
export LLM_MODEL='cody/anthropic::2024-10-22::claude-sonnet-4-latest'
export LLM_API_KEY='your-cody-api-key'
export LLM_BASE_URL='https://sourcegraph.com'
```

## Implementation Details

The Cody provider implementation includes:
1. Custom LiteLLM provider (`CodyLLM`) that handles Cody's authentication format
2. Automatic SSL certificate bypass for corporate environments
3. Proper header formatting (`Authorization: token <token>`)
4. Required `X-Requested-With` header

## Troubleshooting

If you see "api_base is required for Cody provider" error:
- Ensure you entered the base URL during setup
- Check that `~/.openhands/settings.json` contains `llm_base_url`
- Use environment variables as a fallback

## Testing

To verify your setup:
```bash
export LLM_MODEL='cody/anthropic::2024-10-22::claude-sonnet-4-latest'
export LLM_API_KEY='your-api-key'
export LLM_BASE_URL='https://sourcegraph.com'
poetry run openhands -t "Hello from Cody!"
```

A 429 rate limit error confirms authentication is working correctly.