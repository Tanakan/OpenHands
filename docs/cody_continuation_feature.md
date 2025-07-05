# Cody Provider Continuation Feature

This document describes the automatic continuation feature implemented in the Cody provider for handling truncated responses due to `max_tokens` limits.

## Overview

When the Cody API returns a response with `finish_reason="length"`, it indicates the response was truncated due to reaching the `max_tokens` limit. The Cody provider includes an automatic continuation feature that seamlessly handles these truncated responses, ensuring you always get complete responses.

## Features

1. **Automatic Detection**: Detects when responses are truncated (`finish_reason="length"`)
2. **Smart Continuation**: Different strategies for text, tool calls, and mixed responses
3. **Tool Call Handling**: Special logic to handle incomplete JSON in function arguments
4. **Always Enabled**: No configuration needed - works automatically
5. **Unlimited Continuations**: Continues until response is complete
6. **Logging**: Detailed logging for debugging

## Configuration

The continuation feature is **always enabled** and has **no limit** on the number of continuation attempts. This ensures that responses are never truncated, regardless of the `max_tokens` setting.

## How It Works

### 1. Response Type Detection

The continuation handler detects three types of responses:
- **text**: Pure text responses
- **tool_calls**: Only function/tool calls
- **mixed**: Both text content and tool calls

### 2. Continuation Prompts

Based on the response type, different continuation prompts are used:

#### Text Responses
```
"Continue from where you left off."
```

#### Tool Call Responses
```
"Your previous response was truncated. Continue ONLY the incomplete JSON 
for the function arguments, starting from: ...[last 50 chars]
Do not repeat anything before this point. Complete only the JSON."
```

#### Mixed Responses
```
"Continue from exactly where you left off. Last output was: ...[last 100 chars]
Continue without repeating."
```

### 3. Tool Call Merging

For truncated tool calls:
1. Identifies incomplete JSON arguments
2. Attempts to merge the continuation with the incomplete part
3. Handles overlapping content to avoid duplication
4. Preserves tool call IDs and structure

### 4. Usage Tracking

The feature tracks:
- Number of continuations performed
- Accumulated content across all requests
- Combined usage statistics (simplified in current implementation)

## Usage Example

```python
from openhands.core.config import LLMConfig
from openhands.llm import LLM

# No configuration needed - continuation is always enabled

config = LLMConfig(
    model="cody/anthropic::2024-10-22::claude-3-5-sonnet-latest",
    api_key="your-key",
    base_url="https://sourcegraph.com",
    max_output_tokens=100,  # Low limit will trigger automatic continuation
)

llm = LLM(config)

# Will automatically continue if response is truncated
response = llm.completion(messages=[
    {"role": "user", "content": "Write a detailed explanation of Python decorators"}
])
# The response will be complete, regardless of max_output_tokens setting
```

## Implementation Details

### ContinuationHandler Class

The `ContinuationHandler` class manages the continuation logic:

```python
class ContinuationHandler:
    def is_response_truncated(response_json: dict) -> bool
    def detect_response_type(response_json: dict) -> str
    def check_incomplete_tool_calls(tool_calls: list) -> tuple[bool, int]
    def build_continuation_prompt(response_type: str, ...) -> str
    def merge_tool_calls(original_calls: list, ...) -> list
```

### Key Algorithms

1. **JSON Merging**: Attempts multiple strategies to merge incomplete JSON:
   - Direct concatenation
   - Overlap detection and removal
   - Common pattern fixes

2. **Duplicate Prevention**: Checks for overlapping content between original and continuation

3. **State Management**: Maintains accumulated state across continuation requests

## Limitations

1. **Usage Statistics**: Currently simplified - doesn't accurately sum tokens across requests
2. **Streaming**: Not supported in streaming mode
3. **Context Growth**: Each continuation adds to the message history

## Best Practices

1. **Set Reasonable Limits**: While continuation is unlimited, avoid setting `max_tokens` unnecessarily low
2. **Monitor Logs**: Enable debug logging to track continuation behavior
3. **Test Tool Calls**: Thoroughly test function calling with your specific use cases
4. **Consider Context**: Be aware that each continuation consumes more context tokens

## Future Improvements

1. Accurate usage statistics accumulation
2. Streaming support with continuation
3. More sophisticated JSON reconstruction
4. Context optimization strategies