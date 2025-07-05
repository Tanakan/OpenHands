"""
Sourcegraph Cody custom provider for LiteLLM.

This provider handles Cody's specific authentication and API requirements.
"""

import json
import logging
from typing import Callable, Iterator, Optional, Union

import httpx
import litellm
from litellm.llms.custom_llm import CustomLLM, CustomLLMError
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.types.utils import GenericStreamingChunk
from litellm.utils import ModelResponse

logger = logging.getLogger(__name__)


class ContinuationHandler:
    """Handles continuation of truncated responses."""
    
    def __init__(self):
        # Always enabled with unlimited continuations
        pass
    
    def is_response_truncated(self, response_json: dict) -> bool:
        """Check if response was truncated due to length."""
        choices = response_json.get('choices', [])
        if not choices:
            return False
        
        finish_reason = choices[0].get('finish_reason', 'stop')
        return finish_reason == 'length'
    
    def detect_response_type(self, response_json: dict) -> str:
        """Detect the type of response (text, tool_calls, mixed)."""
        choices = response_json.get('choices', [])
        if not choices:
            return 'text'
        
        message = choices[0].get('message', {})
        has_content = bool(message.get('content'))
        has_tool_calls = bool(message.get('tool_calls'))
        
        if has_tool_calls and has_content:
            return 'mixed'
        elif has_tool_calls:
            return 'tool_calls'
        else:
            return 'text'
    
    def check_incomplete_tool_calls(self, tool_calls: list) -> tuple[bool, int]:
        """Check if any tool calls have incomplete JSON arguments."""
        for i, tool_call in enumerate(tool_calls):
            if 'function' in tool_call and 'arguments' in tool_call['function']:
                try:
                    json.loads(tool_call['function']['arguments'])
                except json.JSONDecodeError:
                    return True, i
        return False, -1
    
    def build_continuation_prompt(self, response_type: str, accumulated_content: str, 
                                incomplete_json: Optional[str] = None) -> str:
        """Build appropriate continuation prompt based on response type."""
        if response_type == 'tool_calls' and incomplete_json:
            # For tool calls, be very specific about continuing the JSON
            return (
                "Your previous response was truncated. Continue ONLY the incomplete JSON "
                f"for the function arguments, starting from: ...{incomplete_json[-50:]}"
                "\nDo not repeat anything before this point. Complete only the JSON."
            )
        else:
            # For all other types, use a clear continuation instruction
            return (
                "Your previous response was truncated due to length limit. "
                "Continue generating the response from the exact point where it was cut off. "
                "IMPORTANT: Do not repeat or restate any content that was already generated. "
                "Start with the very next word, character, or token that would have followed."
            )
    
    def merge_tool_calls(self, original_calls: list, continuation_calls: list, 
                        incomplete_index: int) -> list:
        """Merge continued tool calls with original ones."""
        # Keep complete tool calls
        merged = original_calls[:incomplete_index]
        
        # Handle the incomplete tool call
        if incomplete_index < len(original_calls):
            incomplete_call = original_calls[incomplete_index]
            
            # Try to complete the arguments
            if continuation_calls and 'function' in continuation_calls[0]:
                # Get the incomplete arguments
                incomplete_args = incomplete_call.get('function', {}).get('arguments', '')
                continuation_args = continuation_calls[0].get('function', {}).get('arguments', '')
                
                # Try to merge JSON strings
                try:
                    # Remove potential duplicate content
                    merged_args = self._merge_json_strings(incomplete_args, continuation_args)
                    incomplete_call['function']['arguments'] = merged_args
                    merged.append(incomplete_call)
                    
                    # Add any additional tool calls from continuation
                    if len(continuation_calls) > 1:
                        merged.extend(continuation_calls[1:])
                except Exception as e:
                    logger.warning(f"Failed to merge tool call arguments: {e}")
                    # Fall back to using continuation as-is
                    merged.extend(continuation_calls)
            else:
                # No continuation tool calls, keep the incomplete one
                merged.append(incomplete_call)
        
        return merged
    
    def _merge_json_strings(self, incomplete: str, continuation: str) -> str:
        """Merge incomplete JSON with its continuation."""
        # Try direct concatenation first
        combined = incomplete + continuation
        try:
            json.loads(combined)
            return combined
        except json.JSONDecodeError:
            pass
        
        # Try to find overlap
        for i in range(min(50, len(incomplete))):
            overlap_start = len(incomplete) - i
            if continuation.startswith(incomplete[overlap_start:]):
                combined = incomplete[:overlap_start] + continuation
                try:
                    json.loads(combined)
                    return combined
                except json.JSONDecodeError:
                    continue
        
        # Last resort: try to fix common issues
        if not incomplete.rstrip().endswith('}') and continuation.lstrip().startswith('}'):
            return incomplete + continuation
        
        return incomplete + continuation


class CodyLLM(CustomLLM):
    """Custom LLM provider for Sourcegraph Cody."""
    
    def __init__(self):
        super().__init__()
        self.continuation_handler = ContinuationHandler()

    def _prepare_request(
        self,
        model: str,
        messages: list,
        api_base: str,
        api_key: str,
        optional_params: dict,
    ) -> tuple[str, dict, dict]:
        """Prepare the request URL, headers, and data for Cody API."""
        
        # Extract the actual model name from cody/ prefix
        if model.startswith('cody/'):
            model = model[5:]
        
        # Ensure API base has the correct path
        if not api_base:
            raise ValueError("api_base is required for Cody provider")
        
        url = api_base.rstrip('/') + '/.api/llm/chat/completions'
        
        # Prepare headers with Cody authentication
        headers = {
            'Authorization': f'token {api_key}',
            'X-Requested-With': 'openhands 0.1.0',  # Format: <client-name> <client-version>
            'Content-Type': 'application/json',
        }
        
        # Convert system messages to user messages (Cody doesn't support system role)
        filtered_messages = []
        for msg in messages:
            content = msg.get('content', '')
            if not content.strip():
                continue
                
            if msg.get('role') == 'system':
                filtered_messages.append({
                    'role': 'user',
                    'content': f"[System instruction: {content}]"
                })
            else:
                filtered_messages.append(msg)
        
        # Prepare request data
        data = {
            'model': model,
            'messages': filtered_messages,
            'temperature': optional_params.get('temperature', 1.0),
            'top_p': optional_params.get('top_p', 1.0),
        }
        
        # Add max_tokens if specified
        if 'max_tokens' in optional_params:
            data['max_tokens'] = optional_params['max_tokens']
        elif 'max_completion_tokens' in optional_params:
            data['max_tokens'] = optional_params['max_completion_tokens']
        
        # Add optional parameters if present
        for key in ['stream', 'stop', 'n', 'presence_penalty', 'frequency_penalty']:
            if key in optional_params:
                data[key] = optional_params[key]
        
        return url, headers, data

    def completion(
        self,
        model: str,
        messages: list,
        api_base: str,
        custom_prompt_dict: dict,
        model_response: ModelResponse,
        print_verbose: Callable,
        encoding,
        api_key,
        logging_obj,
        optional_params: dict,
        acompletion=None,
        litellm_params=None,
        logger_fn=None,
        headers={},
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        client: Optional[HTTPHandler] = None,
    ) -> ModelResponse:
        """Make a completion request to Cody API with automatic continuation support."""
        
        # Prepare request
        url, request_headers, data = self._prepare_request(
            model, messages, api_base, api_key, optional_params
        )
        
        # Merge any additional headers
        if headers:
            request_headers.update(headers)
        
        # Create client if not provided
        if client is None:
            client = HTTPHandler(timeout=timeout)
        
        # Initialize continuation state
        continuation_count = 0
        accumulated_content = ""
        accumulated_tool_calls = []
        final_response_json = None
        
        # Make the initial request
        try:
            while True:
                response = client.post(
                    url=url,
                    json=data,
                    headers=request_headers,
                )
                
                response.raise_for_status()
                response_json = response.json()
                
                # Log finish reason
                finish_reason = response_json.get('choices', [{}])[0].get('finish_reason', 'unknown')
                logger.debug(f"[Cody] Response finish_reason: {finish_reason}")
                
                # Store first response for metadata
                if final_response_json is None:
                    final_response_json = response_json
                
                # Extract content and tool calls
                current_message = response_json.get('choices', [{}])[0].get('message', {})
                current_content = current_message.get('content', '')
                current_tool_calls = current_message.get('tool_calls', [])
                
                # Accumulate content
                if current_content:
                    accumulated_content += current_content
                
                # Handle tool calls
                if current_tool_calls:
                    if not accumulated_tool_calls:
                        accumulated_tool_calls = current_tool_calls
                    else:
                        # Merge tool calls if continuing
                        has_incomplete, incomplete_idx = self.continuation_handler.check_incomplete_tool_calls(
                            accumulated_tool_calls
                        )
                        if has_incomplete:
                            accumulated_tool_calls = self.continuation_handler.merge_tool_calls(
                                accumulated_tool_calls, current_tool_calls, incomplete_idx
                            )
                        else:
                            accumulated_tool_calls.extend(current_tool_calls)
                
                # Check if continuation is needed (always continue if truncated)
                if not self.continuation_handler.is_response_truncated(response_json):
                    break
                
                # Check for incomplete tool calls
                has_incomplete, incomplete_idx = self.continuation_handler.check_incomplete_tool_calls(
                    accumulated_tool_calls
                )
                
                # Prepare continuation
                continuation_count += 1
                logger.info(f"[Cody] Response truncated, attempting continuation {continuation_count}")
                
                # Build continuation messages
                continuation_messages = messages.copy()
                
                # Add accumulated response as assistant message
                assistant_msg = {
                    "role": "assistant",
                    "content": accumulated_content
                }
                if accumulated_tool_calls:
                    assistant_msg["tool_calls"] = accumulated_tool_calls
                continuation_messages.append(assistant_msg)
                
                # Determine response type and build continuation prompt
                response_type = self.continuation_handler.detect_response_type(response_json)
                
                incomplete_json = None
                if has_incomplete and incomplete_idx < len(accumulated_tool_calls):
                    incomplete_json = accumulated_tool_calls[incomplete_idx].get('function', {}).get('arguments', '')
                
                continuation_prompt = self.continuation_handler.build_continuation_prompt(
                    response_type, accumulated_content, incomplete_json
                )
                
                continuation_messages.append({
                    "role": "user",
                    "content": continuation_prompt
                })
                
                # Update request data
                url, request_headers, data = self._prepare_request(
                    model, continuation_messages, api_base, api_key, optional_params
                )
            
            # Build final response
            from litellm import Choices, Message
            
            # Create the final message
            final_message_data = {}
            if accumulated_content:
                final_message_data['content'] = accumulated_content
            if accumulated_tool_calls:
                final_message_data['tool_calls'] = accumulated_tool_calls
            final_message_data['role'] = 'assistant'
            
            # Determine final finish_reason based on accumulated response
            if accumulated_tool_calls:
                final_finish_reason = 'tool_calls'
            else:
                final_finish_reason = 'stop'
            
            # Create choice object
            choice_obj = Choices(
                index=0,
                message=Message(**final_message_data),
                finish_reason=final_finish_reason
            )
            
            model_response.choices = [choice_obj]
            model_response.id = final_response_json.get('id', '')
            model_response.model = final_response_json.get('model', model)
            
            # Accumulate usage stats
            if continuation_count > 0:
                total_usage = {
                    'prompt_tokens': 0,
                    'completion_tokens': 0,
                    'total_tokens': 0
                }
                # Note: This is simplified - in reality you'd track usage from each request
                model_response.usage = final_response_json.get('usage', total_usage)
            else:
                model_response.usage = final_response_json.get('usage', {})
            
            model_response.created = final_response_json.get('created', 0)
            model_response.object = final_response_json.get('object', 'chat.completion')
            
            if continuation_count > 0:
                logger.info(f"[Cody] Completed after {continuation_count} continuations")
            
            return model_response
            
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                raise CustomLLMError(
                    status_code=401,
                    message="Authentication failed. Please check your Cody API key."
                )
            elif e.response.status_code == 429:
                raise CustomLLMError(
                    status_code=429,
                    message="Rate limit exceeded. Please try again later."
                )
            else:
                raise CustomLLMError(
                    status_code=e.response.status_code,
                    message=f"Cody API error: {e.response.text}"
                )
        except Exception as e:
            raise CustomLLMError(
                status_code=500,
                message=f"Error calling Cody API: {str(e)}"
            )

    def streaming(
        self,
        model: str,
        messages: list,
        api_base: str,
        custom_prompt_dict: dict,
        model_response: ModelResponse,
        print_verbose: Callable,
        encoding,
        api_key,
        logging_obj,
        optional_params: dict,
        acompletion=None,
        litellm_params=None,
        logger_fn=None,
        headers={},
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        client: Optional[HTTPHandler] = None,
    ) -> Iterator[GenericStreamingChunk]:
        """Streaming is not implemented yet for Cody."""
        raise CustomLLMError(
            status_code=501,
            message="Streaming is not implemented for Cody provider yet."
        )

    async def acompletion(
        self,
        model: str,
        messages: list,
        api_base: str,
        custom_prompt_dict: dict,
        model_response: ModelResponse,
        print_verbose: Callable,
        encoding,
        api_key,
        logging_obj,
        optional_params: dict,
        acompletion=None,
        litellm_params=None,
        logger_fn=None,
        headers={},
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        client: Optional[AsyncHTTPHandler] = None,
    ) -> ModelResponse:
        """Make an async completion request to Cody API with automatic continuation support."""
        
        # Prepare request
        url, request_headers, data = self._prepare_request(
            model, messages, api_base, api_key, optional_params
        )
        
        # Merge any additional headers
        if headers:
            request_headers.update(headers)
        
        # Create client if not provided
        if client is None:
            client = AsyncHTTPHandler(timeout=timeout)
        
        # Initialize continuation state
        continuation_count = 0
        accumulated_content = ""
        accumulated_tool_calls = []
        final_response_json = None
        
        # Make the initial request
        try:
            while True:
                response = await client.post(
                    url=url,
                    json=data,
                    headers=request_headers,
                )
                
                response.raise_for_status()
                response_json = response.json()
                
                # Log finish reason
                finish_reason = response_json.get('choices', [{}])[0].get('finish_reason', 'unknown')
                logger.debug(f"[Cody] Response finish_reason: {finish_reason}")
                
                # Store first response for metadata
                if final_response_json is None:
                    final_response_json = response_json
                
                # Extract content and tool calls
                current_message = response_json.get('choices', [{}])[0].get('message', {})
                current_content = current_message.get('content', '')
                current_tool_calls = current_message.get('tool_calls', [])
                
                # Accumulate content
                if current_content:
                    accumulated_content += current_content
                
                # Handle tool calls
                if current_tool_calls:
                    if not accumulated_tool_calls:
                        accumulated_tool_calls = current_tool_calls
                    else:
                        # Merge tool calls if continuing
                        has_incomplete, incomplete_idx = self.continuation_handler.check_incomplete_tool_calls(
                            accumulated_tool_calls
                        )
                        if has_incomplete:
                            accumulated_tool_calls = self.continuation_handler.merge_tool_calls(
                                accumulated_tool_calls, current_tool_calls, incomplete_idx
                            )
                        else:
                            accumulated_tool_calls.extend(current_tool_calls)
                
                # Check if continuation is needed (always continue if truncated)
                if not self.continuation_handler.is_response_truncated(response_json):
                    break
                
                # Check for incomplete tool calls
                has_incomplete, incomplete_idx = self.continuation_handler.check_incomplete_tool_calls(
                    accumulated_tool_calls
                )
                
                # Prepare continuation
                continuation_count += 1
                logger.info(f"[Cody] Response truncated, attempting continuation {continuation_count}")
                
                # Build continuation messages
                continuation_messages = messages.copy()
                
                # Add accumulated response as assistant message
                assistant_msg = {
                    "role": "assistant",
                    "content": accumulated_content
                }
                if accumulated_tool_calls:
                    assistant_msg["tool_calls"] = accumulated_tool_calls
                continuation_messages.append(assistant_msg)
                
                # Determine response type and build continuation prompt
                response_type = self.continuation_handler.detect_response_type(response_json)
                
                incomplete_json = None
                if has_incomplete and incomplete_idx < len(accumulated_tool_calls):
                    incomplete_json = accumulated_tool_calls[incomplete_idx].get('function', {}).get('arguments', '')
                
                continuation_prompt = self.continuation_handler.build_continuation_prompt(
                    response_type, accumulated_content, incomplete_json
                )
                
                continuation_messages.append({
                    "role": "user",
                    "content": continuation_prompt
                })
                
                # Update request data
                url, request_headers, data = self._prepare_request(
                    model, continuation_messages, api_base, api_key, optional_params
                )
            
            # Build final response
            from litellm import Choices, Message
            
            # Create the final message
            final_message_data = {}
            if accumulated_content:
                final_message_data['content'] = accumulated_content
            if accumulated_tool_calls:
                final_message_data['tool_calls'] = accumulated_tool_calls
            final_message_data['role'] = 'assistant'
            
            # Determine final finish_reason based on accumulated response
            if accumulated_tool_calls:
                final_finish_reason = 'tool_calls'
            else:
                final_finish_reason = 'stop'
            
            # Create choice object
            choice_obj = Choices(
                index=0,
                message=Message(**final_message_data),
                finish_reason=final_finish_reason
            )
            
            model_response.choices = [choice_obj]
            model_response.id = final_response_json.get('id', '')
            model_response.model = final_response_json.get('model', model)
            
            # Accumulate usage stats
            if continuation_count > 0:
                total_usage = {
                    'prompt_tokens': 0,
                    'completion_tokens': 0,
                    'total_tokens': 0
                }
                # Note: This is simplified - in reality you'd track usage from each request
                model_response.usage = final_response_json.get('usage', total_usage)
            else:
                model_response.usage = final_response_json.get('usage', {})
            
            model_response.created = final_response_json.get('created', 0)
            model_response.object = final_response_json.get('object', 'chat.completion')
            
            if continuation_count > 0:
                logger.info(f"[Cody] Completed after {continuation_count} continuations")
            
            return model_response
            
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                raise CustomLLMError(
                    status_code=401,
                    message="Authentication failed. Please check your Cody API key."
                )
            elif e.response.status_code == 429:
                raise CustomLLMError(
                    status_code=429,
                    message="Rate limit exceeded. Please try again later."
                )
            else:
                raise CustomLLMError(
                    status_code=e.response.status_code,
                    message=f"Cody API error: {e.response.text}"
                )
        except Exception as e:
            raise CustomLLMError(
                status_code=500,
                message=f"Error calling Cody API: {str(e)}"
            )

    async def astreaming(
        self,
        model: str,
        messages: list,
        api_base: str,
        custom_prompt_dict: dict,
        model_response: ModelResponse,
        print_verbose: Callable,
        encoding,
        api_key,
        logging_obj,
        optional_params: dict,
        acompletion=None,
        litellm_params=None,
        logger_fn=None,
        headers={},
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        client: Optional[AsyncHTTPHandler] = None,
    ):
        """Async streaming is not implemented yet for Cody."""
        raise CustomLLMError(
            status_code=501,
            message="Async streaming is not implemented for Cody provider yet."
        )


# Create a singleton instance
cody_llm = CodyLLM()


def register_cody_provider():
    """Register Cody as a custom provider in LiteLLM."""
    
    # Initialize custom_provider_map if it doesn't exist
    if not hasattr(litellm, 'custom_provider_map'):
        litellm.custom_provider_map = []
    
    # Check if already registered
    for i, provider in enumerate(litellm.custom_provider_map):
        if provider.get('provider') == 'cody':
            # Update existing registration
            litellm.custom_provider_map[i] = {
                'provider': 'cody',
                'custom_handler': cody_llm,
            }
            logger.info("[Cody] Provider updated in LiteLLM")
            return

    # Register new provider
    litellm.custom_provider_map.append({
        'provider': 'cody',
        'custom_handler': cody_llm,
    })
    logger.info("[Cody] Provider registered with LiteLLM")