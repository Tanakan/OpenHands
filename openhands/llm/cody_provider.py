"""
Sourcegraph Cody custom provider for LiteLLM.

This provider handles Cody's specific authentication and API requirements.
"""

import logging
from typing import Callable, Iterator, Optional, Union

import httpx
import litellm
from litellm.llms.custom_llm import CustomLLM, CustomLLMError
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.types.utils import GenericStreamingChunk
from litellm.utils import ModelResponse

logger = logging.getLogger(__name__)


class CodyLLM(CustomLLM):
    """Custom LLM provider for Sourcegraph Cody."""

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
            'X-Requested-With': 'openhands',
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
        """Make a completion request to Cody API."""
        
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
        
        # Make the request
        try:
            response = client.post(
                url=url,
                json=data,
                headers=request_headers,
            )
            
            response.raise_for_status()
            response_json = response.json()
            
            # Update model_response with the response data
            from litellm import Choices, Message
            
            choices = []
            for idx, choice in enumerate(response_json.get('choices', [])):
                choice_obj = Choices(
                    index=choice.get('index', idx),
                    message=Message(**choice.get('message', {})),
                    finish_reason=choice.get('finish_reason', 'stop')
                )
                choices.append(choice_obj)
            
            model_response.choices = choices
            model_response.id = response_json.get('id', '')
            model_response.model = response_json.get('model', model)
            model_response.usage = response_json.get('usage', {})
            model_response.created = response_json.get('created', 0)
            model_response.object = response_json.get('object', 'chat.completion')
            
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
        """Make an async completion request to Cody API."""
        
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
        
        # Make the request
        try:
            response = await client.post(
                url=url,
                json=data,
                headers=request_headers,
            )
            
            response.raise_for_status()
            response_json = response.json()
            
            # Update model_response with the response data
            from litellm import Choices, Message
            
            choices = []
            for idx, choice in enumerate(response_json.get('choices', [])):
                choice_obj = Choices(
                    index=choice.get('index', idx),
                    message=Message(**choice.get('message', {})),
                    finish_reason=choice.get('finish_reason', 'stop')
                )
                choices.append(choice_obj)
            
            model_response.choices = choices
            model_response.id = response_json.get('id', '')
            model_response.model = response_json.get('model', model)
            model_response.usage = response_json.get('usage', {})
            model_response.created = response_json.get('created', 0)
            model_response.object = response_json.get('object', 'chat.completion')
            
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