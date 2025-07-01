"""
Sourcegraph Cody custom provider for LiteLLM.

This provider handles Cody's specific authentication and API requirements.
"""

import json
import logging
import os
import ssl
from typing import Any, Callable, Iterator, Optional, Union

import httpx
import litellm
from litellm.llms.custom_llm import CustomLLM, CustomLLMError
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.types.utils import GenericStreamingChunk
from litellm.utils import ModelResponse

logger = logging.getLogger(__name__)


class CodyLLM(CustomLLM):
    """Custom LLM provider for Sourcegraph Cody."""

    def __init__(self):
        super().__init__()
        self._setup_ssl_bypass()

    def _setup_ssl_bypass(self):
        """Set up SSL bypass for corporate environments."""
        # Store original context
        if not hasattr(ssl, '_original_create_default_https_context'):
            ssl._original_create_default_https_context = ssl._create_default_https_context
        
        # Apply bypass
        ssl._create_default_https_context = ssl._create_unverified_context
        
        # Set environment variables
        os.environ['REQUESTS_CA_BUNDLE'] = ''
        os.environ['SSL_CERT_FILE'] = ''
        os.environ['CURL_CA_BUNDLE'] = ''

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
        
        # Debug logging
        logger.debug(f'[CodyLLM._prepare_request] api_base parameter: {api_base!r}')
        
        # Ensure API base has the correct path
        if not api_base:
            raise ValueError("api_base is required for Cody provider. Please set LLM_BASE_URL environment variable.")
            
        if not api_base.endswith('/.api/llm/chat/completions'):
            url = api_base.rstrip('/') + '/.api/llm/chat/completions'
        else:
            url = api_base
        
        # Prepare headers with Cody authentication
        headers = {
            'Authorization': f'token {api_key}',
            'X-Requested-With': 'openhands 0.46.0',  # Client name must be lowercase
            'Content-Type': 'application/json',
        }
        
        # Filter out system messages since Cody doesn't support them
        # Convert system messages to user messages with a prefix
        filtered_messages = []
        for msg in messages:
            # Skip messages with empty or missing content
            content = msg.get('content', '')
            # Also skip messages that only contain whitespace
            if not content or not content.strip():
                logger.warning(f"Skipping message with empty or whitespace-only content: {msg}")
                continue
                
            if msg.get('role') == 'system':
                # Convert system message to user message with prefix
                filtered_messages.append({
                    'role': 'user',
                    'content': f"[System instruction: {content}]"
                })
            else:
                # Ensure the message has content before adding
                filtered_messages.append({
                    'role': msg.get('role', 'user'),
                    'content': content
                })
        
        # Ensure we have at least one message after filtering
        if not filtered_messages:
            raise ValueError("No valid messages with content after filtering. All messages had empty content.")
        
        # Prepare request data
        data = {
            'model': model,
            'messages': filtered_messages,
            'temperature': optional_params.get('temperature', 1.0),
            'max_tokens': optional_params.get('max_tokens', 4096),
            'top_p': optional_params.get('top_p', 1.0),
        }
        
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
        
        # Debug logging for incoming parameters
        logger.debug(f'[CodyLLM] completion called with:')
        logger.debug(f'[CodyLLM]   model: {model}')
        logger.debug(f'[CodyLLM]   api_base: {api_base!r}')
        logger.debug(f'[CodyLLM]   api_key present: {bool(api_key)}')
        
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
            if print_verbose:
                print_verbose(f"Making request to Cody: {url}")
                print_verbose(f"Headers: {request_headers}")
                print_verbose(f"Data: {data}")
            
            response = client.post(
                url=url,
                json=data,
                headers=request_headers,
            )
            
            response.raise_for_status()
            response_json = response.json()
            
            # Update model_response with the response data
            # LiteLLM expects specific format for choices
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
            if print_verbose:
                print_verbose(f"Making async request to Cody: {url}")
                print_verbose(f"Headers: {request_headers}")
                print_verbose(f"Data: {data}")
            
            response = await client.post(
                url=url,
                json=data,
                headers=request_headers,
            )
            
            response.raise_for_status()
            response_json = response.json()
            
            # Update model_response with the response data
            # LiteLLM expects specific format for choices
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
    
    # Add Cody to the provider map
    if not hasattr(litellm, 'custom_provider_map'):
        litellm.custom_provider_map = []
    
    # Check if already registered
    for provider in litellm.custom_provider_map:
        if provider.get('provider') == 'cody':
            return
    
    # Register the provider
    litellm.custom_provider_map.append({
        'provider': 'cody',
        'custom_handler': cody_llm,
    })
    
    print("[Cody] Provider registered with LiteLLM")