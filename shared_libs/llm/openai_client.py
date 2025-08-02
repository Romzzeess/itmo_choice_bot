import base64
import json
import time
from typing import Any, Dict, List, Optional, Type

from openai import OpenAI
from pydantic import BaseModel, ValidationError

from .base_llm_client import LLMClientBase


class OpenaiClient(LLMClientBase):
    """Concrete LLM client for interfacing with a vLLM-compatible endpoint.

    Uses the OpenAI Python SDK to send requests to a local or remote vLLM server
    that exposes an OpenAI-compatible API surface (e.g., chat completions).
    """

    def __init__(
        self,
        api_key: str,
        model_name: str,
        base_url: str,
        **kwargs: Any,
    ) -> None:
        """Initialize the vLLM client with endpoint and model configuration.

        Args:
            api_key (str): API key or token. For vLLM, this can be a dummy value if authentication is not enforced.
            model_name (str): Name or identifier of the vLLM model (e.g., "qwen-3.4b").
            base_url (str, optional): Base URL of the vLLM server's OpenAI-compatible API.
            **kwargs: Additional provider-specific configuration parameters (e.g., timeout).
        """
        super().__init__(api_key=api_key, model_name=model_name, base_url=base_url, **kwargs)
        # Instantiate the OpenAI client, pointing to the vLLM endpoint
        self.client = OpenAI(base_url=base_url, api_key=api_key)

    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        **kwargs: Any,
    ) -> str:
        """Generate a text completion from the vLLM server.

        Sends a chat completion request to the vLLM endpoint and returns the
        generated text content. Uses the "chat/completions" endpoint.

        Args:
            prompt (str): The user prompt or context to send to the model.
            system_prompt (str, optional): System-level instructions to guide generation.
            **kwargs: Additional parameters supported by the vLLM chat/completions API
                (e.g., top_p, n, stop sequences).

        Returns:
            str: The generated text content from the first choice.

        Raises:
            RuntimeError: If the request to the vLLM server fails or returns a non-200 status.
        """
        try:
            # Build the messages list, inserting a system message if provided
            messages: List[Dict[str, Any]] = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": [
                {"type": "text", "text": prompt},
            ]})

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.0,
                **kwargs,
            )
        except Exception as e:
            time.sleep(60)
            raise RuntimeError(f"OpenAi generate request failed: {e}")

        try:
            return response.choices[0].message.content
        except (AttributeError, KeyError, IndexError) as e:
            raise RuntimeError(f"Failed to parse OpenAI generate response: {e}")

