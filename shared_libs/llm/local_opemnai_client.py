import base64
import json
from typing import Any

from openai import OpenAI
from pydantic import BaseModel, ValidationError

from shared_libs.llm.base import LLMClientBase


class LocalOpenaiClient(LLMClientBase):
    """Concrete LLM client for interfacing with a server-compatible endpoint.

    Uses the OpenAI Python SDK to send requests to a local or remote server server
    that exposes an OpenAI-compatible API surface (e.g., chat completions).
    """

    def __init__(
        self,
        api_key: str,
        model_name: str,
        base_url: str = "http://localhost:8000/v1",
        **kwargs: Any,
    ) -> None:
        """Initialize the server client with endpoint and model configuration.

        Args:
            api_key (str): API key or token. For server, this can be a dummy value if authentication is not enforced.
            model_name (str): Name or identifier of the server model (e.g., "qwen-3.4b").
            base_url (str, optional): Base URL of the server server's OpenAI-compatible API.
                Defaults to "http://localhost:8000/v1".
            **kwargs: Additional provider-specific configuration parameters (e.g., timeout).
        """
        super().__init__(api_key=api_key, model_name=model_name, base_url=base_url, **kwargs)
        # Instantiate the OpenAI client, pointing to the server endpoint
        self.client = OpenAI(base_url=base_url, api_key=api_key)

    def generate(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float = 0.0,
        system_prompt: str = "",
        **kwargs: Any,
    ) -> str:
        """Generate a text completion from the server server.

        Sends a chat completion request to the server endpoint and returns the
        generated text content. Uses the "chat/completions" endpoint.

        Args:
            prompt (str): The user prompt or context to send to the model.
            max_tokens (int): Maximum number of tokens to generate in the response.
            temperature (float, optional): Sampling temperature for generation. Defaults to 0.0.
            system_prompt (str, optional): System-level instructions to guide generation.
            **kwargs: Additional parameters supported by the server chat/completions API
                (e.g., top_p, n, stop sequences).

        Returns:
            str: The generated text content from the first choice.

        Raises:
            RuntimeError: If the request to the server server fails or returns a non-200 status.
        """
        try:
            # Build the messages list, inserting a system message if provided
            messages: list[dict[str, Any]] = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                    ],
                }
            )

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs,
            )
        except Exception as e:
            msg = f"server generate request failed: {e}"
            raise RuntimeError(msg)

        try:
            return response.choices[0].message.content
        except (AttributeError, KeyError, IndexError) as e:
            msg = f"Failed to parse server generate response: {e}"
            raise RuntimeError(msg)

    def generate_json(
        self,
        prompt: str,
        max_tokens: int,
        response_model: type[BaseModel],
        temperature: float = 0.0,
        system_prompt: str = "",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Generate a JSON‐structured completion and validate it with a Pydantic model.

        Args:
            prompt (str): The user prompt.
            max_tokens (int): Maximum number of tokens to generate.
            response_model (Type[BaseModel]): Pydantic model class for validation.
            temperature (float): Sampling temperature.
            system_prompt (str): Optional system‐level instruction.
            **kwargs: Additional parameters for the chat/completions API.

        Returns:
            Dict[str, Any]: The `.dict()` output of the validated Pydantic model.

        Raises:
            RuntimeError: If the API call fails or the response is malformed.
            ValueError: If JSON parsing or validation against `response_model` fails.
        """
        try:
            messages: list[dict[str, Any]] = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs,
            )
        except Exception as e:
            msg = f"server generate_json request failed: {e}"
            raise RuntimeError(msg)

        try:
            json_response = response.choices[0].message.content
        except (AttributeError, IndexError) as e:
            raise RuntimeError("Failed to retrieve content from model response") from e

        try:
            parsed = response_model.model_validate_json(json_response)
            return parsed.model_dump_json()
        except ValidationError as ve:
            msg = f"Invalid JSON structure: {ve}"
            raise ValueError(msg) from ve
        except json.JSONDecodeError as je:
            msg = f"Failed to decode JSON response: {je}"
            raise ValueError(msg) from je

    def _encode_image_to_data_uri(self: str) -> str:
        """Read an image file and return it as a base64-encoded data URI."""
        try:
            with open(self, "rb") as img_file:
                encoded_bytes = base64.b64encode(img_file.read()).decode("utf-8")
                return f"data:image/jpeg;base64,{encoded_bytes}"
        except Exception as e:
            msg = f"Failed to encode image '{self}': {e}"
            raise RuntimeError(msg)

    def generate_with_images(
        self,
        prompt: str,
        images: list[str] | None = None,
        system_prompt: str = "",
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str:
        """Generate a text completion by sending both text and images to the server server.

        Embeds images as base64-encoded data URIs in the request. Constructs a
        chat completion with one system message (if provided), followed by a user
        message that contains the text prompt and any images.

        Args:
            prompt (str): The user prompt or context to send to the model.
            images (Optional[List[str]], optional): List of file paths to images
                to embed. Defaults to an empty list.
            system_prompt (str, optional): System-level instructions to guide generation.
            max_tokens (int, optional): Maximum number of tokens to generate. Defaults to 512.
            temperature (float, optional): Sampling temperature for generation. Defaults to 0.7.

        Returns:
            str: The generated text content from the first choice.

        Raises:
            RuntimeError: If the request to the server server fails or returns an error.
        """
        if images is None:
            images = []

        try:
            # Build the messages list, inserting a system message if provided
            messages: list[dict[str, Any]] = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})

            # Construct the “content” list containing text and image_url objects
            content_elements: list[Any] = [{"type": "text", "text": prompt}]
            for img_path in images:
                data_uri = self._encode_image_to_data_uri(img_path)
                content_elements.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": data_uri},
                    }
                )

            messages.append({"role": "user", "content": content_elements})

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except Exception as e:
            msg = f"server generate_with_images request failed: {e}"
            raise RuntimeError(msg)

        try:
            return response.choices[0].message.content.strip()
        except (AttributeError, KeyError, IndexError) as e:
            msg = f"Failed to parse server generate_with_images response: {e}"
            raise RuntimeError(msg)
