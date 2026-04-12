from __future__ import annotations
import os
from openai import OpenAI
from typing import List, Dict, Any


class OpenAICompatibleClient:
    """Client for OpenAI-compatible chat completion APIs.

    Supports any API that implements the OpenAI chat completions format,
    including OpenAI, Azure OpenAI, local models via vLLM, etc.

    Attributes:
        client: OpenAI client instance
        model: Model identifier to use for completions

    Example:
        >>> client = OpenAICompatibleClient(
        ...     model="gpt-4",
        ...     base_url="https://api.openai.com/v1",
        ...     api_key="sk-..."
        ... )
        >>> response = client.chat([
        ...     {"role": "system", "content": "You are helpful."},
        ...     {"role": "user", "content": "Hello!"}
        ... ])
    """

    def __init__(self, model: str, base_url: str | None = None, api_key: str | None = None):
        """Initialize the LLM client.

        Args:
            model: Model identifier (e.g., "gpt-4", "gpt-3.5-turbo")
            base_url: Optional base URL for the API. If None, uses OPENAI_BASE_URL
                     environment variable or OpenAI's default.
            api_key: Optional API key. If None, uses OPENAI_API_KEY environment variable.
        """
        self.client = OpenAI(
            base_url=base_url or os.getenv("OPENAI_BASE_URL", None),
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
        )
        self.model = model

    def chat(self, messages: List[Dict[str, str]], **kw: Any) -> str:
        """Send a chat completion request.

        Args:
            messages: List of message dicts with 'role' and 'content' keys
            **kw: Additional parameters (temperature, max_tokens, etc.)

        Returns:
            Generated text response

        Example:
            >>> response = client.chat(
            ...     messages=[{"role": "user", "content": "Hello"}],
            ...     temperature=0.8,
            ...     max_tokens=256
            ... )
        """
        resp = self.client.chat.completions.create(model=self.model, messages=messages, **kw)
        return resp.choices[0].message.content or ""
