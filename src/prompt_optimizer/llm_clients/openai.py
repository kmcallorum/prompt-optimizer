"""OpenAI API client."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openai import AsyncOpenAI as AsyncOpenAIType


class OpenAIClient:
    """OpenAI API client.

    Supports dependency injection of the underlying AsyncOpenAI client
    for testing purposes.
    """

    PRICING = {
        "gpt-4": {"input": 30.0, "output": 60.0},
        "gpt-4-turbo": {"input": 10.0, "output": 30.0},
        "gpt-4o": {"input": 2.5, "output": 10.0},
        "gpt-4o-mini": {"input": 0.15, "output": 0.6},
        "gpt-3.5-turbo": {"input": 0.5, "output": 1.5},
    }

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4o",
        client: AsyncOpenAIType | None = None,
    ) -> None:
        """Initialize the OpenAI client.

        Args:
            api_key: API key for OpenAI. Falls back to OPENAI_API_KEY env var.
            model: Model name to use.
            client: Optional pre-configured AsyncOpenAI client for DI/testing.
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if client is not None:
            self.client = client
        else:
            from openai import AsyncOpenAI

            self.client = AsyncOpenAI(api_key=self.api_key)
        self.model = model

    async def generate(
        self,
        prompt: str,
        system: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> str:
        """Generate text from prompt."""
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,  # type: ignore[arg-type]
            max_tokens=max_tokens,
            temperature=temperature,
        )
        content = response.choices[0].message.content
        return content or ""

    def count_tokens(self, text: str) -> int:
        """Approximate token count (4 chars per token)."""
        return len(text) // 4

    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate API cost in USD."""
        pricing = self.PRICING.get(self.model, {"input": 2.5, "output": 10.0})
        input_cost = input_tokens * pricing["input"] / 1_000_000
        output_cost = output_tokens * pricing["output"] / 1_000_000
        return input_cost + output_cost

    @property
    def model_name(self) -> str:
        """Get the model name."""
        return self.model
