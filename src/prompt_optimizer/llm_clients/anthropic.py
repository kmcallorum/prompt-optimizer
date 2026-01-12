"""Anthropic Claude API client."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from anthropic import AsyncAnthropic as AsyncAnthropicType


class AnthropicClient:
    """Claude API client.

    Supports dependency injection of the underlying AsyncAnthropic client
    for testing purposes.
    """

    PRICING = {
        "claude-sonnet-4-20250514": {"input": 3.0, "output": 15.0},
        "claude-opus-4-20250514": {"input": 15.0, "output": 75.0},
        "claude-3-5-sonnet-20241022": {"input": 3.0, "output": 15.0},
        "claude-3-5-haiku-20241022": {"input": 0.8, "output": 4.0},
    }

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-sonnet-4-20250514",
        client: AsyncAnthropicType | None = None,
    ) -> None:
        """Initialize the Anthropic client.

        Args:
            api_key: API key for Anthropic. Falls back to ANTHROPIC_API_KEY env var.
            model: Model name to use.
            client: Optional pre-configured AsyncAnthropic client for DI/testing.
        """
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if client is not None:
            self.client = client
        else:
            from anthropic import AsyncAnthropic

            self.client = AsyncAnthropic(api_key=self.api_key)
        self.model = model

    async def generate(
        self,
        prompt: str,
        system: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> str:
        """Generate text from prompt."""
        response = await self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system or "",
            messages=[{"role": "user", "content": prompt}],
        )
        content = response.content[0]
        if hasattr(content, "text"):
            return content.text
        return str(content)

    def count_tokens(self, text: str) -> int:
        """Approximate token count (4 chars per token)."""
        return len(text) // 4

    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate API cost in USD."""
        pricing = self.PRICING.get(
            self.model, {"input": 3.0, "output": 15.0}
        )
        input_cost = input_tokens * pricing["input"] / 1_000_000
        output_cost = output_tokens * pricing["output"] / 1_000_000
        return input_cost + output_cost

    @property
    def model_name(self) -> str:
        """Get the model name."""
        return self.model
