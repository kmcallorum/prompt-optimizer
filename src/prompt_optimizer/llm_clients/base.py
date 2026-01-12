"""Base LLM client protocol."""

from typing import Protocol


class LLMClient(Protocol):
    """Protocol for LLM clients."""

    async def generate(
        self,
        prompt: str,
        system: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> str:
        """Generate text from prompt."""
        ...

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        ...

    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate API cost."""
        ...

    @property
    def model_name(self) -> str:
        """Get the model name."""
        ...
