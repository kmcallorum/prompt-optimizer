"""Local Ollama client."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import httpx as httpx_types


class OllamaClient:
    """Local Ollama client.

    Supports dependency injection of the underlying httpx.AsyncClient
    for testing purposes.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama3",
        http_client: httpx_types.AsyncClient | None = None,
    ) -> None:
        """Initialize the Ollama client.

        Args:
            base_url: Base URL for Ollama API.
            model: Model name to use.
            http_client: Optional pre-configured httpx.AsyncClient for DI/testing.
        """
        self.base_url = base_url
        self.model = model
        self._http_client = http_client

    async def generate(
        self,
        prompt: str,
        system: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> str:
        """Generate text from prompt."""
        import httpx

        request_json = {
            "model": self.model,
            "prompt": prompt,
            "system": system or "",
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
            },
        }

        if self._http_client is not None:
            response = await self._http_client.post(
                f"{self.base_url}/api/generate",
                json=request_json,
            )
            response.raise_for_status()
            result: str = response.json()["response"]
            return result

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{self.base_url}/api/generate",
                json=request_json,
            )
            response.raise_for_status()
            result = response.json()["response"]
            return result

    def count_tokens(self, text: str) -> int:
        """Approximate token count (4 chars per token)."""
        return len(text) // 4

    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Local model, no cost."""
        return 0.0

    @property
    def model_name(self) -> str:
        """Get the model name."""
        return self.model
