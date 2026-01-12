"""Factory pattern for LLM client creation with dependency injection support."""

from collections.abc import Callable
from typing import Protocol

from prompt_optimizer.llm_clients.base import LLMClient


class ClientFactory(Protocol):
    """Protocol for LLM client factories."""

    def create(
        self,
        llm: str,
        api_key: str | None = None,
    ) -> LLMClient:
        """Create an LLM client.

        Args:
            llm: LLM identifier (e.g., 'claude-sonnet-4', 'gpt-4o', 'ollama:llama3')
            api_key: Optional API key

        Returns:
            Configured LLM client
        """


# Type alias for client creator functions
ClientCreator = Callable[[str, str | None], LLMClient]


class ClientRegistry:
    """Registry for LLM client creators.

    Allows registering custom client creators for different LLM providers.
    """

    def __init__(self) -> None:
        self._creators: dict[str, ClientCreator] = {}
        self._prefixes: list[tuple[str, ClientCreator]] = []

    def register(
        self,
        prefix: str,
        creator: ClientCreator,
    ) -> None:
        """Register a client creator for a prefix.

        Args:
            prefix: Prefix to match (e.g., 'claude', 'gpt', 'ollama')
            creator: Function that creates a client given model and api_key
        """
        self._prefixes.append((prefix, creator))
        self._prefixes.sort(key=lambda x: len(x[0]), reverse=True)

    def get_creator(self, llm: str) -> ClientCreator | None:
        """Get the creator for a given LLM identifier.

        Args:
            llm: LLM identifier

        Returns:
            Creator function or None if not found
        """
        for prefix, creator in self._prefixes:
            if llm.startswith(prefix):
                return creator
        return None


class DefaultClientFactory:
    """Default factory for creating LLM clients.

    Uses a registry to map LLM identifiers to client creators.
    Supports Anthropic, OpenAI, and Ollama by default.
    """

    def __init__(self, registry: ClientRegistry | None = None) -> None:
        """Initialize the factory.

        Args:
            registry: Optional custom registry. Uses default if not provided.
        """
        self._registry = registry or self._create_default_registry()

    def _create_default_registry(self) -> ClientRegistry:
        """Create the default client registry with standard providers."""
        from prompt_optimizer.llm_clients.anthropic import AnthropicClient
        from prompt_optimizer.llm_clients.ollama import OllamaClient
        from prompt_optimizer.llm_clients.openai import OpenAIClient

        registry = ClientRegistry()

        def create_anthropic(llm: str, api_key: str | None) -> LLMClient:
            model = llm.replace("anthropic:", "")
            if model == "claude-sonnet-4":
                model = "claude-sonnet-4-20250514"
            elif model == "claude-opus-4":
                model = "claude-opus-4-20250514"
            return AnthropicClient(api_key=api_key, model=model)

        def create_openai(llm: str, api_key: str | None) -> LLMClient:
            model = llm.replace("openai:", "")
            return OpenAIClient(api_key=api_key, model=model)

        def create_ollama(llm: str, api_key: str | None) -> LLMClient:
            model = llm.replace("ollama:", "")
            return OllamaClient(model=model)

        # Register providers (order matters - longer prefixes first)
        registry.register("anthropic:", create_anthropic)
        registry.register("claude", create_anthropic)
        registry.register("openai:", create_openai)
        registry.register("gpt", create_openai)
        registry.register("ollama:", create_ollama)
        registry.register("ollama", create_ollama)

        return registry

    def create(
        self,
        llm: str,
        api_key: str | None = None,
    ) -> LLMClient:
        """Create an LLM client.

        Args:
            llm: LLM identifier (e.g., 'claude-sonnet-4', 'gpt-4o', 'ollama:llama3')
            api_key: Optional API key

        Returns:
            Configured LLM client

        Raises:
            ValueError: If no creator is registered for the LLM
        """
        creator = self._registry.get_creator(llm)
        if creator:
            return creator(llm, api_key)

        # Default to Anthropic for unknown models
        from prompt_optimizer.llm_clients.anthropic import AnthropicClient

        return AnthropicClient(api_key=api_key, model=llm)


class MockClientFactory:
    """Factory that returns mock clients for testing.

    Tracks all create calls for assertion purposes.
    """

    def __init__(self, mock_client: LLMClient) -> None:
        """Initialize with a mock client.

        Args:
            mock_client: Client to return for all create calls
        """
        self._mock_client = mock_client
        self.create_calls: list[dict[str, str | None]] = []

    def create(
        self,
        llm: str,
        api_key: str | None = None,
    ) -> LLMClient:
        """Return the mock client and track the call.

        Args:
            llm: LLM identifier (tracked but not used)
            api_key: API key (tracked but not used)

        Returns:
            The mock client
        """
        self.create_calls.append({"llm": llm, "api_key": api_key})
        return self._mock_client


# Default factory instance
_default_factory: ClientFactory | None = None


def get_default_factory() -> ClientFactory:
    """Get the default client factory.

    Returns:
        The default ClientFactory instance
    """
    global _default_factory
    if _default_factory is None:
        _default_factory = DefaultClientFactory()
    return _default_factory


def set_default_factory(factory: ClientFactory | None) -> None:
    """Set the default client factory.

    Args:
        factory: Factory to use as default, or None to reset to DefaultClientFactory
    """
    global _default_factory
    _default_factory = factory
