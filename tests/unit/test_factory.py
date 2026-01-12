"""Unit tests for factory module."""

from typing import Any

import pytest

from prompt_optimizer.factory import (
    ClientRegistry,
    DefaultClientFactory,
    MockClientFactory,
    get_default_factory,
    set_default_factory,
)
from prompt_optimizer.llm_clients.anthropic import AnthropicClient
from prompt_optimizer.llm_clients.ollama import OllamaClient
from prompt_optimizer.llm_clients.openai import OpenAIClient

pytestmark = pytest.mark.unit


class MockLLMClient:
    """Mock LLM client for testing."""

    def __init__(self, responses: list[str] | None = None) -> None:
        self.responses = responses or ["Mock response"]
        self.call_count = 0
        self.calls: list[dict[str, Any]] = []

    async def generate(
        self,
        prompt: str,
        system: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> str:
        """Return mock response."""
        self.calls.append({
            "prompt": prompt,
            "system": system,
            "max_tokens": max_tokens,
            "temperature": temperature,
        })
        response = self.responses[self.call_count % len(self.responses)]
        self.call_count += 1
        return response

    def count_tokens(self, text: str) -> int:
        """Count tokens."""
        return len(text) // 4

    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate mock cost."""
        return 0.001

    @property
    def model_name(self) -> str:
        """Return mock model name."""
        return "mock-model"


class TestClientRegistry:
    """Tests for ClientRegistry."""

    def test_register_and_get_creator(self) -> None:
        """Test registering and retrieving creators."""
        registry = ClientRegistry()

        def mock_creator(llm: str, api_key: str | None) -> MockLLMClient:
            return MockLLMClient()

        registry.register("test", mock_creator)
        creator = registry.get_creator("test-model")

        assert creator is not None
        client = creator("test-model", None)
        assert isinstance(client, MockLLMClient)

    def test_get_creator_no_match(self) -> None:
        """Test get_creator returns None when no match."""
        registry = ClientRegistry()
        assert registry.get_creator("unknown") is None

    def test_longer_prefix_matches_first(self) -> None:
        """Test that longer prefixes are matched first."""
        registry = ClientRegistry()

        results: list[str] = []

        def short_creator(llm: str, api_key: str | None) -> MockLLMClient:
            results.append("short")
            return MockLLMClient()

        def long_creator(llm: str, api_key: str | None) -> MockLLMClient:
            results.append("long")
            return MockLLMClient()

        registry.register("test", short_creator)
        registry.register("test-long", long_creator)

        creator = registry.get_creator("test-long-model")
        assert creator is not None
        creator("test-long-model", None)
        assert results == ["long"]


class TestDefaultClientFactory:
    """Tests for DefaultClientFactory."""

    def test_create_anthropic_client(self) -> None:
        """Test creating Anthropic client."""
        factory = DefaultClientFactory()
        client = factory.create("claude-sonnet-4")
        assert isinstance(client, AnthropicClient)
        assert client.model == "claude-sonnet-4-20250514"

    def test_create_anthropic_client_with_prefix(self) -> None:
        """Test creating Anthropic client with anthropic: prefix."""
        factory = DefaultClientFactory()
        client = factory.create("anthropic:claude-3-5-sonnet-20241022")
        assert isinstance(client, AnthropicClient)
        assert client.model == "claude-3-5-sonnet-20241022"

    def test_create_openai_client(self) -> None:
        """Test creating OpenAI client."""
        factory = DefaultClientFactory()
        client = factory.create("gpt-4o")
        assert isinstance(client, OpenAIClient)
        assert client.model == "gpt-4o"

    def test_create_openai_client_with_prefix(self) -> None:
        """Test creating OpenAI client with openai: prefix."""
        factory = DefaultClientFactory()
        client = factory.create("openai:gpt-4-turbo")
        assert isinstance(client, OpenAIClient)
        assert client.model == "gpt-4-turbo"

    def test_create_ollama_client(self) -> None:
        """Test creating Ollama client."""
        factory = DefaultClientFactory()
        client = factory.create("ollama:llama3")
        assert isinstance(client, OllamaClient)
        assert client.model == "llama3"

    def test_create_unknown_defaults_to_anthropic(self) -> None:
        """Test that unknown models default to Anthropic."""
        factory = DefaultClientFactory()
        client = factory.create("some-custom-model")
        assert isinstance(client, AnthropicClient)
        assert client.model == "some-custom-model"

    def test_custom_registry(self) -> None:
        """Test using custom registry."""
        registry = ClientRegistry()

        def custom_creator(llm: str, api_key: str | None) -> MockLLMClient:
            return MockLLMClient()

        registry.register("custom", custom_creator)
        factory = DefaultClientFactory(registry)

        client = factory.create("custom-model")
        assert isinstance(client, MockLLMClient)


class TestMockClientFactory:
    """Tests for MockClientFactory."""

    def test_returns_mock_client(self) -> None:
        """Test that factory returns the mock client."""
        mock = MockLLMClient()
        factory = MockClientFactory(mock)

        client = factory.create("any-model")
        assert client is mock

    def test_tracks_create_calls(self) -> None:
        """Test that factory tracks create calls."""
        mock = MockLLMClient()
        factory = MockClientFactory(mock)

        factory.create("model-1", "key-1")
        factory.create("model-2", None)

        assert len(factory.create_calls) == 2
        assert factory.create_calls[0] == {"llm": "model-1", "api_key": "key-1"}
        assert factory.create_calls[1] == {"llm": "model-2", "api_key": None}


class TestDefaultFactorySingleton:
    """Tests for get/set_default_factory."""

    def test_get_default_factory(self) -> None:
        """Test getting default factory."""
        # Reset to ensure clean state
        set_default_factory(None)
        factory = get_default_factory()
        assert isinstance(factory, DefaultClientFactory)

    def test_set_default_factory(self) -> None:
        """Test setting custom default factory."""
        mock = MockLLMClient()
        custom_factory = MockClientFactory(mock)

        set_default_factory(custom_factory)
        factory = get_default_factory()
        assert factory is custom_factory

        # Clean up
        set_default_factory(None)
