"""Unit tests for LLM clients."""

import pytest

from prompt_optimizer.llm_clients.anthropic import AnthropicClient
from prompt_optimizer.llm_clients.ollama import OllamaClient
from prompt_optimizer.llm_clients.openai import OpenAIClient


class TestAnthropicClient:
    """Tests for AnthropicClient."""

    def test_initialization(self) -> None:
        """Test client initialization."""
        client = AnthropicClient(api_key="test-key", model="claude-sonnet-4-20250514")
        assert client.model == "claude-sonnet-4-20250514"
        assert client.api_key == "test-key"

    def test_model_name_property(self) -> None:
        """Test model_name property."""
        client = AnthropicClient(api_key="test", model="claude-3-5-sonnet-20241022")
        assert client.model_name == "claude-3-5-sonnet-20241022"

    def test_count_tokens(self) -> None:
        """Test token counting."""
        client = AnthropicClient(api_key="test")
        count = client.count_tokens("This is a test string")
        assert count > 0
        assert isinstance(count, int)

    def test_calculate_cost(self) -> None:
        """Test cost calculation."""
        client = AnthropicClient(api_key="test", model="claude-sonnet-4-20250514")
        cost = client.calculate_cost(1000, 500)
        assert cost > 0
        assert isinstance(cost, float)

    def test_calculate_cost_unknown_model(self) -> None:
        """Test cost calculation with unknown model."""
        client = AnthropicClient(api_key="test", model="unknown-model")
        cost = client.calculate_cost(1000, 500)
        assert cost > 0


class TestOpenAIClient:
    """Tests for OpenAIClient."""

    def test_initialization(self) -> None:
        """Test client initialization."""
        client = OpenAIClient(api_key="test-key", model="gpt-4o")
        assert client.model == "gpt-4o"
        assert client.api_key == "test-key"

    def test_model_name_property(self) -> None:
        """Test model_name property."""
        client = OpenAIClient(api_key="test", model="gpt-4-turbo")
        assert client.model_name == "gpt-4-turbo"

    def test_count_tokens(self) -> None:
        """Test token counting."""
        client = OpenAIClient(api_key="test")
        count = client.count_tokens("This is a test string")
        assert count > 0

    def test_calculate_cost(self) -> None:
        """Test cost calculation."""
        client = OpenAIClient(api_key="test", model="gpt-4o")
        cost = client.calculate_cost(1000, 500)
        assert cost > 0

    def test_calculate_cost_unknown_model(self) -> None:
        """Test cost calculation with unknown model."""
        client = OpenAIClient(api_key="test", model="unknown-model")
        cost = client.calculate_cost(1000, 500)
        assert cost > 0


class TestOllamaClient:
    """Tests for OllamaClient."""

    def test_initialization(self) -> None:
        """Test client initialization."""
        client = OllamaClient(base_url="http://localhost:11434", model="llama3")
        assert client.model == "llama3"
        assert client.base_url == "http://localhost:11434"

    def test_model_name_property(self) -> None:
        """Test model_name property."""
        client = OllamaClient(model="mistral")
        assert client.model_name == "mistral"

    def test_count_tokens(self) -> None:
        """Test token counting."""
        client = OllamaClient()
        count = client.count_tokens("This is a test string")
        assert count > 0

    def test_calculate_cost(self) -> None:
        """Test cost calculation (should be 0 for local)."""
        client = OllamaClient()
        cost = client.calculate_cost(1000, 500)
        assert cost == 0.0
