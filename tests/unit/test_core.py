"""Unit tests for core module."""


from prompt_optimizer.core import generate_variants, get_llm_client
from prompt_optimizer.llm_clients.anthropic import AnthropicClient
from prompt_optimizer.llm_clients.ollama import OllamaClient
from prompt_optimizer.llm_clients.openai import OpenAIClient
from prompt_optimizer.prompt import Prompt


class TestGenerateVariants:
    """Tests for generate_variants function."""

    def test_generate_single_variant(self, sample_prompt: Prompt) -> None:
        """Test generating a single variant."""
        variants = generate_variants(sample_prompt, ["concise"])
        assert len(variants) == 1
        assert variants[0].strategy == "concise"
        assert "concise" in variants[0].template.lower()

    def test_generate_multiple_variants(self, sample_prompt: Prompt) -> None:
        """Test generating multiple variants."""
        strategies = ["concise", "detailed", "cot"]
        variants = generate_variants(sample_prompt, strategies)
        assert len(variants) == 3
        assert [v.strategy for v in variants] == strategies

    def test_variant_contains_base_template(self, sample_prompt: Prompt) -> None:
        """Test variant contains base template."""
        variants = generate_variants(sample_prompt, ["concise"])
        assert sample_prompt.template in variants[0].template

    def test_variant_preserves_system_message(self, sample_prompt: Prompt) -> None:
        """Test variant preserves system message."""
        variants = generate_variants(sample_prompt, ["detailed"])
        assert variants[0].system_message == sample_prompt.system_message


class TestGetLLMClient:
    """Tests for get_llm_client function."""

    def test_get_anthropic_client(self) -> None:
        """Test getting Anthropic client."""
        client = get_llm_client("claude-sonnet-4")
        assert isinstance(client, AnthropicClient)

    def test_get_openai_client(self) -> None:
        """Test getting OpenAI client."""
        client = get_llm_client("gpt-4o")
        assert isinstance(client, OpenAIClient)

    def test_get_ollama_client(self) -> None:
        """Test getting Ollama client."""
        client = get_llm_client("ollama:llama3")
        assert isinstance(client, OllamaClient)

    def test_default_to_anthropic(self) -> None:
        """Test defaulting to Anthropic for unknown models."""
        client = get_llm_client("unknown-model")
        assert isinstance(client, AnthropicClient)
