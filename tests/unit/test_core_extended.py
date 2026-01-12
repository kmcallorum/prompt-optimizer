"""Extended tests for core module to achieve full coverage."""

from unittest.mock import AsyncMock, patch

import pytest

from prompt_optimizer.core import (
    get_llm_client,
    optimize_prompt,
    optimize_prompt_async,
)
from prompt_optimizer.prompt import Prompt, TestCase


class TestGetLLMClientExtended:
    """Extended tests for get_llm_client."""

    def test_anthropic_sonnet_shorthand(self) -> None:
        """Test claude-sonnet-4 shorthand."""
        client = get_llm_client("claude-sonnet-4")
        assert client.model == "claude-sonnet-4-20250514"

    def test_anthropic_opus_shorthand(self) -> None:
        """Test claude-opus-4 shorthand."""
        client = get_llm_client("claude-opus-4")
        assert client.model == "claude-opus-4-20250514"

    def test_anthropic_prefix(self) -> None:
        """Test anthropic: prefix."""
        client = get_llm_client("anthropic:claude-3-5-sonnet-20241022")
        assert client.model == "claude-3-5-sonnet-20241022"

    def test_openai_prefix(self) -> None:
        """Test openai: prefix."""
        client = get_llm_client("openai:gpt-4-turbo")
        assert client.model == "gpt-4-turbo"


class TestOptimizePromptSync:
    """Tests for synchronous optimize_prompt wrapper."""

    def test_sync_wrapper(self) -> None:
        """Test that sync wrapper calls async function."""
        prompt = Prompt(template="Test {{ q }}", name="test")
        test_cases = [TestCase(input_variables={"q": "question"})]

        mock_client = AsyncMock()
        mock_client.generate = AsyncMock(return_value="answer")
        mock_client.count_tokens = lambda x: 10
        mock_client.calculate_cost = lambda i, o: 0.001

        with patch(
            "prompt_optimizer.core.get_llm_client", return_value=mock_client
        ):
            results = optimize_prompt(
                prompt=prompt,
                test_cases=test_cases,
                strategies=["concise"],
                llm="claude-sonnet-4",
            )

        assert results.variants_tested == 1
        assert results.best_variant is not None


@pytest.mark.asyncio
async def test_optimize_prompt_async_full() -> None:
    """Test full async optimization flow."""
    prompt = Prompt(template="Answer {{ q }}", name="test")
    test_cases = [
        TestCase(input_variables={"q": "What is 1+1?"}, expected_output="2"),
        TestCase(input_variables={"q": "What is 2+2?"}, expected_output="4"),
    ]

    mock_client = AsyncMock()
    mock_client.generate = AsyncMock(side_effect=["2", "4", "Two", "Four", "2", "4"])
    mock_client.count_tokens = lambda x: len(x) // 4
    mock_client.calculate_cost = lambda i, o: 0.001

    with patch("prompt_optimizer.core.get_llm_client", return_value=mock_client):
        results = await optimize_prompt_async(
            prompt=prompt,
            test_cases=test_cases,
            strategies=["concise", "detailed", "cot"],
            llm="claude-sonnet-4",
        )

    assert results.variants_tested == 3
    assert results.test_cases_run == 2
    assert results.best_variant is not None
    assert results.total_cost > 0


@pytest.mark.asyncio
async def test_optimize_prompt_async_with_weights() -> None:
    """Test optimization with custom weights."""
    prompt = Prompt(template="Test {{ q }}", name="test")
    test_cases = [TestCase(input_variables={"q": "q"})]

    mock_client = AsyncMock()
    mock_client.generate = AsyncMock(return_value="answer")
    mock_client.count_tokens = lambda x: 10
    mock_client.calculate_cost = lambda i, o: 0.001

    with patch("prompt_optimizer.core.get_llm_client", return_value=mock_client):
        results = await optimize_prompt_async(
            prompt=prompt,
            test_cases=test_cases,
            strategies=["concise"],
            weights={"accuracy": 0.8, "conciseness": 0.2},
        )

    assert results.best_variant is not None


@pytest.mark.asyncio
async def test_optimize_prompt_async_default_strategies() -> None:
    """Test optimization with default strategies (None)."""
    prompt = Prompt(template="Test {{ q }}", name="test")
    test_cases = [TestCase(input_variables={"q": "q"})]

    mock_client = AsyncMock()
    mock_client.generate = AsyncMock(return_value="answer")
    mock_client.count_tokens = lambda x: 10
    mock_client.calculate_cost = lambda i, o: 0.001

    with patch("prompt_optimizer.core.get_llm_client", return_value=mock_client):
        results = await optimize_prompt_async(
            prompt=prompt,
            test_cases=test_cases,
            # strategies=None by default
        )

    # Should use default strategies: concise, detailed
    assert results.variants_tested == 2
    assert results.best_variant is not None
