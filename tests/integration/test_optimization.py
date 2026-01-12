"""Integration tests for full optimization flow."""

import pytest

from prompt_optimizer.core import (
    evaluate_all_variants,
    evaluate_variant,
    generate_variants,
)
from prompt_optimizer.evaluator import select_best_variant
from prompt_optimizer.prompt import Prompt, TestCase

# Import MockLLMClient from conftest
from tests.conftest import MockLLMClient


@pytest.mark.asyncio
async def test_evaluate_single_variant() -> None:
    """Test evaluating a single variant."""
    prompt = Prompt(
        template="Answer: {{ question }}",
        name="test",
    )
    variant = generate_variants(prompt, ["concise"])[0]

    test_case = TestCase(
        input_variables={"question": "What is 2+2?"},
        expected_output="4",
    )

    mock_client = MockLLMClient(responses=["4"])
    result = await evaluate_variant(variant, test_case, mock_client)

    assert result.output == "4"
    assert result.scores["accuracy"] == 1.0
    assert result.latency_ms > 0
    assert result.cost_usd > 0


@pytest.mark.asyncio
async def test_evaluate_all_variants() -> None:
    """Test evaluating multiple variants against multiple test cases."""
    prompt = Prompt(
        template="Answer: {{ question }}",
        name="test",
    )
    variants = generate_variants(prompt, ["concise", "detailed"])

    test_cases = [
        TestCase(
            input_variables={"question": "What is 2+2?"},
            expected_output="4",
        ),
        TestCase(
            input_variables={"question": "What is 3+3?"},
            expected_output="6",
        ),
    ]

    mock_client = MockLLMClient(responses=["4", "6", "4", "6"])
    results = await evaluate_all_variants(variants, test_cases, mock_client)

    # 2 variants * 2 test cases = 4 results
    assert len(results) == 4
    assert mock_client.call_count == 4


@pytest.mark.asyncio
async def test_full_optimization_flow() -> None:
    """Test complete optimization workflow."""
    # Create prompt
    prompt = Prompt(
        template="Question: {{ question }}\nAnswer:",
        name="qa-prompt",
    )

    # Create test cases
    test_cases = [
        TestCase(
            input_variables={"question": "What is 2+2?"},
            expected_output="4",
        ),
        TestCase(
            input_variables={"question": "Capital of France?"},
            expected_output="Paris",
        ),
    ]

    # Generate variants
    variants = generate_variants(prompt, ["concise", "detailed", "cot"])
    assert len(variants) == 3

    # Evaluate with mock client
    mock_client = MockLLMClient(responses=["4", "Paris"] * 3)
    results = await evaluate_all_variants(variants, test_cases, mock_client)

    assert len(results) == 6  # 3 variants * 2 test cases

    # Select best
    best_variant, score = select_best_variant(results)
    assert best_variant is not None
    assert score > 0


@pytest.mark.asyncio
async def test_optimization_with_concurrency() -> None:
    """Test that concurrency limiting works."""
    prompt = Prompt(template="Test {{ n }}", name="test")
    variants = generate_variants(prompt, ["concise"])

    test_cases = [
        TestCase(input_variables={"n": str(i)})
        for i in range(10)
    ]

    mock_client = MockLLMClient(responses=["response"] * 10)
    results = await evaluate_all_variants(
        variants, test_cases, mock_client, concurrency=2
    )

    assert len(results) == 10
    assert mock_client.call_count == 10


@pytest.mark.asyncio
async def test_variant_rendering_in_evaluation() -> None:
    """Test that variants are properly rendered during evaluation."""
    prompt = Prompt(
        template="Hello {{ name }}!",
        variables={"name": "default"},
        name="greeting",
    )
    variant = generate_variants(prompt, ["concise"])[0]

    test_case = TestCase(
        input_variables={"name": "Claude"},
    )

    mock_client = MockLLMClient()
    await evaluate_variant(variant, test_case, mock_client)

    # Check that the rendered prompt was passed to the client
    assert len(mock_client.calls) == 1
    assert "Claude" in mock_client.calls[0]["prompt"]
