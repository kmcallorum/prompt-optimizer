"""Pytest fixtures for prompt-optimizer tests."""

from typing import Any

import pytest

from prompt_optimizer.prompt import Prompt, PromptVariant, TestCase, TestSuite


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
        self.calls.append(
            {
                "prompt": prompt,
                "system": system,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
        )
        response = self.responses[self.call_count % len(self.responses)]
        self.call_count += 1
        return response

    def count_tokens(self, text: str) -> int:
        """Count tokens (4 chars per token approximation)."""
        return len(text) // 4

    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate mock cost."""
        return 0.001

    @property
    def model_name(self) -> str:
        """Return mock model name."""
        return "mock-model"


@pytest.fixture
def mock_llm_client() -> MockLLMClient:
    """Create a mock LLM client."""
    return MockLLMClient()


@pytest.fixture
def sample_prompt() -> Prompt:
    """Create a sample prompt for testing."""
    return Prompt(
        template="Answer the question: {{ question }}",
        variables={"question": "What is 2+2?"},
        system_message="You are a helpful assistant.",
        name="test-prompt",
    )


@pytest.fixture
def sample_test_case() -> TestCase:
    """Create a sample test case."""
    return TestCase(
        input_variables={"question": "What is 2+2?"},
        expected_output="4",
        expected_properties={"length": "<20 words"},
    )


@pytest.fixture
def sample_test_suite(sample_test_case: TestCase) -> TestSuite:
    """Create a sample test suite."""
    return TestSuite(
        name="Sample Tests",
        test_cases=[
            sample_test_case,
            TestCase(
                input_variables={"question": "What is the capital of France?"},
                expected_output="Paris",
            ),
        ],
    )


@pytest.fixture
def sample_variant(sample_prompt: Prompt) -> PromptVariant:
    """Create a sample prompt variant."""
    return PromptVariant(
        base_prompt=sample_prompt,
        strategy="concise",
        template=sample_prompt.template + "\n\nBe concise.",
        system_message=sample_prompt.system_message,
    )
