"""Pytest fixtures for prompt-optimizer tests."""

from typing import Any

import pytest

from prompt_optimizer.prompt import Prompt, PromptVariant, TestCase, TestSuite


class MockLLMClient:
    """Mock LLM client for testing.

    Implements the LLMClient protocol for use in tests.
    """

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


class MockMetricsRecorder:
    """Mock metrics recorder for testing.

    Implements the MetricsRecorder protocol and tracks all calls for assertions.
    """

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def init_metrics(self, version: str = "0.3.0") -> None:
        """Track init_metrics call."""
        self.calls.append({"method": "init_metrics", "version": version})

    def record_optimization_start(self, prompt_name: str) -> None:
        """Track optimization start."""
        self.calls.append({
            "method": "record_optimization_start",
            "prompt_name": prompt_name,
        })

    def record_optimization_complete(
        self,
        prompt_name: str,
        duration_seconds: float,
        success: bool = True,
    ) -> None:
        """Track optimization complete."""
        self.calls.append({
            "method": "record_optimization_complete",
            "prompt_name": prompt_name,
            "duration_seconds": duration_seconds,
            "success": success,
        })

    def record_variant_evaluation(
        self,
        prompt_name: str,
        strategy: str,
        scores: dict[str, float],
    ) -> None:
        """Track variant evaluation."""
        self.calls.append({
            "method": "record_variant_evaluation",
            "prompt_name": prompt_name,
            "strategy": strategy,
            "scores": scores,
        })

    def record_test_case_result(
        self,
        prompt_name: str,
        passed: bool,
    ) -> None:
        """Track test case result."""
        self.calls.append({
            "method": "record_test_case_result",
            "prompt_name": prompt_name,
            "passed": passed,
        })

    def record_llm_request(
        self,
        llm: str,
        operation: str,
        duration_seconds: float,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float,
    ) -> None:
        """Track LLM request."""
        self.calls.append({
            "method": "record_llm_request",
            "llm": llm,
            "operation": operation,
            "duration_seconds": duration_seconds,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": cost_usd,
        })

    def record_judge_evaluation(
        self,
        judge_llm: str,
        prompt_name: str,
        scores: dict[str, float],
    ) -> None:
        """Track judge evaluation."""
        self.calls.append({
            "method": "record_judge_evaluation",
            "judge_llm": judge_llm,
            "prompt_name": prompt_name,
            "scores": scores,
        })

    def record_best_variant(
        self,
        prompt_name: str,
        strategy: str,
        score: float,
    ) -> None:
        """Track best variant."""
        self.calls.append({
            "method": "record_best_variant",
            "prompt_name": prompt_name,
            "strategy": strategy,
            "score": score,
        })

    def get_calls(self, method: str) -> list[dict[str, Any]]:
        """Get all calls to a specific method."""
        return [c for c in self.calls if c["method"] == method]


class MockClientFactory:
    """Mock client factory for testing.

    Returns a pre-configured mock client for all create() calls.
    """

    def __init__(self, mock_client: MockLLMClient | None = None) -> None:
        self.mock_client = mock_client or MockLLMClient()
        self.create_calls: list[dict[str, Any]] = []

    def create(
        self,
        llm: str,
        api_key: str | None = None,
    ) -> MockLLMClient:
        """Return the mock client and track the call."""
        self.create_calls.append({"llm": llm, "api_key": api_key})
        return self.mock_client


@pytest.fixture
def mock_llm_client() -> MockLLMClient:
    """Create a mock LLM client."""
    return MockLLMClient()


@pytest.fixture
def mock_metrics() -> MockMetricsRecorder:
    """Create a mock metrics recorder."""
    return MockMetricsRecorder()


@pytest.fixture
def mock_client_factory(mock_llm_client: MockLLMClient) -> MockClientFactory:
    """Create a mock client factory."""
    return MockClientFactory(mock_llm_client)


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
