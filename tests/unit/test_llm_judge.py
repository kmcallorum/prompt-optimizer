"""Unit tests for LLM judge module."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from prompt_optimizer.llm_judge import LLMJudge, llm_evaluate_response
from prompt_optimizer.prompt import TestCase

pytestmark = pytest.mark.unit


class TestLLMJudge:
    """Tests for LLMJudge class."""

    @pytest.fixture
    def mock_client(self) -> MagicMock:
        """Create a mock LLM client."""
        client = MagicMock()
        client.generate = AsyncMock()
        return client

    @pytest.fixture
    def judge(self, mock_client: MagicMock) -> LLMJudge:
        """Create an LLM judge with mock client."""
        return LLMJudge(mock_client)

    @pytest.fixture
    def test_case(self) -> TestCase:
        """Create a test case."""
        return TestCase(
            input_variables={"question": "What is 2+2?"},
            expected_output="4",
            expected_properties={"length": "<10 words"},
        )

    @pytest.mark.asyncio
    async def test_evaluate_parses_json_response(
        self, judge: LLMJudge, mock_client: MagicMock, test_case: TestCase
    ) -> None:
        """Test that evaluate parses JSON response from judge."""
        mock_client.generate.return_value = (
            '{"accuracy": 0.9, "relevance": 0.8, "coherence": 0.85, '
            '"completeness": 0.7, "conciseness": 0.95}'
        )

        scores = await judge.evaluate("4", test_case, "What is 2+2?")

        assert "accuracy" in scores
        assert scores["accuracy"] == 0.9
        assert scores["relevance"] == 0.8

    @pytest.mark.asyncio
    async def test_evaluate_with_custom_criteria(
        self, mock_client: MagicMock, test_case: TestCase
    ) -> None:
        """Test evaluation with custom criteria filter."""
        judge = LLMJudge(mock_client, criteria=["accuracy", "conciseness"])
        mock_client.generate.return_value = (
            '{"accuracy": 0.9, "relevance": 0.8, "coherence": 0.85, '
            '"completeness": 0.7, "conciseness": 0.95}'
        )

        scores = await judge.evaluate("4", test_case, "What is 2+2?")

        # Should only include requested criteria
        assert "accuracy" in scores
        assert "conciseness" in scores
        assert "relevance" not in scores
        assert "coherence" not in scores

    @pytest.mark.asyncio
    async def test_evaluate_handles_malformed_json(
        self, judge: LLMJudge, mock_client: MagicMock, test_case: TestCase
    ) -> None:
        """Test that evaluate handles malformed JSON gracefully."""
        mock_client.generate.return_value = "accuracy: 0.9, relevance: 0.8"

        scores = await judge.evaluate("4", test_case, "What is 2+2?")

        # Should parse individual scores
        assert "accuracy" in scores
        assert scores["accuracy"] == 0.9

    @pytest.mark.asyncio
    async def test_evaluate_returns_defaults_on_parse_failure(
        self, judge: LLMJudge, mock_client: MagicMock, test_case: TestCase
    ) -> None:
        """Test that evaluate returns defaults when parsing fails completely."""
        mock_client.generate.return_value = "I cannot evaluate this response."

        scores = await judge.evaluate("4", test_case, "What is 2+2?")

        # Should return default scores
        assert scores["accuracy"] == 0.5
        assert scores["relevance"] == 0.5

    @pytest.mark.asyncio
    async def test_evaluate_clamps_scores(
        self, judge: LLMJudge, mock_client: MagicMock, test_case: TestCase
    ) -> None:
        """Test that scores are clamped to 0.0-1.0 range."""
        mock_client.generate.return_value = (
            '{"accuracy": 1.5, "relevance": -0.2, "coherence": 0.5, '
            '"completeness": 0.5, "conciseness": 0.5}'
        )

        scores = await judge.evaluate("4", test_case, "What is 2+2?")

        assert scores["accuracy"] == 1.0  # Clamped from 1.5
        assert scores["relevance"] == 0.0  # Clamped from -0.2


class TestParseScores:
    """Tests for _parse_scores method."""

    @pytest.fixture
    def judge(self) -> LLMJudge:
        """Create judge with mock client."""
        mock_client = MagicMock()
        return LLMJudge(mock_client)

    def test_parse_valid_json(self, judge: LLMJudge) -> None:
        """Test parsing valid JSON."""
        response = '{"accuracy": 0.9, "relevance": 0.8}'
        scores = judge._parse_scores(response)

        assert scores["accuracy"] == 0.9
        assert scores["relevance"] == 0.8

    def test_parse_json_with_surrounding_text(self, judge: LLMJudge) -> None:
        """Test parsing JSON embedded in text."""
        response = 'Here are the scores: {"accuracy": 0.85, "relevance": 0.7} Done.'
        scores = judge._parse_scores(response)

        assert scores["accuracy"] == 0.85
        assert scores["relevance"] == 0.7

    def test_parse_key_value_pairs(self, judge: LLMJudge) -> None:
        """Test parsing key=value format."""
        response = "accuracy = 0.9\nrelevance = 0.8"
        scores = judge._parse_scores(response)

        assert scores["accuracy"] == 0.9
        assert scores["relevance"] == 0.8

    def test_parse_quoted_keys(self, judge: LLMJudge) -> None:
        """Test parsing quoted keys."""
        response = '"accuracy": 0.75, "relevance": 0.65'
        scores = judge._parse_scores(response)

        assert scores["accuracy"] == 0.75
        assert scores["relevance"] == 0.65

    def test_parse_returns_defaults_for_unparseable(self, judge: LLMJudge) -> None:
        """Test that unparseable response returns defaults."""
        response = "This response has no scores."
        scores = judge._parse_scores(response)

        assert scores["accuracy"] == 0.5
        assert scores["relevance"] == 0.5
        assert scores["coherence"] == 0.5


class TestLlmEvaluateResponse:
    """Tests for llm_evaluate_response convenience function."""

    @pytest.mark.asyncio
    async def test_convenience_function(self) -> None:
        """Test the convenience function creates judge and evaluates."""
        mock_client = MagicMock()
        mock_client.generate = AsyncMock(
            return_value='{"accuracy": 0.9, "relevance": 0.8, "coherence": 0.85, '
            '"completeness": 0.7, "conciseness": 0.95}'
        )

        test_case = TestCase(
            input_variables={"q": "test"},
            expected_output="answer",
        )

        scores = await llm_evaluate_response(
            response="answer",
            test_case=test_case,
            rendered_prompt="test prompt",
            judge_client=mock_client,
        )

        assert "accuracy" in scores
        assert scores["accuracy"] == 0.9

    @pytest.mark.asyncio
    async def test_convenience_function_with_criteria(self) -> None:
        """Test convenience function with custom criteria."""
        mock_client = MagicMock()
        mock_client.generate = AsyncMock(
            return_value='{"accuracy": 0.9, "relevance": 0.8, "coherence": 0.85, '
            '"completeness": 0.7, "conciseness": 0.95}'
        )

        test_case = TestCase(input_variables={})

        scores = await llm_evaluate_response(
            response="test",
            test_case=test_case,
            rendered_prompt="prompt",
            judge_client=mock_client,
            criteria=["accuracy"],
        )

        assert "accuracy" in scores
        assert "relevance" not in scores
