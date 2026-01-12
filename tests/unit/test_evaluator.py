"""Unit tests for evaluator module."""

import pytest

from prompt_optimizer.evaluator import (
    EvaluationResult,
    accuracy_scorer,
    conciseness_scorer,
    evaluate_response,
    includes_scorer,
    select_best_variant,
)
from prompt_optimizer.prompt import Prompt, PromptVariant, TestCase

pytestmark = pytest.mark.unit


class TestAccuracyScorer:
    """Tests for accuracy scoring."""

    def test_exact_match(self) -> None:
        """Test exact match returns 1.0."""
        tc = TestCase(input_variables={}, expected_output="Paris")
        score = accuracy_scorer("Paris", tc)
        assert score == 1.0

    def test_case_insensitive(self) -> None:
        """Test case insensitive matching."""
        tc = TestCase(input_variables={}, expected_output="Paris")
        score = accuracy_scorer("PARIS", tc)
        assert score == 1.0

    def test_partial_match(self) -> None:
        """Test partial match returns less than 1.0."""
        tc = TestCase(input_variables={}, expected_output="Paris")
        score = accuracy_scorer("The capital is Paris", tc)
        assert 0.0 < score < 1.0

    def test_no_expected_output(self) -> None:
        """Test no expected output returns 1.0."""
        tc = TestCase(input_variables={})
        score = accuracy_scorer("anything", tc)
        assert score == 1.0


class TestConcisenessScorer:
    """Tests for conciseness scoring."""

    def test_short_response(self) -> None:
        """Test short response scores well."""
        tc = TestCase(input_variables={})
        score = conciseness_scorer("Short answer", tc)
        assert score == 1.0

    def test_length_constraint_met(self) -> None:
        """Test length constraint is met."""
        tc = TestCase(
            input_variables={},
            expected_properties={"length": "<20 words"},
        )
        score = conciseness_scorer("This is a short answer.", tc)
        assert score > 0.5

    def test_length_constraint_exceeded(self) -> None:
        """Test length constraint exceeded."""
        tc = TestCase(
            input_variables={},
            expected_properties={"length": "<5 words"},
        )
        long_response = " ".join(["word"] * 50)
        score = conciseness_scorer(long_response, tc)
        assert score < 0.5

    def test_range_constraint_met(self) -> None:
        """Test range constraint met."""
        tc = TestCase(
            input_variables={},
            expected_properties={"length": "5-10 words"},
        )
        response = "This is a seven word test response."
        score = conciseness_scorer(response, tc)
        assert score == 1.0

    def test_range_constraint_under(self) -> None:
        """Test range constraint when under minimum."""
        tc = TestCase(
            input_variables={},
            expected_properties={"length": "10-20 words"},
        )
        response = "Short."
        score = conciseness_scorer(response, tc)
        assert 0 < score < 1.0

    def test_range_constraint_over(self) -> None:
        """Test range constraint when over maximum."""
        tc = TestCase(
            input_variables={},
            expected_properties={"length": "2-5 words"},
        )
        response = " ".join(["word"] * 20)
        score = conciseness_scorer(response, tc)
        assert 0 < score < 1.0

    def test_medium_length_response(self) -> None:
        """Test medium length response without constraints."""
        tc = TestCase(input_variables={})
        response = " ".join(["word"] * 75)
        score = conciseness_scorer(response, tc)
        assert score == 0.8

    def test_long_response(self) -> None:
        """Test long response without constraints."""
        tc = TestCase(input_variables={})
        response = " ".join(["word"] * 150)
        score = conciseness_scorer(response, tc)
        assert score == 0.6

    def test_very_long_response(self) -> None:
        """Test very long response without constraints."""
        tc = TestCase(input_variables={})
        response = " ".join(["word"] * 250)
        score = conciseness_scorer(response, tc)
        assert score == 0.4


class TestIncludesScorer:
    """Tests for includes scoring."""

    def test_all_keywords_present(self) -> None:
        """Test all keywords present."""
        tc = TestCase(
            input_variables={},
            expected_properties={"includes": ["python", "code"]},
        )
        score = includes_scorer("Python is great for writing code.", tc)
        assert score == 1.0

    def test_partial_keywords(self) -> None:
        """Test partial keywords match."""
        tc = TestCase(
            input_variables={},
            expected_properties={"includes": ["python", "java", "rust"]},
        )
        score = includes_scorer("Python is popular.", tc)
        assert score == pytest.approx(1 / 3)

    def test_no_keywords_required(self) -> None:
        """Test no keywords required."""
        tc = TestCase(input_variables={})
        score = includes_scorer("anything", tc)
        assert score == 1.0


class TestEvaluateResponse:
    """Tests for evaluate_response function."""

    def test_default_criteria(self) -> None:
        """Test evaluation with default criteria."""
        tc = TestCase(input_variables={}, expected_output="test")
        scores = evaluate_response("test", tc)
        assert "accuracy" in scores
        assert "conciseness" in scores

    def test_custom_criteria(self) -> None:
        """Test evaluation with custom criteria."""
        tc = TestCase(
            input_variables={},
            expected_output="test",
            expected_properties={"includes": ["keyword"]},
        )
        scores = evaluate_response("test with keyword", tc, ["accuracy", "includes"])
        assert "accuracy" in scores
        assert "includes" in scores


class TestSelectBestVariant:
    """Tests for select_best_variant function."""

    def test_select_best(self, sample_prompt: Prompt) -> None:
        """Test selecting best variant."""
        v1 = PromptVariant(
            base_prompt=sample_prompt,
            strategy="concise",
            template="concise",
        )
        v2 = PromptVariant(
            base_prompt=sample_prompt,
            strategy="detailed",
            template="detailed",
        )

        tc = TestCase(input_variables={})

        results = [
            EvaluationResult(
                variant=v1,
                test_case=tc,
                output="short",
                scores={"accuracy": 0.9, "conciseness": 0.9},
                latency_ms=100,
                token_count=10,
                cost_usd=0.001,
            ),
            EvaluationResult(
                variant=v2,
                test_case=tc,
                output="long detailed response",
                scores={"accuracy": 0.7, "conciseness": 0.5},
                latency_ms=200,
                token_count=50,
                cost_usd=0.005,
            ),
        ]

        best, score = select_best_variant(results)
        assert best.strategy == "concise"
        assert score > 0

    def test_empty_results_raises(self) -> None:
        """Test empty results raises error."""
        with pytest.raises(ValueError):
            select_best_variant([])
