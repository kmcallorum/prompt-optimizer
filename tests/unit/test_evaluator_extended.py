"""Extended tests for evaluator module."""


from prompt_optimizer.evaluator import (
    EvaluationResult,
    evaluate_response,
    select_best_variant,
)
from prompt_optimizer.prompt import Prompt, PromptVariant, TestCase


class TestSelectBestVariantExtended:
    """Extended tests for select_best_variant."""

    def test_custom_weights(self) -> None:
        """Test selecting with custom weights."""
        prompt = Prompt(template="Test", name="test")
        v1 = PromptVariant(base_prompt=prompt, strategy="fast", template="fast")
        v2 = PromptVariant(base_prompt=prompt, strategy="accurate", template="accurate")

        tc = TestCase(input_variables={})

        results = [
            EvaluationResult(
                variant=v1,
                test_case=tc,
                output="quick",
                scores={"accuracy": 0.6, "conciseness": 0.9},
                latency_ms=50,
                token_count=5,
                cost_usd=0.0001,
            ),
            EvaluationResult(
                variant=v2,
                test_case=tc,
                output="detailed accurate response",
                scores={"accuracy": 0.95, "conciseness": 0.4},
                latency_ms=200,
                token_count=20,
                cost_usd=0.001,
            ),
        ]

        # With high accuracy weight, v2 should win
        best, score = select_best_variant(
            results, weights={"accuracy": 0.9, "conciseness": 0.1}
        )
        assert best.strategy == "accurate"

        # With high conciseness weight, v1 should win
        best, score = select_best_variant(
            results, weights={"accuracy": 0.1, "conciseness": 0.9}
        )
        assert best.strategy == "fast"

    def test_multiple_results_per_variant(self) -> None:
        """Test averaging scores across multiple test cases."""
        prompt = Prompt(template="Test", name="test")
        variant = PromptVariant(base_prompt=prompt, strategy="test", template="test")

        results = [
            EvaluationResult(
                variant=variant,
                test_case=TestCase(input_variables={"n": "1"}),
                output="out1",
                scores={"accuracy": 1.0},
                latency_ms=100,
                token_count=10,
                cost_usd=0.001,
            ),
            EvaluationResult(
                variant=variant,
                test_case=TestCase(input_variables={"n": "2"}),
                output="out2",
                scores={"accuracy": 0.5},
                latency_ms=100,
                token_count=10,
                cost_usd=0.001,
            ),
        ]

        best, score = select_best_variant(results)
        assert best.strategy == "test"
        # Score should be averaged


class TestEvaluateResponseExtended:
    """Extended tests for evaluate_response."""

    def test_with_includes_in_properties(self) -> None:
        """Test that includes criterion is auto-added when in properties."""
        tc = TestCase(
            input_variables={},
            expected_properties={"includes": ["python", "code"]},
        )

        scores = evaluate_response("Python is great for code.", tc)

        assert "includes" in scores
        assert scores["includes"] == 1.0

    def test_unknown_criterion_ignored(self) -> None:
        """Test that unknown criteria are ignored."""
        tc = TestCase(input_variables={}, expected_output="test")

        scores = evaluate_response("test", tc, ["accuracy", "unknown_criterion"])

        assert "accuracy" in scores
        assert "unknown_criterion" not in scores
