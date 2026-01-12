"""Unit tests for metrics module."""

from prompt_optimizer.metrics import (
    JUDGE_EVALUATIONS_TOTAL,
    LLM_REQUESTS_TOTAL,
    OPTIMIZATIONS_TOTAL,
    TEST_CASE_FAIL_TOTAL,
    TEST_CASE_PASS_TOTAL,
    TEST_CASES_RUN_TOTAL,
    VARIANTS_EVALUATED_TOTAL,
    init_metrics,
    record_best_variant,
    record_judge_evaluation,
    record_llm_request,
    record_optimization_complete,
    record_optimization_start,
    record_test_case_result,
    record_variant_evaluation,
)


class TestInitMetrics:
    """Tests for init_metrics function."""

    def test_init_metrics(self) -> None:
        """Test that init_metrics sets version info."""
        init_metrics("1.0.0")
        # Just verify it runs without error


class TestRecordOptimization:
    """Tests for optimization recording functions."""

    def test_record_optimization_start(self) -> None:
        """Test recording optimization start."""
        before = OPTIMIZATIONS_TOTAL.labels(
            prompt_name="test", status="started"
        )._value.get()
        record_optimization_start("test")
        after = OPTIMIZATIONS_TOTAL.labels(
            prompt_name="test", status="started"
        )._value.get()
        assert after == before + 1

    def test_record_optimization_complete_success(self) -> None:
        """Test recording successful optimization completion."""
        before = OPTIMIZATIONS_TOTAL.labels(
            prompt_name="test", status="success"
        )._value.get()
        record_optimization_complete("test", 10.5, success=True)
        after = OPTIMIZATIONS_TOTAL.labels(
            prompt_name="test", status="success"
        )._value.get()
        assert after == before + 1

    def test_record_optimization_complete_failure(self) -> None:
        """Test recording failed optimization completion."""
        before = OPTIMIZATIONS_TOTAL.labels(
            prompt_name="test", status="failure"
        )._value.get()
        record_optimization_complete("test", 5.0, success=False)
        after = OPTIMIZATIONS_TOTAL.labels(
            prompt_name="test", status="failure"
        )._value.get()
        assert after == before + 1


class TestRecordVariantEvaluation:
    """Tests for variant evaluation recording."""

    def test_record_variant_evaluation(self) -> None:
        """Test recording variant evaluation."""
        before = VARIANTS_EVALUATED_TOTAL.labels(
            prompt_name="test", strategy="concise"
        )._value.get()

        record_variant_evaluation(
            "test", "concise", {"accuracy": 0.9, "conciseness": 0.8}
        )

        after = VARIANTS_EVALUATED_TOTAL.labels(
            prompt_name="test", strategy="concise"
        )._value.get()
        assert after == before + 1


class TestRecordTestCaseResult:
    """Tests for test case result recording."""

    def test_record_test_case_pass(self) -> None:
        """Test recording passed test case."""
        before_run = TEST_CASES_RUN_TOTAL.labels(prompt_name="test")._value.get()
        before_pass = TEST_CASE_PASS_TOTAL.labels(prompt_name="test")._value.get()

        record_test_case_result("test", passed=True)

        after_run = TEST_CASES_RUN_TOTAL.labels(prompt_name="test")._value.get()
        after_pass = TEST_CASE_PASS_TOTAL.labels(prompt_name="test")._value.get()

        assert after_run == before_run + 1
        assert after_pass == before_pass + 1

    def test_record_test_case_fail(self) -> None:
        """Test recording failed test case."""
        before_run = TEST_CASES_RUN_TOTAL.labels(prompt_name="test")._value.get()
        before_fail = TEST_CASE_FAIL_TOTAL.labels(prompt_name="test")._value.get()

        record_test_case_result("test", passed=False)

        after_run = TEST_CASES_RUN_TOTAL.labels(prompt_name="test")._value.get()
        after_fail = TEST_CASE_FAIL_TOTAL.labels(prompt_name="test")._value.get()

        assert after_run == before_run + 1
        assert after_fail == before_fail + 1


class TestRecordLlmRequest:
    """Tests for LLM request recording."""

    def test_record_llm_request(self) -> None:
        """Test recording LLM request metrics."""
        before_requests = LLM_REQUESTS_TOTAL.labels(
            llm="test-model", operation="generate"
        )._value.get()

        record_llm_request(
            llm="test-model",
            operation="generate",
            duration_seconds=1.5,
            input_tokens=100,
            output_tokens=50,
            cost_usd=0.001,
        )

        after_requests = LLM_REQUESTS_TOTAL.labels(
            llm="test-model", operation="generate"
        )._value.get()
        assert after_requests == before_requests + 1


class TestRecordJudgeEvaluation:
    """Tests for judge evaluation recording."""

    def test_record_judge_evaluation(self) -> None:
        """Test recording judge evaluation."""
        before = JUDGE_EVALUATIONS_TOTAL.labels(judge_llm="gpt-4")._value.get()

        record_judge_evaluation(
            "gpt-4", "test-prompt", {"accuracy": 0.9, "relevance": 0.85}
        )

        after = JUDGE_EVALUATIONS_TOTAL.labels(judge_llm="gpt-4")._value.get()
        assert after == before + 1


class TestRecordBestVariant:
    """Tests for best variant recording."""

    def test_record_best_variant(self) -> None:
        """Test recording best variant."""
        record_best_variant("test-prompt", "detailed", 0.92)
        # Just verify it runs without error
