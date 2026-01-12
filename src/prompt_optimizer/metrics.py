"""Prometheus metrics for prompt-optimizer with dependency injection support."""

from __future__ import annotations

from typing import Protocol

from prometheus_client import Counter, Gauge, Histogram, Info


class MetricsRecorder(Protocol):
    """Protocol for metrics recording.

    Allows swapping out the metrics implementation for testing
    or using different metrics backends.
    """

    def init_metrics(self, version: str) -> None:
        """Initialize package metrics."""

    def record_optimization_start(self, prompt_name: str) -> None:
        """Record the start of an optimization run."""

    def record_optimization_complete(
        self,
        prompt_name: str,
        duration_seconds: float,
        success: bool = True,
    ) -> None:
        """Record the completion of an optimization run."""

    def record_variant_evaluation(
        self,
        prompt_name: str,
        strategy: str,
        scores: dict[str, float],
    ) -> None:
        """Record a variant evaluation."""

    def record_test_case_result(
        self,
        prompt_name: str,
        passed: bool,
    ) -> None:
        """Record a test case result."""

    def record_llm_request(
        self,
        llm: str,
        operation: str,
        duration_seconds: float,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float,
    ) -> None:
        """Record an LLM API request."""

    def record_judge_evaluation(
        self,
        judge_llm: str,
        prompt_name: str,
        scores: dict[str, float],
    ) -> None:
        """Record an LLM judge evaluation."""

    def record_best_variant(
        self,
        prompt_name: str,
        strategy: str,
        score: float,
    ) -> None:
        """Record the best performing variant."""


class PrometheusMetricsRecorder:
    """Prometheus-based metrics recorder implementation."""

    def __init__(self) -> None:
        """Initialize Prometheus metrics."""
        # Package info
        self._package_info = Info(
            "prompt_optimizer",
            "Prompt optimizer package information",
        )

        # Optimization metrics
        self._optimizations_total = Counter(
            "prompt_optimizer_optimizations_total",
            "Total number of optimization runs",
            ["prompt_name", "status"],
        )

        self._optimization_duration_seconds = Histogram(
            "prompt_optimizer_optimization_duration_seconds",
            "Duration of optimization runs in seconds",
            ["prompt_name"],
            buckets=[1, 5, 10, 30, 60, 120, 300, 600],
        )

        # Variant metrics
        self._variants_evaluated_total = Counter(
            "prompt_optimizer_variants_evaluated_total",
            "Total number of variants evaluated",
            ["prompt_name", "strategy"],
        )

        self._variant_score = Gauge(
            "prompt_optimizer_variant_score",
            "Score of a variant evaluation",
            ["prompt_name", "strategy", "criterion"],
        )

        # Test case metrics
        self._test_cases_run_total = Counter(
            "prompt_optimizer_test_cases_run_total",
            "Total number of test cases run",
            ["prompt_name"],
        )

        self._test_case_pass_total = Counter(
            "prompt_optimizer_test_case_pass_total",
            "Total number of test cases passed",
            ["prompt_name"],
        )

        self._test_case_fail_total = Counter(
            "prompt_optimizer_test_case_fail_total",
            "Total number of test cases failed",
            ["prompt_name"],
        )

        # LLM metrics
        self._llm_requests_total = Counter(
            "prompt_optimizer_llm_requests_total",
            "Total number of LLM API requests",
            ["llm", "operation"],
        )

        self._llm_request_duration_seconds = Histogram(
            "prompt_optimizer_llm_request_duration_seconds",
            "Duration of LLM API requests in seconds",
            ["llm"],
            buckets=[0.1, 0.5, 1, 2, 5, 10, 30, 60],
        )

        self._llm_tokens_total = Counter(
            "prompt_optimizer_llm_tokens_total",
            "Total number of tokens used",
            ["llm", "token_type"],
        )

        self._llm_cost_usd_total = Counter(
            "prompt_optimizer_llm_cost_usd_total",
            "Total cost in USD",
            ["llm"],
        )

        # Judge metrics
        self._judge_evaluations_total = Counter(
            "prompt_optimizer_judge_evaluations_total",
            "Total number of LLM judge evaluations",
            ["judge_llm"],
        )

        self._judge_score = Gauge(
            "prompt_optimizer_judge_score",
            "Score from LLM judge evaluation",
            ["prompt_name", "criterion"],
        )

        # Best variant tracking
        self._best_variant_score = Gauge(
            "prompt_optimizer_best_variant_score",
            "Score of the best performing variant",
            ["prompt_name"],
        )

        self._best_variant_strategy = Info(
            "prompt_optimizer_best_variant",
            "Information about the best performing variant",
        )

    def init_metrics(self, version: str = "0.3.0") -> None:
        """Initialize package metrics."""
        self._package_info.info({"version": version})

    def record_optimization_start(self, prompt_name: str) -> None:
        """Record the start of an optimization run."""
        self._optimizations_total.labels(
            prompt_name=prompt_name, status="started"
        ).inc()

    def record_optimization_complete(
        self,
        prompt_name: str,
        duration_seconds: float,
        success: bool = True,
    ) -> None:
        """Record the completion of an optimization run."""
        status = "success" if success else "failure"
        self._optimizations_total.labels(prompt_name=prompt_name, status=status).inc()
        self._optimization_duration_seconds.labels(prompt_name=prompt_name).observe(
            duration_seconds
        )

    def record_variant_evaluation(
        self,
        prompt_name: str,
        strategy: str,
        scores: dict[str, float],
    ) -> None:
        """Record a variant evaluation."""
        self._variants_evaluated_total.labels(
            prompt_name=prompt_name, strategy=strategy
        ).inc()

        for criterion, score in scores.items():
            self._variant_score.labels(
                prompt_name=prompt_name,
                strategy=strategy,
                criterion=criterion,
            ).set(score)

    def record_test_case_result(
        self,
        prompt_name: str,
        passed: bool,
    ) -> None:
        """Record a test case result."""
        self._test_cases_run_total.labels(prompt_name=prompt_name).inc()

        if passed:
            self._test_case_pass_total.labels(prompt_name=prompt_name).inc()
        else:
            self._test_case_fail_total.labels(prompt_name=prompt_name).inc()

    def record_llm_request(
        self,
        llm: str,
        operation: str,
        duration_seconds: float,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float,
    ) -> None:
        """Record an LLM API request."""
        self._llm_requests_total.labels(llm=llm, operation=operation).inc()
        self._llm_request_duration_seconds.labels(llm=llm).observe(duration_seconds)
        self._llm_tokens_total.labels(
            llm=llm, token_type="input"  # noqa: S106
        ).inc(input_tokens)
        self._llm_tokens_total.labels(
            llm=llm, token_type="output"  # noqa: S106
        ).inc(output_tokens)
        self._llm_cost_usd_total.labels(llm=llm).inc(cost_usd)

    def record_judge_evaluation(
        self,
        judge_llm: str,
        prompt_name: str,
        scores: dict[str, float],
    ) -> None:
        """Record an LLM judge evaluation."""
        self._judge_evaluations_total.labels(judge_llm=judge_llm).inc()

        for criterion, score in scores.items():
            self._judge_score.labels(
                prompt_name=prompt_name,
                criterion=criterion,
            ).set(score)

    def record_best_variant(
        self,
        prompt_name: str,
        strategy: str,
        score: float,
    ) -> None:
        """Record the best performing variant."""
        self._best_variant_score.labels(prompt_name=prompt_name).set(score)
        self._best_variant_strategy.info({
            "prompt_name": prompt_name,
            "strategy": strategy,
        })


class NullMetricsRecorder:
    """No-op metrics recorder for testing or when metrics are disabled."""

    def init_metrics(self, version: str = "0.3.0") -> None:
        """No-op."""
        pass

    def record_optimization_start(self, prompt_name: str) -> None:
        """No-op."""
        pass

    def record_optimization_complete(
        self,
        prompt_name: str,
        duration_seconds: float,
        success: bool = True,
    ) -> None:
        """No-op."""
        pass

    def record_variant_evaluation(
        self,
        prompt_name: str,
        strategy: str,
        scores: dict[str, float],
    ) -> None:
        """No-op."""
        pass

    def record_test_case_result(
        self,
        prompt_name: str,
        passed: bool,
    ) -> None:
        """No-op."""
        pass

    def record_llm_request(
        self,
        llm: str,
        operation: str,
        duration_seconds: float,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float,
    ) -> None:
        """No-op."""
        pass

    def record_judge_evaluation(
        self,
        judge_llm: str,
        prompt_name: str,
        scores: dict[str, float],
    ) -> None:
        """No-op."""
        pass

    def record_best_variant(
        self,
        prompt_name: str,
        strategy: str,
        score: float,
    ) -> None:
        """No-op."""
        pass


# Default metrics recorder instance
_default_recorder: MetricsRecorder | None = None


def get_metrics_recorder() -> MetricsRecorder:
    """Get the default metrics recorder.

    Returns:
        The default MetricsRecorder instance
    """
    global _default_recorder
    if _default_recorder is None:
        _default_recorder = PrometheusMetricsRecorder()
    return _default_recorder


def set_metrics_recorder(recorder: MetricsRecorder | None) -> None:
    """Set the default metrics recorder.

    Args:
        recorder: Recorder to use as default, or None to reset
    """
    global _default_recorder
    _default_recorder = recorder


# Backward-compatible module-level metrics (for existing code)
# These reference a lazily-created PrometheusMetricsRecorder
PACKAGE_INFO = Info(
    "prompt_optimizer",
    "Prompt optimizer package information",
)

OPTIMIZATIONS_TOTAL = Counter(
    "prompt_optimizer_optimizations_total",
    "Total number of optimization runs",
    ["prompt_name", "status"],
)

OPTIMIZATION_DURATION_SECONDS = Histogram(
    "prompt_optimizer_optimization_duration_seconds",
    "Duration of optimization runs in seconds",
    ["prompt_name"],
    buckets=[1, 5, 10, 30, 60, 120, 300, 600],
)

VARIANTS_EVALUATED_TOTAL = Counter(
    "prompt_optimizer_variants_evaluated_total",
    "Total number of variants evaluated",
    ["prompt_name", "strategy"],
)

VARIANT_SCORE = Gauge(
    "prompt_optimizer_variant_score",
    "Score of a variant evaluation",
    ["prompt_name", "strategy", "criterion"],
)

TEST_CASES_RUN_TOTAL = Counter(
    "prompt_optimizer_test_cases_run_total",
    "Total number of test cases run",
    ["prompt_name"],
)

TEST_CASE_PASS_TOTAL = Counter(
    "prompt_optimizer_test_case_pass_total",
    "Total number of test cases passed",
    ["prompt_name"],
)

TEST_CASE_FAIL_TOTAL = Counter(
    "prompt_optimizer_test_case_fail_total",
    "Total number of test cases failed",
    ["prompt_name"],
)

LLM_REQUESTS_TOTAL = Counter(
    "prompt_optimizer_llm_requests_total",
    "Total number of LLM API requests",
    ["llm", "operation"],
)

LLM_REQUEST_DURATION_SECONDS = Histogram(
    "prompt_optimizer_llm_request_duration_seconds",
    "Duration of LLM API requests in seconds",
    ["llm"],
    buckets=[0.1, 0.5, 1, 2, 5, 10, 30, 60],
)

LLM_TOKENS_TOTAL = Counter(
    "prompt_optimizer_llm_tokens_total",
    "Total number of tokens used",
    ["llm", "token_type"],
)

LLM_COST_USD_TOTAL = Counter(
    "prompt_optimizer_llm_cost_usd_total",
    "Total cost in USD",
    ["llm"],
)

JUDGE_EVALUATIONS_TOTAL = Counter(
    "prompt_optimizer_judge_evaluations_total",
    "Total number of LLM judge evaluations",
    ["judge_llm"],
)

JUDGE_SCORE = Gauge(
    "prompt_optimizer_judge_score",
    "Score from LLM judge evaluation",
    ["prompt_name", "criterion"],
)

BEST_VARIANT_SCORE = Gauge(
    "prompt_optimizer_best_variant_score",
    "Score of the best performing variant",
    ["prompt_name"],
)

BEST_VARIANT_STRATEGY = Info(
    "prompt_optimizer_best_variant",
    "Information about the best performing variant",
)


# Backward-compatible module-level functions
def init_metrics(version: str = "0.3.0") -> None:
    """Initialize package metrics."""
    PACKAGE_INFO.info({"version": version})


def record_optimization_start(prompt_name: str) -> None:
    """Record the start of an optimization run."""
    OPTIMIZATIONS_TOTAL.labels(prompt_name=prompt_name, status="started").inc()


def record_optimization_complete(
    prompt_name: str,
    duration_seconds: float,
    success: bool = True,
) -> None:
    """Record the completion of an optimization run."""
    status = "success" if success else "failure"
    OPTIMIZATIONS_TOTAL.labels(prompt_name=prompt_name, status=status).inc()
    OPTIMIZATION_DURATION_SECONDS.labels(prompt_name=prompt_name).observe(
        duration_seconds
    )


def record_variant_evaluation(
    prompt_name: str,
    strategy: str,
    scores: dict[str, float],
) -> None:
    """Record a variant evaluation."""
    VARIANTS_EVALUATED_TOTAL.labels(
        prompt_name=prompt_name, strategy=strategy
    ).inc()

    for criterion, score in scores.items():
        VARIANT_SCORE.labels(
            prompt_name=prompt_name,
            strategy=strategy,
            criterion=criterion,
        ).set(score)


def record_test_case_result(
    prompt_name: str,
    passed: bool,
) -> None:
    """Record a test case result."""
    TEST_CASES_RUN_TOTAL.labels(prompt_name=prompt_name).inc()

    if passed:
        TEST_CASE_PASS_TOTAL.labels(prompt_name=prompt_name).inc()
    else:
        TEST_CASE_FAIL_TOTAL.labels(prompt_name=prompt_name).inc()


def record_llm_request(
    llm: str,
    operation: str,
    duration_seconds: float,
    input_tokens: int,
    output_tokens: int,
    cost_usd: float,
) -> None:
    """Record an LLM API request."""
    LLM_REQUESTS_TOTAL.labels(llm=llm, operation=operation).inc()
    LLM_REQUEST_DURATION_SECONDS.labels(llm=llm).observe(duration_seconds)
    LLM_TOKENS_TOTAL.labels(llm=llm, token_type="input").inc(input_tokens)  # noqa: S106
    LLM_TOKENS_TOTAL.labels(llm=llm, token_type="output").inc(output_tokens)  # noqa: S106
    LLM_COST_USD_TOTAL.labels(llm=llm).inc(cost_usd)


def record_judge_evaluation(
    judge_llm: str,
    prompt_name: str,
    scores: dict[str, float],
) -> None:
    """Record an LLM judge evaluation."""
    JUDGE_EVALUATIONS_TOTAL.labels(judge_llm=judge_llm).inc()

    for criterion, score in scores.items():
        JUDGE_SCORE.labels(
            prompt_name=prompt_name,
            criterion=criterion,
        ).set(score)


def record_best_variant(
    prompt_name: str,
    strategy: str,
    score: float,
) -> None:
    """Record the best performing variant."""
    BEST_VARIANT_SCORE.labels(prompt_name=prompt_name).set(score)
    BEST_VARIANT_STRATEGY.info({
        "prompt_name": prompt_name,
        "strategy": strategy,
    })
