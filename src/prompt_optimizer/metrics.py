"""Prometheus metrics for prompt-optimizer."""

from prometheus_client import Counter, Gauge, Histogram, Info

# Package info
PACKAGE_INFO = Info(
    "prompt_optimizer",
    "Prompt optimizer package information",
)

# Optimization metrics
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

# Variant metrics
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

# Test case metrics
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

# LLM metrics
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

# Judge metrics
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

# Best variant tracking
BEST_VARIANT_SCORE = Gauge(
    "prompt_optimizer_best_variant_score",
    "Score of the best performing variant",
    ["prompt_name"],
)

BEST_VARIANT_STRATEGY = Info(
    "prompt_optimizer_best_variant",
    "Information about the best performing variant",
)


def init_metrics(version: str = "0.2.0") -> None:
    """Initialize package metrics.

    Args:
        version: Package version string
    """
    PACKAGE_INFO.info({"version": version})


def record_optimization_start(prompt_name: str) -> None:
    """Record the start of an optimization run.

    Args:
        prompt_name: Name of the prompt being optimized
    """
    OPTIMIZATIONS_TOTAL.labels(prompt_name=prompt_name, status="started").inc()


def record_optimization_complete(
    prompt_name: str,
    duration_seconds: float,
    success: bool = True,
) -> None:
    """Record the completion of an optimization run.

    Args:
        prompt_name: Name of the prompt
        duration_seconds: Duration of the optimization
        success: Whether the optimization succeeded
    """
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
    """Record a variant evaluation.

    Args:
        prompt_name: Name of the prompt
        strategy: Strategy used for the variant
        scores: Dictionary of criterion scores
    """
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
    """Record a test case result.

    Args:
        prompt_name: Name of the prompt
        passed: Whether the test case passed
    """
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
    """Record an LLM API request.

    Args:
        llm: Name of the LLM
        operation: Type of operation (generate, judge)
        duration_seconds: Duration of the request
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        cost_usd: Cost in USD
    """
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
    """Record an LLM judge evaluation.

    Args:
        judge_llm: Name of the judge LLM
        prompt_name: Name of the prompt being evaluated
        scores: Dictionary of criterion scores
    """
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
    """Record the best performing variant.

    Args:
        prompt_name: Name of the prompt
        strategy: Strategy of the best variant
        score: Score of the best variant
    """
    BEST_VARIANT_SCORE.labels(prompt_name=prompt_name).set(score)
    BEST_VARIANT_STRATEGY.info({
        "prompt_name": prompt_name,
        "strategy": strategy,
    })
