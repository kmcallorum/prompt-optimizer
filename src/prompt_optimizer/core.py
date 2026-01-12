"""Core optimization logic with dependency injection support."""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

from prompt_optimizer.evaluator import (
    EvaluationResult,
    OptimizationResults,
    evaluate_response,
    select_best_variant,
)
from prompt_optimizer.llm_clients.base import LLMClient
from prompt_optimizer.prompt import Prompt, PromptVariant, TestCase

if TYPE_CHECKING:
    from prompt_optimizer.factory import ClientFactory
    from prompt_optimizer.llm_judge import LLMJudge
    from prompt_optimizer.metrics import MetricsRecorder

STRATEGY_TRANSFORMS: dict[str, dict[str, str]] = {
    "concise": {
        "prefix": "",
        "suffix": "\n\nBe concise and direct. Answer in as few words as possible.",
    },
    "detailed": {
        "prefix": "",
        "suffix": "\n\nProvide a detailed and thorough response with explanations.",
    },
    "cot": {
        "prefix": "Think step by step.\n\n",
        "suffix": "\n\nShow your reasoning before giving the final answer.",
    },
    "structured": {
        "prefix": "",
        "suffix": "\n\nFormat your response with clear sections and bullet points.",
    },
    "few_shot": {
        "prefix": "Here are some examples of good responses:\n\n",
        "suffix": "\n\nNow respond in a similar style.",
    },
}


def generate_variants(
    prompt: Prompt,
    strategies: list[str],
) -> list[PromptVariant]:
    """Generate prompt variations using specified strategies.

    Args:
        prompt: Base prompt to create variants from
        strategies: List of strategy names
            (concise, detailed, cot, structured, few_shot)

    Returns:
        List of prompt variants
    """
    variants: list[PromptVariant] = []

    for strategy in strategies:
        transform = STRATEGY_TRANSFORMS.get(strategy, {"prefix": "", "suffix": ""})
        new_template = transform["prefix"] + prompt.template + transform["suffix"]

        variant = PromptVariant(
            base_prompt=prompt,
            strategy=strategy,
            template=new_template,
            system_message=prompt.system_message,
        )
        variants.append(variant)

    return variants


async def evaluate_variant(
    variant: PromptVariant,
    test_case: TestCase,
    llm_client: LLMClient,
    criteria: list[str] | None = None,
    judge: LLMJudge | None = None,
    metrics: MetricsRecorder | None = None,
) -> EvaluationResult:
    """Evaluate a variant against a test case.

    Args:
        variant: Prompt variant to evaluate
        test_case: Test case with input and expected output
        llm_client: LLM client to use for generation
        criteria: List of criteria to evaluate
        judge: Optional LLM judge for AI-based evaluation
        metrics: Optional metrics recorder for observability

    Returns:
        Evaluation result with scores and metadata
    """
    rendered = variant.render(**test_case.input_variables)

    start = time.time()
    response = await llm_client.generate(
        prompt=rendered,
        system=variant.system_message,
    )
    latency = time.time() - start

    # Use LLM judge if provided, otherwise use rule-based evaluation
    if judge is not None:
        scores = await judge.evaluate(response, test_case, rendered)
    else:
        scores = evaluate_response(response, test_case, criteria)

    input_tokens = llm_client.count_tokens(rendered)
    output_tokens = llm_client.count_tokens(response)
    cost = llm_client.calculate_cost(input_tokens, output_tokens)

    # Record metrics if recorder provided
    if metrics is not None:
        prompt_name = variant.base_prompt.name
        metrics.record_llm_request(
            llm=llm_client.model_name,
            operation="generate",
            duration_seconds=latency,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
        )
        metrics.record_variant_evaluation(prompt_name, variant.strategy, scores)

        # Record test case result (pass if accuracy > 0.7)
        accuracy = scores.get("accuracy", 0.0)
        metrics.record_test_case_result(prompt_name, passed=accuracy > 0.7)

    return EvaluationResult(
        variant=variant,
        test_case=test_case,
        output=response,
        scores=scores,
        latency_ms=latency * 1000,
        token_count=output_tokens,
        cost_usd=cost,
    )


async def evaluate_all_variants(
    variants: list[PromptVariant],
    test_cases: list[TestCase],
    llm_client: LLMClient,
    criteria: list[str] | None = None,
    concurrency: int = 5,
    judge: LLMJudge | None = None,
    metrics: MetricsRecorder | None = None,
) -> list[EvaluationResult]:
    """Evaluate all variants against all test cases.

    Args:
        variants: List of prompt variants
        test_cases: List of test cases
        llm_client: LLM client for generation
        criteria: Evaluation criteria
        concurrency: Maximum concurrent API calls
        judge: Optional LLM judge for AI-based evaluation
        metrics: Optional metrics recorder for observability

    Returns:
        List of all evaluation results
    """
    semaphore = asyncio.Semaphore(concurrency)

    async def rate_limited_eval(
        variant: PromptVariant,
        test_case: TestCase,
    ) -> EvaluationResult:
        async with semaphore:
            return await evaluate_variant(
                variant, test_case, llm_client, criteria, judge, metrics
            )

    tasks = [
        rate_limited_eval(variant, test_case)
        for variant in variants
        for test_case in test_cases
    ]

    return await asyncio.gather(*tasks)


def get_llm_client(
    llm: str,
    api_key: str | None = None,
    factory: ClientFactory | None = None,
) -> LLMClient:
    """Get LLM client by name.

    Args:
        llm: LLM identifier (e.g., 'claude-sonnet-4', 'gpt-4o', 'ollama:llama3')
        api_key: Optional API key
        factory: Optional client factory for dependency injection

    Returns:
        Configured LLM client
    """
    if factory is not None:
        return factory.create(llm, api_key)

    # Use default factory
    from prompt_optimizer.factory import get_default_factory

    return get_default_factory().create(llm, api_key)


async def optimize_prompt_async(
    prompt: Prompt,
    test_cases: list[TestCase],
    strategies: list[str] | None = None,
    llm: str = "claude-sonnet-4",
    api_key: str | None = None,
    weights: dict[str, float] | None = None,
    criteria: list[str] | None = None,
    judge_llm: str | None = None,
    judge_api_key: str | None = None,
    client_factory: ClientFactory | None = None,
    metrics: MetricsRecorder | None = None,
    llm_client: LLMClient | None = None,
    judge: LLMJudge | None = None,
) -> OptimizationResults:
    """Optimize a prompt by testing multiple variants.

    Args:
        prompt: Base prompt to optimize
        test_cases: Test cases to evaluate against
        strategies: Variant strategies to test
        llm: LLM to use for generation (ignored if llm_client provided)
        api_key: Optional API key for generation LLM
        weights: Scoring weights
        criteria: Evaluation criteria
        judge_llm: Optional LLM to use as judge for AI-based evaluation
        judge_api_key: Optional API key for judge LLM
        client_factory: Optional factory for creating LLM clients
        metrics: Optional metrics recorder for observability
        llm_client: Optional pre-configured LLM client (for DI)
        judge: Optional pre-configured LLM judge (for DI)

    Returns:
        Optimization results with best variant
    """
    if strategies is None:
        strategies = ["concise", "detailed"]

    # Use default metrics if none provided (for backward compatibility)
    if metrics is None:
        from prompt_optimizer.metrics import (
            record_best_variant,
            record_optimization_complete,
            record_optimization_start,
        )

        # Use module-level functions for backward compatibility
        record_optimization_start(prompt.name)
        use_module_metrics = True
    else:
        metrics.record_optimization_start(prompt.name)
        use_module_metrics = False

    start_time = time.time()

    # Create or use provided LLM client
    if llm_client is None:
        llm_client = get_llm_client(llm, api_key, client_factory)

    # Create judge if specified and not provided
    if judge is None and judge_llm:
        from prompt_optimizer.llm_judge import LLMJudge

        judge_client = get_llm_client(judge_llm, judge_api_key, client_factory)
        judge = LLMJudge(judge_client, criteria)

    try:
        variants = generate_variants(prompt, strategies)
        results = await evaluate_all_variants(
            variants, test_cases, llm_client, criteria, judge=judge, metrics=metrics
        )
        best_variant, best_score = select_best_variant(results, weights)
        total_time = time.time() - start_time
        total_cost = sum(r.cost_usd for r in results)

        # Record successful optimization
        if use_module_metrics:
            record_optimization_complete(prompt.name, total_time, success=True)
            record_best_variant(prompt.name, best_variant.strategy, best_score)
        elif metrics is not None:
            metrics.record_optimization_complete(prompt.name, total_time, success=True)
            metrics.record_best_variant(prompt.name, best_variant.strategy, best_score)

        return OptimizationResults(
            base_prompt_name=prompt.name,
            variants_tested=len(variants),
            test_cases_run=len(test_cases),
            best_variant=best_variant,
            best_weighted_score=best_score,
            all_results=results,
            total_cost=total_cost,
            total_time_seconds=total_time,
        )
    except Exception:
        # Record failed optimization
        total_time = time.time() - start_time
        if use_module_metrics:
            from prompt_optimizer.metrics import record_optimization_complete

            record_optimization_complete(prompt.name, total_time, success=False)
        elif metrics is not None:
            metrics.record_optimization_complete(prompt.name, total_time, success=False)
        raise


def optimize_prompt(
    prompt: Prompt,
    test_cases: list[TestCase],
    strategies: list[str] | None = None,
    llm: str = "claude-sonnet-4",
    api_key: str | None = None,
    weights: dict[str, float] | None = None,
    criteria: list[str] | None = None,
    judge_llm: str | None = None,
    judge_api_key: str | None = None,
    client_factory: ClientFactory | None = None,
    metrics: MetricsRecorder | None = None,
    llm_client: LLMClient | None = None,
    judge: LLMJudge | None = None,
) -> OptimizationResults:
    """Synchronous wrapper for optimize_prompt_async.

    Args:
        prompt: Base prompt to optimize
        test_cases: Test cases to evaluate against
        strategies: Variant strategies to test
        llm: LLM to use for generation (ignored if llm_client provided)
        api_key: Optional API key for generation LLM
        weights: Scoring weights
        criteria: Evaluation criteria
        judge_llm: Optional LLM to use as judge for AI-based evaluation
        judge_api_key: Optional API key for judge LLM
        client_factory: Optional factory for creating LLM clients
        metrics: Optional metrics recorder for observability
        llm_client: Optional pre-configured LLM client (for DI)
        judge: Optional pre-configured LLM judge (for DI)

    Returns:
        Optimization results with best variant
    """
    return asyncio.run(
        optimize_prompt_async(
            prompt=prompt,
            test_cases=test_cases,
            strategies=strategies,
            llm=llm,
            api_key=api_key,
            weights=weights,
            criteria=criteria,
            judge_llm=judge_llm,
            judge_api_key=judge_api_key,
            client_factory=client_factory,
            metrics=metrics,
            llm_client=llm_client,
            judge=judge,
        )
    )
