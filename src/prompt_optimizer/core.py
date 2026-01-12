"""Core optimization logic."""

import asyncio
import time

from prompt_optimizer.evaluator import (
    EvaluationResult,
    OptimizationResults,
    evaluate_response,
    select_best_variant,
)
from prompt_optimizer.llm_clients.anthropic import AnthropicClient
from prompt_optimizer.llm_clients.base import LLMClient
from prompt_optimizer.llm_clients.ollama import OllamaClient
from prompt_optimizer.llm_clients.openai import OpenAIClient
from prompt_optimizer.llm_judge import LLMJudge
from prompt_optimizer.prompt import Prompt, PromptVariant, TestCase

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
) -> EvaluationResult:
    """Evaluate a variant against a test case.

    Args:
        variant: Prompt variant to evaluate
        test_case: Test case with input and expected output
        llm_client: LLM client to use for generation
        criteria: List of criteria to evaluate
        judge: Optional LLM judge for AI-based evaluation

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
) -> list[EvaluationResult]:
    """Evaluate all variants against all test cases.

    Args:
        variants: List of prompt variants
        test_cases: List of test cases
        llm_client: LLM client for generation
        criteria: Evaluation criteria
        concurrency: Maximum concurrent API calls
        judge: Optional LLM judge for AI-based evaluation

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
                variant, test_case, llm_client, criteria, judge
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
) -> AnthropicClient | OpenAIClient | OllamaClient:
    """Get LLM client by name.

    Args:
        llm: LLM identifier (e.g., 'claude-sonnet-4', 'gpt-4o', 'ollama:llama3')
        api_key: Optional API key

    Returns:
        Configured LLM client
    """
    if llm.startswith("claude") or llm.startswith("anthropic"):
        model = llm.replace("anthropic:", "")
        if model == "claude-sonnet-4":
            model = "claude-sonnet-4-20250514"
        elif model == "claude-opus-4":
            model = "claude-opus-4-20250514"
        return AnthropicClient(api_key=api_key, model=model)
    elif llm.startswith("gpt") or llm.startswith("openai"):
        model = llm.replace("openai:", "")
        return OpenAIClient(api_key=api_key, model=model)
    elif llm.startswith("ollama"):
        model = llm.replace("ollama:", "")
        return OllamaClient(model=model)
    else:
        return AnthropicClient(api_key=api_key, model=llm)


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
) -> OptimizationResults:
    """Optimize a prompt by testing multiple variants.

    Args:
        prompt: Base prompt to optimize
        test_cases: Test cases to evaluate against
        strategies: Variant strategies to test
        llm: LLM to use for generation
        api_key: Optional API key for generation LLM
        weights: Scoring weights
        criteria: Evaluation criteria
        judge_llm: Optional LLM to use as judge for AI-based evaluation
        judge_api_key: Optional API key for judge LLM

    Returns:
        Optimization results with best variant
    """
    if strategies is None:
        strategies = ["concise", "detailed"]

    start_time = time.time()
    llm_client = get_llm_client(llm, api_key)

    # Create judge if specified
    judge: LLMJudge | None = None
    if judge_llm:
        judge_client = get_llm_client(judge_llm, judge_api_key)
        judge = LLMJudge(judge_client, criteria)

    variants = generate_variants(prompt, strategies)
    results = await evaluate_all_variants(
        variants, test_cases, llm_client, criteria, judge=judge
    )
    best_variant, best_score = select_best_variant(results, weights)
    total_time = time.time() - start_time
    total_cost = sum(r.cost_usd for r in results)

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
) -> OptimizationResults:
    """Synchronous wrapper for optimize_prompt_async.

    Args:
        prompt: Base prompt to optimize
        test_cases: Test cases to evaluate against
        strategies: Variant strategies to test
        llm: LLM to use for generation
        api_key: Optional API key for generation LLM
        weights: Scoring weights
        criteria: Evaluation criteria
        judge_llm: Optional LLM to use as judge for AI-based evaluation
        judge_api_key: Optional API key for judge LLM

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
        )
    )
