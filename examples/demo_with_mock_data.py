#!/usr/bin/env python3
"""Demo script showing full optimization flow with mock data and Prometheus metrics.

This example demonstrates:
1. Mock LLM clients that simulate real API responses
2. Prometheus metrics collection and exposure
3. Full optimization workflow without needing API keys

Run with: python examples/demo_with_mock_data.py

Then scrape metrics at: http://localhost:8000/metrics
"""

import asyncio
import time
from typing import Any

from prometheus_client import start_http_server

from prompt_optimizer import (
    MockClientFactory,
    Prompt,
    TestCase,
    init_metrics,
    optimize_prompt,
    set_default_factory,
)


class DemoMockLLMClient:
    """Mock LLM client with realistic demo responses."""

    def __init__(self) -> None:
        self.call_count = 0
        self.calls: list[dict[str, Any]] = []

        # Predefined realistic responses for different prompts
        self._responses = [
            # Concise strategy responses
            "Machine learning enables computers to learn from data automatically.",
            "Python is a readable, high-level programming language.",
            # Detailed strategy responses
            "Machine learning is an AI subset where systems improve "
            "through experience, accessing and learning from data.",
            "Python is a versatile, high-level language with clear syntax, supporting "
            "multiple paradigms and featuring an extensive standard library.",
            # Chain-of-thought responses
            "Let me analyze this: The text discusses ML fundamentals. Key points: "
            "1) subset of AI, 2) learns from experience, 3) no explicit programming. "
            "Summary: ML allows systems to automatically learn from data.",
            "Breaking down: 1) Python is high-level and interpreted, 2) known for "
            "readability, 3) supports multiple paradigms. Summary: Python is a clear, "
            "versatile programming language with extensive libraries.",
        ]

    async def generate(
        self,
        prompt: str,
        system: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> str:
        """Simulate LLM response with realistic delay."""
        # Track the call
        self.calls.append({
            "prompt": prompt[:100] + "..." if len(prompt) > 100 else prompt,
            "system": system,
            "max_tokens": max_tokens,
            "temperature": temperature,
        })

        # Simulate API latency (50-200ms)
        await asyncio.sleep(0.05 + (self.call_count % 3) * 0.05)

        # Return cycling through responses
        response = self._responses[self.call_count % len(self._responses)]
        self.call_count += 1
        return response

    def count_tokens(self, text: str) -> int:
        """Estimate tokens (roughly 4 chars per token)."""
        return len(text) // 4

    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate simulated cost based on Claude pricing."""
        # Simulated pricing: $3/1M input, $15/1M output
        return (input_tokens * 3 + output_tokens * 15) / 1_000_000

    @property
    def model_name(self) -> str:
        """Return mock model name."""
        return "mock-claude-sonnet-4"


def create_demo_prompt() -> Prompt:
    """Create a sample prompt for the demo."""
    return Prompt(
        template="""Summarize this text in {{ length }}:

{{ text }}""",
        variables={"length": "one sentence", "text": ""},
        system_message="You are a helpful summarization assistant.",
        name="demo-summarizer",
    )


def create_demo_test_cases() -> list[TestCase]:
    """Create sample test cases for the demo."""
    return [
        TestCase(
            input_variables={
                "text": (
                    "Machine learning is a subset of artificial intelligence "
                    "that enables systems to learn and improve from experience "
                    "without being explicitly programmed. It focuses on developing "
                    "computer programs that can access data and use it to learn "
                    "for themselves."
                ),
                "length": "one sentence",
            },
            expected_output="machine learning",
            expected_properties={"length": "<30 words"},
        ),
        TestCase(
            input_variables={
                "text": (
                    "Python is a high-level, interpreted programming language "
                    "known for its clear syntax and readability. It supports "
                    "multiple programming paradigms and has a large standard library."
                ),
                "length": "one sentence",
            },
            expected_output="Python",
            expected_properties={"length": "<30 words"},
        ),
        TestCase(
            input_variables={
                "text": (
                    "Cloud computing delivers computing services over the internet, "
                    "including servers, storage, databases, networking, and software. "
                    "It offers faster innovation and flexible resources."
                ),
                "length": "one sentence",
            },
            expected_output="cloud",
            expected_properties={"length": "<25 words"},
        ),
    ]


def main() -> None:
    """Run the demo with mock data and Prometheus metrics."""
    print("=" * 60)
    print("Prompt Optimizer Demo with Mock Data")
    print("=" * 60)

    # Start Prometheus metrics server
    metrics_port = 9090
    print(f"\n[1/5] Starting Prometheus metrics server on port {metrics_port}...")
    start_http_server(metrics_port)
    print(f"      Metrics available at: http://localhost:{metrics_port}/metrics")

    # Initialize metrics
    print("\n[2/5] Initializing metrics...")
    init_metrics("0.3.3-demo")

    # Set up mock LLM client
    print("\n[3/5] Setting up mock LLM client...")
    mock_client = DemoMockLLMClient()
    mock_factory = MockClientFactory(mock_client)  # type: ignore[arg-type]
    set_default_factory(mock_factory)  # type: ignore[arg-type]
    print(f"      Using mock model: {mock_client.model_name}")

    # Create demo data
    print("\n[4/5] Creating demo prompt and test cases...")
    prompt = create_demo_prompt()
    test_cases = create_demo_test_cases()
    print(f"      Prompt: {prompt.name}")
    print(f"      Test cases: {len(test_cases)}")

    # Run optimization
    print("\n[5/5] Running optimization...")
    print("-" * 60)

    start_time = time.time()
    results = optimize_prompt(
        prompt,
        test_cases,
        strategies=["concise", "detailed", "cot"],
        llm="mock-claude-sonnet-4",  # Will use our mock
    )
    elapsed = time.time() - start_time

    # Display results
    print("\n" + "=" * 60)
    print("OPTIMIZATION RESULTS")
    print("=" * 60)
    print(f"Variants tested:     {results.variants_tested}")
    print(f"Test cases run:      {results.test_cases_run}")
    print(f"Total cost:          ${results.total_cost:.6f}")
    print(f"Total time:          {results.total_time_seconds:.2f}s")
    print(f"Actual elapsed:      {elapsed:.2f}s")
    print("-" * 60)
    print(f"Best variant:        {results.best_variant.strategy}")
    print(f"Best score:          {results.best_weighted_score:.2%}")
    print("-" * 60)
    print("Best template:")
    print(results.best_variant.template)

    # Show individual results
    print("\n" + "=" * 60)
    print("DETAILED RESULTS BY VARIANT")
    print("=" * 60)

    # Group results by variant
    by_variant: dict[str, list[Any]] = {}
    for r in results.all_results:
        strategy = r.variant.strategy
        if strategy not in by_variant:
            by_variant[strategy] = []
        by_variant[strategy].append(r)

    for strategy, variant_results in by_variant.items():
        print(f"\n[{strategy.upper()}]")
        avg_score = sum(
            sum(r.scores.values()) / len(r.scores) for r in variant_results
        ) / len(variant_results)
        print(f"  Average score: {avg_score:.2%}")
        for i, r in enumerate(variant_results, 1):
            print(f"  Test {i}: {r.scores} | {r.latency_ms:.0f}ms")

    # Show mock client stats
    print("\n" + "=" * 60)
    print("MOCK CLIENT STATISTICS")
    print("=" * 60)
    print(f"Total API calls:     {mock_client.call_count}")
    print(f"Factory create calls: {len(mock_factory.create_calls)}")

    # Prometheus metrics info
    print("\n" + "=" * 60)
    print("PROMETHEUS METRICS")
    print("=" * 60)
    print(f"Metrics are being exposed at http://localhost:{metrics_port}/metrics")
    print("\nKey metrics to look for:")
    print("  - prompt_optimizer_optimizations_total")
    print("  - prompt_optimizer_variants_evaluated_total")
    print("  - prompt_optimizer_test_cases_run_total")
    print("  - prompt_optimizer_llm_requests_total")
    print("  - prompt_optimizer_optimization_duration_seconds")
    print("\nPress Ctrl+C to stop the metrics server...")

    # Keep server running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")


if __name__ == "__main__":
    main()
