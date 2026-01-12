#!/usr/bin/env python3
"""Basic example of prompt optimization."""

from prompt_optimizer import Prompt, TestCase, optimize_prompt


def main() -> None:
    """Run basic optimization example."""
    # Define a prompt for summarization
    prompt = Prompt(
        template="""Summarize this text in {{ length }}:

{{ text }}""",
        variables={"length": "one sentence", "text": ""},
        system_message="You are a helpful summarization assistant.",
        name="summarizer",
    )

    # Define test cases
    test_cases = [
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
            expected_properties={"length": "<30 words"},
        ),
    ]

    # Run optimization
    print("Starting optimization...")
    results = optimize_prompt(
        prompt,
        test_cases,
        strategies=["concise", "detailed"],
        llm="claude-sonnet-4",
    )

    # Display results
    print(f"\nOptimization complete!")
    print(f"Variants tested: {results.variants_tested}")
    print(f"Test cases run: {results.test_cases_run}")
    print(f"Total cost: ${results.total_cost:.4f}")
    print(f"Total time: {results.total_time_seconds:.2f}s")
    print(f"\nBest variant: {results.best_variant.strategy}")
    print(f"Score: {results.best_weighted_score:.2%}")
    print(f"\nBest template:\n{results.best_variant.template}")


if __name__ == "__main__":
    main()
