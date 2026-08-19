#!/usr/bin/env python3
"""Example of using custom evaluation criteria."""

import asyncio

from prompt_optimizer import Prompt, TestCase
from prompt_optimizer.core import (
    evaluate_all_variants,
    generate_variants,
    get_llm_client,
)
from prompt_optimizer.evaluator import EVALUATORS, select_best_variant


def formal_tone_scorer(response: str, test_case: TestCase) -> float:
    """Custom scorer that checks for formal tone indicators."""
    formal_indicators = [
        "therefore",
        "consequently",
        "furthermore",
        "however",
        "moreover",
        "thus",
        "hence",
    ]
    informal_indicators = [
        "gonna",
        "wanna",
        "kinda",
        "sorta",
        "yeah",
        "nope",
        "!",
    ]

    response_lower = response.lower()

    formal_count = sum(1 for word in formal_indicators if word in response_lower)
    informal_count = sum(1 for word in informal_indicators if word in response_lower)

    if formal_count + informal_count == 0:
        return 0.5  # Neutral

    return formal_count / (formal_count + informal_count + 1)


def json_format_scorer(response: str, test_case: TestCase) -> float:
    """Check if response is valid JSON format."""
    import json

    try:
        json.loads(response)
        return 1.0
    except json.JSONDecodeError:
        # Check for partial JSON structure
        if response.strip().startswith("{") and response.strip().endswith("}"):
            return 0.5
        return 0.0


async def main() -> None:
    """Run custom evaluator example."""
    # Register custom evaluators
    EVALUATORS["formal_tone"] = formal_tone_scorer
    EVALUATORS["json_format"] = json_format_scorer

    # Create a prompt that requires formal tone
    prompt = Prompt(
        template="""Write a formal {{ document_type }} about {{ topic }}.

Requirements:
- Use professional language
- Be concise but thorough
- Avoid colloquialisms""",
        name="formal-writer",
    )

    test_cases = [
        TestCase(
            input_variables={
                "document_type": "executive summary",
                "topic": "quarterly sales performance",
            },
            expected_properties={
                "tone": "formal",
                "length": "50-150 words",
            },
        ),
        TestCase(
            input_variables={
                "document_type": "memo",
                "topic": "new office policy",
            },
            expected_properties={
                "tone": "formal",
            },
        ),
    ]

    # Generate variants
    variants = generate_variants(prompt, ["concise", "detailed", "structured"])

    # Get LLM client
    llm_client = get_llm_client("claude-sonnet-4")

    print("Evaluating with custom criteria...")

    # Evaluate with custom criteria
    results = await evaluate_all_variants(
        variants,
        test_cases,
        llm_client,
        criteria=["accuracy", "conciseness", "formal_tone"],
    )

    # Select best with custom weights
    custom_weights = {
        "accuracy": 0.3,
        "conciseness": 0.2,
        "formal_tone": 0.4,
        "cost": 0.05,
        "latency": 0.05,
    }

    best_variant, score = select_best_variant(results, weights=custom_weights)

    print(f"\nBest variant: {best_variant.strategy}")
    print(f"Weighted score: {score:.2%}")

    # Show individual scores
    print("\nDetailed scores by variant:")
    for variant_name in ["concise", "detailed", "structured"]:
        vr = [r for r in results if r.variant.strategy == variant_name]
        if vr:
            avg_formal = sum(r.scores.get("formal_tone", 0) for r in vr) / len(vr)
            avg_accuracy = sum(r.scores.get("accuracy", 0) for r in vr) / len(vr)
            print(f"  {variant_name}: acc={avg_accuracy:.2%}, formal={avg_formal:.2%}")


if __name__ == "__main__":
    asyncio.run(main())
