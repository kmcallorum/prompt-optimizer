"""Evaluation and scoring functions."""

from collections.abc import Callable
from difflib import SequenceMatcher

from pydantic import BaseModel

from prompt_optimizer.prompt import PromptVariant, TestCase


class EvaluationResult(BaseModel):
    """Results from evaluating a prompt variant."""

    variant: PromptVariant
    test_case: TestCase
    output: str
    scores: dict[str, float]
    latency_ms: float
    token_count: int
    cost_usd: float


class OptimizationResults(BaseModel):
    """Complete optimization run results."""

    base_prompt_name: str
    variants_tested: int
    test_cases_run: int
    best_variant: PromptVariant
    best_weighted_score: float
    all_results: list[EvaluationResult]
    total_cost: float
    total_time_seconds: float


def accuracy_scorer(response: str, test_case: TestCase) -> float:
    """Score accuracy by comparing to expected output using sequence matching."""
    if test_case.expected_output is None:
        return 1.0
    expected = test_case.expected_output.lower()
    ratio = SequenceMatcher(None, response.lower(), expected).ratio()
    return ratio


def conciseness_scorer(response: str, test_case: TestCase) -> float:
    """Score conciseness based on word count."""
    word_count = len(response.split())
    expected_props = test_case.expected_properties

    if "length" in expected_props:
        length_spec = expected_props["length"]
        if isinstance(length_spec, str):
            if length_spec.startswith("<"):
                cleaned = length_spec.replace("<", "").replace("words", "").strip()
                max_words = int(cleaned)
                return min(1.0, max_words / max(word_count, 1))
            elif "-" in length_spec:
                parts = length_spec.replace("words", "").strip().split("-")
                min_words, max_words = int(parts[0]), int(parts[1])
                if min_words <= word_count <= max_words:
                    return 1.0
                elif word_count < min_words:
                    return word_count / min_words
                else:
                    return max_words / word_count

    if word_count <= 50:
        return 1.0
    elif word_count <= 100:
        return 0.8
    elif word_count <= 200:
        return 0.6
    else:
        return 0.4


def includes_scorer(response: str, test_case: TestCase) -> float:
    """Score based on required keywords being present."""
    expected_props = test_case.expected_properties
    includes = expected_props.get("includes", [])

    if not includes:
        return 1.0

    response_lower = response.lower()
    found = sum(1 for keyword in includes if keyword.lower() in response_lower)
    return found / len(includes)


EVALUATORS: dict[str, Callable[[str, TestCase], float]] = {
    "accuracy": accuracy_scorer,
    "conciseness": conciseness_scorer,
    "includes": includes_scorer,
}

# LLM judge criteria (evaluated by AI, not rule-based)
LLM_JUDGE_CRITERIA = [
    "accuracy",
    "relevance",
    "coherence",
    "completeness",
    "conciseness",
]


def evaluate_response(
    response: str,
    test_case: TestCase,
    criteria: list[str] | None = None,
) -> dict[str, float]:
    """Evaluate a response against a test case.

    Args:
        response: Generated text from LLM
        test_case: Test case with expected outputs/properties
        criteria: List of criteria to evaluate (defaults to all applicable)

    Returns:
        Dictionary of criterion names to scores (0.0 to 1.0)
    """
    if criteria is None:
        criteria = ["accuracy", "conciseness"]
        if "includes" in test_case.expected_properties:
            criteria.append("includes")

    scores: dict[str, float] = {}
    for criterion in criteria:
        if criterion in EVALUATORS:
            scores[criterion] = EVALUATORS[criterion](response, test_case)

    return scores


def select_best_variant(
    results: list[EvaluationResult],
    weights: dict[str, float] | None = None,
) -> tuple[PromptVariant, float]:
    """Select best variant using weighted scoring.

    Args:
        results: List of evaluation results
        weights: Weights for each criterion (defaults to balanced)

    Returns:
        Tuple of (best variant, weighted score)
    """
    if weights is None:
        weights = {"accuracy": 0.5, "conciseness": 0.2, "cost": 0.2, "latency": 0.1}

    variant_scores: dict[str, tuple[PromptVariant, list[float]]] = {}

    for result in results:
        key = result.variant.strategy
        if key not in variant_scores:
            variant_scores[key] = (result.variant, [])

        weighted = 0.0
        for criterion, weight in weights.items():
            if criterion == "cost":
                cost_score = 1.0 - min(result.cost_usd / 0.01, 1.0)
                weighted += cost_score * weight
            elif criterion == "latency":
                latency_score = 1.0 - min(result.latency_ms / 5000, 1.0)
                weighted += latency_score * weight
            else:
                weighted += result.scores.get(criterion, 0.0) * weight

        variant_scores[key][1].append(weighted)

    best_variant = None
    best_score = -1.0

    for variant, scores_list in variant_scores.values():
        avg_score = sum(scores_list) / len(scores_list) if scores_list else 0.0
        if avg_score > best_score:
            best_score = avg_score
            best_variant = variant

    if best_variant is None:
        raise ValueError("No results to select from")

    return best_variant, best_score
