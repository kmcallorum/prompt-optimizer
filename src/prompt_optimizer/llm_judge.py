"""LLM-as-judge evaluation for AI-based scoring."""

import json
import re
from typing import Any

from prompt_optimizer.llm_clients.base import LLMClient
from prompt_optimizer.prompt import TestCase

JUDGE_SYSTEM_PROMPT = (
    "You are an expert evaluator assessing AI-generated responses. "
    "You must evaluate responses objectively and return scores in valid JSON format. "
    "Be strict but fair in your assessments."
)

JUDGE_TEMPLATE = """Evaluate this AI-generated response against the given criteria.

## Original Prompt
{prompt}

## AI Response
{response}

## Expected Output (if provided)
{expected}

## Required Properties
{properties}

## Evaluation Criteria
Score each criterion from 0.0 to 1.0:

1. **accuracy** - How well does the response match the expected output?
2. **relevance** - How relevant and on-topic is the response?
3. **coherence** - How well-structured and logically coherent is the response?
4. **completeness** - Does the response fully address all aspects of the prompt?
5. **conciseness** - Is the response appropriately concise?

Return ONLY a JSON object with scores, no other text:
{{"accuracy": 0.0, "relevance": 0.0, "coherence": 0.0, "completeness": 0.0, \
"conciseness": 0.0}}"""


class LLMJudge:
    """LLM-based response evaluator."""

    def __init__(
        self,
        client: LLMClient,
        criteria: list[str] | None = None,
    ) -> None:
        """Initialize LLM judge.

        Args:
            client: LLM client to use for evaluation
            criteria: List of criteria to evaluate (defaults to all)
        """
        self.client = client
        self.criteria = criteria or [
            "accuracy",
            "relevance",
            "coherence",
            "completeness",
            "conciseness",
        ]

    async def evaluate(
        self,
        response: str,
        test_case: TestCase,
        rendered_prompt: str,
    ) -> dict[str, float]:
        """Evaluate a response using the LLM judge.

        Args:
            response: Generated response to evaluate
            test_case: Test case with expected outputs/properties
            rendered_prompt: The prompt that was sent to generate the response

        Returns:
            Dictionary of criterion names to scores (0.0 to 1.0)
        """
        expected = test_case.expected_output or "Not specified"
        properties = json.dumps(test_case.expected_properties, indent=2)
        props_str = properties if test_case.expected_properties else "None"

        judge_prompt = JUDGE_TEMPLATE.format(
            prompt=rendered_prompt,
            response=response,
            expected=expected,
            properties=props_str,
        )

        judge_response = await self.client.generate(
            prompt=judge_prompt,
            system=JUDGE_SYSTEM_PROMPT,
        )

        scores = self._parse_scores(judge_response)

        # Filter to requested criteria only
        return {k: v for k, v in scores.items() if k in self.criteria}

    def _parse_scores(self, response: str) -> dict[str, float]:
        """Parse scores from judge response.

        Args:
            response: Raw response from judge LLM

        Returns:
            Dictionary of scores
        """
        # Try to extract JSON from response
        json_match = re.search(r'\{[^{}]*\}', response)
        if json_match:
            try:
                parsed: dict[str, Any] = json.loads(json_match.group())
                # Validate and clamp scores
                validated: dict[str, float] = {}
                for key, value in parsed.items():
                    if isinstance(value, (int, float)):
                        validated[key] = max(0.0, min(1.0, float(value)))
                return validated
            except json.JSONDecodeError:
                pass

        # Fallback: try to parse individual scores
        fallback_scores: dict[str, float] = {}
        patterns = [
            r'"?(\w+)"?\s*:\s*([0-9.]+)',
            r'(\w+)\s*=\s*([0-9.]+)',
        ]

        for pattern in patterns:
            matches = re.findall(pattern, response)
            for key, value in matches:
                try:
                    fallback_scores[key.lower()] = max(0.0, min(1.0, float(value)))
                except ValueError:
                    continue

        # Return defaults if parsing failed
        if not fallback_scores:
            return {
                "accuracy": 0.5,
                "relevance": 0.5,
                "coherence": 0.5,
                "completeness": 0.5,
                "conciseness": 0.5,
            }

        return fallback_scores


async def llm_evaluate_response(
    response: str,
    test_case: TestCase,
    rendered_prompt: str,
    judge_client: LLMClient,
    criteria: list[str] | None = None,
) -> dict[str, float]:
    """Convenience function for LLM-based evaluation.

    Args:
        response: Generated response to evaluate
        test_case: Test case with expected outputs/properties
        rendered_prompt: The prompt that was sent to generate the response
        judge_client: LLM client to use as judge
        criteria: List of criteria to evaluate

    Returns:
        Dictionary of criterion names to scores (0.0 to 1.0)
    """
    judge = LLMJudge(judge_client, criteria)
    return await judge.evaluate(response, test_case, rendered_prompt)
