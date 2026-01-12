"""Prompt Optimizer - A CLI tool and library for optimizing LLM prompts."""

from prompt_optimizer.core import evaluate_variant, generate_variants, optimize_prompt
from prompt_optimizer.evaluator import EvaluationResult, OptimizationResults
from prompt_optimizer.prompt import Prompt, PromptVariant, TestCase, TestSuite

__version__ = "0.1.0"

__all__ = [
    "Prompt",
    "PromptVariant",
    "TestCase",
    "TestSuite",
    "EvaluationResult",
    "OptimizationResults",
    "optimize_prompt",
    "generate_variants",
    "evaluate_variant",
]
