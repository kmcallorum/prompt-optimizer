"""Prompt Optimizer - A CLI tool and library for optimizing LLM prompts."""

from prometheus_client import start_http_server

from prompt_optimizer.core import evaluate_variant, generate_variants, optimize_prompt
from prompt_optimizer.evaluator import EvaluationResult, OptimizationResults
from prompt_optimizer.factory import (
    ClientFactory,
    ClientRegistry,
    DefaultClientFactory,
    MockClientFactory,
    get_default_factory,
    set_default_factory,
)
from prompt_optimizer.llm_clients.base import LLMClient
from prompt_optimizer.llm_judge import LLMJudge
from prompt_optimizer.metrics import (
    MetricsRecorder,
    NullMetricsRecorder,
    PrometheusMetricsRecorder,
    get_metrics_recorder,
    init_metrics,
    set_metrics_recorder,
)
from prompt_optimizer.prompt import Prompt, PromptVariant, TestCase, TestSuite

__version__ = "0.3.2"

__all__ = [
    # Core models
    "Prompt",
    "PromptVariant",
    "TestCase",
    "TestSuite",
    "EvaluationResult",
    "OptimizationResults",
    # Core functions
    "optimize_prompt",
    "generate_variants",
    "evaluate_variant",
    # LLM clients
    "LLMClient",
    "LLMJudge",
    # Dependency injection - Client factory
    "ClientFactory",
    "ClientRegistry",
    "DefaultClientFactory",
    "MockClientFactory",
    "get_default_factory",
    "set_default_factory",
    # Dependency injection - Metrics
    "MetricsRecorder",
    "PrometheusMetricsRecorder",
    "NullMetricsRecorder",
    "get_metrics_recorder",
    "set_metrics_recorder",
    "init_metrics",
    "start_http_server",
]
