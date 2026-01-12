"""Unit tests for reporters module."""

from pathlib import Path

from rich.console import Console

from prompt_optimizer.evaluator import EvaluationResult, OptimizationResults
from prompt_optimizer.prompt import Prompt, PromptVariant, TestCase
from prompt_optimizer.reporters import (
    display_comparison,
    display_results,
    generate_html_report,
    results_to_dict,
    save_html_report,
    save_json_report,
)


def create_sample_results() -> OptimizationResults:
    """Create sample optimization results for testing."""
    prompt = Prompt(template="Test {{ var }}", name="test-prompt")
    variant = PromptVariant(
        base_prompt=prompt,
        strategy="concise",
        template="Test concise",
    )
    test_case = TestCase(input_variables={"var": "value"}, expected_output="expected")

    eval_result = EvaluationResult(
        variant=variant,
        test_case=test_case,
        output="test output",
        scores={"accuracy": 0.9, "conciseness": 0.8},
        latency_ms=100.0,
        token_count=10,
        cost_usd=0.001,
    )

    return OptimizationResults(
        base_prompt_name="test-prompt",
        variants_tested=1,
        test_cases_run=1,
        best_variant=variant,
        best_weighted_score=0.85,
        all_results=[eval_result],
        total_cost=0.001,
        total_time_seconds=1.5,
    )


class TestResultsToDict:
    """Tests for results_to_dict function."""

    def test_converts_results(self) -> None:
        """Test conversion to dictionary."""
        results = create_sample_results()
        data = results_to_dict(results)

        assert data["base_prompt_name"] == "test-prompt"
        assert data["variants_tested"] == 1
        assert data["test_cases_run"] == 1
        assert data["best_variant"]["strategy"] == "concise"
        assert len(data["all_results"]) == 1


class TestSaveJsonReport:
    """Tests for save_json_report function."""

    def test_saves_json(self, tmp_path: Path) -> None:
        """Test saving JSON report."""
        results = create_sample_results()
        output_path = tmp_path / "report.json"

        save_json_report(results, output_path)

        assert output_path.exists()
        content = output_path.read_text()
        assert "test-prompt" in content


class TestSaveHtmlReport:
    """Tests for save_html_report function."""

    def test_saves_html(self, tmp_path: Path) -> None:
        """Test saving HTML report."""
        results = create_sample_results()
        output_path = tmp_path / "report.html"

        save_html_report(results, output_path)

        assert output_path.exists()
        content = output_path.read_text()
        assert "<html>" in content
        assert "test-prompt" in content


class TestGenerateHtmlReport:
    """Tests for generate_html_report function."""

    def test_generates_html(self) -> None:
        """Test HTML generation."""
        results = create_sample_results()
        html = generate_html_report(results)

        assert "<html>" in html
        assert "Prompt Optimization Report" in html
        assert "test-prompt" in html
        assert "concise" in html


class TestDisplayResults:
    """Tests for display_results function."""

    def test_displays_without_error(self) -> None:
        """Test that display doesn't raise errors."""
        results = create_sample_results()
        console = Console(force_terminal=True, width=80)

        # Should not raise
        display_results(console, results)


class TestDisplayComparison:
    """Tests for display_comparison function."""

    def test_displays_comparison(self) -> None:
        """Test comparison display."""
        prompt = Prompt(template="Test", name="test")
        variant = PromptVariant(
            base_prompt=prompt,
            strategy="concise",
            template="Test",
        )
        test_case = TestCase(input_variables={})

        results1 = [
            EvaluationResult(
                variant=variant,
                test_case=test_case,
                output="output1",
                scores={"accuracy": 0.9},
                latency_ms=100,
                token_count=10,
                cost_usd=0.001,
            )
        ]
        results2 = [
            EvaluationResult(
                variant=variant,
                test_case=test_case,
                output="output2",
                scores={"accuracy": 0.8},
                latency_ms=150,
                token_count=15,
                cost_usd=0.002,
            )
        ]

        console = Console(force_terminal=True, width=80)

        # Should not raise
        display_comparison(console, results1, results2, "prompt1", "prompt2")
