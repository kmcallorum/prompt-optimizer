"""Tests for async CLI commands using mocked LLM clients."""

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from click.testing import CliRunner

from prompt_optimizer.cli import cli

pytestmark = pytest.mark.unit


@pytest.fixture
def runner() -> CliRunner:
    """Create CLI test runner."""
    return CliRunner()


@pytest.fixture
def prompt_file(tmp_path: Path) -> Path:
    """Create a sample prompt file."""
    f = tmp_path / "prompt.yaml"
    f.write_text(
        """template: "Answer: {{ question }}"
name: test-prompt
"""
    )
    return f


@pytest.fixture
def test_file(tmp_path: Path) -> Path:
    """Create a sample test file."""
    f = tmp_path / "tests.yaml"
    f.write_text(
        """test_cases:
  - input_variables:
      question: "What is 2+2?"
    expected_output: "4"
"""
    )
    return f


class TestTestCommand:
    """Tests for the test command."""

    def test_test_command(
        self, runner: CliRunner, prompt_file: Path, test_file: Path
    ) -> None:
        """Test the test command with mocked LLM."""
        mock_client = AsyncMock()
        mock_client.generate = AsyncMock(return_value="4")
        mock_client.count_tokens = lambda x: len(x) // 4
        mock_client.calculate_cost = lambda i, o: 0.001

        with patch(
            "prompt_optimizer.cli.get_llm_client", return_value=mock_client
        ):
            result = runner.invoke(
                cli,
                ["test", str(prompt_file), "--test-cases", str(test_file)],
            )

        assert result.exit_code == 0
        assert "Testing prompt" in result.output


class TestOptimizeCommand:
    """Tests for the optimize command."""

    def test_optimize_command(
        self, runner: CliRunner, prompt_file: Path, test_file: Path, tmp_path: Path
    ) -> None:
        """Test the optimize command with mocked LLM."""
        mock_client = AsyncMock()
        mock_client.generate = AsyncMock(return_value="4")
        mock_client.count_tokens = lambda x: len(x) // 4
        mock_client.calculate_cost = lambda i, o: 0.001

        output_file = tmp_path / "results.json"

        with patch(
            "prompt_optimizer.core.get_llm_client", return_value=mock_client
        ):
            result = runner.invoke(
                cli,
                [
                    "optimize",
                    str(prompt_file),
                    "--test-cases",
                    str(test_file),
                    "--strategies",
                    "concise,detailed",
                    "--output",
                    str(output_file),
                ],
            )

        assert result.exit_code == 0
        assert "Optimizing" in result.output
        assert output_file.exists()

    def test_optimize_html_output(
        self, runner: CliRunner, prompt_file: Path, test_file: Path, tmp_path: Path
    ) -> None:
        """Test optimize with HTML output."""
        mock_client = AsyncMock()
        mock_client.generate = AsyncMock(return_value="4")
        mock_client.count_tokens = lambda x: len(x) // 4
        mock_client.calculate_cost = lambda i, o: 0.001

        output_file = tmp_path / "results.html"

        with patch(
            "prompt_optimizer.core.get_llm_client", return_value=mock_client
        ):
            result = runner.invoke(
                cli,
                [
                    "optimize",
                    str(prompt_file),
                    "--test-cases",
                    str(test_file),
                    "--output",
                    str(output_file),
                ],
            )

        assert result.exit_code == 0
        assert output_file.exists()
        assert "<html>" in output_file.read_text()


class TestCompareCommand:
    """Tests for the compare command."""

    def test_compare_command(
        self, runner: CliRunner, prompt_file: Path, test_file: Path, tmp_path: Path
    ) -> None:
        """Test the compare command with mocked LLM."""
        prompt2 = tmp_path / "prompt2.yaml"
        prompt2.write_text(
            """template: "Question: {{ question }}"
name: test-prompt-2
"""
        )

        mock_client = AsyncMock()
        mock_client.generate = AsyncMock(return_value="4")
        mock_client.count_tokens = lambda x: len(x) // 4
        mock_client.calculate_cost = lambda i, o: 0.001

        with patch(
            "prompt_optimizer.cli.get_llm_client", return_value=mock_client
        ):
            result = runner.invoke(
                cli,
                [
                    "compare",
                    str(prompt_file),
                    str(prompt2),
                    "--test-cases",
                    str(test_file),
                ],
            )

        assert result.exit_code == 0
        assert "Comparing" in result.output


class TestHistoryCommand:
    """Tests for the history command with data."""

    def test_history_with_versions(self, runner: CliRunner, tmp_path: Path) -> None:
        """Test history command when versions exist."""
        from prompt_optimizer.prompt import Prompt
        from prompt_optimizer.storage import PromptStorage

        storage = PromptStorage(tmp_path / ".prompt-optimizer")
        prompt = Prompt(template="Test", name="my-prompt")
        storage.save_prompt(prompt)
        storage.save_prompt(prompt)

        result = runner.invoke(
            cli,
            ["history", "my-prompt", "--storage", str(tmp_path / ".prompt-optimizer")],
        )

        assert result.exit_code == 0
        assert "my-prompt" in result.output
        assert "Version" in result.output
