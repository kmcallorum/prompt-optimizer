"""Unit tests for CLI module."""

from pathlib import Path

import pytest
from click.testing import CliRunner

from prompt_optimizer.cli import cli, load_prompt_from_yaml, load_test_cases_from_yaml


@pytest.fixture
def runner() -> CliRunner:
    """Create CLI test runner."""
    return CliRunner()


@pytest.fixture
def sample_prompt_yaml(tmp_path: Path) -> Path:
    """Create a sample prompt YAML file."""
    prompt_file = tmp_path / "prompt.yaml"
    prompt_file.write_text(
        """template: |
  Answer: {{ question }}

system_message: "Be helpful."

variables:
  question: ""

name: test-prompt
"""
    )
    return prompt_file


@pytest.fixture
def sample_tests_yaml(tmp_path: Path) -> Path:
    """Create a sample tests YAML file."""
    tests_file = tmp_path / "tests.yaml"
    tests_file.write_text(
        """name: "Test Suite"

test_cases:
  - input_variables:
      question: "What is 2+2?"
    expected_output: "4"

  - input_variables:
      question: "Capital of France?"
    expected_output: "Paris"
"""
    )
    return tests_file


class TestLoadPromptFromYaml:
    """Tests for loading prompts from YAML."""

    def test_load_prompt(self, sample_prompt_yaml: Path) -> None:
        """Test loading a valid prompt."""
        prompt = load_prompt_from_yaml(sample_prompt_yaml)
        assert prompt.name == "test-prompt"
        assert "{{ question }}" in prompt.template
        assert prompt.system_message == "Be helpful."

    def test_load_prompt_minimal(self, tmp_path: Path) -> None:
        """Test loading minimal prompt."""
        minimal = tmp_path / "minimal.yaml"
        minimal.write_text("template: Hello World")
        prompt = load_prompt_from_yaml(minimal)
        assert prompt.template == "Hello World"


class TestLoadTestCasesFromYaml:
    """Tests for loading test cases from YAML."""

    def test_load_test_cases(self, sample_tests_yaml: Path) -> None:
        """Test loading test cases."""
        cases = load_test_cases_from_yaml(sample_tests_yaml)
        assert len(cases) == 2
        assert cases[0].expected_output == "4"

    def test_load_list_format(self, tmp_path: Path) -> None:
        """Test loading test cases in list format."""
        list_file = tmp_path / "list.yaml"
        list_file.write_text(
            """- input_variables:
    question: "Q1"
  expected_output: "A1"
- input_variables:
    question: "Q2"
"""
        )
        cases = load_test_cases_from_yaml(list_file)
        assert len(cases) == 2


class TestCliCommands:
    """Tests for CLI commands."""

    def test_cli_version(self, runner: CliRunner) -> None:
        """Test version flag."""
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "0.1.0" in result.output

    def test_cli_help(self, runner: CliRunner) -> None:
        """Test help output."""
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "Prompt Optimizer" in result.output

    def test_init_command(self, runner: CliRunner, tmp_path: Path) -> None:
        """Test init command."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(cli, ["init"])
            assert result.exit_code == 0
            assert Path(".prompt-optimizer").exists()
            assert Path("prompts/example.yaml").exists()
            assert Path("tests/example_tests.yaml").exists()

    def test_show_command(
        self, runner: CliRunner, sample_prompt_yaml: Path
    ) -> None:
        """Test show command."""
        result = runner.invoke(cli, ["show", str(sample_prompt_yaml)])
        assert result.exit_code == 0

    def test_history_empty(self, runner: CliRunner, tmp_path: Path) -> None:
        """Test history with no versions."""
        storage_path = tmp_path / ".prompt-optimizer"
        storage_path.mkdir()
        result = runner.invoke(
            cli, ["history", "nonexistent", "--storage", str(storage_path)]
        )
        assert result.exit_code == 0
        assert "No history found" in result.output

    def test_report_terminal(self, runner: CliRunner, tmp_path: Path) -> None:
        """Test report command with terminal format."""
        results_file = tmp_path / "results.json"
        results_file.write_text(
            """{
            "base_prompt_name": "test",
            "variants_tested": 2,
            "test_cases_run": 3,
            "total_cost": 0.001,
            "best_variant": {
                "strategy": "concise",
                "weighted_score": 0.85
            }
        }"""
        )
        result = runner.invoke(cli, ["report", str(results_file)])
        assert result.exit_code == 0
        assert "test" in result.output

    def test_report_json(self, runner: CliRunner, tmp_path: Path) -> None:
        """Test report command with JSON format."""
        results_file = tmp_path / "results.json"
        results_file.write_text('{"base_prompt_name": "test"}')

        output_file = tmp_path / "output.json"
        result = runner.invoke(
            cli,
            ["report", str(results_file), "--format", "json", "--output", str(output_file)],
        )
        assert result.exit_code == 0
        assert output_file.exists()

    def test_report_html(self, runner: CliRunner, tmp_path: Path) -> None:
        """Test report command with HTML format."""
        results_file = tmp_path / "results.json"
        results_file.write_text('{"base_prompt_name": "test"}')

        output_file = tmp_path / "output.html"
        result = runner.invoke(
            cli,
            ["report", str(results_file), "--format", "html", "--output", str(output_file)],
        )
        assert result.exit_code == 0
        assert output_file.exists()

    def test_init_existing_files(self, runner: CliRunner, tmp_path: Path) -> None:
        """Test init command with existing files doesn't overwrite."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            # First init
            runner.invoke(cli, ["init"])

            # Modify example file
            prompt_file = Path("prompts/example.yaml")
            prompt_file.write_text("template: Custom content")

            # Second init
            result = runner.invoke(cli, ["init"])
            assert result.exit_code == 0

            # File should not be overwritten
            assert prompt_file.read_text() == "template: Custom content"

    def test_load_invalid_test_format(self, runner: CliRunner, tmp_path: Path) -> None:
        """Test loading invalid test case format raises error."""
        invalid_file = tmp_path / "invalid.yaml"
        invalid_file.write_text("invalid_key: value")

        with pytest.raises(ValueError, match="Invalid test cases format"):
            load_test_cases_from_yaml(invalid_file)
