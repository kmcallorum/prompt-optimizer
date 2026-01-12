"""Unit tests for prompt module."""

import pytest

from prompt_optimizer.prompt import Prompt, PromptVariant, TestCase, TestSuite

pytestmark = pytest.mark.unit


class TestPrompt:
    """Tests for Prompt class."""

    def test_prompt_creation(self) -> None:
        """Test basic prompt creation."""
        prompt = Prompt(
            template="Hello {{ name }}!",
            variables={"name": "World"},
        )
        assert prompt.template == "Hello {{ name }}!"
        assert prompt.variables == {"name": "World"}
        assert prompt.system_message is None

    def test_prompt_rendering(self) -> None:
        """Test prompt template rendering."""
        prompt = Prompt(
            template="Hello {{ name }}!",
            variables={"name": "World"},
        )
        rendered = prompt.render()
        assert rendered == "Hello World!"

    def test_prompt_rendering_with_override(self) -> None:
        """Test rendering with variable override."""
        prompt = Prompt(
            template="Hello {{ name }}!",
            variables={"name": "World"},
        )
        rendered = prompt.render(name="Claude")
        assert rendered == "Hello Claude!"

    def test_prompt_with_system_message(self) -> None:
        """Test prompt with system message."""
        prompt = Prompt(
            template="Answer: {{ question }}",
            system_message="You are helpful.",
        )
        assert prompt.system_message == "You are helpful."

    def test_prompt_with_metadata(self) -> None:
        """Test prompt with metadata."""
        prompt = Prompt(
            template="Test",
            metadata={"author": "test", "version": "1.0"},
        )
        assert prompt.metadata["author"] == "test"
        assert prompt.metadata["version"] == "1.0"


class TestPromptVariant:
    """Tests for PromptVariant class."""

    def test_variant_creation(self, sample_prompt: Prompt) -> None:
        """Test variant creation."""
        variant = PromptVariant(
            base_prompt=sample_prompt,
            strategy="concise",
            template=sample_prompt.template + "\n\nBe concise.",
        )
        assert variant.strategy == "concise"
        assert "Be concise" in variant.template

    def test_variant_rendering(self, sample_prompt: Prompt) -> None:
        """Test variant rendering."""
        variant = PromptVariant(
            base_prompt=sample_prompt,
            strategy="detailed",
            template="Detailed: {{ question }}",
        )
        rendered = variant.render(question="What is AI?")
        assert rendered == "Detailed: What is AI?"


class TestTestCase:
    """Tests for TestCase class."""

    def test_test_case_creation(self) -> None:
        """Test basic test case creation."""
        tc = TestCase(
            input_variables={"question": "What is 2+2?"},
            expected_output="4",
        )
        assert tc.input_variables["question"] == "What is 2+2?"
        assert tc.expected_output == "4"

    def test_test_case_with_properties(self) -> None:
        """Test case with expected properties."""
        tc = TestCase(
            input_variables={"text": "Hello"},
            expected_properties={"length": "<10 words", "tone": "friendly"},
        )
        assert tc.expected_properties["length"] == "<10 words"
        assert tc.expected_properties["tone"] == "friendly"


class TestTestSuite:
    """Tests for TestSuite class."""

    def test_suite_creation(self) -> None:
        """Test suite creation."""
        suite = TestSuite(
            name="My Tests",
            test_cases=[
                TestCase(input_variables={"q": "1"}),
                TestCase(input_variables={"q": "2"}),
            ],
        )
        assert suite.name == "My Tests"
        assert len(suite.test_cases) == 2
