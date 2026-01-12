"""Prompt models and test case definitions."""

from typing import Any

from jinja2 import Template
from pydantic import BaseModel, Field


class Prompt(BaseModel):
    """A prompt template with variables."""

    template: str = Field(..., description="Prompt text with {variable} placeholders")
    variables: dict[str, str] = Field(default_factory=dict)
    system_message: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    name: str = Field(default="unnamed")

    def render(self, **kwargs: str) -> str:
        """Render template with variables."""
        merged_vars = {**self.variables, **kwargs}
        jinja_template = Template(self.template)
        return jinja_template.render(**merged_vars)


class PromptVariant(BaseModel):
    """A specific variation of a prompt."""

    base_prompt: Prompt
    strategy: str
    template: str
    system_message: str | None = None

    def render(self, **kwargs: str) -> str:
        """Render the variant template with variables."""
        merged_vars = {**self.base_prompt.variables, **kwargs}
        jinja_template = Template(self.template)
        return jinja_template.render(**merged_vars)


class TestCase(BaseModel):
    """A single test case for prompt evaluation."""

    input_variables: dict[str, str]
    expected_output: str | None = None
    expected_properties: dict[str, Any] = Field(default_factory=dict)


class TestSuite(BaseModel):
    """Collection of test cases."""

    name: str
    test_cases: list[TestCase]
