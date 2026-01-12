# tests/quality/test_code_health.py
"""Code health tests using pytest-agents.

These tests require pytest-agents to be properly configured with agent definitions.
They are skipped if the pytest_agents_agent fixture is not available.
"""

import pytest

# Check if pytest-agents fixture is available
pytest_agents_available = pytest.importorskip(
    "pytest_agents", reason="pytest-agents not properly configured"
)


@pytest.mark.skip(reason="pytest-agents fixture not yet configured")
@pytest.mark.agent_pm
def test_no_critical_issues(pytest_agents_agent):
    """Ensure no FIXMEs in prompt-optimizer codebase."""
    result = pytest_agents_agent.invoke_agent(
        'pm', 'track_tasks',
        {'path': './src/prompt_optimizer'}
    )

    tasks = result['data']['tasks']
    fixmes = [t for t in tasks if t['type'] == 'fixme']
    hacks = [t for t in tasks if t['type'] == 'hack']

    assert len(fixmes) == 0, f"Critical FIXMEs found: {fixmes}"
    assert len(hacks) < 3, f"Too many HACKs: {hacks}"


@pytest.mark.skip(reason="pytest-agents fixture not yet configured")
@pytest.mark.agent_research
def test_documentation_complete(pytest_agents_agent):
    """Verify README documents all features."""
    result = pytest_agents_agent.invoke_agent(
        'research', 'analyze_document',
        {'path': './README.md'}
    )

    # Check that DI factory is documented
    readme_text = result['data']['content']
    has_factory = 'factory' in readme_text.lower()
    has_di = 'dependency injection' in readme_text.lower()
    assert has_factory or has_di
