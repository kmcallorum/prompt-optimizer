# prompt-optimizer-cli

[![PyPI](https://img.shields.io/pypi/v/prompt-optimizer-cli.svg)](https://pypi.org/project/prompt-optimizer-cli/)
[![CI](https://github.com/Lanier-Developments/prompt-optimizer/actions/workflows/ci.yml/badge.svg)](https://github.com/Lanier-Developments/prompt-optimizer/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/Lanier-Developments/prompt-optimizer/graph/badge.svg)](https://codecov.io/gh/Lanier-Developments/prompt-optimizer)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Type Checked](https://img.shields.io/badge/type%20checked-mypy-blue.svg)](https://mypy-lang.org/)
[![pytest-agents](https://img.shields.io/badge/pytest--agents-enabled-brightgreen.svg)](https://github.com/Lanier-Developments/pytest-agents)

A CLI tool and Python library for optimizing LLM prompts through systematic testing, version control, and performance metrics. Think "pytest for prompts" - test multiple prompt variations, measure quality, and automatically select the best performer.

## Features

- **Prompt Testing**: Run multiple prompt variations against test cases
- **Quality Metrics**: Score outputs on accuracy, conciseness, tone, and cost
- **LLM-as-Judge**: AI-powered evaluation using any LLM as a judge
- **Prometheus Metrics**: Built-in observability with Prometheus metrics
- **Version Control**: Track prompt evolution with history and diffs
- **Auto-Selection**: Identify and select the best-performing prompt variant
- **CLI & Library**: Use as a command-line tool or Python import
- **Multi-LLM Support**: Works with Anthropic Claude, OpenAI GPT, and local Ollama models

## Quick Start

```bash
# Install from PyPI
pip install prompt-optimizer-cli

# Initialize a project
prompt-optimizer init

# Optimize a prompt
prompt-optimizer optimize prompts/example.yaml \
    --test-cases tests/example_tests.yaml \
    --strategies concise,detailed \
    --llm claude-sonnet-4 \
    --output results.json
```

## Installation

### From PyPI

```bash
pip install prompt-optimizer-cli
```

### From Source

```bash
git clone https://github.com/Lanier-Developments/prompt-optimizer.git
cd prompt-optimizer
pip install -e .
```

### With Development Dependencies

```bash
pip install -e ".[dev]"
```

### Using Docker

```bash
docker-compose build
docker-compose run prompt-optimizer --help
```

## Usage

### CLI Commands

```bash
# Initialize new project with example files
prompt-optimizer init

# Test a prompt against test cases
prompt-optimizer test prompt.yaml --test-cases tests.yaml --llm claude-sonnet-4

# Optimize with multiple strategies
prompt-optimizer optimize prompt.yaml \
    --strategies concise,detailed,cot \
    --test-cases tests.yaml \
    --llm claude-sonnet-4 \
    --output results.json

# Use LLM-as-judge for AI-powered evaluation
prompt-optimizer optimize prompt.yaml \
    --test-cases tests.yaml \
    --llm claude-sonnet-4 \
    --judge gpt-4o \
    --output results.json

# Compare two prompts
prompt-optimizer compare prompt1.yaml prompt2.yaml --test-cases tests.yaml

# View prompt history
prompt-optimizer history my-prompt

# Generate report from results
prompt-optimizer report results.json --format html --output report.html

# Display a prompt file
prompt-optimizer show prompt.yaml
```

### Python Library

```python
from prompt_optimizer import Prompt, TestCase, optimize_prompt

# Define a prompt
prompt = Prompt(
    template="Summarize this text in {{ length }}: {{ text }}",
    variables={"length": "one sentence", "text": ""},
    system_message="You are a helpful summarization assistant.",
    name="summarizer",
)

# Define test cases
test_cases = [
    TestCase(
        input_variables={
            "text": "Long article text here...",
            "length": "one sentence"
        },
        expected_properties={"length": "<30 words"}
    )
]

# Run optimization
results = optimize_prompt(
    prompt,
    test_cases,
    strategies=["concise", "detailed"],
    llm="claude-sonnet-4"
)

print(f"Best variant: {results.best_variant.strategy}")
print(f"Score: {results.best_weighted_score:.2%}")
```

## File Formats

### Prompt File (YAML)

```yaml
template: |
  Answer the following question: {{ question }}

  Requirements:
  - Be concise
  - Be accurate

system_message: "You are a helpful AI assistant."

variables:
  question: ""

metadata:
  author: "developer"
  version: "1.0"
  tags: ["qa", "concise"]
```

### Test Cases (YAML)

```yaml
name: "QA Test Suite"

test_cases:
  - input_variables:
      question: "What is the capital of France?"
    expected_output: "Paris"
    expected_properties:
      tone: "neutral"
      length: "<20 words"

  - input_variables:
      question: "Explain quantum computing"
    expected_properties:
      length: "50-150 words"
      includes: ["qubits", "superposition"]
```

## Supported LLMs

| Provider | Models | Environment Variable |
|----------|--------|---------------------|
| Anthropic | claude-sonnet-4, claude-opus-4 | `ANTHROPIC_API_KEY` |
| OpenAI | gpt-4o, gpt-4-turbo, gpt-3.5-turbo | `OPENAI_API_KEY` |
| Ollama | llama3, mistral, etc. | N/A (local) |

Specify the LLM with the `--llm` flag:

```bash
prompt-optimizer optimize prompt.yaml --llm claude-sonnet-4
prompt-optimizer optimize prompt.yaml --llm gpt-4o
prompt-optimizer optimize prompt.yaml --llm ollama:llama3
```

## Optimization Strategies

| Strategy | Description |
|----------|-------------|
| `concise` | Makes responses shorter and more direct |
| `detailed` | Adds context and thorough explanations |
| `cot` | Adds chain-of-thought reasoning |
| `structured` | Formats output with sections and bullet points |
| `few_shot` | Adds example-based prompting |

## Evaluation Criteria

Built-in scoring functions:

- **accuracy**: Compares output to expected result using sequence matching
- **conciseness**: Scores based on word count and length constraints
- **includes**: Checks for required keywords in response

Custom evaluators can be added:

```python
from prompt_optimizer.evaluator import EVALUATORS

def custom_scorer(response: str, test_case: TestCase) -> float:
    # Your scoring logic
    return 0.8

EVALUATORS["custom"] = custom_scorer
```

## LLM-as-Judge

Use an LLM to evaluate response quality instead of rule-based scoring:

```bash
# Use GPT-4 as judge while testing with Claude
prompt-optimizer optimize prompt.yaml \
    --test-cases tests.yaml \
    --llm claude-sonnet-4 \
    --judge gpt-4o
```

```python
from prompt_optimizer import optimize_prompt, Prompt, TestCase

results = optimize_prompt(
    prompt=my_prompt,
    test_cases=test_cases,
    llm="claude-sonnet-4",
    judge_llm="gpt-4o",  # AI-based evaluation
)
```

The LLM judge evaluates responses on:
- **accuracy** - How well the response matches expected output
- **relevance** - How on-topic the response is
- **coherence** - How well-structured and logical the response is
- **completeness** - Whether all aspects of the prompt are addressed
- **conciseness** - Whether the response is appropriately brief

## Prometheus Metrics

Built-in observability for production deployments:

```bash
# Start metrics server
prompt-optimizer metrics --port 8000

# Metrics available at http://localhost:8000/metrics
```

```python
from prompt_optimizer import init_metrics, start_http_server

# Initialize and start metrics server
init_metrics()
start_http_server(8000)

# Run optimizations - metrics are automatically recorded
results = optimize_prompt(...)
```

Available metrics:
- `prompt_optimizer_optimizations_total` - Total optimization runs
- `prompt_optimizer_optimization_duration_seconds` - Optimization duration
- `prompt_optimizer_variants_evaluated_total` - Variants evaluated
- `prompt_optimizer_test_cases_run_total` - Test cases run
- `prompt_optimizer_llm_requests_total` - LLM API requests
- `prompt_optimizer_llm_tokens_total` - Tokens used (input/output)
- `prompt_optimizer_llm_cost_usd_total` - Total cost in USD
- `prompt_optimizer_best_variant_score` - Best variant score

## Configuration

Environment variables:

```bash
export ANTHROPIC_API_KEY=your-api-key
export OPENAI_API_KEY=your-api-key
```

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run with coverage
pytest --cov=src/prompt_optimizer --cov-report=html

# Lint
ruff check src tests

# Type check
mypy src
```

## Project Structure

```
prompt-optimizer/
├── src/prompt_optimizer/
│   ├── __init__.py
│   ├── cli.py              # Click-based CLI
│   ├── core.py             # Core optimization logic
│   ├── prompt.py           # Prompt models
│   ├── evaluator.py        # Scoring functions
│   ├── storage.py          # Version control
│   ├── reporters.py        # Result reporting
│   └── llm_clients/        # LLM integrations
├── tests/
├── examples/
├── Dockerfile
└── docker-compose.yml
```

## License

MIT
