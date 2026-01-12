# prompt-optimizer - AGENTS.md

## Project Vision
A CLI tool and Python library that helps developers optimize LLM prompts through systematic testing, version control, and performance metrics. Think "pytest for prompts" - test multiple prompt variations, measure quality, and automatically select the best performer.

## EXECUTION MODE: AUTONOMOUS
Claude should make ALL changes without asking for approval unless a critical architectural decision arises.
Quality gates at the end determine success. If all tests pass and linting succeeds, the implementation is acceptable.

---

## Core Functionality

### What It Does
1. **Prompt Testing**: Run multiple prompt variations against test cases
2. **Quality Metrics**: Score outputs (accuracy, conciseness, tone, cost)
3. **Version Control**: Track prompt evolution with git-like diffs
4. **Auto-Selection**: Identify best-performing prompt variant
5. **CLI & Library**: Usable as command-line tool or Python import

### What It Does NOT Do
- ❌ NO web UI (CLI only, maybe simple HTML report)
- ❌ NO authentication/authorization (local tool)
- ❌ NO complex database (SQLite is fine, prefer JSON files)
- ❌ NO custom API server (just a library + CLI)
- ❌ NO LLM training or fine-tuning (just prompt optimization)

---

## Project Structure

```
prompt-optimizer/
├── src/
│   └── prompt_optimizer/
│       ├── __init__.py
│       ├── cli.py              # Click-based CLI
│       ├── core.py             # Core optimization logic
│       ├── prompt.py           # Prompt class and variants
│       ├── evaluator.py        # Scoring and evaluation
│       ├── storage.py          # Prompt version storage
│       ├── reporters.py        # Results reporting
│       └── llm_clients/
│           ├── __init__.py
│           ├── base.py         # Abstract LLM client
│           ├── anthropic.py    # Claude client
│           ├── openai.py       # OpenAI client
│           └── ollama.py       # Local Ollama client
├── tests/
│   ├── unit/
│   ├── integration/
│   └── fixtures/
├── examples/
│   ├── basic_optimization.py
│   ├── custom_evaluator.py
│   └── version_control.py
├── pyproject.toml
├── README.md
├── AGENTS.md                    # This file
├── Dockerfile
└── docker-compose.yml
```

---

## Technical Stack

### Core Dependencies (MUST USE)
```toml
[project.dependencies]
click = ">=8.1.0"              # CLI framework (simple, not argparse)
pydantic = ">=2.0.0"           # Data validation
anthropic = ">=0.18.0"         # Claude API client
openai = ">=1.0.0"             # OpenAI API client
httpx = ">=0.27.0"             # For Ollama client
rich = ">=13.0.0"              # Pretty terminal output
jinja2 = ">=3.1.0"             # Prompt templating
pyyaml = ">=6.0.0"             # Config files
typing-extensions = ">=4.9.0"  # Type hints
```

### Dev Dependencies
```toml
[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-cov>=4.1.0",
    "pytest-asyncio>=0.23.0",
    "ruff>=0.1.0",
    "mypy>=1.8.0",
]
```

### DO NOT ADD
- ❌ No FastAPI/Flask (not building an API)
- ❌ No SQLAlchemy (too heavy, use SQLite directly if needed)
- ❌ No Celery/Redis (no distributed tasks)
- ❌ No complex ORMs
- ❌ No custom config systems beyond YAML

---

## Architecture Constraints

### DO: Simple and Direct
```python
# GOOD: Simple, readable
def optimize_prompt(prompt: Prompt, test_cases: list[TestCase]) -> Results:
    variants = generate_variants(prompt)
    scores = evaluate_all(variants, test_cases)
    return select_best(scores)

# BAD: Over-engineered
class PromptOptimizationStrategy(ABC):
    @abstractmethod
    def optimize(self) -> OptimizationResult:
        pass

class GeneticAlgorithmOptimizer(PromptOptimizationStrategy):
    # ... 500 lines of complexity
```

### DO: Functional Core, Imperative Shell
- Pure functions for evaluation logic
- Side effects (API calls, file I/O) in clearly marked places
- Easy to test without mocks

### DO: Type Hints Everywhere
```python
from typing import Protocol, TypeVar, Generic

class LLMClient(Protocol):
    async def generate(self, prompt: str, **kwargs) -> str:
        ...

def evaluate_response(
    response: str,
    expected: str,
    criteria: dict[str, float]
) -> float:
    ...
```

### DON'T: Over-Abstract
- No "PromptFactory" classes
- No "EvaluatorRegistry" singletons
- No "StrategyPattern" unless absolutely necessary
- Prefer composition over inheritance

---

## Core Classes & Data Models

### Prompt Model
```python
from pydantic import BaseModel, Field
from typing import Optional

class Prompt(BaseModel):
    """A prompt template with variables."""
    
    template: str = Field(..., description="Prompt text with {variable} placeholders")
    variables: dict[str, str] = Field(default_factory=dict)
    system_message: Optional[str] = None
    metadata: dict[str, any] = Field(default_factory=dict)
    
    def render(self, **kwargs) -> str:
        """Render template with variables."""
        # Use jinja2 for rendering
        pass
    
    def variants(self, strategies: list[str]) -> list['Prompt']:
        """Generate prompt variations."""
        # Apply transformation strategies
        pass

class PromptVariant(BaseModel):
    """A specific variation of a prompt."""
    base_prompt: Prompt
    strategy: str  # e.g., "concise", "detailed", "step_by_step"
    rendered: str
```

### Test Cases
```python
class TestCase(BaseModel):
    """A single test case for prompt evaluation."""
    
    input_variables: dict[str, str]
    expected_output: Optional[str] = None
    expected_properties: dict[str, any] = Field(default_factory=dict)
    # e.g., {"tone": "professional", "length": "<100 words"}
    
class TestSuite(BaseModel):
    """Collection of test cases."""
    name: str
    test_cases: list[TestCase]
```

### Evaluation Results
```python
class EvaluationResult(BaseModel):
    """Results from evaluating a prompt variant."""
    
    variant: PromptVariant
    test_case: TestCase
    output: str
    scores: dict[str, float]  # {"accuracy": 0.9, "conciseness": 0.7}
    latency_ms: float
    token_count: int
    cost_usd: float

class OptimizationResults(BaseModel):
    """Complete optimization run results."""
    
    base_prompt: Prompt
    variants_tested: int
    test_cases_run: int
    best_variant: PromptVariant
    all_results: list[EvaluationResult]
    total_cost: float
    total_time_seconds: float
```

---

## CLI Interface

### Commands Structure
```bash
# Initialize new project
prompt-optimizer init

# Test a single prompt
prompt-optimizer test prompt.yaml --test-cases tests.yaml

# Optimize with multiple strategies
prompt-optimizer optimize prompt.yaml \
    --strategies concise,detailed,cot \
    --test-cases tests.yaml \
    --llm claude-sonnet-4 \
    --output results.json

# Compare two prompts
prompt-optimizer compare prompt1.yaml prompt2.yaml \
    --test-cases tests.yaml

# Show optimization history
prompt-optimizer history prompt.yaml

# Generate report
prompt-optimizer report results.json --format html
```

### CLI Implementation Style
```python
import click
from rich.console import Console
from rich.table import Table

@click.group()
def cli():
    """Prompt optimization CLI."""
    pass

@cli.command()
@click.argument('prompt_file', type=click.Path(exists=True))
@click.option('--strategies', default='concise,detailed')
@click.option('--llm', default='claude-sonnet-4')
def optimize(prompt_file: str, strategies: str, llm: str):
    """Optimize a prompt with multiple strategies."""
    console = Console()
    
    # Load prompt
    prompt = load_prompt(prompt_file)
    
    # Run optimization
    with console.status("Optimizing..."):
        results = run_optimization(prompt, strategies.split(','), llm)
    
    # Display results
    display_results(console, results)
```

Use **rich** library for:
- Progress bars during evaluation
- Pretty tables for results
- Syntax highlighting for prompts
- Color-coded output

---

## Core Algorithms

### Prompt Variant Generation
```python
def generate_variants(prompt: Prompt, strategies: list[str]) -> list[PromptVariant]:
    """
    Generate variations using strategies:
    
    - 'concise': Remove unnecessary words, make direct
    - 'detailed': Add context and examples
    - 'cot': Add chain-of-thought reasoning
    - 'structured': Add XML/JSON structure requirements
    - 'few_shot': Add few-shot examples
    """
    variants = []
    
    for strategy in strategies:
        if strategy == 'concise':
            # Apply conciseness transformation
            pass
        elif strategy == 'detailed':
            # Add detail transformation
            pass
        # ... etc
    
    return variants
```

### Evaluation Scoring
```python
async def evaluate_variant(
    variant: PromptVariant,
    test_case: TestCase,
    llm_client: LLMClient,
    criteria: dict[str, Callable]
) -> EvaluationResult:
    """
    Evaluate a variant against a test case.
    
    Criteria can include:
    - accuracy: Compare to expected output (semantic similarity)
    - conciseness: Token count, word count
    - tone: Sentiment analysis
    - structure: Check for required format
    - cost: Track API costs
    - latency: Response time
    """
    
    # Render prompt
    rendered = variant.render(**test_case.input_variables)
    
    # Call LLM
    start = time.time()
    response = await llm_client.generate(rendered)
    latency = time.time() - start
    
    # Score response
    scores = {}
    for criterion_name, criterion_func in criteria.items():
        scores[criterion_name] = criterion_func(response, test_case)
    
    return EvaluationResult(
        variant=variant,
        test_case=test_case,
        output=response,
        scores=scores,
        latency_ms=latency * 1000,
        token_count=count_tokens(response),
        cost_usd=calculate_cost(response, llm_client)
    )
```

### Best Variant Selection
```python
def select_best_variant(
    results: list[EvaluationResult],
    weights: dict[str, float] = None
) -> PromptVariant:
    """
    Select best variant using weighted scoring.
    
    Default weights:
    - accuracy: 0.5
    - conciseness: 0.2
    - cost: 0.2
    - latency: 0.1
    """
    if weights is None:
        weights = {"accuracy": 0.5, "conciseness": 0.2, "cost": 0.2, "latency": 0.1}
    
    # Calculate weighted scores
    variant_scores = {}
    for result in results:
        weighted = sum(
            result.scores.get(k, 0) * v 
            for k, v in weights.items()
        )
        variant_scores[result.variant] = weighted
    
    return max(variant_scores.items(), key=lambda x: x[1])[0]
```

---

## LLM Client Architecture

### Base Protocol
```python
from typing import Protocol

class LLMClient(Protocol):
    """Protocol for LLM clients."""
    
    async def generate(
        self, 
        prompt: str,
        system: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """Generate text from prompt."""
        ...
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        ...
    
    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate API cost."""
        ...
```

### Anthropic Implementation
```python
from anthropic import AsyncAnthropic

class AnthropicClient:
    """Claude API client."""
    
    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514"):
        self.client = AsyncAnthropic(api_key=api_key)
        self.model = model
    
    async def generate(self, prompt: str, system: str | None = None, **kwargs) -> str:
        response = await self.client.messages.create(
            model=self.model,
            max_tokens=kwargs.get('max_tokens', 1024),
            temperature=kwargs.get('temperature', 0.7),
            system=system or "",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text
    
    def count_tokens(self, text: str) -> int:
        # Use anthropic's token counting
        return self.client.count_tokens(text)
    
    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        # Pricing as of 2025
        input_cost = input_tokens * 0.000003  # $3 per MTok
        output_cost = output_tokens * 0.000015  # $15 per MTok
        return input_cost + output_cost
```

### OpenAI Implementation
```python
from openai import AsyncOpenAI

class OpenAIClient:
    """OpenAI API client."""
    
    def __init__(self, api_key: str, model: str = "gpt-4"):
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model
    
    async def generate(self, prompt: str, system: str | None = None, **kwargs) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=kwargs.get('max_tokens', 1024),
            temperature=kwargs.get('temperature', 0.7)
        )
        return response.choices[0].message.content
```

### Ollama Implementation (Local)
```python
import httpx

class OllamaClient:
    """Local Ollama client."""
    
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3"):
        self.base_url = base_url
        self.model = model
    
    async def generate(self, prompt: str, system: str | None = None, **kwargs) -> str:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "system": system or "",
                    "stream": False
                }
            )
            return response.json()["response"]
    
    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        return 0.0  # Local model, no cost
```

---

## File Formats

### Prompt Files (YAML)
```yaml
# prompt.yaml
template: |
  You are a helpful assistant. 
  Answer the following question: {question}
  
  Requirements:
  - Be concise
  - Cite sources if applicable

system_message: "You are a helpful AI assistant."

variables:
  question: ""

metadata:
  author: "mac"
  version: "1.0"
  tags: ["qa", "concise"]
```

### Test Cases (YAML)
```yaml
# tests.yaml
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
      tone: "educational"
      length: "50-150 words"
      includes: ["qubits", "superposition"]
```

### Results (JSON)
```json
{
  "optimization_id": "uuid",
  "timestamp": "2025-01-12T...",
  "base_prompt": {...},
  "variants_tested": 3,
  "best_variant": {
    "strategy": "concise",
    "weighted_score": 0.85
  },
  "all_results": [...]
}
```

---

## Version Control / Storage

### Simple JSON-based Storage
```python
class PromptStorage:
    """Version control for prompts."""
    
    def __init__(self, storage_path: Path = Path(".prompt-optimizer")):
        self.storage_path = storage_path
        self.storage_path.mkdir(exist_ok=True)
    
    def save_prompt(self, prompt: Prompt, version: str = None) -> str:
        """Save prompt and return version ID."""
        version_id = version or generate_version_id()
        
        prompt_file = self.storage_path / f"{prompt.name}_{version_id}.json"
        prompt_file.write_text(prompt.model_dump_json(indent=2))
        
        return version_id
    
    def load_prompt(self, name: str, version: str = None) -> Prompt:
        """Load specific version or latest."""
        if version:
            prompt_file = self.storage_path / f"{name}_{version}.json"
        else:
            # Find latest version
            versions = list(self.storage_path.glob(f"{name}_*.json"))
            prompt_file = max(versions, key=lambda p: p.stat().st_mtime)
        
        return Prompt.model_validate_json(prompt_file.read_text())
    
    def history(self, name: str) -> list[dict]:
        """Get version history."""
        versions = list(self.storage_path.glob(f"{name}_*.json"))
        return [
            {
                "version": v.stem.split("_")[1],
                "modified": v.stat().st_mtime,
                "size": v.stat().st_size
            }
            for v in sorted(versions, key=lambda p: p.stat().st_mtime, reverse=True)
        ]
```

---

## Error Handling

### Strategy
- Let exceptions bubble naturally
- Use specific exception types
- Log errors, don't swallow
- Return error status in CLI, don't crash

```python
class PromptOptimizerError(Exception):
    """Base exception."""
    pass

class LLMAPIError(PromptOptimizerError):
    """LLM API call failed."""
    pass

class EvaluationError(PromptOptimizerError):
    """Evaluation criteria failed."""
    pass

# In code
try:
    response = await llm_client.generate(prompt)
except httpx.HTTPError as e:
    raise LLMAPIError(f"API call failed: {e}") from e
```

---

## Testing Strategy

### Unit Tests
```python
# tests/unit/test_prompt.py
def test_prompt_rendering():
    prompt = Prompt(
        template="Hello {name}!",
        variables={"name": "World"}
    )
    assert prompt.render() == "Hello World!"

def test_variant_generation():
    prompt = Prompt(template="Explain quantum computing")
    variants = generate_variants(prompt, ["concise", "detailed"])
    assert len(variants) == 2
    assert variants[0].strategy == "concise"
```

### Integration Tests
```python
# tests/integration/test_optimization.py
@pytest.mark.asyncio
async def test_full_optimization_flow(tmp_path):
    # Create test prompt
    prompt = Prompt(template="Answer: {question}")
    
    # Create test cases
    test_cases = [
        TestCase(input_variables={"question": "What is 2+2?"}, expected_output="4")
    ]
    
    # Run optimization with mock LLM
    results = await optimize_prompt(prompt, test_cases, strategies=["concise"])
    
    assert results.best_variant is not None
    assert len(results.all_results) > 0
```

### Fixtures
```python
# tests/conftest.py
@pytest.fixture
def mock_llm_client():
    class MockClient:
        async def generate(self, prompt: str, **kwargs) -> str:
            return "Mock response"
        
        def count_tokens(self, text: str) -> int:
            return len(text.split())
        
        def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
            return 0.001
    
    return MockClient()
```

---

## Quality Gates (Must Pass)

### Automated Checks
```bash
# All must pass for approval
make lint        # Ruff passes
make type-check  # Mypy passes
make test        # All tests pass
make coverage    # >80% coverage
make docker      # Docker builds successfully
```

### Code Quality Standards
- Type hints on all functions
- Docstrings on public APIs
- No `# type: ignore` unless absolutely necessary
- No security vulnerabilities (Ruff security rules)
- Max line length: 88 (Black standard)
- No unused imports

---

## Documentation Requirements

### README.md Structure
```markdown
# prompt-optimizer

One-line description.

## Features
- Bullet list of capabilities

## Quick Start
```bash
pip install prompt-optimizer
prompt-optimizer optimize prompt.yaml --test-cases tests.yaml
```

## Installation
## Usage Examples
## Configuration
## API Reference (link to docs)
## Contributing
## License
```

### Docstring Style
```python
def evaluate_response(response: str, expected: str, criteria: dict[str, float]) -> float:
    """Evaluate LLM response against expected output.
    
    Args:
        response: Generated text from LLM
        expected: Expected output text
        criteria: Scoring criteria with weights
        
    Returns:
        Weighted score between 0.0 and 1.0
        
    Raises:
        EvaluationError: If criteria cannot be applied
        
    Example:
        >>> score = evaluate_response("Paris", "Paris", {"accuracy": 1.0})
        >>> assert score == 1.0
    """
```

---

## Docker Support

### Dockerfile
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir -e .

# Copy source
COPY src/ src/

# CLI entrypoint
ENTRYPOINT ["prompt-optimizer"]
```

### docker-compose.yml
```yaml
version: '3.8'

services:
  prompt-optimizer:
    build: .
    volumes:
      - ./prompts:/app/prompts
      - ./results:/app/results
    environment:
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    command: optimize /app/prompts/example.yaml
```

---

## Examples to Include

### 1. Basic Optimization
```python
# examples/basic_optimization.py
from prompt_optimizer import Prompt, TestCase, optimize_prompt

# Define prompt
prompt = Prompt(
    template="Summarize this text in {length}: {text}",
    variables={"length": "one sentence", "text": ""}
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
print(f"Score: {results.best_variant.weighted_score}")
```

### 2. Custom Evaluator
```python
# examples/custom_evaluator.py
def custom_tone_evaluator(response: str, test_case: TestCase) -> float:
    """Custom tone scoring function."""
    expected_tone = test_case.expected_properties.get("tone")
    
    # Use sentiment analysis or LLM-based evaluation
    detected_tone = detect_tone(response)
    
    return 1.0 if detected_tone == expected_tone else 0.0

# Use in optimization
results = optimize_prompt(
    prompt,
    test_cases,
    custom_criteria={"tone": custom_tone_evaluator}
)
```

---

## Anti-Patterns to Avoid

### ❌ DON'T: Over-engineer the evaluator
```python
# BAD
class AbstractEvaluatorFactory:
    @staticmethod
    def create_evaluator(type: str) -> BaseEvaluator:
        ...

class EvaluatorRegistry:
    _evaluators = {}
    ...
```

### ✅ DO: Keep it simple
```python
# GOOD
EVALUATORS = {
    "accuracy": accuracy_scorer,
    "conciseness": conciseness_scorer,
    "tone": tone_scorer
}

def evaluate(response: str, criteria: dict) -> dict[str, float]:
    return {
        name: EVALUATORS[name](response)
        for name in criteria.keys()
    }
```

### ❌ DON'T: Create unnecessary abstractions
```python
# BAD
class PromptVariantGeneratorStrategyInterface(ABC):
    @abstractmethod
    def generate(self) -> PromptVariant:
        pass
```

### ✅ DO: Use simple functions
```python
# GOOD
def make_concise(prompt: str) -> str:
    """Remove unnecessary words."""
    # Simple string manipulation
    return prompt

STRATEGIES = {
    "concise": make_concise,
    "detailed": make_detailed,
}
```

---

## Performance Considerations

### Async Everything
```python
# LLM calls are I/O bound - use async
async def evaluate_all_variants(variants, test_cases, llm_client):
    tasks = [
        evaluate_variant(variant, test_case, llm_client)
        for variant in variants
        for test_case in test_cases
    ]
    return await asyncio.gather(*tasks)
```

### Caching
```python
# Cache LLM responses to avoid duplicate API calls
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_generate(prompt: str, model: str) -> str:
    # Hash prompt + model, cache result
    pass
```

### Rate Limiting
```python
# Respect API rate limits
from asyncio import Semaphore

async def rate_limited_generate(
    prompt: str,
    llm_client: LLMClient,
    semaphore: Semaphore
):
    async with semaphore:
        return await llm_client.generate(prompt)
```

---

## Success Criteria

The project is complete when:

1. ✅ CLI works: `prompt-optimizer optimize prompt.yaml`
2. ✅ Library works: `from prompt_optimizer import optimize_prompt`
3. ✅ All 3 LLM clients work (Anthropic, OpenAI, Ollama)
4. ✅ All tests pass (`pytest`)
5. ✅ Type checking passes (`mypy`)
6. ✅ Linting passes (`ruff`)
7. ✅ Coverage > 80%
8. ✅ Docker builds successfully
9. ✅ README has clear examples
10. ✅ Can be installed from source: `pip install -e .`

---

## Timeline Expectation

With this detailed AGENTS.md, Claude should be able to build this in:
- **Target: 2 hours**
- **Max acceptable: 4 hours**

If it takes longer, the AGENTS.md needs more detail.

---

## Notes for Claude

- Prefer simplicity over cleverness
- Write code that's easy to test
- Use type hints religiously
- Keep functions small and focused
- Avoid premature optimization
- If unsure about a design decision, pick the simpler option
- The goal is a working tool, not a framework

When complete, the tool should be immediately usable by someone who wants to optimize their LLM prompts without deep ML knowledge.
