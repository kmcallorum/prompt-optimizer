"""Click-based CLI for prompt-optimizer."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import click
import yaml
from prometheus_client import start_http_server
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.syntax import Syntax

from prompt_optimizer.core import (
    evaluate_all_variants,
    generate_variants,
    get_llm_client,
    optimize_prompt_async,
)
from prompt_optimizer.llm_judge import LLMJudge
from prompt_optimizer.metrics import init_metrics
from prompt_optimizer.prompt import Prompt, TestCase
from prompt_optimizer.reporters import (
    display_comparison,
    display_results,
    save_html_report,
    save_json_report,
)
from prompt_optimizer.storage import PromptStorage

console = Console()


def load_prompt_from_yaml(path: Path) -> Prompt:
    """Load a prompt from a YAML file."""
    data = yaml.safe_load(path.read_text())
    return Prompt(
        template=data.get("template", ""),
        variables=data.get("variables", {}),
        system_message=data.get("system_message"),
        metadata=data.get("metadata", {}),
        name=data.get("name", path.stem),
    )


def load_test_cases_from_yaml(path: Path) -> list[TestCase]:
    """Load test cases from a YAML file."""
    data = yaml.safe_load(path.read_text())

    if "test_cases" in data:
        return [TestCase(**tc) for tc in data["test_cases"]]
    elif isinstance(data, list):
        return [TestCase(**tc) for tc in data]
    else:
        raise ValueError("Invalid test cases format")


@click.group()
@click.version_option(version="0.3.0")
def cli() -> None:
    """Prompt Optimizer - Test and optimize your LLM prompts."""
    pass


@cli.command()
@click.option("--path", default=".", help="Directory to initialize")
def init(path: str) -> None:
    """Initialize a new prompt-optimizer project."""
    project_path = Path(path)
    storage_path = project_path / ".prompt-optimizer"
    storage_path.mkdir(exist_ok=True)

    prompts_path = project_path / "prompts"
    prompts_path.mkdir(exist_ok=True)

    tests_path = project_path / "tests"
    tests_path.mkdir(exist_ok=True)

    example_prompt = prompts_path / "example.yaml"
    if not example_prompt.exists():
        example_prompt.write_text(
            """template: |
  Answer the following question: {{ question }}

  Requirements:
  - Be concise
  - Be accurate

system_message: "You are a helpful AI assistant."

variables:
  question: ""

metadata:
  author: "prompt-optimizer"
  version: "1.0"
"""
        )

    example_tests = tests_path / "example_tests.yaml"
    if not example_tests.exists():
        example_tests.write_text(
            """name: "Example Test Suite"

test_cases:
  - input_variables:
      question: "What is 2 + 2?"
    expected_output: "4"
    expected_properties:
      length: "<20 words"

  - input_variables:
      question: "What is the capital of France?"
    expected_output: "Paris"
    expected_properties:
      length: "<20 words"
"""
        )

    abs_path = project_path.absolute()
    console.print(
        Panel(
            f"Initialized prompt-optimizer project in [cyan]{abs_path}[/cyan]\n\n"
            "Created:\n"
            "  - .prompt-optimizer/ (storage directory)\n"
            "  - prompts/example.yaml (example prompt)\n"
            "  - tests/example_tests.yaml (example tests)\n\n"
            "Get started:\n"
            "  prompt-optimizer test prompts/example.yaml --test-cases tests.yaml",
            title="Project Initialized",
        )
    )


@cli.command()
@click.argument("prompt_file", type=click.Path(exists=True))
@click.option("--test-cases", required=True, type=click.Path(exists=True))
@click.option("--llm", default="claude-sonnet-4", help="LLM to use for generation")
@click.option("--judge", default=None, help="LLM judge for AI-based evaluation")
def test(prompt_file: str, test_cases: str, llm: str, judge: str | None) -> None:
    """Test a single prompt against test cases."""
    prompt = load_prompt_from_yaml(Path(prompt_file))
    cases = load_test_cases_from_yaml(Path(test_cases))

    console.print(f"Testing prompt: [cyan]{prompt.name}[/cyan]")
    console.print(f"Test cases: [cyan]{len(cases)}[/cyan]")
    console.print(f"LLM: [cyan]{llm}[/cyan]")
    if judge:
        console.print(f"Judge: [cyan]{judge}[/cyan] (AI-based evaluation)")

    async def run_test() -> None:
        llm_client = get_llm_client(llm)
        variants = [
            generate_variants(prompt, ["concise"])[0]
        ]  # Just test the base prompt

        # Create judge if specified
        judge_obj: LLMJudge | None = None
        if judge:
            judge_client = get_llm_client(judge)
            judge_obj = LLMJudge(judge_client)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            progress.add_task("Running tests...", total=None)
            results = await evaluate_all_variants(
                variants, cases, llm_client, judge=judge_obj
            )

        console.print()
        for i, result in enumerate(results, 1):
            acc = result.scores.get("accuracy", 0)
            status = "[green]PASS[/green]" if acc > 0.7 else "[red]FAIL[/red]"
            console.print(f"Test {i}: {status} (accuracy: {acc:.2%})")
            if judge:
                # Show all AI judge scores
                pairs = [f"{k}: {v:.2f}" for k, v in result.scores.items()]
                console.print(f"  Scores: {', '.join(pairs)}")
            console.print(f"  Output: {result.output[:100]}...")

    asyncio.run(run_test())


@cli.command()
@click.argument("prompt_file", type=click.Path(exists=True))
@click.option("--test-cases", required=True, type=click.Path(exists=True))
@click.option(
    "--strategies",
    default="concise,detailed",
    help="Comma-separated list of strategies",
)
@click.option("--llm", default="claude-sonnet-4", help="LLM to use for generation")
@click.option("--judge", default=None, help="LLM judge for AI-based evaluation")
@click.option("--output", type=click.Path(), help="Output file for results (JSON)")
def optimize(
    prompt_file: str,
    test_cases: str,
    strategies: str,
    llm: str,
    judge: str | None,
    output: str | None,
) -> None:
    """Optimize a prompt with multiple strategies."""
    prompt = load_prompt_from_yaml(Path(prompt_file))
    cases = load_test_cases_from_yaml(Path(test_cases))
    strategy_list = [s.strip() for s in strategies.split(",")]

    console.print(f"Optimizing: [cyan]{prompt.name}[/cyan]")
    console.print(f"Strategies: [cyan]{', '.join(strategy_list)}[/cyan]")
    console.print(f"Test cases: [cyan]{len(cases)}[/cyan]")
    console.print(f"LLM: [cyan]{llm}[/cyan]")
    if judge:
        console.print(f"Judge: [cyan]{judge}[/cyan] (AI-based evaluation)")

    async def run_optimization() -> None:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            progress.add_task("Optimizing...", total=None)
            results = await optimize_prompt_async(
                prompt=prompt,
                test_cases=cases,
                strategies=strategy_list,
                llm=llm,
                judge_llm=judge,
            )

        display_results(console, results)

        if output:
            output_path = Path(output)
            if output_path.suffix == ".html":
                save_html_report(results, output_path)
            else:
                save_json_report(results, output_path)
            console.print(f"\nResults saved to: [cyan]{output}[/cyan]")

    asyncio.run(run_optimization())


@cli.command()
@click.argument("prompt1", type=click.Path(exists=True))
@click.argument("prompt2", type=click.Path(exists=True))
@click.option("--test-cases", required=True, type=click.Path(exists=True))
@click.option("--llm", default="claude-sonnet-4", help="LLM to use")
def compare(prompt1: str, prompt2: str, test_cases: str, llm: str) -> None:
    """Compare two prompts against the same test cases."""
    p1 = load_prompt_from_yaml(Path(prompt1))
    p2 = load_prompt_from_yaml(Path(prompt2))
    cases = load_test_cases_from_yaml(Path(test_cases))

    console.print(f"Comparing: [cyan]{p1.name}[/cyan] vs [cyan]{p2.name}[/cyan]")
    console.print(f"Test cases: [cyan]{len(cases)}[/cyan]")

    async def run_comparison() -> None:
        llm_client = get_llm_client(llm)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            progress.add_task("Comparing prompts...", total=None)

            v1 = generate_variants(p1, ["concise"])[0]
            v2 = generate_variants(p2, ["concise"])[0]

            results1 = await evaluate_all_variants([v1], cases, llm_client)
            results2 = await evaluate_all_variants([v2], cases, llm_client)

        display_comparison(console, results1, results2, p1.name, p2.name)

    asyncio.run(run_comparison())


@cli.command()
@click.argument("prompt_name")
@click.option("--storage", default=".prompt-optimizer", help="Storage directory")
def history(prompt_name: str, storage: str) -> None:
    """Show optimization history for a prompt."""
    storage_obj = PromptStorage(Path(storage))
    versions = storage_obj.history(prompt_name)

    if not versions:
        console.print(f"No history found for: [cyan]{prompt_name}[/cyan]")
        return

    from datetime import datetime

    from rich.table import Table

    table = Table(title=f"History: {prompt_name}")
    table.add_column("Version", style="cyan")
    table.add_column("Modified", style="green")
    table.add_column("Size", justify="right")

    for v in versions:
        modified = datetime.fromtimestamp(float(v["modified"])).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        table.add_row(str(v["version"]), modified, f"{v['size']} bytes")

    console.print(table)


@cli.command()
@click.argument("results_file", type=click.Path(exists=True))
@click.option(
    "--format", "fmt", default="terminal",
    type=click.Choice(["terminal", "html", "json"])
)
@click.option("--output", type=click.Path(), help="Output file path")
def report(results_file: str, fmt: str, output: str | None) -> None:
    """Generate a report from results file."""
    data = json.loads(Path(results_file).read_text())

    if fmt == "terminal":
        console.print(Panel("[bold]Results Report[/bold]", title="Prompt Optimizer"))
        base_prompt = data.get('base_prompt_name', 'N/A')
        console.print(f"Base prompt: [cyan]{base_prompt}[/cyan]")
        console.print(f"Variants tested: [cyan]{data.get('variants_tested', 0)}[/cyan]")
        console.print(f"Test cases: [cyan]{data.get('test_cases_run', 0)}[/cyan]")
        console.print(f"Total cost: [cyan]${data.get('total_cost', 0):.4f}[/cyan]")

        if "best_variant" in data:
            bv = data["best_variant"]
            console.print(f"\nBest variant: [green]{bv.get('strategy', 'N/A')}[/green]")
            console.print(f"Score: [green]{bv.get('weighted_score', 0):.2%}[/green]")

    elif fmt == "html":

        output_path = Path(output) if output else Path("report.html")
        html = f"""<!DOCTYPE html>
<html>
<head><title>Report</title></head>
<body>
<h1>Optimization Report</h1>
<pre>{json.dumps(data, indent=2)}</pre>
</body>
</html>"""
        output_path.write_text(html)
        console.print(f"HTML report saved to: [cyan]{output_path}[/cyan]")

    elif fmt == "json":
        output_path = Path(output) if output else Path("report.json")
        output_path.write_text(json.dumps(data, indent=2))
        console.print(f"JSON report saved to: [cyan]{output_path}[/cyan]")


@cli.command()
@click.argument("prompt_file", type=click.Path(exists=True))
def show(prompt_file: str) -> None:
    """Display a prompt file with syntax highlighting."""
    content = Path(prompt_file).read_text()
    syntax = Syntax(content, "yaml", theme="monokai", line_numbers=True)
    console.print(syntax)


@cli.command()
@click.option("--port", default=8000, help="Port for metrics server")
def metrics(port: int) -> None:
    """Start Prometheus metrics server."""
    init_metrics()
    console.print(f"Starting Prometheus metrics server on port [cyan]{port}[/cyan]")
    console.print(f"Metrics available at: [cyan]http://localhost:{port}/metrics[/cyan]")
    console.print("Press Ctrl+C to stop")

    start_http_server(port)

    # Keep the server running
    try:
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        console.print("\nMetrics server stopped")


if __name__ == "__main__":  # pragma: no cover
    cli()
