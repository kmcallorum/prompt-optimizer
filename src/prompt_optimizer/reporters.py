"""Results reporting utilities."""

import json
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from prompt_optimizer.evaluator import EvaluationResult, OptimizationResults


def display_results(console: Console, results: OptimizationResults) -> None:
    """Display optimization results in a rich table."""
    console.print()
    console.print(
        Panel(
            f"[bold]Optimization Complete[/bold]\n"
            f"Variants tested: {results.variants_tested}\n"
            f"Test cases: {results.test_cases_run}\n"
            f"Total cost: ${results.total_cost:.4f}\n"
            f"Total time: {results.total_time_seconds:.2f}s",
            title="Summary",
        )
    )

    table = Table(title="Variant Scores")
    table.add_column("Strategy", style="cyan")
    table.add_column("Accuracy", justify="right")
    table.add_column("Conciseness", justify="right")
    table.add_column("Avg Score", justify="right", style="green")
    table.add_column("Cost", justify="right")

    variant_scores: dict[str, dict[str, list[float]]] = {}
    for result in results.all_results:
        strategy = result.variant.strategy
        if strategy not in variant_scores:
            variant_scores[strategy] = {
                "accuracy": [],
                "conciseness": [],
                "cost": [],
            }
        variant_scores[strategy]["accuracy"].append(
            result.scores.get("accuracy", 0.0)
        )
        variant_scores[strategy]["conciseness"].append(
            result.scores.get("conciseness", 0.0)
        )
        variant_scores[strategy]["cost"].append(result.cost_usd)

    for strategy, scores in variant_scores.items():
        avg_acc = sum(scores["accuracy"]) / len(scores["accuracy"])
        avg_conc = sum(scores["conciseness"]) / len(scores["conciseness"])
        avg_score = (avg_acc + avg_conc) / 2
        total_cost = sum(scores["cost"])

        is_best = strategy == results.best_variant.strategy
        style = "bold green" if is_best else ""

        table.add_row(
            f"{'* ' if is_best else ''}{strategy}",
            f"{avg_acc:.2%}",
            f"{avg_conc:.2%}",
            f"{avg_score:.2%}",
            f"${total_cost:.4f}",
            style=style,
        )

    console.print(table)

    console.print()
    console.print(
        f"[bold green]Best variant:[/bold green] {results.best_variant.strategy} "
        f"(score: {results.best_weighted_score:.2%})"
    )


def display_comparison(
    console: Console,
    results1: list[EvaluationResult],
    results2: list[EvaluationResult],
    name1: str,
    name2: str,
) -> None:
    """Display comparison between two prompts."""
    table = Table(title=f"Comparison: {name1} vs {name2}")
    table.add_column("Metric", style="cyan")
    table.add_column(name1, justify="right")
    table.add_column(name2, justify="right")
    table.add_column("Winner", justify="center")

    def avg_metric(results: list[EvaluationResult], metric: str) -> float:
        values = [r.scores.get(metric, 0.0) for r in results]
        return sum(values) / len(values) if values else 0.0

    metrics = ["accuracy", "conciseness"]
    for metric in metrics:
        v1 = avg_metric(results1, metric)
        v2 = avg_metric(results2, metric)
        winner = name1 if v1 > v2 else (name2 if v2 > v1 else "Tie")
        table.add_row(metric.capitalize(), f"{v1:.2%}", f"{v2:.2%}", winner)

    cost1 = sum(r.cost_usd for r in results1)
    cost2 = sum(r.cost_usd for r in results2)
    winner = name1 if cost1 < cost2 else (name2 if cost2 < cost1 else "Tie")
    table.add_row("Total Cost", f"${cost1:.4f}", f"${cost2:.4f}", winner)

    console.print(table)


def results_to_dict(results: OptimizationResults) -> dict[str, Any]:
    """Convert optimization results to a dictionary for JSON export."""
    return {
        "base_prompt_name": results.base_prompt_name,
        "variants_tested": results.variants_tested,
        "test_cases_run": results.test_cases_run,
        "best_variant": {
            "strategy": results.best_variant.strategy,
            "template": results.best_variant.template,
            "weighted_score": results.best_weighted_score,
        },
        "total_cost": results.total_cost,
        "total_time_seconds": results.total_time_seconds,
        "all_results": [
            {
                "strategy": r.variant.strategy,
                "output": r.output,
                "scores": r.scores,
                "latency_ms": r.latency_ms,
                "token_count": r.token_count,
                "cost_usd": r.cost_usd,
            }
            for r in results.all_results
        ],
    }


def save_json_report(results: OptimizationResults, path: Path) -> None:
    """Save results as JSON."""
    data = results_to_dict(results)
    path.write_text(json.dumps(data, indent=2))


def generate_html_report(results: OptimizationResults) -> str:
    """Generate an HTML report."""
    data = results_to_dict(results)

    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Prompt Optimization Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, sans-serif;
            margin: 40px;
        }}
        h1 {{ color: #333; }}
        .summary {{
            background: #f5f5f5;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
        }}
        .best {{
            background: #e8f5e9;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #4caf50;
        }}
        table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background: #333; color: white; }}
        tr:nth-child(even) {{ background: #f9f9f9; }}
        .score {{ font-weight: bold; }}
        .good {{ color: #4caf50; }}
        .medium {{ color: #ff9800; }}
        .poor {{ color: #f44336; }}
    </style>
</head>
<body>
    <h1>Prompt Optimization Report</h1>

    <div class="summary">
        <h2>Summary</h2>
        <p><strong>Base Prompt:</strong> {data['base_prompt_name']}</p>
        <p><strong>Variants Tested:</strong> {data['variants_tested']}</p>
        <p><strong>Test Cases:</strong> {data['test_cases_run']}</p>
        <p><strong>Total Cost:</strong> ${data['total_cost']:.4f}</p>
        <p><strong>Total Time:</strong> {data['total_time_seconds']:.2f}s</p>
    </div>

    <div class="best">
        <h2>Best Variant: {data['best_variant']['strategy']}</h2>
        <p><strong>Score:</strong> {data['best_variant']['weighted_score']:.2%}</p>
        <pre>{data['best_variant']['template']}</pre>
    </div>

    <h2>All Results</h2>
    <table>
        <tr>
            <th>Strategy</th>
            <th>Accuracy</th>
            <th>Conciseness</th>
            <th>Latency (ms)</th>
            <th>Cost</th>
        </tr>"""

    for r in data["all_results"]:
        acc = r["scores"].get("accuracy", 0)
        conc = r["scores"].get("conciseness", 0)
        acc_class = "good" if acc > 0.8 else ("medium" if acc > 0.5 else "poor")
        conc_class = "good" if conc > 0.8 else ("medium" if conc > 0.5 else "poor")

        html += f"""
        <tr>
            <td>{r['strategy']}</td>
            <td class="score {acc_class}">{acc:.2%}</td>
            <td class="score {conc_class}">{conc:.2%}</td>
            <td>{r['latency_ms']:.0f}</td>
            <td>${r['cost_usd']:.4f}</td>
        </tr>"""

    html += """
    </table>
</body>
</html>"""

    return html


def save_html_report(results: OptimizationResults, path: Path) -> None:
    """Save results as HTML."""
    html = generate_html_report(results)
    path.write_text(html)
