"""
Pydantic AI — Advanced Research Multi-Agent System

Usage:
  python runner.py --topic "AI safety" --depth standard --mode basic
  python runner.py --topic "quantum computing" --depth deep --mode stream

Modes:
  basic   — run full pipeline, print summary table and final report
  stream  — demonstrate agent.run_stream() by streaming the writer output token by token
"""
import argparse
import asyncio
import sys

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from pipeline import PipelineResult, run_research

console = Console()

BRAND = "#e92063"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pydantic AI — Advanced Research Multi-Agent System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--topic", required=True, help="Research topic to investigate")
    parser.add_argument(
        "--depth",
        choices=["quick", "standard", "deep"],
        default="standard",
        help="Research depth (default: standard)",
    )
    parser.add_argument(
        "--mode",
        choices=["basic", "stream"],
        default="basic",
        help="Run mode: basic (full pipeline) or stream (writer streaming demo)",
    )
    return parser.parse_args()


def print_results(result: PipelineResult) -> None:
    table = Table(
        title="Research Pipeline Summary",
        border_style=BRAND,
        header_style=f"bold {BRAND}",
    )
    table.add_column("Field", style="cyan", no_wrap=True)
    table.add_column("Value", style="white")

    src_count = len(result.research.sources) if result.research else 0
    finding_count = len(result.analysis.key_findings) if result.analysis else 0
    confidence = f"{result.analysis.confidence:.2f}" if result.analysis else "N/A"
    score = f"{result.critique.score:.2f}" if result.critique else "N/A"
    approved_str = "[green]Yes[/]" if result.approved else "[yellow]No[/]"

    table.add_row("Topic", result.topic)
    table.add_row("Sources Found", str(src_count))
    table.add_row("Key Findings", str(finding_count))
    table.add_row("Analysis Confidence", confidence)
    table.add_row("Quality Score", score)
    table.add_row("Approved", approved_str)
    table.add_row("Revision Cycles", str(result.revision_cycles))

    console.print(table)
    console.print()

    if result.final_markdown:
        console.print(
            Panel(
                Markdown(result.final_markdown),
                title=f"[bold]Final Report — {result.topic}[/]",
                border_style="green",
                padding=(1, 2),
            )
        )

    if result.critique:
        console.print(
            Panel(
                f"[bold]Strengths:[/]\n"
                + "\n".join(f"  + {s}" for s in result.critique.strengths)
                + "\n\n[bold]Weaknesses:[/]\n"
                + "\n".join(f"  - {w}" for w in result.critique.weaknesses),
                title="[dim]Critic Feedback[/]",
                border_style="dim",
            )
        )


async def run_stream_mode(topic: str, depth: str) -> None:
    """Demonstrate Pydantic AI's streaming API.

    agent.run_stream() returns an async context manager. Inside,
    stream.stream_text(delta=True) yields incremental text chunks
    before the final validated output is assembled.

    Note: We use output_type=str here so we can stream raw text.
    Structured output streaming yields partial JSON, which is less readable.
    """
    import httpx

    from agents.supervisor import supervisor_agent
    from agents.researcher import researcher_agent
    from agents.analyzer import analyzer_agent
    from agents.writer import writer_agent
    from config import config, ResearchConfig
    from deps import ResearchDependencies
    from pydantic_ai import Agent
    from pydantic_ai.models.openai import OpenAIModel

    console.print(
        Panel(
            "[dim]Stream mode: runs steps 1-3 via full pipeline, "
            "then streams the writer agent output token by token.[/]",
            border_style=BRAND,
        )
    )

    # Run steps 1-3 to get analysis
    console.print("[dim]Running pipeline steps 1-3 (supervisor → researcher → analyzer)...[/]\n")
    result = await run_research(topic, depth)

    if not result.analysis:
        console.print("[red]Pipeline did not reach analysis stage. Cannot stream.[/]")
        return

    # Create a plain-text writer agent for streaming demo
    stream_writer: Agent[ResearchDependencies, str] = Agent(
        model=OpenAIModel(config.writer_model),
        output_type=str,
        deps_type=ResearchDependencies,
        system_prompt=(
            "You are a Research Writer. Write a comprehensive, engaging research report "
            "in Markdown format. Include an introduction, main findings, analysis, and conclusion."
        ),
    )

    async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as http:
        deps = ResearchDependencies(http_client=http, config=ResearchConfig())
        findings_text = "\n".join(f"- {f}" for f in result.analysis.key_findings)

        console.print(f"\n[bold {BRAND}]Streaming writer output:[/]\n")
        console.print("─" * 60)

        async with stream_writer.run_stream(
            f"Write a comprehensive research report on: {topic}\n\n"
            f"Key findings to cover:\n{findings_text}\n\n"
            f"Sources: {', '.join(result.research.sources[:8]) if result.research else ''}",
            deps=deps,
        ) as stream:
            async for chunk in stream.stream_text(delta=True):
                console.print(chunk, end="", markup=False)

        console.print("\n" + "─" * 60)
        console.print(f"\n[green]Stream complete.[/]")


def main() -> None:
    args = parse_args()

    console.print(
        Panel(
            f"[bold]Pydantic AI Research System[/]\n\n"
            f"[cyan]Topic:[/]  {args.topic}\n"
            f"[cyan]Depth:[/]  {args.depth}\n"
            f"[cyan]Mode:[/]   {args.mode}\n\n"
            "[dim]Pipeline: Supervisor → Researcher → Analyzer → Writer → Critic[/]",
            title=f"[bold {BRAND}]Pydantic AI[/] Multi-Agent Research System",
            border_style=BRAND,
        )
    )

    if args.mode == "stream":
        asyncio.run(run_stream_mode(args.topic, args.depth))
    else:
        result = asyncio.run(run_research(args.topic, args.depth))
        print_results(result)

    sys.exit(0)


if __name__ == "__main__":
    main()
