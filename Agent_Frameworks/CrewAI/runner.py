"""
Runner — CLI entrypoint that demonstrates the two execution modes.

Modes:
  flow  (default) — run via ResearchFlow, the full @start/@listen/@router pipeline
  crew            — run the ResearchCrew directly, bypassing the Flow wrapper

Usage:
  python runner.py --topic "quantum computing" --mode flow
  python runner.py --topic "AI in healthcare" --mode crew --depth deep
  python runner.py --topic "climate solutions" --mode flow --depth quick
  python runner.py --topic "space exploration" --mode crew --depth standard
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

console = Console()


def _ensure_logs_dir() -> None:
    """Create logs/ before the Crew is built — output_log_file needs it to exist."""
    Path("logs").mkdir(exist_ok=True)


# ── Run modes ─────────────────────────────────────────────────────────────────

def run_flow(topic: str, depth: str) -> None:
    """
    Demonstrates: ResearchFlow.kickoff(), @start, @listen, @router, Flow state.
    This is the recommended production pattern in CrewAI.
    """
    from flow import ResearchFlow

    console.print(Panel(
        f"[bold purple]Mode: ResearchFlow[/]\n"
        f"[cyan]Topic:[/]  {topic}\n"
        f"[cyan]Depth:[/]  {depth}\n\n"
        "[dim]Execution path: @start → @listen (crew) → @router → @listen (publish|revise)[/]",
        title="[bold]CrewAI Advanced Research System[/]",
        border_style="purple",
    ))

    flow = ResearchFlow()
    flow.kickoff(inputs={"topic": topic, "depth": depth})

    # Flow state is accessible after kickoff
    console.print()
    _print_flow_summary(flow.state)


def run_crew(topic: str, depth: str) -> None:
    """
    Demonstrates: crew.kickoff(), per-task Pydantic outputs, CrewOutput structure.
    Runs the hierarchical crew directly, without the Flow wrapper.
    """
    from crew import build_research_crew
    from tasks import analysis_task, editing_task, research_task, writing_task

    console.print(Panel(
        f"[bold purple]Mode: Direct Crew[/]\n"
        f"[cyan]Topic:[/]  {topic}\n"
        f"[cyan]Depth:[/]  {depth}\n\n"
        "[dim]Runs Process.hierarchical crew directly (no Flow wrapper)[/]",
        title="[bold]CrewAI Advanced Research System[/]",
        border_style="purple",
    ))

    research_crew = build_research_crew()
    result = research_crew.kickoff(inputs={"topic": topic, "depth": depth})

    _print_crew_outputs(result, [research_task, analysis_task, writing_task, editing_task])


# ── Output helpers ─────────────────────────────────────────────────────────────

def _print_flow_summary(state) -> None:
    """Print a summary table of the flow state after completion."""
    table = Table(title="Flow State Summary", border_style="purple")
    table.add_column("Field", style="cyan")
    table.add_column("Value")
    table.add_row("Topic",          state.topic)
    table.add_row("Depth",          state.depth)
    table.add_row("Quality Score",  f"{state.quality_score:.2f}")
    table.add_row("Approved",       "[green]Yes[/]" if state.approved else "[yellow]No[/]")
    table.add_row("Run Complete",   "[green]Yes[/]" if state.run_complete else "[red]No[/]")
    table.add_row("Report Length",  f"{len(state.final_report)} chars")
    console.print(table)

    if state.final_report:
        console.print()
        console.print(Panel(
            Markdown(state.final_report),
            title="[bold green]Final Report[/]",
            border_style="green",
        ))


def _print_crew_outputs(result, tasks: list) -> None:
    """
    Print a summary table of per-task Pydantic outputs.
    Demonstrates accessing task.output.pydantic after crew.kickoff().
    """
    console.print()
    table = Table(title="Per-Task Output Summary", border_style="purple")
    table.add_column("Task",   style="cyan",  no_wrap=True)
    table.add_column("Agent",  style="magenta")
    table.add_column("Output Type", style="yellow")
    table.add_column("Preview", style="dim")

    for task in tasks:
        if task.output:
            pydantic_obj = getattr(task.output, "pydantic", None)
            output_type = type(pydantic_obj).__name__ if pydantic_obj else "str (raw)"
            raw_preview = str(getattr(task.output, "raw", ""))[:60] + "..."
            agent_role = getattr(task.agent, "role", "unknown")[:28]
            desc_preview = task.description.split("\n")[0][:35] + "..."
            table.add_row(desc_preview, agent_role, output_type, raw_preview)

    console.print(table)

    # Show typed outputs for research and analysis tasks
    try:
        from tasks import research_task, analysis_task

        if research_task.output and research_task.output.pydantic:
            rg = research_task.output.pydantic
            console.print(Panel(
                f"[cyan]Sources found:[/] {len(rg.sources)}\n"
                f"[cyan]Search queries:[/] {len(rg.search_queries_used)}\n"
                f"[cyan]Coverage:[/] {rg.coverage_assessment[:150]}",
                title="[bold]ResearchGathering (Pydantic output)[/]",
                border_style="cyan",
            ))

        if analysis_task.output and analysis_task.output.pydantic:
            ar = analysis_task.output.pydantic
            findings_preview = "\n".join(f"  • {f}" for f in ar.key_findings[:5])
            console.print(Panel(
                f"[cyan]Key findings ({len(ar.key_findings)}):[/]\n{findings_preview}\n\n"
                f"[cyan]Confidence:[/] {ar.confidence_score:.2f}",
                title="[bold]AnalysisResult (Pydantic output)[/]",
                border_style="cyan",
            ))
    except Exception:
        pass

    if result.raw:
        console.print()
        console.print(Panel(
            Markdown(result.raw),
            title="[bold green]Final Report (CrewOutput.raw)[/]",
            border_style="green",
        ))


# ── CLI entrypoint ─────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="CrewAI Advanced Research Multi-Agent System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--topic",
        required=True,
        help="Research topic (e.g. 'quantum computing', 'AI in healthcare')",
    )
    parser.add_argument(
        "--mode",
        choices=["flow", "crew"],
        default="flow",
        help="Execution mode: flow (ResearchFlow, default) or crew (direct crew kickoff)",
    )
    parser.add_argument(
        "--depth",
        choices=["quick", "standard", "deep"],
        default="standard",
        help="Research depth: quick (3-4 sources), standard (5+), deep (5+ including 3 arXiv papers)",
    )
    args = parser.parse_args()

    # Create logs/ directory before any crew is built
    _ensure_logs_dir()

    console.print(f"[dim]Mode: {args.mode} | Depth: {args.depth}[/]\n")

    if args.mode == "flow":
        run_flow(args.topic, args.depth)
    else:
        run_crew(args.topic, args.depth)


if __name__ == "__main__":
    main()
