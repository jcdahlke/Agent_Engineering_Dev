"""
Runner — CLI entrypoint demonstrating Microsoft Agent Framework features.

Modes:
  basic  — run the full pipeline, print final report and summary
  stream — stream tokens live as the orchestrator thinks and calls specialists
  hitl   — same as basic; publish_report tool requires human approval before
           executing (approval_mode="always_require" handles the pause automatically)

Usage:
  python runner.py --topic "quantum computing" --mode basic
  python runner.py --topic "AI in healthcare" --mode stream --depth deep
  python runner.py --topic "climate solutions" --mode hitl --max-iter 4

Agent Framework async notes:
  All agent.run() calls return coroutines. runner.py wraps them in asyncio.run()
  for the CLI entrypoint. In a web app, await them directly in your async handler.
"""
from __future__ import annotations

import argparse
import asyncio

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from workflow import build_pipeline

console = Console()


# ── Run modes ─────────────────────────────────────────────────────────────────

async def run_basic(topic: str, depth: str, max_iter: int) -> None:
    """
    Demonstrates: agent.run() (non-streaming), shared state inspection,
    MaxIterationsMiddleware termination.
    """
    console.print(Panel(
        f"[bold cyan]Mode: Basic Invoke[/]\nTopic: {topic}\nDepth: {depth}",
        title="[bold]Microsoft Agent Framework — Research System[/]",
        border_style="cyan",
    ))

    orchestrator, state = build_pipeline(topic, depth, max_iter)

    initial_prompt = (
        f"Research the following topic and produce a comprehensive report:\n\n"
        f"TOPIC: {topic}\n"
        f"DEPTH: {depth}\n"
        f"MAX ITERATIONS: {max_iter}\n\n"
        f"Follow the routing strategy in your instructions. "
        f"Start by calling call_researcher."
    )

    console.print("\n[dim]Starting orchestrator run…[/]\n")

    try:
        result = await orchestrator.run(initial_prompt)
        if result and result.text:
            console.print(f"[dim]Orchestrator final message:[/] {result.text[:300]}\n")
    except Exception as exc:
        # MaxIterationsMiddleware raises MiddlewareTermination — surface gracefully
        console.print(f"[yellow]Run ended: {exc}[/]\n")

    _print_summary(state)


async def run_streaming(topic: str, depth: str, max_iter: int) -> None:
    """
    Demonstrates: agent.run(stream=True) — each chunk has .text for partial
    token output. Prints tokens as they arrive for live visibility.

    Agent Framework streaming is chunk-based: each chunk may be a partial
    token, a tool call announcement, or a tool result. Check chunk.text
    for non-None values before printing.
    """
    console.print(Panel(
        f"[bold cyan]Mode: Streaming[/]\nTopic: {topic}",
        title="[bold]Microsoft Agent Framework — Streaming[/]",
        border_style="cyan",
    ))

    orchestrator, state = build_pipeline(topic, depth, max_iter)

    initial_prompt = (
        f"Research the following topic and produce a comprehensive report:\n"
        f"TOPIC: {topic}\nDEPTH: {depth}\n\n"
        f"Start by calling call_researcher."
    )

    console.print("\n[bold yellow]── Streaming orchestrator output ──[/]\n")

    try:
        async for chunk in orchestrator.run(initial_prompt, stream=True):
            if chunk.text:
                console.print(chunk.text, end="", highlight=False)
    except Exception as exc:
        console.print(f"\n[yellow]Stream ended: {exc}[/]")

    console.print("\n")
    _print_summary(state)


async def run_hitl(topic: str, depth: str, max_iter: int) -> None:
    """
    Demonstrates: @tool(approval_mode="always_require") — the framework pauses
    before executing publish_report and prompts for human confirmation.

    In Agent Framework, HITL is tool-centric: the approval gate is declared
    on the tool itself. The framework handles the pause/resume mechanics.
    The runner just needs to handle the approval prompt the framework surfaces.

    Contrast with LangGraph:
      - LangGraph: interrupt() pauses the entire graph mid-node; state is
        preserved via MemorySaver; Command(resume=...) injects the response.
      - Agent Framework: pauses only for the specific tool call; the agent's
        conversation context is preserved internally by the framework.
    """
    console.print(Panel(
        f"[bold cyan]Mode: Human-in-the-Loop[/]\nTopic: {topic}\n\n"
        "[dim]The publish_report tool requires human approval before execution.\n"
        "The framework will pause and prompt you when the Writer is ready.[/]",
        title="[bold]Microsoft Agent Framework — HITL[/]",
        border_style="cyan",
    ))

    # Fewer iterations so the pipeline reaches publish_report quickly for demos
    effective_iter = min(max_iter, 4)
    orchestrator, state = build_pipeline(topic, depth, effective_iter)

    initial_prompt = (
        f"Research the following topic and produce a comprehensive report:\n"
        f"TOPIC: {topic}\nDEPTH: {depth}\n\n"
        f"After the Critic approves the draft, have the Writer call publish_report "
        f"to trigger the human approval gate."
    )

    console.print("\n[bold yellow]── Running until publish_report approval gate ──[/]\n")
    console.print(
        "[dim]The framework will pause and ask for your approval "
        "before publish_report executes.[/]\n"
    )

    try:
        result = await orchestrator.run(initial_prompt)
        if result and result.text:
            console.print(f"[green]Run completed.[/] {result.text[:300]}")
    except Exception as exc:
        console.print(f"[yellow]Run ended: {exc}[/]")

    _print_summary(state)


# ── Output helpers ────────────────────────────────────────────────────────────

def _print_summary(state: dict) -> None:
    console.print()
    table = Table(title="Research Run Summary", border_style="cyan")
    table.add_column("Field",          style="cyan",  no_wrap=True)
    table.add_column("Value",          style="white")

    table.add_row("Topic",             state.get("topic", "—"))
    table.add_row("Depth",             state.get("depth", "—"))
    table.add_row("Iterations",        str(state.get("iteration_count", 0)))
    table.add_row("Content blocks",    str(len(state.get("raw_results", []))))
    table.add_row("Analysis blocks",   str(len(state.get("key_findings", []))))
    table.add_row("Quality score",     f"{state.get('quality_score', 0.0):.2f}")
    table.add_row("Critic approved",   "[green]Yes[/]" if state.get("approved") else "[yellow]No[/]")
    table.add_row("Task complete",     "[green]Yes[/]" if state.get("task_complete") else "[red]No[/]")
    console.print(table)

    draft = state.get("report_draft", "")
    if draft:
        console.print()
        console.print(Panel(
            Markdown(draft),
            title="[bold green]Final Report[/]",
            border_style="green",
        ))
    else:
        console.print("\n[yellow]No report draft in state. The pipeline may not have reached the Writer.[/]")

    critique = state.get("critique", "")
    if critique:
        console.print()
        console.print(Panel(
            critique[:800],
            title="[bold red]Critic Feedback[/]",
            border_style="red",
        ))


# ── CLI entrypoint ────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Microsoft Agent Framework — Advanced Research Multi-Agent System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--topic",  required=True, help="Research topic")
    parser.add_argument(
        "--mode",
        choices=["basic", "stream", "hitl"],
        default="basic",
        help="Execution mode (default: basic)",
    )
    parser.add_argument(
        "--depth",
        choices=["quick", "standard", "deep"],
        default="standard",
        help="Research depth (default: standard)",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=5,
        dest="max_iter",
        help="Maximum orchestrator iterations (default: 5)",
    )
    args = parser.parse_args()

    console.print(f"[dim]Topic: {args.topic} | Mode: {args.mode} | Depth: {args.depth}[/]\n")

    modes = {
        "basic":  run_basic,
        "stream": run_streaming,
        "hitl":   run_hitl,
    }

    asyncio.run(modes[args.mode](args.topic, args.depth, args.max_iter))


if __name__ == "__main__":
    main()
