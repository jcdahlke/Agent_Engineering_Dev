"""
CLI entrypoint for the LlamaIndex Research Multi-Agent System.

Usage:
  python runner.py --topic "quantum computing" --mode basic
  python runner.py --topic "AI in healthcare" --mode stream --depth deep
  python runner.py --topic "climate solutions" --mode debug

Modes:
  basic   — run to completion, render the final report with rich
  stream  — print live ProgressEvent updates as each step completes
  debug   — enable LlamaDebugHandler, print full LLM call trace after run
"""

from __future__ import annotations

import argparse
import asyncio

from rich import print as rprint
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from workflow import ProgressEvent, ResearchReport, ResearchWorkflow

console = Console()


# ── Mode 1: basic ─────────────────────────────────────────────────────────────

def run_basic(topic: str, depth: str) -> None:
    console.print(Panel(
        f"[bold #f59e0b]LlamaIndex Research Agent[/bold #f59e0b]\n"
        f"Topic: [cyan]{topic}[/cyan] | Depth: [yellow]{depth}[/yellow]",
        border_style="#f59e0b",
    ))

    workflow = ResearchWorkflow(debug=False, timeout=600)

    with console.status("[#f59e0b]Running research workflow...[/amber]"):
        report: ResearchReport = asyncio.run(
            workflow.run(topic=topic, depth=depth)
        )

    _render_report(report)


# ── Mode 2: stream ────────────────────────────────────────────────────────────

async def _run_stream_async(topic: str, depth: str) -> None:
    console.print(Panel(
        f"[bold #f59e0b]LlamaIndex Research Agent — Stream Mode[/bold #f59e0b]\n"
        f"Topic: [cyan]{topic}[/cyan] | Depth: [yellow]{depth}[/yellow]",
        border_style="#f59e0b",
    ))

    workflow = ResearchWorkflow(debug=False, timeout=600)
    handler = workflow.run(topic=topic, depth=depth)

    step_icons = {
        "orchestrator": "🧠",
        "web_researcher": "🌐",
        "rag_analyst": "📚",
        "synthesizer": "🔀",
        "report_writer": "✍️",
    }

    # Iterate over the async event stream — ProgressEvents surface here
    # as each @step emits ctx.write_event_to_stream(ProgressEvent(...))
    async for event in handler.stream_events():
        if isinstance(event, ProgressEvent):
            icon = step_icons.get(event.step, "•")
            console.print(f"  {icon} [bold]{event.step}[/bold]: {event.message}")
            if event.data:
                for k, v in event.data.items():
                    if isinstance(v, list) and v:
                        console.print(f"    [dim]{k}: {', '.join(str(x) for x in v[:3])}{'...' if len(v) > 3 else ''}[/dim]")

    report = await handler
    _render_report(report)


def run_stream(topic: str, depth: str) -> None:
    asyncio.run(_run_stream_async(topic, depth))


# ── Mode 3: debug ─────────────────────────────────────────────────────────────

def run_debug(topic: str, depth: str) -> None:
    from llama_index.core.callbacks import CBEventType

    console.print(Panel(
        f"[bold #f59e0b]LlamaIndex Research Agent — Debug Mode[/bold #f59e0b]\n"
        f"Topic: [cyan]{topic}[/cyan] | Depth: [yellow]{depth}[/yellow]\n"
        "[dim]LlamaDebugHandler enabled — full LLM call trace will print after run[/dim]",
        border_style="#f59e0b",
    ))

    # debug=True enables LlamaDebugHandler(print_trace_on_end=True) which
    # prints a full trace of all LLM, embedding, and retrieval events when
    # the workflow completes. This is unique to LlamaIndex.
    workflow = ResearchWorkflow(debug=True, timeout=600)

    with console.status("[#f59e0b]Running debug workflow...[/amber]"):
        report: ResearchReport = asyncio.run(
            workflow.run(topic=topic, depth=depth)
        )

    # Retrieve all LLM call pairs from the debug handler
    try:
        llm_event_pairs = workflow.debug_handler.get_event_pairs(CBEventType.LLM)

        console.print("\n")
        console.print(Panel("[bold]LlamaDebugHandler — LLM Call Trace[/bold]", border_style="dim"))

        table = Table(show_header=True, header_style="bold #f59e0b")
        table.add_column("#", style="dim", width=4)
        table.add_column("Event Type", width=16)
        table.add_column("Model", width=16)
        table.add_column("Tokens In", justify="right", width=10)
        table.add_column("Tokens Out", justify="right", width=10)
        table.add_column("Payload Preview", width=50)

        for i, pair in enumerate(llm_event_pairs, 1):
            start, end = pair[0], pair[1] if len(pair) > 1 else None
            payload = start.payload or {}
            model = payload.get("model", "unknown")
            tokens_in = str(payload.get("prompt_tokens", "—"))
            tokens_out = str(payload.get("completion_tokens", "—"))
            preview = str(payload.get("messages", [{"content": ""}])[0].get("content", ""))[:60]
            table.add_row(str(i), "LLM", model, tokens_in, tokens_out, preview)

        console.print(table)
        console.print(f"\n[dim]Total LLM calls captured: {len(llm_event_pairs)}[/dim]")
    except Exception:
        console.print("[dim]Debug handler data not available — see trace above.[/dim]")

    _render_report(report)


# ── Report renderer ───────────────────────────────────────────────────────────

def _render_report(report: ResearchReport) -> None:
    console.print("\n")
    console.print(Panel(
        f"[bold #f59e0b]{report.title}[/bold #f59e0b]",
        subtitle=f"Quality: {report.quality_score:.0%} | RAG queries: {report.rag_queries_run} | ~{report.word_count_estimate} words",
        border_style="green",
    ))

    console.print(Markdown(f"## Executive Summary\n\n{report.executive_summary}"))
    console.print(Markdown(f"### Methodology\n\n{report.methodology}"))

    for section in report.sections:
        console.print(Markdown(f"## {section.title}\n\n{section.content}"))

    console.print(Markdown(f"## Conclusion\n\n{report.conclusion}"))

    if report.citations:
        citations_md = "\n".join(f"- {c}" for c in report.citations[:10])
        console.print(Markdown(f"## Sources\n\n{citations_md}"))


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="LlamaIndex Advanced Research Multi-Agent System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python runner.py --topic "quantum computing" --mode basic
  python runner.py --topic "AI in healthcare" --mode stream --depth deep
  python runner.py --topic "climate solutions" --mode debug
        """,
    )
    parser.add_argument("--topic", required=True, help="Research topic")
    parser.add_argument(
        "--mode",
        choices=["basic", "stream", "debug"],
        default="basic",
        help="Execution mode (default: basic)",
    )
    parser.add_argument(
        "--depth",
        choices=["quick", "standard", "deep"],
        default="standard",
        help="Research depth controlling number of sub-questions (default: standard)",
    )
    args = parser.parse_args()

    if args.mode == "basic":
        run_basic(args.topic, args.depth)
    elif args.mode == "stream":
        run_stream(args.topic, args.depth)
    elif args.mode == "debug":
        run_debug(args.topic, args.depth)


if __name__ == "__main__":
    main()
