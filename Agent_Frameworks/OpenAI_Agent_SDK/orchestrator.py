"""
Orchestrator — wires all agents together and exposes the run_research() entry point.

OpenAI Agents SDK patterns demonstrated:
  - set_default_openai_key(): explicit API key setup from config
  - agent.clone(handoffs=[...]): resolves the Critic ↔ Writer circular reference
  - handoff(agent, tool_description_override): named handoff with custom description
  - trace("Research Pipeline"): wraps the entire run for the OpenAI tracing dashboard
  - Runner.run(agent, input, context=ctx): standard async execution
  - Runner.run_streamed(agent, input, context=ctx): streaming execution with events
  - result.final_output_as(CritiqueResult): type-safe structured output extraction
  - InputGuardrailTripwireTriggered: exception when guardrail rejects the input
"""
from __future__ import annotations

# ── Fix: prevent local pipeline/ agents/ from shadowing openai-agents SDK ────
# The local agents/ directory has the same name as the installed SDK package.
# We reorder sys.path so site-packages comes before the script directory,
# ensuring `from agents import Agent` finds the SDK, not our local folder.
import sys as _sys
import os as _os

def _fix_sys_path() -> None:
    _here = _os.path.normpath(_os.path.dirname(_os.path.abspath(__file__)))
    _cwd  = _os.path.normpath(_os.path.abspath(""))
    _local = {_here, _cwd, ""}
    _non_local = [p for p in _sys.path if _os.path.normpath(p) not in _local and p != ""]
    _local_paths = [p for p in _sys.path if _os.path.normpath(p) in _local or p == ""]
    _sys.path[:] = _non_local + _local_paths

_fix_sys_path()
del _fix_sys_path
# ─────────────────────────────────────────────────────────────────────────────

import asyncio
import traceback as _traceback
from typing import Any

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from agents import (
    Agent,
    InputGuardrailTripwireTriggered,
    Runner,
    handoff,
    set_default_openai_key,
    trace,
)

from pipeline.analyzer import AnalysisResult, create_analyzer
from pipeline.critic import CritiqueResult, create_critic
from pipeline.researcher import create_researcher
from pipeline.supervisor import create_supervisor
from pipeline.writer import create_writer
from config import settings
from context import ResearchContext
from guardrails import topic_safety_guardrail
from hooks import ResearchHooks

console = Console()


def build_pipeline(ctx: ResearchContext) -> Agent:
    """
    Construct the full agent pipeline with correct handoff wiring.

    Creation order matters for the circular Critic ↔ Writer back-edge:
      1. Create Critic with no handoffs (Writer doesn't exist yet)
      2. Create Writer pointing at Critic
      3. Clone Critic with Writer in handoffs (resolves the circular ref)
      4. Create Analyzer → Writer chain
      5. Create Researcher → Analyzer chain
      6. Create Supervisor (entry point) with hooks + guardrail
    """
    set_default_openai_key(settings.openai_api_key)

    critic_initial = create_critic(writer_agent=None, model=settings.critic_model)
    writer_initial = create_writer(critic_agent=critic_initial, model=settings.writer_model)

    # Clone critic with the back-edge to writer_initial wired in
    critic = critic_initial.clone(
        handoffs=[
            handoff(
                writer_initial,
                tool_description_override=(
                    "Send the report back to the Writer with specific revision instructions."
                ),
            )
        ]
    )

    # Clone writer to use the correct critic (the one that has the back-edge to Writer)
    writer = writer_initial.clone(
        handoffs=[
            handoff(
                critic,
                tool_description_override=(
                    "Transfer control to the Critic to evaluate the report. "
                    "Call this immediately after save_draft_report() — do NOT produce a text response first."
                ),
            )
        ]
    )

    analyzer_base = create_analyzer(writer_agent=writer, model=settings.analyst_model)
    analyzer = analyzer_base.clone(
        handoffs=[
            handoff(
                writer,
                tool_description_override=(
                    "Transfer control to the Writer to compose the research report. "
                    "Call this immediately after save_analysis() — do NOT produce a text response first."
                ),
            )
        ]
    )
    researcher = create_researcher(analyzer_agent=analyzer, model=settings.researcher_model)

    hooks      = ResearchHooks()
    supervisor = create_supervisor(
        researcher_agent=researcher,
        hooks=hooks,
        input_guardrails=[topic_safety_guardrail],
        model=settings.supervisor_model,
    )
    return supervisor


async def run_research(topic: str, depth: str = "standard", mode: str = "basic") -> ResearchContext:
    """Execute the full research pipeline and return the populated ResearchContext."""
    ctx = ResearchContext(
        topic=topic,
        depth=depth,
        max_revisions=settings.max_revisions,
    )
    supervisor = build_pipeline(ctx)
    prompt = (
        f"Research this topic in {depth} depth: {topic}\n\n"
        f"Gather comprehensive information, analyze it deeply, write a thorough "
        f"report, and have it reviewed. Deliver the final approved research report."
    )

    console.print(f"[dim]▷ Runner starting (max_turns={settings.max_turns})…[/]")

    try:
        with trace("Research Pipeline", group_id=f"research-{topic[:30]}"):
            if mode == "stream":
                await _run_streamed(supervisor, prompt, ctx)
            else:
                result = await Runner.run(
                    supervisor, prompt, context=ctx, max_turns=settings.max_turns
                )
                console.print(
                    f"[dim]▷ Runner.run() returned. "
                    f"final_output type: {type(result.final_output).__name__}[/]"
                )
                if mode == "verbose":
                    _print_verbose_result(result)

    except InputGuardrailTripwireTriggered as exc:
        validation = exc.guardrail_result.output.output_info
        console.print(
            Panel(
                f"[bold red]Topic rejected by safety guardrail[/]\n\n"
                f"Reason: {getattr(validation, 'reason', 'Not provided')}\n"
                f"Suggestion: {getattr(validation, 'suggested_alternative', 'N/A')}",
                title="[red]Guardrail Triggered[/]",
                border_style="red",
            )
        )
    except Exception as exc:
        console.print(
            Panel(
                f"[bold red]{type(exc).__name__}[/]: {exc}\n\n"
                + _traceback.format_exc(),
                title="[red]Unexpected Error[/]",
                border_style="red",
            )
        )

    _dump_agent_log(ctx)
    return ctx


def _dump_agent_log(ctx: ResearchContext) -> None:
    """Print the full tool-call log and context snapshot for debugging."""
    console.print()
    console.print("[bold yellow]══════════ DEBUG LOG ══════════[/]")
    console.print(f"  agents_visited (hooks) : {ctx.agents_visited}")
    console.print(f"  raw_research keys      : {list(ctx.raw_research.keys())}")
    console.print(f"  analysis saved         : {len(ctx.analysis)} chars")
    console.print(f"  draft_report saved     : {len(ctx.draft_report)} chars")
    console.print(f"  final_report saved     : {len(ctx.final_report)} chars")
    console.print(f"  revision_count         : {ctx.revision_count}")
    console.print(f"  quality_score          : {ctx.quality_score}")
    console.print(f"  completed              : {ctx.completed}")
    console.print()
    if ctx.agent_log:
        console.print("  [bold]Tool calls recorded:[/]")
        for entry in ctx.agent_log:
            console.print(f"    {entry}")
    else:
        console.print("  [red]agent_log is EMPTY — no @function_tool ran at all.[/]")
    console.print("[bold yellow]═══════════════════════════════[/]")
    console.print()


async def _run_streamed(supervisor: Agent, prompt: str, ctx: ResearchContext) -> None:
    """Stream mode: print tokens in real time as each agent responds."""
    console.print("[dim]Streaming output (tokens printed as they arrive)…[/]\n")
    current_agent_name = supervisor.name

    result = Runner.run_streamed(supervisor, prompt, context=ctx, max_turns=settings.max_turns)

    async for event in result.stream_events():
        event_type = getattr(event, "type", None)

        if event_type == "agent_updated_stream_event":
            new_agent = getattr(event, "new_agent", None)
            if new_agent and new_agent.name != current_agent_name:
                console.print(f"\n\n[bold cyan]── {new_agent.name} ──[/]\n")
                current_agent_name = new_agent.name

        elif event_type == "raw_response_event":
            data = getattr(event, "data", None)
            if data and hasattr(data, "choices") and data.choices:
                delta = data.choices[0].delta
                content = getattr(delta, "content", None)
                if content:
                    console.print(content, end="", markup=False)

    console.print()


def _print_verbose_result(result: Any) -> None:
    messages = getattr(result, "new_messages", []) or []
    for msg in messages:
        role    = getattr(msg, "role", "unknown")
        content = getattr(msg, "content", "")
        if isinstance(content, list):
            for block in content:
                text = getattr(block, "text", None)
                if text:
                    console.print(f"[dim]{role}:[/] {text[:300]}")
        elif content:
            console.print(f"[dim]{role}:[/] {str(content)[:300]}")


def print_final_results(ctx: ResearchContext) -> None:
    """Pretty-print the completed ResearchContext to the terminal."""
    console.print()

    table = Table(title="Research Run Summary", border_style="blue")
    table.add_column("Field",  style="cyan", no_wrap=True)
    table.add_column("Value",  style="white")
    table.add_row("Topic",          ctx.topic)
    table.add_row("Depth",          ctx.depth)
    table.add_row("Quality Score",  f"{ctx.quality_score:.1f} / 10.0")
    table.add_row("Sources Found",  str(len(ctx.source_urls)))
    table.add_row("Revisions",      str(ctx.revision_count))
    table.add_row("Agents Visited", " → ".join(ctx.agents_visited))
    table.add_row("Completed",      "[green]Yes[/]" if ctx.completed else "[yellow]No[/]")
    console.print(table)

    if ctx.critique:
        console.print(Panel(ctx.critique, title="[magenta]Critic Summary[/]", expand=False))

    report = ctx.final_report or ctx.draft_report
    title  = "Final Report" if ctx.final_report else "Draft Report (not yet approved)"
    style  = "green" if ctx.final_report else "yellow"
    if report:
        console.print()
        console.print(Panel(
            Markdown(report),
            title=f"[{style}]{title} — {ctx.topic}[/]",
            border_style=style,
            expand=True,
        ))
    else:
        console.print("[dim]No report generated.[/]")
