"""
Runner — CLI entrypoint that demonstrates every major LangGraph feature.

Modes:
  basic   — synchronous invoke, print final report
  stream  — stream node updates and custom progress events
  hitl    — human-in-the-loop: interrupt → human decision → resume
  resume  — reload a prior session from checkpoint and continue

Usage:
  python runner.py --topic "quantum computing" --mode basic
  python runner.py --topic "AI in healthcare" --mode stream --depth deep
  python runner.py --topic "climate solutions" --mode hitl
  python runner.py --topic "any topic" --mode resume --thread-id <saved-id>
"""
from __future__ import annotations

import argparse
import uuid

from langchain_core.messages import HumanMessage
from langgraph.types import Command
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from graph import app
from state import ResearchState

console = Console()


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_config(thread_id: str) -> dict:
    """LangGraph uses thread_id to namespace checkpoints. Same ID = same session."""
    return {"configurable": {"thread_id": thread_id}}


def make_initial_state(topic: str, depth: str = "standard", max_iter: int = 5) -> dict:
    return {
        "messages": [HumanMessage(content=f"Research this topic: {topic}")],
        "research_topic": topic,
        "research_depth": depth,
        "max_iterations": max_iter,
        # Accumulator fields must start as empty lists (not None)
        "sources_found": [],
        "raw_content": [],
        "key_findings": [],
        "data_tables": [],
        "code_outputs": [],
        "report_draft": "",
        "report_sections": [],
        "critique": "",
        "revision_needed": False,
        "quality_score": 0.0,
        "current_agent": "",
        "next_agent": "",
        "iteration_count": 0,
        "task_complete": False,
        "human_feedback": None,
        "awaiting_approval": False,
    }


def print_result_summary(result: dict) -> None:
    console.print()
    if result.get("report_draft"):
        preview = result["report_draft"]
        console.print(Panel(Markdown(preview), title="[green]Final Report[/]", expand=True))
    console.print(f"  Quality score : [bold]{result.get('quality_score', 0.0):.2f}[/]")
    console.print(f"  Sources found : {len(result.get('sources_found', []))}")
    console.print(f"  Key findings  : {len(result.get('key_findings', []))}")
    console.print(f"  Iterations    : {result.get('iteration_count', 0)}")


# ── Run modes ─────────────────────────────────────────────────────────────────

def run_basic(topic: str, depth: str, thread_id: str) -> None:
    """
    Demonstrates: app.invoke(), graph visualization, final state inspection.
    """
    console.print(Panel(
        f"[bold blue]Mode: Basic Invoke[/]\nTopic: {topic}",
        expand=False,
    ))

    # Graph visualization — great for understanding the topology
    console.print("\n[dim]Graph structure (ASCII):[/]")
    app.get_graph().print_ascii()
    console.print()

    config = make_config(thread_id)
    result = app.invoke(make_initial_state(topic, depth), config)
    print_result_summary(result)


def run_streaming(topic: str, depth: str, thread_id: str) -> None:
    """
    Demonstrates:
      stream_mode="updates" — see each node's state delta as it completes
      stream_mode="custom"  — structured progress events from StreamWriter calls
    """
    console.print(Panel(
        f"[bold blue]Mode: Streaming[/]\nTopic: {topic}",
        expand=False,
    ))
    config = make_config(thread_id)
    initial = make_initial_state(topic, depth)

    # ── updates stream ────────────────────────────────────────────────────────
    console.print("\n[bold yellow]── stream_mode='updates' ──[/]")
    last_result: dict = {}
    for chunk in app.stream(initial, config, stream_mode="updates"):
        for node_name, update in chunk.items():
            keys_updated = [k for k, v in update.items() if v]
            agent = update.get("current_agent", node_name)
            console.print(
                f"  [cyan]{agent:12}[/] updated: [dim]{', '.join(keys_updated)}[/]"
            )
            last_result.update(update)

    # ── custom events stream (new thread so we see the events) ───────────────
    console.print("\n[bold yellow]── stream_mode='custom' (progress events) ──[/]")
    config2 = make_config(thread_id + "-custom")
    for event in app.stream(make_initial_state(topic, depth), config2, stream_mode="custom"):
        if isinstance(event, dict):
            evt = event.pop("event", "event")
            console.print(f"  [magenta]{evt:25}[/] {event}")

    print_result_summary(last_result)


def run_hitl(topic: str, depth: str, thread_id: str) -> None:
    """
    Demonstrates:
      interrupt()            — pauses execution inside critic_node
      Command(resume=...)    — resumes with human decision
      get_state_history()    — time-travel: inspect every past checkpoint
    """
    console.print(Panel(
        f"[bold blue]Mode: Human-in-the-Loop[/]\nTopic: {topic}",
        expand=False,
    ))
    config = make_config(thread_id)
    # Low max_iterations so the critic triggers quickly in demos
    initial = make_initial_state(topic, depth, max_iter=3)

    console.print("\nRunning until interrupt or completion…\n")
    result = app.invoke(initial, config)

    # Check if execution was paused by interrupt()
    if "__interrupt__" in result:
        interrupt_data = result["__interrupt__"][0].value
        console.print(Panel(
            f"[yellow]HUMAN REVIEW REQUIRED[/]\n\n"
            f"Score  : [bold]{interrupt_data.get('score', 'N/A')}[/]\n\n"
            f"Strengths:\n" +
            "\n".join(f"  ✓ {s}" for s in interrupt_data.get("strengths", [])) +
            f"\n\nWeaknesses:\n" +
            "\n".join(f"  ✗ {w}" for w in interrupt_data.get("weaknesses", [])) +
            f"\n\nCritique:\n{interrupt_data.get('critique', '')}\n\n"
            f"{interrupt_data.get('prompt', '')}",
            title="[red]Interrupt — Awaiting Human Decision[/]",
        ))
        console.print("\n[dim]Draft preview:[/]")
        console.print(interrupt_data.get("draft_preview", "")[:600])
        console.print()

        user_input = console.input(
            "[bold]Your decision[/] (approve / revise: <notes> / reject): "
        ).strip() or "approve"

        console.print(f"\nResuming with: [italic]{user_input}[/]\n")
        # Resume — Command(resume=...) injects the value as interrupt()'s return
        result = app.invoke(Command(resume=user_input), config)

    print_result_summary(result)

    # ── State history: time-travel debugging ──────────────────────────────────
    console.print("\n[bold yellow]── State History (time-travel checkpoints) ──[/]")
    history = list(app.get_state_history(config))
    table = Table("Step", "Next Node", "Iteration", "Sources", "Score", title="Checkpoint History")
    for i, snapshot in enumerate(reversed(history[:10])):
        vals = snapshot.values
        next_nodes = list(snapshot.next) if snapshot.next else ["—"]
        table.add_row(
            str(i),
            next_nodes[0],
            str(vals.get("iteration_count", 0)),
            str(len(vals.get("sources_found", []))),
            f"{vals.get('quality_score', 0.0):.2f}",
        )
    console.print(table)


def run_resume(topic: str, depth: str, thread_id: str) -> None:
    """
    Demonstrates: reloading a saved checkpoint so a session continues exactly
    where it left off — even after restarting the Python process (with SqliteSaver).
    """
    console.print(Panel(
        f"[bold blue]Mode: Resume Session[/]\nThread: {thread_id}",
        expand=False,
    ))
    config = make_config(thread_id)
    snapshot = app.get_state(config)

    if not snapshot or not snapshot.values:
        console.print(
            "[yellow]No existing session for this thread ID. Starting fresh.[/]\n"
        )
        result = app.invoke(make_initial_state(topic, depth), config)
    else:
        vals = snapshot.values
        console.print("[green]Resuming existing session:[/]")
        console.print(f"  Sources    : {len(vals.get('sources_found', []))}")
        console.print(f"  Findings   : {len(vals.get('key_findings', []))}")
        console.print(f"  Draft      : {'Yes' if vals.get('report_draft') else 'No'}")
        console.print(f"  Next nodes : {list(snapshot.next)}\n")
        # Passing None continues from the last checkpoint
        result = app.invoke(None, config)

    if result:
        print_result_summary(result)


# ── CLI entrypoint ────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="LangGraph Advanced Research Multi-Agent System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--topic", required=True, help="Research topic")
    parser.add_argument(
        "--mode",
        choices=["basic", "stream", "hitl", "resume"],
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
        "--thread-id",
        default=None,
        dest="thread_id",
        help="Checkpoint thread ID (auto-generated if not provided)",
    )
    args = parser.parse_args()

    thread_id = args.thread_id or f"research-{uuid.uuid4().hex[:8]}"
    console.print(f"[dim]Thread ID: {thread_id}[/]\n")

    modes = {
        "basic": run_basic,
        "stream": run_streaming,
        "hitl": run_hitl,
        "resume": run_resume,
    }
    modes[args.mode](args.topic, args.depth, thread_id)


if __name__ == "__main__":
    main()
