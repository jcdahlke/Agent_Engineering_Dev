"""
Runner — CLI entrypoint for the OpenAI Agent SDK Research System.

Usage:
  python runner.py --topic "quantum computing" --mode basic
  python runner.py --topic "AI in healthcare" --mode stream --depth deep
  python runner.py --topic "climate change solutions" --mode verbose

Modes:
  basic   — synchronous run; prints final report + summary table when complete
  stream  — real-time token streaming; prints each agent's response as it arrives
  verbose — same as basic but ResearchHooks logs every lifecycle event to console
"""
from __future__ import annotations

# ── Fix: prevent local agents/ from shadowing openai-agents SDK ──────────────
# Must be the very first code executed so all subsequent imports find the SDK.
import sys as _sys
import os as _os

def _fix_sys_path() -> None:
    _here  = _os.path.normpath(_os.path.dirname(_os.path.abspath(__file__)))
    _cwd   = _os.path.normpath(_os.path.abspath(""))
    _local = {_here, _cwd, ""}
    _non_local   = [p for p in _sys.path if _os.path.normpath(p) not in _local and p != ""]
    _local_paths = [p for p in _sys.path if _os.path.normpath(p) in _local or p == ""]
    _sys.path[:] = _non_local + _local_paths

_fix_sys_path()
del _fix_sys_path
# ─────────────────────────────────────────────────────────────────────────────

import argparse
import asyncio
import sys

from rich.console import Console
from rich.panel import Panel

from orchestrator import print_final_results, run_research

console = Console()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OpenAI Agent SDK — Advanced Research Multi-Agent System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--topic",
        required=True,
        help='Research topic (e.g. "quantum computing" or "AI in healthcare")',
    )
    parser.add_argument(
        "--mode",
        choices=["basic", "stream", "verbose"],
        default="basic",
        help="Execution mode (default: basic)",
    )
    parser.add_argument(
        "--depth",
        choices=["minimal", "standard", "deep"],
        default="standard",
        help="Research depth (default: standard)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    console.print(
        Panel(
            f"[bold]OpenAI Agent SDK Research System[/]\n\n"
            f"[cyan]Topic :[/] {args.topic}\n"
            f"[cyan]Mode  :[/] {args.mode}\n"
            f"[cyan]Depth :[/] {args.depth}\n\n"
            f"[dim]Pipeline: Supervisor → Researcher → Analyzer → Writer → Critic[/]",
            title="[bold blue]Starting Research Pipeline[/]",
            border_style="blue",
        )
    )

    ctx = asyncio.run(run_research(args.topic, args.depth, args.mode))
    print_final_results(ctx)

    sys.exit(0 if ctx.completed else 1)


if __name__ == "__main__":
    main()
