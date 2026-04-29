"""
AgentHooks subclass — provides rich terminal feedback for every lifecycle event.

OpenAI Agents SDK patterns demonstrated:
  - AgentHooks[TContext]: generic base class for lifecycle callbacks
  - on_start: fired when an agent begins its turn
  - on_end: fired when an agent produces its final output for that turn
  - on_tool_start / on_tool_end: fired around every @function_tool call
  - on_handoff: fired when THIS agent receives a handoff from another agent

All hooks are async and receive RunContextWrapper[ResearchContext] so they can
read (but should not mutate) the shared pipeline state.
"""
from __future__ import annotations

from typing import Any

from rich.console import Console
from rich.panel import Panel

from agents import Agent, AgentHooks, RunContextWrapper

from context import ResearchContext

console = Console()

# Colour-code each agent for visual distinction in the terminal
_AGENT_COLOURS: dict[str, str] = {
    "Supervisor": "bold blue",
    "Researcher": "bold cyan",
    "Analyzer":   "bold yellow",
    "Writer":     "bold green",
    "Critic":     "bold magenta",
}


def _colour(agent_name: str) -> str:
    return _AGENT_COLOURS.get(agent_name, "bold white")


class ResearchHooks(AgentHooks[ResearchContext]):
    """
    Rich-console lifecycle logger.
    Attach this to the Supervisor agent so it fires for all agents in the run.
    """

    async def on_start(
        self,
        context: RunContextWrapper[ResearchContext],
        agent: Agent,
    ) -> None:
        context.context.visit(agent.name)
        col = _colour(agent.name)
        console.print(f"\n[{col}]▶ {agent.name}[/] starting…")

    async def on_end(
        self,
        context: RunContextWrapper[ResearchContext],
        agent: Agent,
        output: Any,
    ) -> None:
        col = _colour(agent.name)
        console.print(f"[{col}]✓ {agent.name}[/] finished.")

    async def on_tool_start(
        self,
        context: RunContextWrapper[ResearchContext],
        agent: Agent,
        tool: Any,
    ) -> None:
        tool_name = getattr(tool, "name", str(tool))
        col = _colour(agent.name)
        console.print(f"  [{col}]→ tool[/] [dim]{tool_name}[/dim]")

    async def on_tool_end(
        self,
        context: RunContextWrapper[ResearchContext],
        agent: Agent,
        tool: Any,
        result: str,
    ) -> None:
        preview = (result[:80] + "…") if len(result) > 80 else result
        console.print(f"  [dim]  ↳ {preview}[/dim]")

    async def on_handoff(
        self,
        context: RunContextWrapper[ResearchContext],
        agent: Agent,
        source: Agent,
    ) -> None:
        src_col = _colour(source.name)
        dst_col = _colour(agent.name)
        console.print(
            Panel(
                f"[{src_col}]{source.name}[/] → [{dst_col}]{agent.name}[/]",
                title="[dim]Handoff[/dim]",
                border_style="dim",
                expand=False,
            )
        )
