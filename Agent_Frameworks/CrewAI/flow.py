"""
ResearchFlow — CrewAI's production-ready orchestration layer.

Flows are the highest-level abstraction in CrewAI. They sit above Crews and
provide state management, event-driven execution, and conditional routing.

Key decorators demonstrated:

  @start()
    Marks the entry point method. Called when flow.kickoff(inputs={...}) is invoked.
    Return value is passed to all @listen methods that subscribe to it.

  @listen(method_or_string)
    Subscribes to the output of a method (or a routing string). Fires automatically
    when the referenced method completes or when a @router emits the matching string.

  @router(method)
    Like @listen, but returns a routing string that determines which @listen("string")
    method fires next. Enables conditional branching without explicit if/else wiring.

Flow state (ResearchFlowState):
    A Pydantic BaseModel that persists across all methods in a single run.
    Access via self.state inside any Flow method. Unlike crew task context
    (which is per-task text injection), Flow state is a typed Python object.
"""
from __future__ import annotations

from pydantic import BaseModel
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from crewai.flow.flow import Flow, listen, router, start

console = Console()


class ResearchFlowState(BaseModel):
    """Typed state threaded through the entire flow run."""
    topic: str = ""
    depth: str = "standard"
    crew_result: str = ""
    final_report: str = ""
    quality_score: float = 0.0
    approved: bool = False
    run_complete: bool = False


class ResearchFlow(Flow[ResearchFlowState]):
    """
    Production orchestrator for the multi-agent research system.

    Execution path:
      kickoff_research (@start)
        → run_research_crew (@listen)
          → quality_gate (@router)
            → publish_report (@listen "publish_report")   if approved
            → handle_revision (@listen "handle_revision") if needs work
    """

    @start()
    def kickoff_research(self) -> dict:
        """
        Flow entry point. Called by flow.kickoff(inputs={...}).
        Inputs are automatically mapped onto self.state before this fires.
        Returns a dict that becomes the input to the crew.
        """
        console.print(Panel(
            f"[bold purple]ResearchFlow starting[/]\n"
            f"[cyan]Topic:[/] {self.state.topic}\n"
            f"[cyan]Depth:[/] {self.state.depth}",
            title="[bold]Flow: @start[/]",
            border_style="purple",
        ))
        return {
            "topic": self.state.topic,
            "depth": self.state.depth,
        }

    @listen(kickoff_research)
    def run_research_crew(self, crew_inputs: dict) -> str:
        """
        Listens for kickoff_research to complete, then runs the full ResearchCrew.
        After the crew finishes, extracts the editorial review from the editing task's
        Pydantic output to populate flow state for the quality gate.
        """
        console.print(
            "\n[dim purple]Flow: @listen(kickoff_research) → launching ResearchCrew...[/]\n"
        )

        # Import here to avoid circular imports at module load time
        from crew import build_research_crew
        from tasks import editing_task

        research_crew = build_research_crew()
        result = research_crew.kickoff(inputs=crew_inputs)
        self.state.crew_result = result.raw or ""

        # Extract quality score from the editor task's Pydantic output
        try:
            if editing_task.output and editing_task.output.pydantic:
                review = editing_task.output.pydantic
                self.state.quality_score = float(review.quality_score)
                self.state.approved = bool(review.approved)
            else:
                # Fallback if pydantic output wasn't produced
                self.state.quality_score = 0.5
                self.state.approved = True
        except Exception:
            self.state.quality_score = 0.5
            self.state.approved = True

        console.print(
            f"\n[dim purple]Flow: crew finished — "
            f"score={self.state.quality_score:.2f}, "
            f"approved={self.state.approved}[/]"
        )
        return self.state.crew_result

    @router(run_research_crew)
    def quality_gate(self, report: str) -> str:
        """
        Routes execution based on editorial quality.

        Returns a string that maps to a @listen("string") method:
          "publish_report"  → report meets quality bar
          "handle_revision" → report needs more work

        This is how CrewAI Flows implement conditional branching — the router
        returns a string key, not a boolean. Any @listen that declares that
        string fires next.
        """
        console.print(
            f"\n[dim purple]Flow: @router — "
            f"quality_score={self.state.quality_score:.2f}, "
            f"approved={self.state.approved}[/]"
        )
        if self.state.approved and self.state.quality_score >= 0.7:
            return "publish_report"
        return "handle_revision"

    @listen("publish_report")
    def publish_report(self, report: str) -> None:
        """
        Fires when the @router returns "publish_report".
        Presents the final approved report.
        """
        self.state.final_report = report
        self.state.run_complete = True

        console.print(Panel(
            f"[green]Report approved[/] "
            f"(quality score: [bold]{self.state.quality_score:.2f}[/])\n\n"
            + Markdown(report[:1200]).__str__()
            + ("\n\n[dim]...truncated, see full output above[/]" if len(report) > 1200 else ""),
            title="[bold green]Flow: Final Report Published[/]",
            border_style="green",
        ))

    @listen("handle_revision")
    def handle_revision(self, report: str) -> None:
        """
        Fires when the @router returns "handle_revision".
        In a production flow, this could trigger a second crew run with
        the editor's revision_instructions as additional context.
        Here we deliver the best-available draft with a quality warning.
        """
        self.state.final_report = report
        self.state.run_complete = True

        console.print(Panel(
            f"[yellow]Quality gate: below threshold[/] "
            f"(score: [bold]{self.state.quality_score:.2f}[/] — target: 0.7)\n\n"
            f"Delivering best-available draft. In a production Flow, this method\n"
            f"would @listen to a second revision crew with the editor's feedback.\n\n"
            + str(report)[:800],
            title="[bold yellow]Flow: Revision Required[/]",
            border_style="yellow",
        ))
