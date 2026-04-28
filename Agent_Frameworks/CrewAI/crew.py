"""
ResearchCrew assembly — the core of the CrewAI system.

Key concepts demonstrated:

  Process.hierarchical
    An auto-generated manager agent (using manager_llm) dynamically reads
    all tasks and delegates them to specialist agents. The manager can re-order,
    reassign, and retry tasks based on agent capabilities and outputs.
    Compare to Process.sequential where tasks run strictly in the declared order.

  memory=True
    Activates three memory stores:
      - Short-term  : in-session context management (automatic)
      - Long-term   : SQLite + embeddings written to MEMORY_DIR; agents recall
                      relevant past runs before each task (adds OpenAI embedding calls)
      - Entity      : extracts and stores named entities (people, orgs, concepts)

  max_rpm
    Built-in rate limiter that throttles all LLM calls for the entire crew
    collectively. Prevents hitting OpenAI's requests-per-minute limits.

  step_callback
    Fires after every agent ReAct step (thought → tool call → observation loop).
    Gives real-time visibility into agent reasoning without waiting for task completion.

  output_log_file
    Writes the full execution log to a file for post-run inspection.
    The logs/ directory must exist before the Crew is instantiated.
"""
from __future__ import annotations

from crewai import Crew, LLM, Process

from agents import data_analyst, editor_critic, senior_researcher, technical_writer
from config import settings
from tasks import (
    analysis_task,
    editing_task,
    final_summary_task,
    research_task,
    step_callback,
    writing_task,
)


def build_research_crew() -> Crew:
    """
    Build and return the ResearchCrew.

    Called fresh each run so that task output state is clean.
    The Crew is stateful after kickoff() — don't reuse across runs.

    To disable long-term memory for faster testing (no embedding API calls):
      Pass memory=False or remove the memory= argument entirely.
    """
    manager_llm = LLM(
        model=settings.manager_model,
        temperature=0,  # deterministic delegation decisions
    )

    return Crew(
        agents=[senior_researcher, data_analyst, technical_writer, editor_critic],
        tasks=[research_task, analysis_task, writing_task, editing_task, final_summary_task],

        # ── Hierarchical process ───────────────────────────────────────────────
        # The auto-manager reads all agent roles and task descriptions, then
        # dynamically assigns work. manager_llm is required; omitting it raises
        # ValueError. Pass manager_agent= instead to use a custom Agent object.
        process=Process.hierarchical,
        manager_llm=manager_llm,

        # ── Memory ────────────────────────────────────────────────────────────
        # Set to False for fast testing (skips embedding API calls)
        memory=True,

        # ── Rate limiting ──────────────────────────────────────────────────────
        max_rpm=settings.max_rpm,

        # ── Callbacks ─────────────────────────────────────────────────────────
        # step_callback fires after every ReAct step across all agents
        step_callback=step_callback,

        # ── Observability ─────────────────────────────────────────────────────
        verbose=settings.verbose,
        output_log_file="logs/crew_run.log",

        # ── Execution limits ──────────────────────────────────────────────────
        max_iter=25,
    )
