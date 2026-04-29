"""
Workflow assembly — wires all agents into the orchestrated research pipeline.

Microsoft Agent Framework orchestration pattern: HANDOFF MESH (Hub-and-Spoke)

  The Orchestrator sits at the center. It has handoff tools that directly call
  each specialist, receive results, and then decide the next step. Control
  always returns to the Orchestrator — a centralized hub-and-spoke pattern.

  This is one of 5 official Agent Framework orchestration patterns:

  Pattern         Description                        Best For
  ──────────────  ─────────────────────────────────  ─────────────────────────
  Sequential      Fixed pipeline: A → B → C → D      Linear workflows
  Concurrent      asyncio.gather of parallel agents   Independent subtasks
  Handoff (this)  Agents hand control via tool calls  Dynamic routing
  Group Chat      Shared conversation with moderator  Multi-agent debate / collab
  Magentic One    Planner + specialist scratchpad     Complex long-horizon tasks

Shared state:
  research_state (plain dict) plays the role of LangGraph's TypedDict state.
  It is passed by reference to build_orchestrator(), which creates closure-based
  handoff tools that read and write it. After orchestrator.run() returns, the
  caller inspects research_state for the final report, quality score, etc.

  Trade-off vs LangGraph: simpler setup, but no automatic reducers, no
  checkpointing, and no time-travel debugging. State is lost if the process exits.
"""
from __future__ import annotations

from agents.orchestrator import build_orchestrator


def build_pipeline(
    topic: str,
    depth: str = "standard",
    max_iterations: int = 5,
) -> tuple:
    """
    Build the research pipeline for a given topic.

    Returns
    -------
    orchestrator : Agent
        The configured Orchestrator agent. Await orchestrator.run(prompt).
    research_state : dict
        Mutable shared state. Inspect after run() for results.
    """
    research_state = {
        "topic": topic,
        "depth": depth,
        "max_iterations": max_iterations,
        "raw_results": [],       # list[str] — content blocks from Researcher
        "key_findings": [],      # list[str] — structured analysis blocks
        "sources": [],           # list[str] — URLs accumulated
        "report_draft": "",      # str — latest Writer output
        "critique": "",          # str — latest Critic output
        "quality_score": 0.0,    # float — parsed from Critic's SCORE: line
        "approved": False,       # bool — parsed from Critic's DECISION: line
        "task_complete": False,  # bool — set by finish_research handoff tool
        "iteration_count": 0,    # int — incremented by each handoff tool call
    }

    orchestrator, _middleware = build_orchestrator(
        research_state=research_state,
        max_iterations=max_iterations,
    )

    return orchestrator, research_state
