"""
ResearchState — the shared state that flows through every node in the graph.

Key LangGraph concept: fields annotated with a reducer function use that function
to MERGE updates from nodes rather than simply overwriting. This is what makes
multi-agent state accumulation safe and predictable.

  - add_messages: deduplicates messages by ID; handles OpenAI tool-call round-trips
  - operator.add:  list concatenation — each node's output is APPENDED, never lost
  - (no annotation): last-write wins — for scalar "current state" fields
"""
from __future__ import annotations

from operator import add
from typing import Annotated, Optional

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class ResearchState(TypedDict):
    # ── Conversation messages ─────────────────────────────────────────────────
    messages: Annotated[list[BaseMessage], add_messages]

    # ── Task inputs (set once at graph entry) ─────────────────────────────────
    research_topic: str
    research_depth: str       # "quick" | "standard" | "deep"
    max_iterations: int

    # ── Researcher outputs (accumulate across calls) ──────────────────────────
    sources_found: Annotated[list[str], add]
    raw_content: Annotated[list[str], add]

    # ── Analyzer outputs ──────────────────────────────────────────────────────
    key_findings: Annotated[list[str], add]
    data_tables: Annotated[list[dict], add]
    code_outputs: Annotated[list[str], add]

    # ── Writer outputs ────────────────────────────────────────────────────────
    report_draft: str                           # overwritten each write cycle
    report_sections: Annotated[list[str], add]

    # ── Critic / quality control ──────────────────────────────────────────────
    critique: str
    revision_needed: bool
    quality_score: float                        # 0.0–1.0

    # ── Supervisor / routing control ──────────────────────────────────────────
    current_agent: str
    next_agent: str
    iteration_count: int
    task_complete: bool

    # ── Human-in-the-loop ─────────────────────────────────────────────────────
    human_feedback: Optional[str]
    awaiting_approval: bool
