"""
Critic agent — reviews the report draft and triggers human-in-the-loop when needed.

LangGraph patterns demonstrated:
  - interrupt(): pauses graph execution mid-node and surfaces data to the caller.
    The graph state is fully checkpointed at the interrupt point. The caller
    resumes with Command(resume=<value>) and execution continues from here.
  - with_structured_output: CritiqueResult gives downstream routing code a typed,
    reliable signal rather than fragile string parsing.
"""
from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.types import StreamWriter
from langchain_openai import ChatOpenAI
from langgraph.types import interrupt
from pydantic import BaseModel, Field

from config import settings
from state import ResearchState

_SYSTEM = """\
You are a Research Critic. Evaluate this research report rigorously and honestly.

Scoring rubric:
  0.9–1.0  Publication-ready: comprehensive, well-cited, no significant gaps
  0.7–0.89 Good quality: minor issues only
  0.5–0.69 Adequate but notable gaps, weak citations, or unclear sections
  0.3–0.49 Major issues: incomplete coverage, unsubstantiated claims, poor structure
  0.0–0.29 Fundamental problems: off-topic, fabricated facts, or unusable

Set approved=True only if score >= 0.7.
Set requires_human=True if:
  - score < 0.4 (too poor to auto-approve revision loop)
  - The report makes strong medical, legal, or financial claims
  - There are factual contradictions you cannot resolve

Critique must be specific: cite sections and explain exactly what needs improving."""


class CritiqueResult(BaseModel):
    score: float = Field(ge=0.0, le=1.0, description="Quality score 0.0–1.0")
    approved: bool = Field(description="True if report is ready for delivery")
    critique: str = Field(description="Specific revision instructions (empty if approved)")
    strengths: list[str] = Field(description="What the report does well")
    weaknesses: list[str] = Field(default_factory=list, description="Specific areas needing work")
    requires_human: bool = Field(
        description="True if human review is required before proceeding"
    )


def critic_node(state: ResearchState, writer: StreamWriter = lambda _: None) -> dict:
    draft = state.get("report_draft", "")

    if not draft:
        writer({"event": "critic_skip", "reason": "no_draft"})
        return {
            "critique": "No draft to review. Please generate a report first.",
            "revision_needed": True,
            "quality_score": 0.0,
            "current_agent": "critic",
        }

    writer({"event": "critic_start", "draft_chars": len(draft)})

    llm = ChatOpenAI(
        model=settings.critic_model,
        temperature=0,
        api_key=settings.openai_api_key,
    ).with_structured_output(CritiqueResult)

    result: CritiqueResult = llm.invoke([
        SystemMessage(content=_SYSTEM),
        HumanMessage(
            content=(
                f"Review this research report on '{state['research_topic']}':\n\n"
                f"{draft[:8000]}"
            )
        ),
    ])

    writer({"event": "critic_scored", "score": result.score, "approved": result.approved})

    human_feedback: str | None = None

    if result.requires_human:
        writer({"event": "hitl_triggered", "score": result.score})

        # ── HUMAN-IN-THE-LOOP ────────────────────────────────────────────────
        # interrupt() checkpoints the full graph state here. Execution pauses.
        # The caller (runner.py) receives {"__interrupt__": [...]} and must
        # resume with: app.invoke(Command(resume=<user_input>), config)
        # ─────────────────────────────────────────────────────────────────────
        human_feedback = interrupt({
            "draft_preview": draft[:2000],
            "critique": result.critique,
            "score": result.score,
            "strengths": result.strengths,
            "weaknesses": result.weaknesses,
            "prompt": (
                f"Research report requires your review.\n"
                f"Quality score: {result.score:.2f}/1.00\n\n"
                "Reply with one of:\n"
                "  approve              — accept as-is\n"
                "  revise: <notes>      — request specific changes\n"
                "  reject               — discard and restart research\n"
            ),
        })

        # Process human decision (execution resumes here after Command(resume=...))
        if isinstance(human_feedback, str):
            decision = human_feedback.lower().strip()
            if decision.startswith("approve"):
                result.approved = True
                result.revision_needed = False
            elif decision.startswith("revise:"):
                result.critique = human_feedback[7:].strip()
                result.approved = False
            else:
                result.approved = False
                result.critique = f"Human decision: {human_feedback}"

    return {
        "messages": [
            AIMessage(
                content=(
                    f"Critique complete. Score: {result.score:.2f}. "
                    f"{'✓ Approved.' if result.approved else f'Revision needed: {result.critique[:120]}'}"
                )
            )
        ],
        "critique": result.critique,
        "revision_needed": not result.approved,
        "quality_score": result.score,
        "human_feedback": human_feedback,
        "awaiting_approval": False,
        "current_agent": "critic",
    }
