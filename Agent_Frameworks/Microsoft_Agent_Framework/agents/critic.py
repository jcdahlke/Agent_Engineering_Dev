"""
Critic agent — scores the research report and decides on approval.

Microsoft Agent Framework patterns demonstrated:
  - Pure text-output agent (no tools needed — pure LLM reasoning)
  - Structured output via prompt engineering: SCORE: X.XX and DECISION: lines
    are parsed by the Orchestrator's call_critic handoff tool using regex.
  - No tool loop: the agent makes one LLM call and returns its structured
    critique directly — Agent Framework handles the single-turn case cleanly.

Note on HITL in this system:
  The primary HITL gate is the publish_report tool in writer.py.
  The Critic provides the quality signal (0.0–1.0 score + APPROVED/REVISION).
  The Orchestrator decides whether to call publish_report (triggering HITL)
  or to loop back for revision. Two separate checks — quality assessment and
  human approval — are both handled by the framework automatically.
"""
from __future__ import annotations

from agent_framework.openai import OpenAIChatClient

from config import settings

_CRITIC_INSTRUCTIONS = """\
You are a Research Critic. Evaluate research reports rigorously and honestly.

Scoring rubric:
  0.9–1.0  Publication-ready: comprehensive, well-cited, no significant gaps
  0.7–0.89 Good quality: only minor issues
  0.5–0.69 Adequate but notable gaps or unclear sections
  0.3–0.49 Major issues: incomplete coverage or poor structure
  0.0–0.29 Fundamental problems: off-topic or unusable

Always output your critique in this EXACT format so it can be parsed:

SCORE: [X.XX]
DECISION: [APPROVED or REVISION NEEDED]

STRENGTHS:
- [strength 1]
- [strength 2]

WEAKNESSES:
- [weakness 1 with specific section reference]
- [weakness 2]

CRITIQUE:
[Specific revision instructions — cite sections and exact issues. Leave empty if APPROVED.]

Rules:
  - Set DECISION to APPROVED only if SCORE >= 0.70.
  - Be specific in WEAKNESSES — "needs more detail" is not actionable.
  - CRITIQUE should give the Writer concrete steps to improve.
"""


def build_critic():
    """Build and return the Critic agent."""
    client = OpenAIChatClient(
        model=settings.critic_model,
        api_key=settings.openai_api_key,
    )
    return client.as_agent(
        name="Critic",
        instructions=_CRITIC_INSTRUCTIONS,
        tools=[],
    )
