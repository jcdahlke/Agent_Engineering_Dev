"""
Orchestrator agent — routes the research pipeline via handoff tools.

Microsoft Agent Framework Handoff Pattern demonstrated:
  - The Orchestrator is itself an agent with handoff tools.
  - Each handoff tool is an async closure that directly awaits a specialist
    agent's .run() call, accumulates results into research_state, then
    returns the specialist's output back to the Orchestrator's LLM.
  - This creates a hub-and-spoke "mesh" — the Orchestrator retains full
    control and explicitly decides each next step.
  - MaxIterationsMiddleware sits in the middleware pipeline and raises
    MiddlewareTermination after N cycles, cleanly stopping the run.

Pattern contrast vs LangGraph:
  - LangGraph: supervisor uses tool NAMES as routing signals in graph.py's
    routing functions. The handoff tool itself does no real work.
  - Agent Framework: the handoff tool ACTUALLY RUNS the specialist via
    await and returns its output. No separate graph router is needed.

Shared state:
  research_state is a plain mutable dict passed by reference. The handoff
  tool closures capture it and read/write it directly. This replaces
  LangGraph's TypedDict state — simpler but without automatic reducers.
  The trade-off: more manual bookkeeping, less framework magic.
"""
from __future__ import annotations

import re
from typing import Annotated

from agent_framework import AgentMiddleware, AgentContext, MiddlewareTermination, tool
from agent_framework.openai import OpenAIChatClient

from config import settings


# ── Middleware: iteration ceiling ──────────────────────────────────────────────

class MaxIterationsMiddleware(AgentMiddleware):
    """
    Terminates the Orchestrator after max_iterations tool-call cycles.

    Agent Framework middleware sits between the Orchestrator's LLM decision
    and its execution. process() is called before each agent action. Raising
    MiddlewareTermination stops the run and surfaces the last state cleanly.

    This is the framework's pluggable termination mechanism — equivalent to
    LangGraph's conditional edge that checks iteration_count >= max_iterations.
    """

    def __init__(self, max_iterations: int = 5) -> None:
        self.max_iterations = max_iterations
        self._count = 0

    async def process(self, context: AgentContext, call_next) -> None:
        if self._count >= self.max_iterations:
            raise MiddlewareTermination(
                f"Reached maximum iterations ({self.max_iterations}). Stopping."
            )
        await call_next()
        self._count += 1

    def reset(self) -> None:
        self._count = 0


# ── Orchestrator factory ───────────────────────────────────────────────────────

def build_orchestrator(
    research_state: dict,
    max_iterations: int = 5,
) -> tuple:
    """
    Build and return (orchestrator_agent, middleware_instance).

    Handoff tools are defined as async closures so they capture research_state
    by reference — every tool call reads and writes the same dict the caller
    sees after the run completes. This is the state-sharing mechanism.

    Parameters
    ----------
    research_state : dict
        Mutable shared state. Keys: topic, depth, raw_results, key_findings,
        sources, report_draft, critique, quality_score, approved,
        task_complete, iteration_count.
    max_iterations : int
        MaxIterationsMiddleware ceiling.

    Returns
    -------
    (orchestrator, middleware)
    """
    from agents.researcher import build_researcher
    from agents.analyzer import build_analyzer
    from agents.writer import build_writer
    from agents.critic import build_critic

    middleware = MaxIterationsMiddleware(max_iterations=max_iterations)

    researcher = build_researcher()
    analyzer   = build_analyzer()
    writer     = build_writer()
    critic     = build_critic()

    # ── Handoff tools (async closures over research_state) ────────────────────

    @tool
    async def call_researcher(
        subtopic: Annotated[str, "Specific aspect or subtopic to research"],
    ) -> str:
        """Dispatch to the Researcher agent to gather information from web, arXiv,
        and Wikipedia. Use this when more information is needed on the topic.
        Returns a structured summary of what was found, including source URLs."""
        prompt = (
            f"Research the following for a report on '{research_state['topic']}':\n"
            f"{subtopic}\n\n"
            f"Research depth: {research_state.get('depth', 'standard')}\n"
            f"Content blocks already gathered: {len(research_state.get('raw_results', []))}"
        )
        result = await researcher.run(prompt)
        text = result.text or ""
        research_state.setdefault("raw_results", []).append(text)
        research_state["iteration_count"] = research_state.get("iteration_count", 0) + 1
        return text or "Researcher returned empty result."

    @tool
    async def call_analyzer(
        instruction: Annotated[str, "What to analyze or what findings to extract"],
    ) -> str:
        """Dispatch to the Analyzer agent to extract key findings and structure data
        from gathered research. Use after sufficient research has been gathered.
        Returns structured KEY FINDINGS, DATA POINTS, CONFIDENCE, and GAPS."""
        content = "\n\n---\n\n".join(research_state.get("raw_results", []))[:8000]
        prompt = (
            f"Analyze the following research content about '{research_state['topic']}'.\n"
            f"Instruction: {instruction}\n\n"
            f"Content:\n{content or '(No research content yet — use your knowledge of the topic)'}"
        )
        result = await analyzer.run(prompt)
        text = result.text or ""
        research_state.setdefault("key_findings", []).append(text)
        research_state["iteration_count"] = research_state.get("iteration_count", 0) + 1
        return text or "Analyzer returned empty result."

    @tool
    async def call_writer(
        instruction: Annotated[str, "Instructions for the report — focus, tone, length, critique to address"],
    ) -> str:
        """Dispatch to the Writer agent to synthesize research findings into a
        structured Markdown research report. Use after analysis has produced key findings.
        The Writer will also call publish_report, triggering the human approval gate.
        Returns the complete report draft in Markdown."""
        findings_text = "\n\n".join(research_state.get("key_findings", []))[:6000]
        critique = research_state.get("critique", "")
        prompt = (
            f"Write a comprehensive research report on: '{research_state['topic']}'\n\n"
            f"Key findings:\n{findings_text or '(Use your knowledge of the topic)'}\n\n"
            f"Instruction: {instruction}"
            + (f"\n\nAddress this previous critique:\n{critique}" if critique else "")
        )
        result = await writer.run(prompt)
        text = result.text or ""
        research_state["report_draft"] = text
        research_state["iteration_count"] = research_state.get("iteration_count", 0) + 1
        return text or "Writer returned empty result."

    @tool
    async def call_critic(
        focus: Annotated[str, "Specific aspects to evaluate (e.g. 'accuracy and completeness')"],
    ) -> str:
        """Dispatch to the Critic agent to evaluate the current report draft for
        quality, accuracy, and completeness. Use after the Writer has produced a draft.
        Returns SCORE (0.0–1.0), DECISION (APPROVED/REVISION NEEDED), and specific critique."""
        draft = research_state.get("report_draft", "")
        if not draft:
            return "No report draft available to critique. Call call_writer first."
        prompt = (
            f"Critique this research report on '{research_state['topic']}'.\n"
            f"Focus: {focus}\n\n"
            f"Draft:\n{draft[:8000]}"
        )
        result = await critic.run(prompt)
        text = result.text or ""
        research_state["critique"] = text

        # Parse structured signals from the critic's response
        score_match = re.search(r"SCORE:\s*([0-9]+(?:\.[0-9]+)?)", text)
        if score_match:
            research_state["quality_score"] = float(score_match.group(1))
        research_state["approved"] = "APPROVED" in text.upper() and "REVISION NEEDED" not in text.upper()
        research_state["iteration_count"] = research_state.get("iteration_count", 0) + 1
        return text or "Critic returned empty result."

    @tool
    async def finish_research(
        summary: Annotated[str, "Brief summary of what was accomplished in this research run"],
    ) -> str:
        """Mark the research as complete. Call this when the Critic has APPROVED the
        report (quality_score >= 0.7) or when max iterations are nearly exhausted.
        This signals the Orchestrator to stop the run gracefully."""
        research_state["task_complete"] = True
        return f"Research complete. {summary}"

    # ── Build orchestrator agent ───────────────────────────────────────────────
    client = OpenAIChatClient(
        model=settings.orchestrator_model,
        api_key=settings.openai_api_key,
    )

    handoff_tools = [call_researcher, call_analyzer, call_writer, call_critic, finish_research]

    orchestrator = client.as_agent(
        name="Orchestrator",
        instructions=_build_system_prompt(research_state),
        tools=handoff_tools,
        middleware=[middleware],
    )

    return orchestrator, middleware


def _build_system_prompt(state: dict) -> str:
    return f"""\
You are the Research Orchestrator coordinating a multi-agent research pipeline.

Your specialists (all called via handoff tools):
  call_researcher  — gathers raw information (web, arXiv, Wikipedia)
  call_analyzer    — extracts key findings and structures data
  call_writer      — produces a structured Markdown research report
                     (also triggers human approval via publish_report)
  call_critic      — scores the draft (0.0–1.0) and approves or requests revision
  finish_research  — marks the task complete and stops the run

Current research topic: {state.get('topic', '(not set)')}
Research depth: {state.get('depth', 'standard')}
Max iterations: {state.get('max_iterations', 5)}

Routing strategy — follow this order:
  1. Call call_researcher if you have fewer than 3 content blocks gathered.
  2. Call call_analyzer after gathering sufficient research (3+ content blocks).
  3. Call call_writer once analysis has produced key findings.
  4. Call call_critic after the Writer produces a draft.
  5. If APPROVED (quality_score >= 0.7): call finish_research.
  6. If REVISION NEEDED: call call_writer with the critique instructions.
  7. If research coverage is insufficient: call call_researcher on a specific subtopic.
  8. If approaching iteration limit: call call_writer then finish_research.

You MUST use a tool on every turn. Never respond with plain text only.
"""
