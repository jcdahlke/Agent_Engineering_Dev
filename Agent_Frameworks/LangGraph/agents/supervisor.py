"""
Supervisor agent — the orchestrator of the research pipeline.

LangGraph patterns demonstrated:
  - Handoff tools: the supervisor's LLM picks a tool call; the tool name becomes
    the routing signal read by route_from_supervisor() in graph.py.
  - InjectedStore / InMemoryStore: long-term memory persists findings across
    separate research sessions (different thread IDs).
  - parallel_tool_calls=False: forces exactly one tool call per turn so routing
    is always unambiguous.
"""
from __future__ import annotations

import os
from typing import Annotated

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.store.base import BaseStore

from config import settings
from state import ResearchState

# ── Handoff tools (no logic — signal only) ───────────────────────────────────

@tool
def call_researcher(reason: str) -> str:
    """Dispatch to the Researcher to gather information from web, arXiv, and Wikipedia."""
    return reason


@tool
def call_analyzer(reason: str) -> str:
    """Dispatch to the Analyzer to extract key findings and structured data from gathered content."""
    return reason


@tool
def call_writer(reason: str) -> str:
    """Dispatch to the Writer to synthesize findings into a structured research report."""
    return reason


@tool
def call_critic(reason: str) -> str:
    """Dispatch to the Critic to review the report draft for quality and accuracy."""
    return reason


@tool
def finish_research(summary: str) -> str:
    """Mark the research complete and deliver the final approved report."""
    return summary


HANDOFF_TOOLS = [call_researcher, call_analyzer, call_writer, call_critic, finish_research]

# Maps tool name → graph node name (used by route_from_supervisor in graph.py)
ROUTING_MAP: dict[str, str] = {
    "call_researcher": "researcher",
    "call_analyzer": "analyzer",
    "call_writer": "writer",
    "call_critic": "critic",
    "finish_research": "__end__",
}

_SYSTEM_TEMPLATE = """\
You are the Research Supervisor coordinating a multi-agent research pipeline.

Your specialists:
  researcher — gathers raw information (web, arXiv, Wikipedia)
  analyzer   — extracts key findings and runs data analysis
  writer     — produces a structured Markdown research report
  critic     — scores the draft (0–1) and approves or requests revision

Current state:
  Topic       : {topic}
  Depth       : {depth}
  Iteration   : {iteration} / {max_iterations}
  Sources     : {sources}
  Findings    : {findings}
  Has draft   : {has_draft}
  Quality score: {quality_score:.2f}
  Last critique: {critique}

Routing guide:
  1. Start with researcher if sources < 3.
  2. Move to analyzer after gathering sufficient sources.
  3. Move to writer once analysis has produced key findings.
  4. Move to critic after writer produces a draft.
  5. After critic feedback, decide whether to re-research, rewrite, or finish.
  6. Call finish_research when quality_score >= 0.7 and the critic approved.
  7. If iteration >= max_iterations, call writer or finish_research to avoid infinite loops.

You MUST call exactly one tool. Never respond with plain text only.
{past_context}\
"""


def supervisor_node(state: ResearchState, store: BaseStore) -> dict:
    llm = ChatOpenAI(
        model=settings.supervisor_model,
        temperature=0,
        api_key=settings.openai_api_key,
    ).bind_tools(HANDOFF_TOOLS, tool_choice="required", parallel_tool_calls=False)

    # Retrieve relevant past sessions from long-term store
    past_context = ""
    try:
        namespace = ("research", "history")
        # Search by prefix key listing (InMemoryStore doesn't require embeddings for list)
        results = list(store.search(namespace, query=state["research_topic"], limit=2))
        if results:
            lines = [
                f"  - {r.value.get('topic', '')}: {', '.join(r.value.get('findings', []))}"
                for r in results
            ]
            past_context = "\nRelevant past research:\n" + "\n".join(lines)
    except Exception:
        pass  # Store search is best-effort; don't break the pipeline

    system_content = _SYSTEM_TEMPLATE.format(
        topic=state["research_topic"],
        depth=state.get("research_depth", "standard"),
        iteration=state.get("iteration_count", 0),
        max_iterations=state.get("max_iterations", 5),
        sources=len(state.get("sources_found", [])),
        findings=len(state.get("key_findings", [])),
        has_draft="Yes" if state.get("report_draft") else "No",
        quality_score=state.get("quality_score", 0.0),
        critique=state.get("critique", "none yet"),
        past_context=past_context,
    )

    # The supervisor makes a pure routing decision from the structured state fields
    # summarised in the system prompt — it does NOT need the shared message history.
    # Including accumulated state["messages"] would interleave tool-call/ToolMessage
    # pairs from other agents and trigger OpenAI 400 errors on malformed sequences.
    from langchain_core.messages import HumanMessage as _HumanMessage
    messages = [
        SystemMessage(content=system_content),
        _HumanMessage(content="Based on the current research state above, choose the next action."),
    ]
    response: AIMessage = llm.invoke(messages)

    # Persist findings to long-term store for future sessions
    if state.get("key_findings"):
        try:
            store.put(
                ("research", "history"),
                state["research_topic"][:60],
                {
                    "topic": state["research_topic"],
                    "findings": state["key_findings"][:3],
                },
            )
        except Exception:
            pass

    tool_name = response.tool_calls[0]["name"] if response.tool_calls else "call_researcher"
    task_complete = tool_name == "finish_research"

    # OpenAI requires every tool_call AIMessage to be immediately followed by
    # a ToolMessage for each tool_call_id — otherwise the next LLM call errors.
    tool_responses = [
        ToolMessage(
            content=tc["args"].get("reason", tc["args"].get("summary", "dispatching")),
            tool_call_id=tc["id"],
        )
        for tc in response.tool_calls
    ]

    return {
        "messages": [response] + tool_responses,
        "next_agent": ROUTING_MAP.get(tool_name, "researcher"),
        "current_agent": "supervisor",
        "iteration_count": state.get("iteration_count", 0) + 1,
        "task_complete": task_complete,
    }
