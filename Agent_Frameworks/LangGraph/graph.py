"""
Graph assembly — wires all agent nodes into a compiled LangGraph application.

LangGraph patterns demonstrated:
  - StateGraph with START/END constants
  - add_node with RetryPolicy for automatic LLM-error retries
  - add_edge (fixed) vs add_conditional_edges (routing functions)
  - graph.compile() with MemorySaver (per-thread checkpointing) and
    InMemoryStore (cross-thread long-term memory)
  - app.get_graph().print_ascii() / draw_mermaid_png() for visualization
"""
from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.store.memory import InMemoryStore
from langgraph.types import RetryPolicy

from agents.critic import critic_node
from agents.analyzer import analyzer_node
from agents.researcher import researcher_node
from agents.supervisor import ROUTING_MAP, supervisor_node
from agents.writer import writer_node
from state import ResearchState


# ── Routing functions ─────────────────────────────────────────────────────────

def route_from_supervisor(state: ResearchState) -> str:
    """Read the supervisor's tool call and map it to the next graph node."""
    if state.get("task_complete"):
        return END

    # Force an exit if we've exceeded the iteration ceiling
    iteration = state.get("iteration_count", 0)
    max_iter = state.get("max_iterations", 5)
    if iteration >= max_iter:
        return "writer" if not state.get("report_draft") else END

    messages = state.get("messages", [])
    if messages:
        last = messages[-1]
        tool_calls = getattr(last, "tool_calls", None)
        if tool_calls:
            tool_name = tool_calls[0]["name"]
            target = ROUTING_MAP.get(tool_name)
            if target == "__end__":
                return END
            if target:
                return target

    # Safe fallback: if no tool call found, send back to researcher
    return "researcher"


def route_from_critic(state: ResearchState) -> str:
    """Route to supervisor for revision, or END when the report is approved."""
    if state.get("revision_needed", True):
        return "supervisor"
    return END


# ── Graph construction ────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    graph = StateGraph(ResearchState)

    # RetryPolicy: automatically retries on transient errors (rate limits, timeouts)
    llm_retry = RetryPolicy(max_attempts=3, backoff_factor=2.0)
    write_retry = RetryPolicy(max_attempts=2, backoff_factor=1.5)

    graph.add_node("supervisor", supervisor_node, retry=llm_retry)
    graph.add_node("researcher", researcher_node, retry=llm_retry)
    graph.add_node("analyzer", analyzer_node, retry=llm_retry)
    graph.add_node("writer", writer_node, retry=write_retry)
    graph.add_node("critic", critic_node, retry=write_retry)

    # Fixed edges — always go here after this node
    graph.add_edge(START, "supervisor")
    graph.add_edge("researcher", "supervisor")
    graph.add_edge("analyzer", "supervisor")
    graph.add_edge("writer", "critic")

    # Conditional edges — routing function decides the next node
    graph.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {
            "researcher": "researcher",
            "analyzer": "analyzer",
            "writer": "writer",
            "critic": "critic",
            END: END,
        },
    )
    graph.add_conditional_edges(
        "critic",
        route_from_critic,
        {"supervisor": "supervisor", END: END},
    )

    return graph


# ── Singletons (import these in runner.py) ───────────────────────────────────

# MemorySaver: persists full graph state per thread_id in memory.
# Swap for SqliteSaver("research.db") to persist across Python process restarts.
checkpointer = MemorySaver()

# InMemoryStore: cross-thread key-value store for long-term agent memory.
# supervisor_node uses this to recall past research sessions.
store = InMemoryStore()

app = build_graph().compile(
    checkpointer=checkpointer,
    store=store,
    name="ResearchSupervisor",
)
