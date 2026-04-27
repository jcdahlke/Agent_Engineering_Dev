# Advanced Research Multi-Agent System — LangGraph

A production-quality multi-agent research pipeline built with **LangGraph**, demonstrating the full breadth of the framework's features. Designed as a teaching example for the BYU Agent Engineering course.

---

## Architecture

```
START
  │
  ▼
supervisor ──(conditional)──► researcher ──┐
            ──(conditional)──► analyzer  ──┤──► supervisor (loop)
            ──(conditional)──► writer    ──► critic ──(approved)──► END
            ──(conditional)──► critic       │
            ──(finish)──────► END           └──(revision needed)──► supervisor
```

| Agent | Model | Role |
|---|---|---|
| **Supervisor** | gpt-4o-mini | Orchestrates the pipeline; routes via handoff tool calls |
| **Researcher** | gpt-4o-mini | Manual ReAct loop; web/arXiv/Wikipedia search + scraping |
| **Analyzer** | gpt-4o | Structured output; Python REPL for data analysis |
| **Writer** | gpt-4o | Generates structured Markdown report |
| **Critic** | gpt-4o-mini | Scores draft 0–1; triggers human-in-the-loop when needed |

---

## Setup

```bash
# 1. Clone and navigate
cd Agent_Frameworks/LangGraph

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure API keys
cp .env.example .env
# Edit .env and set OPENAI_API_KEY (and optionally TAVILY_API_KEY)
```

---

## Usage

```bash
# Basic synchronous invoke — shows graph ASCII + final report
python runner.py --topic "quantum computing" --mode basic

# Streaming — see each node's updates + custom progress events in real time
python runner.py --topic "AI in healthcare" --mode stream --depth deep

# Human-in-the-loop — pause for human review, then resume
python runner.py --topic "climate change solutions" --mode hitl

# Resume a saved session (same thread-id picks up from checkpoint)
python runner.py --topic "any topic" --mode resume --thread-id research-abc123

# Depth options: quick | standard (default) | deep
```

---

## LangGraph Features Demonstrated

| Feature | Where |
|---|---|
| `StateGraph` + `START`/`END` | `graph.py` |
| `TypedDict` state + `Annotated` reducers (`add_messages`, `operator.add`) | `state.py` |
| Conditional edges + routing functions | `graph.py` |
| Supervisor pattern via handoff tools | `agents/supervisor.py` |
| Manual ReAct tool loop | `agents/researcher.py` |
| `with_structured_output` (Pydantic) | `agents/analyzer.py`, `writer.py`, `critic.py` |
| `interrupt()` + `Command(resume=...)` for HITL | `agents/critic.py`, `runner.py` |
| `MemorySaver` per-thread checkpointing | `graph.py` |
| `InMemoryStore` cross-thread long-term memory | `graph.py`, `agents/supervisor.py` |
| `StreamWriter` custom events | `agents/researcher.py`, `analyzer.py`, `writer.py` |
| `stream_mode="updates"` and `"custom"` | `runner.py` |
| `RetryPolicy` on nodes | `graph.py` |
| `app.get_graph().print_ascii()` | `runner.py` |
| `app.get_state_history()` (time-travel) | `runner.py` |

---

## File Structure

```
LangGraph/
├── .env.example        API key template
├── .gitignore          Excludes .env, __pycache__, *.db
├── requirements.txt    All Python dependencies
├── config.py           pydantic-settings config loader
├── state.py            ResearchState TypedDict with reducers
├── tools.py            All @tool definitions
├── graph.py            StateGraph assembly + compilation
├── runner.py           CLI: basic, stream, hitl, resume modes
├── agents/
│   ├── supervisor.py   Orchestrator (handoff tools + InMemoryStore)
│   ├── researcher.py   Manual ReAct search loop
│   ├── analyzer.py     Structured analysis + Python REPL
│   ├── writer.py       Structured report generation
│   └── critic.py       Quality scoring + HITL interrupt
└── presentation/
    └── index.html      Standalone HTML slide deck
```

---

## Presentation

Open `presentation/index.html` in any browser for a slide deck covering:
- Graph architecture diagram
- How the system works
- LangGraph pros, cons, and use cases

---

## Key Concepts

**Why `Annotated` reducers?**
In a multi-agent graph, multiple nodes update the same state. Without reducers, each node's write overwrites the previous one. `operator.add` on `sources_found` means Researcher's new URLs are *appended* to the list across multiple calls — never lost.

**Why handoff tools for routing?**
The supervisor's LLM picks a handoff tool (`call_researcher`, `call_writer`, etc.). The tool *name* becomes the routing signal in `route_from_supervisor()`. This gives the LLM a structured, type-safe way to express routing decisions — no fragile string parsing.

**Why `interrupt()` instead of a separate approval node?**
`interrupt()` pauses inside a node mid-execution. The full graph state is checkpointed. After the human responds, `Command(resume=value)` re-enters the exact same node line where `interrupt()` was called. This enables true mid-node human oversight without restructuring the graph.
