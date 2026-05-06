# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Teaching code for the **BYU Agent Engineering** course (TA: jcdahlke). The headline artifact is a **side-by-side framework comparison**: each subdirectory under [Agent_Frameworks/](Agent_Frameworks/) builds the **same multi-agent research pipeline** so students can read the same problem expressed in six different agent SDKs and compare primitives directly.

The pipeline everywhere is: **Supervisor/Orchestrator → Researcher → Analyzer → Writer → Critic** (with a Critic→Writer revision loop bounded by a `max_iterations`/`max_revisions` knob). The behavior is identical; the **APIs, state model, and orchestration pattern are deliberately different per framework**. When changing one framework, treat the others as separate codebases — they intentionally diverge.

| Directory | Status | Orchestration pattern shown | State carrier |
| --- | --- | --- | --- |
| [Agent_Frameworks/OpenAI_Agent_SDK/](Agent_Frameworks/OpenAI_Agent_SDK/) | implemented | Handoff tools + `agent.clone()` for circular Critic↔Writer | `ResearchContext` dataclass via `RunContextWrapper[T]` |
| [Agent_Frameworks/Pydantic_AI/](Agent_Frameworks/Pydantic_AI/) | implemented | Imperative composition (await each agent like a function) | `ResearchDependencies` via `deps=` + `message_history=` |
| [Agent_Frameworks/LangGraph/](Agent_Frameworks/LangGraph/) | implemented | `StateGraph` with conditional edges, `interrupt()` HITL, `MemorySaver` checkpointing | `ResearchState` TypedDict with `Annotated` reducers (`add_messages`, `operator.add`) |
| [Agent_Frameworks/LlamaIndex/](Agent_Frameworks/LlamaIndex/) | implemented | Event-driven `Workflow` (`@step` methods routed by typed `Event` payloads) | `Context` + RAG index built from web chunks |
| [Agent_Frameworks/CrewAI/](Agent_Frameworks/CrewAI/) | implemented | `Process.hierarchical` (auto-generated manager) + `@start/@listen/@router` Flow | `Crew` internal memory (short/long/entity) + Flow `state` |
| [Agent_Frameworks/Microsoft_Agent_Framework/](Agent_Frameworks/Microsoft_Agent_Framework/) | implemented | Handoff mesh (hub-and-spoke), `MaxIterationsMiddleware`, approval-mode HITL | Plain dict closed over by handoff tools |
| `Agent_Frameworks/AutoGPT/`, `Haystack/`, `MetaGPT/`, `Smolagents/` | **empty stubs** | — | — |

Other top-level work:

- [agents_chat_demo.py](agents_chat_demo.py) — Gradio demo of **four conversation-state strategies** in the OpenAI Agents SDK: `result.to_input_list()`, `AsyncSQLiteSession`, `OpenAIConversationsSession` (server-managed), and `previous_response_id` (Responses-API chaining).
- [OpenAI_Agents_Comparison.ipynb](OpenAI_Agents_Comparison.ipynb) — notebook comparing the OpenAI Agent SDK to the course's hand-rolled patterns.
- [ollama/](ollama/) — phishing-detection benchmark of local Ollama models via promptfoo.
- [openai_skill_tester/](openai_skill_tester/) — sandbox for testing Codex skills (`math_functions.py` + slide deck).
- `orbit/` — a vendored Python venv (gitignored, treat as opaque).
- `orbit_wars/` — currently empty.

## Per-framework file layout (NOT identical — read this before editing)

The READMEs sometimes describe an idealized layout; the actual layouts diverge. Map for what's really there:

```text
OpenAI_Agent_SDK/        Pydantic_AI/             LangGraph/
├── runner.py            ├── runner.py            ├── runner.py
├── orchestrator.py      ├── pipeline.py          ├── graph.py
├── pipeline/  ← agents! ├── agents/              ├── agents/
├── agents/  ← STUB ONLY ├── deps.py              ├── state.py
├── tools.py             ├── models.py            ├── tools.py
├── context.py           ├── tools.py             └── config.py
├── guardrails.py        └── config.py
├── hooks.py
└── config.py

LlamaIndex/              CrewAI/                  Microsoft_Agent_Framework/
├── runner.py            ├── runner.py            ├── runner.py
├── workflow.py          ├── crew.py              ├── workflow.py
├── agents/              ├── flow.py              ├── agents/
│   ├── orchestrator.py  ├── agents.py            │   └── orchestrator.py + 4 specialists
│   ├── web_researcher   ├── tasks.py             ├── tools.py
│   ├── rag_analyst      ├── tools.py             └── config.py
│   ├── synthesizer      ├── runner.py
│   └── report_writer    └── config.py
├── tools.py
└── config.py
```

**Critical: `OpenAI_Agent_SDK`'s local `agents/` directory is a deliberately empty stub** (see [agents/\_\_init\_\_.py](Agent_Frameworks/OpenAI_Agent_SDK/agents/__init__.py)). The installed `openai-agents` SDK exposes its API as `from agents import Agent, Runner, ...`. To prevent the local directory from shadowing the SDK, all local agent modules live in [pipeline/](Agent_Frameworks/OpenAI_Agent_SDK/pipeline/), and both [runner.py](Agent_Frameworks/OpenAI_Agent_SDK/runner.py) and [orchestrator.py](Agent_Frameworks/OpenAI_Agent_SDK/orchestrator.py) start with a `_fix_sys_path()` shim that demotes the script directory in `sys.path`. **Do not** add imports to the stub `__init__.py`, **do not** rename `pipeline/` to `agents/`, and **do not** remove the sys.path shim — all three break the SDK import.

The SDK's README lists `OpenAI_Agent_SDK/agents/` as the agent location; that's stale documentation. Trust [pipeline/](Agent_Frameworks/OpenAI_Agent_SDK/pipeline/) over [README.md](Agent_Frameworks/OpenAI_Agent_SDK/README.md).

## Running things

Each framework directory is **self-contained** and assumes you `cd` into it before running. They do not share a venv consistently:

- **Repo root** has a `.venv/` that is reused by most frameworks.
- **CrewAI has its own `.venv/`** at `Agent_Frameworks/CrewAI/.venv/` (separate from root). Activate that one when working in CrewAI; CrewAI's deps conflict with several siblings.
- **`orbit/`** is also a venv (gitignored) — don't activate it accidentally.

`.env` files live **per framework subdirectory**, not at the root. The root `.env` exists but is not read by the per-framework `config.py` modules — each one calls `load_dotenv(Path(__file__).parent / ".env")`. Copy keys into each subdirectory you intend to run.

```powershell
cd Agent_Frameworks\OpenAI_Agent_SDK
pip install -r requirements.txt
copy .env.example .env                   # add OPENAI_API_KEY (and optionally TAVILY_API_KEY)
python runner.py --topic "quantum computing" --mode basic
```

Common runner flags (exact set varies — check each framework's `parse_args`):

| Framework | `--mode` choices | `--depth` choices | Other flags |
| --- | --- | --- | --- |
| OpenAI_Agent_SDK | `basic` \| `stream` \| `verbose` | `minimal` \| `standard` \| `deep` | — |
| Pydantic_AI | `basic` \| `stream` | `quick` \| `standard` \| `deep` | — |
| LangGraph | `basic` \| `stream` \| `hitl` \| `resume` | `quick` \| `standard` \| `deep` | `--thread-id` (for `resume`) |
| LlamaIndex | `basic` \| `stream` \| `debug` | `quick` \| `standard` \| `deep` | — |
| CrewAI | `flow` \| `crew` | `quick` \| `standard` \| `deep` | — |
| Microsoft_Agent_Framework | `basic` \| `stream` \| `hitl` | `quick` \| `standard` \| `deep` | `--max-iter` |

## API keys and external dependencies

`OPENAI_API_KEY` is required by every framework. Optional keys differ:

| Framework | Optional keys | Falls back to |
| --- | --- | --- |
| OpenAI_Agent_SDK | — | DuckDuckGo (`ddgs`) + arxiv |
| Pydantic_AI | — | DuckDuckGo + arxiv |
| LangGraph | `TAVILY_API_KEY` | DuckDuckGo |
| LlamaIndex | — | DuckDuckGo |
| CrewAI | `SERPER_API_KEY` (Google Search) | `WebsiteSearchTool` + arxiv |
| Microsoft_Agent_Framework | `TAVILY_API_KEY` | DuckDuckGo |

The default model split across frameworks is consistent: cheap tasks (supervisor / researcher / critic) use `gpt-4o-mini`, structured/long-output tasks (analyzer / writer) use `gpt-4o`. Override per-agent via env vars (e.g. `WRITER_MODEL=gpt-4o-mini` for cheaper testing). Microsoft_Agent_Framework defaults its **orchestrator** to full `gpt-4o` (it does the routing reasoning itself).

## Things that will bite you

- **CrewAI**: `runner.py` calls `Path("logs").mkdir(exist_ok=True)` *before* `build_research_crew()` because `output_log_file` requires the directory to exist at Crew-instantiation time. Don't reorder. Also: `memory=True` writes to `.crew_memory/` and triggers OpenAI embedding calls — disable memory for fast iteration.
- **CrewAI**: `config.py` calls `load_dotenv()` *before* the `pydantic_settings` import because CrewAI reads `OPENAI_API_KEY` at import time. Order is load-bearing.
- **Microsoft Agent Framework**: does **not** auto-load `.env` — `config.py` calls `load_dotenv()` explicitly before `BaseSettings` reads env vars. Same ordering rule.
- **LangGraph `Annotated` reducers**: removing `Annotated[list[str], add]` on a state field silently changes the field to last-write-wins, which makes Researcher's appended URLs disappear when Analyzer runs. Don't drop reducers.
- **OpenAI Agents SDK circular handoff**: Critic↔Writer is resolved by `agent.clone(handoffs=[...])` after both exist — see [orchestrator.py](Agent_Frameworks/OpenAI_Agent_SDK/orchestrator.py) `build_pipeline()`. If you add another back-edge, follow the same pattern (create stub → create dependent → `clone()` to wire the back-edge).
- **Pydantic AI**: agents are awaited like plain async functions; history chaining requires `message_history=result.new_messages()` between calls. Forgetting this drops conversation context.
- **LlamaIndex `Workflow`**: routing is by **Python type annotations on `@step` methods**, not by an explicit edge table. Adding a new event type means a `@step` somewhere has to declare it as a parameter, or it goes to `/dev/null`.

## Conventions to preserve

- Every `<framework>/` ships a `presentation/index.html` — a **standalone slide deck** (no build step, no bundler). Edit the HTML directly. The deck is part of the teaching artifact, not auxiliary; keep it in sync with the code.
- Every README has a **"features demonstrated" table** mapping framework primitives to file locations. If you change the example, update that table — the table is the contract for what the demo claims to teach.
- Frameworks favor **clarity over robustness**. Resist adding retries, circuit breakers, fancy error handling, or abstraction layers. The point is for a student to read the file end-to-end.
- The five role names (Supervisor, Researcher, Analyzer, Writer, Critic) and their responsibilities are stable across frameworks. Don't rename them; the side-by-side comparison breaks.
- Don't bulk-refactor across frameworks. Cross-framework "consistency" PRs defeat the point — each framework is meant to be read on its own terms.

## ollama/ benchmark (separate workflow)

Standalone phishing-detection benchmark, unrelated to `Agent_Frameworks/`. Pipeline: edit `OLLAMA_MODELS` in [generate_config.py](ollama/generate_config.py) → `python generate_config.py` writes [promptfoo.yaml](ollama/promptfoo.yaml) → `promptfoo eval` (Node 18+ required). Models are referenced as `ollama:chat:<name>` where `<name>` must match `ollama list` exactly. Two grading gotchas worth knowing:

- Classification check requires the response to **start with** `phishing` or `legitimate` (after stripping leading punctuation). "not phishing, this is legitimate" fails — by design.
- Category match uses word-boundary regex, so `social_engineering` does **not** match `social_engineering_advanced`. Models that emit "social engineering" with a space instead of underscore fail; check `promptfoo view` raw output before assuming a model is bad.
