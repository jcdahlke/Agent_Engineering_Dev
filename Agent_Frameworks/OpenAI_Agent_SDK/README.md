# OpenAI Agent SDK — Advanced Research Multi-Agent System

A production-quality research pipeline built on the **OpenAI Agents SDK** that showcases every major SDK primitive: handoffs, function tools, input guardrails, structured outputs, streaming, tracing, and lifecycle hooks.

---

## Architecture

```
Input Topic
    │
    ▼  [InputGuardrail — TopicValidator]
    │
    ▼
┌──────────┐   handoff   ┌────────────┐   handoff   ┌──────────┐
│Supervisor│ ──────────▶ │ Researcher │ ──────────▶ │ Analyzer │
└──────────┘             └────────────┘             └──────────┘
                                                          │ handoff
                                                          ▼
                         ┌─────────────────────────────────┐
               revision  │             Writer              │
             ◀ ─ ─ ─ ─  │   get_analysis · save_draft    │
             (score < 7) └─────────────────────────────────┘
                                        │ handoff
                                        ▼
                         ┌─────────────────────────────────┐
                         │             Critic              │
                         │ output_type=CritiqueResult      │
                         └─────────────────────────────────┘
                                        │
                                 score ≥ 7 → DONE
```

---

## OpenAI Agents SDK Features Demonstrated

| Feature | Where |
|---------|-------|
| `Agent(name, instructions, tools, handoffs, model, model_settings, hooks, input_guardrails)` | All 5 agents |
| `@function_tool` — auto JSON schema from type hints | `tools.py` |
| `RunContextWrapper[ResearchContext]` — typed shared state | `tools.py` (all tools) |
| `handoff(agent, tool_description_override)` | `orchestrator.py` |
| `agent.clone(handoffs=[...])` — resolve circular Critic↔Writer ref | `orchestrator.py` |
| `@input_guardrail` + `GuardrailFunctionOutput` | `guardrails.py` |
| `output_type=PydanticModel` + `final_output_as()` | `agents/analyzer.py`, `agents/critic.py` |
| `AgentHooks` — on_start, on_end, on_tool_start, on_tool_end, on_handoff | `hooks.py` |
| `ModelSettings(temperature, max_tokens)` | Per-agent in `orchestrator.py` |
| `Runner.run(agent, input, context=ctx)` | basic + verbose modes |
| `Runner.run_streamed(...)` + `stream_events()` | stream mode |
| `trace("Research Pipeline")` | `orchestrator.py` |
| `set_default_openai_key()` | `orchestrator.py` |
| `InputGuardrailTripwireTriggered` exception handling | `orchestrator.py` |

---

## File Structure

```
OpenAI_Agent_SDK/
├── .env.example          ← API key template
├── .gitignore
├── requirements.txt
├── config.py             ← Pydantic BaseSettings
├── context.py            ← ResearchContext dataclass
├── tools.py              ← 10 @function_tool definitions
├── guardrails.py         ← Input guardrail + TopicValidation model
├── hooks.py              ← ResearchHooks(AgentHooks) rich logging
├── agents/
│   ├── __init__.py
│   ├── instructions.py   ← System prompt strings
│   ├── supervisor.py     ← Triage / entry-point agent
│   ├── researcher.py     ← Web + arXiv research agent
│   ├── analyzer.py       ← Synthesis agent (structured output)
│   ├── writer.py         ← Report writing agent
│   └── critic.py         ← Quality review agent (structured output)
├── orchestrator.py       ← Agent wiring + run_research()
├── runner.py             ← CLI entry point
├── presentation/
│   └── index.html        ← 9-slide standalone presentation
└── README.md
```

---

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

# 3. Run
python runner.py --topic "quantum computing" --mode basic
```

---

## Usage

```bash
# Basic run — print final report when complete
python runner.py --topic "AI in healthcare" --mode basic

# Stream mode — see tokens printed in real time as agents respond
python runner.py --topic "quantum computing" --mode stream --depth deep

# Verbose mode — full lifecycle event logging (hooks) to console
python runner.py --topic "climate change solutions" --mode verbose

# Research depths
# minimal  → 2 web searches, quick synthesis
# standard → 3-5 searches, thorough analysis (default)
# deep     → 5+ searches including ArXiv papers
```

---

## Presentation

Open `presentation/index.html` in any browser. No build step required.

Navigate with **arrow keys** or the **prev/next buttons**. The 9 slides cover:
1. Title & feature overview
2. What is OpenAI Agent SDK
3. System architecture (SVG graph)
4. Agent roles table
5. SDK feature code snippets
6. How a run works (8 steps)
7. Pros
8. Cons
9. Use cases & getting started
