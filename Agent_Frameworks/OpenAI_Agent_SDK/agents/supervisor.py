"""
Supervisor agent — the entry point and triage node.

SDK patterns:
  - No tools: the Supervisor's entire job is to hand off to the Researcher.
  - input_guardrails: topic_safety_guardrail runs before ANY LLM call.
  - ModelSettings: temperature=0 for deterministic routing decisions.
  - hooks: ResearchHooks attached here fires for ALL agents in the run,
    not just the Supervisor.
"""
from agents import Agent, ModelSettings

from agents.instructions import SUPERVISOR


def create_supervisor(
    researcher_agent: Agent,
    hooks=None,
    input_guardrails=None,
    model: str = "gpt-4o-mini",
) -> Agent:
    return Agent(
        name="Supervisor",
        model=model,
        instructions=SUPERVISOR,
        model_settings=ModelSettings(temperature=0),
        tools=[],
        handoffs=[researcher_agent],
        hooks=hooks,
        input_guardrails=input_guardrails or [],
    )
