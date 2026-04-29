from agents import Agent, ModelSettings
from pipeline.instructions import SUPERVISOR


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
