from agents import Agent, ModelSettings
from pipeline.instructions import WRITER
from tools import (
    get_analysis,
    get_draft_report,
    save_draft_report,
)


def create_writer(
    critic_agent: Agent,
    model: str = "gpt-4o",
) -> Agent:
    return Agent(
        name="Writer",
        model=model,
        instructions=WRITER,
        model_settings=ModelSettings(temperature=0.7),
        tools=[
            get_analysis,
            get_draft_report,
            save_draft_report,
        ],
        handoffs=[critic_agent],
    )
