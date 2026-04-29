"""
Writer agent — produces the Markdown research report.

SDK patterns:
  - handoffs=[critic]: single forward handoff after draft is saved
  - ModelSettings: temperature=0.7 for fluent, engaging prose
  - Tools read from context; write to context — no external I/O
"""
from agents import Agent, ModelSettings

from agents.instructions import WRITER
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
