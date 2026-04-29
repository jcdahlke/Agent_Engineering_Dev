from agents import Agent, ModelSettings
from pipeline.instructions import RESEARCHER
from tools import (
    arxiv_search,
    fetch_webpage,
    list_research_keys,
    save_research_notes,
    web_search,
)


def create_researcher(
    analyzer_agent: Agent,
    model: str = "gpt-4o-mini",
) -> Agent:
    return Agent(
        name="Researcher",
        model=model,
        instructions=RESEARCHER,
        model_settings=ModelSettings(temperature=0.3),
        tools=[
            web_search,
            fetch_webpage,
            arxiv_search,
            save_research_notes,
            list_research_keys,
        ],
        handoffs=[analyzer_agent],
    )
