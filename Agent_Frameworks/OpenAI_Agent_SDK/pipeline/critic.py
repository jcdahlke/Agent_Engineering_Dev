from pydantic import BaseModel, Field

from agents import Agent, ModelSettings
from pipeline.instructions import CRITIC
from tools import (
    get_draft_report,
    save_final_report,
)


class CritiqueResult(BaseModel):
    """Structured critique the Critic must produce for every evaluation."""
    quality_score: float = Field(ge=0.0, le=10.0, description="Overall quality 0–10")
    accuracy_score: float = Field(ge=0.0, le=10.0)
    completeness_score: float = Field(ge=0.0, le=10.0)
    clarity_score: float = Field(ge=0.0, le=10.0)
    strengths: list[str]
    weaknesses: list[str]
    revision_instructions: str
    approved: bool


def create_critic(
    writer_agent: Agent | None = None,
    model: str = "gpt-4o-mini",
) -> Agent:
    handoffs = [writer_agent] if writer_agent is not None else []
    return Agent(
        name="Critic",
        model=model,
        instructions=CRITIC,
        model_settings=ModelSettings(temperature=0),
        tools=[
            get_draft_report,
            save_final_report,
        ],
        handoffs=handoffs,
        output_type=CritiqueResult,
    )
