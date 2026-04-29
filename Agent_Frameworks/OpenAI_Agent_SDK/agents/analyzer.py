"""
Analyzer agent — synthesizes research notes into structured analysis.

SDK patterns:
  - output_type=AnalysisResult: forces the model to return valid JSON matching
    the Pydantic model; accessible via result.final_output_as(AnalysisResult)
  - handoffs=[writer]: single forward handoff after analysis is saved
  - ModelSettings: temperature=0.1 for precise analytical reasoning
"""
from pydantic import BaseModel

from agents import Agent, ModelSettings

from agents.instructions import ANALYZER
from tools import (
    get_analysis,
    list_research_keys,
    load_research_notes,
    save_analysis,
)


class AnalysisResult(BaseModel):
    """Structured output the Analyzer must produce before handing off."""
    key_findings: list[str]
    central_themes: list[str]
    important_statistics: list[str]
    knowledge_gaps: list[str]
    confidence_level: str       # "high" | "medium" | "low"
    recommended_report_sections: list[str]
    analysis_summary: str


def create_analyzer(
    writer_agent: Agent,
    model: str = "gpt-4o",
) -> Agent:
    return Agent(
        name="Analyzer",
        model=model,
        instructions=ANALYZER,
        model_settings=ModelSettings(temperature=0.1),
        tools=[
            list_research_keys,
            load_research_notes,
            save_analysis,
            get_analysis,
        ],
        handoffs=[writer_agent],
        output_type=AnalysisResult,
    )
