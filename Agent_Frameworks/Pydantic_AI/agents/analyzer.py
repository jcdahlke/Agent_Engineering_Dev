from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models.openai import OpenAIModel

from config import config
from deps import ResearchDependencies
from models import AnalysisResult

analyzer_agent: Agent[ResearchDependencies, AnalysisResult] = Agent(
    model=OpenAIModel(config.analyst_model),
    output_type=AnalysisResult,
    deps_type=ResearchDependencies,
)


@analyzer_agent.system_prompt
async def analyzer_system_prompt(ctx: RunContext[ResearchDependencies]) -> str:
    return f"""You are a Research Analyst. Extract structured insights from gathered research content.

Session: {ctx.deps.session_id}
Sources available: {len(ctx.deps.sources)}

Your job: produce a structured AnalysisResult with:
  - key_findings: 5-12 specific, factual statements (not vague generalizations)
  - themes: 2-5 overarching themes across all findings
  - confidence: your confidence in the analysis quality (0.0-1.0)
  - needs_more_research: true if critical gaps remain
  - analysis_summary: a concise 2-4 sentence narrative

Be precise. Extract concrete facts, statistics, and named concepts — not summaries."""


@analyzer_agent.output_validator
async def validate_analysis(
    ctx: RunContext[ResearchDependencies], result: AnalysisResult
) -> AnalysisResult:
    """Raise ModelRetry if the analysis does not meet the quality bar.

    ModelRetry sends the error message back to the model, which then
    regenerates its output — Pydantic AI's built-in retry on validation failure.
    """
    if len(result.key_findings) < 3:
        raise ModelRetry(
            f"Only {len(result.key_findings)} key findings returned. "
            "Please extract at least 3 specific, factual findings from the research content."
        )
    if result.confidence < 0.1:
        raise ModelRetry(
            "Confidence score is too low (< 0.1). If the content is sparse, "
            "still provide your best analysis and set confidence accordingly (0.1-0.5 for sparse)."
        )
    return result
