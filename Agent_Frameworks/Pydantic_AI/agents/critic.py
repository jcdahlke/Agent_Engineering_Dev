from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models.openai import OpenAIModel

from config import config
from deps import ResearchDependencies
from models import CritiqueResult

critic_agent: Agent[ResearchDependencies, CritiqueResult] = Agent(
    model=OpenAIModel(config.critic_model),
    output_type=CritiqueResult,
    deps_type=ResearchDependencies,
)


@critic_agent.system_prompt
async def critic_system_prompt(ctx: RunContext[ResearchDependencies]) -> str:
    return f"""You are a Research Critic. Evaluate research reports rigorously and fairly.

Session: {ctx.deps.session_id}

Scoring rubric:
  0.9 - 1.0  Publication-ready. Comprehensive, accurate, well-cited, clear.
  0.7 - 0.89 Good quality. Minor gaps or style issues but solid content.
  0.5 - 0.69 Adequate. Notable gaps in coverage or depth.
  0.3 - 0.49 Major issues. Missing key information or significant inaccuracies.
  0.0 - 0.29 Fundamental problems. Needs complete rewrite.

Set approved=True ONLY if score >= 0.7.
Always provide at least 2 strengths and 2 weaknesses.
If not approved, revision_instructions must be specific and actionable."""


@critic_agent.output_validator
async def validate_critique(
    ctx: RunContext[ResearchDependencies], result: CritiqueResult
) -> CritiqueResult:
    """Enforce that rejection always comes with actionable instructions.

    Raises ModelRetry if the critic rejects the report but gives no guidance —
    ensuring the revision loop has concrete direction.
    """
    if not result.approved and not result.revision_instructions.strip():
        raise ModelRetry(
            "You marked the report as not approved (approved=False) but provided no "
            "revision_instructions. Please specify exactly what needs to be improved "
            "so the Writer agent can act on your feedback."
        )
    if result.approved and result.score < 0.7:
        raise ModelRetry(
            f"Inconsistent result: approved=True but score={result.score:.2f} < 0.7. "
            "Set approved=False if the score is below 0.7, or raise the score if the "
            "report truly meets the quality bar."
        )
    return result
