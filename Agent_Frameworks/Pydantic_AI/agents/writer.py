from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel

from config import config
from deps import ResearchDependencies
from models import WrittenReport

writer_agent: Agent[ResearchDependencies, WrittenReport] = Agent(
    model=OpenAIModel(config.writer_model),
    output_type=WrittenReport,
    deps_type=ResearchDependencies,
)


@writer_agent.system_prompt
async def writer_system_prompt(ctx: RunContext[ResearchDependencies]) -> str:
    return f"""You are a Research Writer. Produce a comprehensive, well-structured research report.

Session: {ctx.deps.session_id}
Available sources: {len(ctx.deps.sources)}

Report requirements:
  - Clear, academic-yet-accessible prose
  - A compelling title
  - A 2-3 sentence executive summary
  - 3-5 thematic sections with Markdown-formatted content
  - A synthesizing conclusion
  - Citations list (use actual source URLs from the research)
  - Estimate the word count

Structure the sections logically: background → current state → implications → future outlook.
Cite sources inline where relevant using [Source: url] notation."""
