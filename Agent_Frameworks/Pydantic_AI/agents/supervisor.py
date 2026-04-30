from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel

from config import config
from deps import ResearchDependencies

supervisor_agent: Agent[ResearchDependencies, str] = Agent(
    model=OpenAIModel(config.supervisor_model),
    output_type=str,
    deps_type=ResearchDependencies,
)


@supervisor_agent.system_prompt
async def supervisor_system_prompt(ctx: RunContext[ResearchDependencies]) -> str:
    depth_guidance = {
        "quick": "Focus on 2-3 core sub-topics. Keep the plan concise.",
        "standard": "Cover 4-5 sub-topics with moderate depth on each.",
        "deep": "Explore 6+ sub-topics, include academic and technical angles.",
    }
    return f"""You are a Research Supervisor coordinating a multi-agent research pipeline.

Your task: given a research topic, produce a clear, structured research plan that
the Researcher agent will follow. Be specific about:
  1. The 3-6 sub-topics or angles to investigate
  2. Key questions that need answering
  3. What the final report should cover

Research depth: {ctx.deps.config.research_depth}
Guidance: {depth_guidance.get(ctx.deps.config.research_depth, depth_guidance['standard'])}
Session: {ctx.deps.session_id}

Output a concise research plan as plain text. No JSON, no headers — just a clear
numbered list of research directions and expected deliverables."""
