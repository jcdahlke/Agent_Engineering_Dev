from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel

import tools as _tools
from config import config
from deps import ResearchDependencies
from models import ResearchFindings

researcher_agent: Agent[ResearchDependencies, ResearchFindings] = Agent(
    model=OpenAIModel(config.researcher_model),
    output_type=ResearchFindings,
    deps_type=ResearchDependencies,
)


@researcher_agent.system_prompt
async def researcher_system_prompt(ctx: RunContext[ResearchDependencies]) -> str:
    search_counts = {"quick": "2-3", "standard": "4-6", "deep": "8+"}
    count = search_counts.get(ctx.deps.config.research_depth, "4-6")
    return f"""You are a Research Agent. Your job is to gather comprehensive, accurate
information on the given topic using your available tools.

Research depth: {ctx.deps.config.research_depth}
Target search queries: {count}
Session: {ctx.deps.session_id}

Tool usage strategy:
  1. Start with web_search for broad coverage
  2. Use arxiv_search for academic/technical depth (if relevant)
  3. Use wikipedia_search for background definitions and context
  4. Use scrape_webpage on 1-2 high-value URLs for detailed content

Populate the ResearchFindings object with:
  - sources: all URLs you found
  - raw_content: extracted text chunks (one per source, up to 500 chars each)
  - search_queries_run: total number of search calls made
  - coverage_summary: brief summary of what you found"""


@researcher_agent.tool
async def web_search(
    ctx: RunContext[ResearchDependencies], query: str, max_results: int = 5
) -> str:
    """Search the web using DuckDuckGo. Returns titles, URLs, and snippets."""
    return await _tools.web_search(ctx, query, max_results)


@researcher_agent.tool
async def arxiv_search(
    ctx: RunContext[ResearchDependencies], query: str, max_results: int = 4
) -> str:
    """Search academic papers on arXiv for peer-reviewed research."""
    return await _tools.arxiv_search(ctx, query, max_results)


@researcher_agent.tool
async def wikipedia_search(
    ctx: RunContext[ResearchDependencies], query: str
) -> str:
    """Look up a topic on Wikipedia for background context and definitions."""
    return await _tools.wikipedia_search(ctx, query)


@researcher_agent.tool
async def scrape_webpage(ctx: RunContext[ResearchDependencies], url: str) -> str:
    """Fetch a webpage and return its readable text. Uses injected HTTP client."""
    return await _tools.scrape_webpage(ctx, url)
