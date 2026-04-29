"""
Researcher agent — gathers information from multiple sources.

Microsoft Agent Framework patterns demonstrated:
  - client.as_agent() with a tool list; no graph wiring required
  - The agent runs its own internal ReAct loop (think → call tool → observe)
    automatically — unlike LangGraph where the loop is implemented manually.
  - The researcher is a "leaf" agent: it runs to completion and returns a
    text summary. The Orchestrator decides what to do with that summary.
  - Streaming: the Orchestrator can call researcher.run(..., stream=True)
    to surface live tokens from inside the specialist's tool loop.
"""
from __future__ import annotations

from agent_framework.openai import OpenAIChatClient

from config import settings
from tools import get_researcher_tools

_RESEARCHER_INSTRUCTIONS = """\
You are a Research Specialist. Gather comprehensive information about the given topic.

Use your tools systematically:
  1. Start with a broad web_search to get an overview.
  2. Do 2-3 targeted web searches on specific subtopics or angles.
  3. Use arxiv_search for technical or scientific topics.
  4. Use wikipedia_search for background definitions and context.
  5. Use scrape_webpage on the most relevant URLs to get full content.

Depth guidelines (provided in the prompt):
  quick    → 2-3 searches total
  standard → 4-6 searches, scrape 2 pages
  deep     → 8+ searches, scrape key pages, include arXiv papers

When done, write a structured summary of EVERYTHING you found, including:
  - Key facts and data points found
  - Source URLs you visited
  - Any contradictions or gaps you noticed

Do NOT stop early — gather as much as the depth level requires.
"""


def build_researcher():
    """Build and return the Researcher agent."""
    client = OpenAIChatClient(
        model=settings.researcher_model,
        api_key=settings.openai_api_key,
    )
    return client.as_agent(
        name="Researcher",
        instructions=_RESEARCHER_INSTRUCTIONS,
        tools=get_researcher_tools(),
    )
