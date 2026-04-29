"""
Web Researcher step — gathers raw text from the web for each sub-question.

Demonstrates LlamaIndex's FunctionTool and FunctionAgent:
- FunctionTool.from_defaults wraps plain Python functions as LLM-callable tools
- FunctionAgent uses OpenAI's native function-calling (not text-based ReAct)
  and is the current recommended agent type in LlamaIndex 0.12+.

The agent runs once per sub-question and accumulates text chunks for indexing.
"""

from __future__ import annotations

from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.callbacks import CallbackManager
from llama_index.core.tools import FunctionTool
from llama_index.llms.openai import OpenAI

from config import settings
from tools import duckduckgo_search, scrape_webpage, wikipedia_search

RESEARCHER_SYSTEM = """\
You are a thorough web research assistant. For each question you receive:
1. Search DuckDuckGo for current information.
2. If relevant Wikipedia pages exist, fetch their summaries.
3. Scrape the most promising URL from your search results for deeper content.
4. Return a comprehensive summary of everything you found.

Be thorough — gather as much relevant text as possible.
"""


def _build_researcher_agent(callback_manager: CallbackManager) -> FunctionAgent:
    tools = [
        FunctionTool.from_defaults(
            fn=duckduckgo_search,
            name="duckduckgo_search",
            description="Search the web for current information. Returns titles, URLs, and snippets.",
        ),
        FunctionTool.from_defaults(
            fn=wikipedia_search,
            name="wikipedia_search",
            description="Retrieve a Wikipedia summary for a topic. Good for background knowledge.",
        ),
        FunctionTool.from_defaults(
            fn=scrape_webpage,
            name="scrape_webpage",
            description="Fetch the full text of a webpage given its URL. Use on promising search results.",
        ),
    ]

    llm = OpenAI(
        model=settings.researcher_model,
        temperature=0.3,
        callback_manager=callback_manager,
    )

    return FunctionAgent(
        tools=tools,
        llm=llm,
        system_prompt=RESEARCHER_SYSTEM,
    )


async def run_web_researcher(
    topic: str,
    sub_questions: list[str],
    depth: str,
    callback_manager: CallbackManager,
) -> tuple[list[str], list[str]]:
    agent = _build_researcher_agent(callback_manager)
    raw_chunks: list[str] = []
    sources: list[str] = []

    for question in sub_questions:
        query = f"Research topic: {topic}\n\nQuestion to answer: {question}"
        response = await agent.run(user_msg=query)
        text = str(response)
        if text.strip():
            raw_chunks.append(f"[Question: {question}]\n{text}")
            sources.append(f"Web research for: {question}")

    # Add a broad topic search for general context
    broad_response = await agent.run(
        user_msg=(
            f"Provide a comprehensive overview of: {topic}\n"
            "Search broadly and scrape the most informative page you find."
        )
    )
    broad_text = str(broad_response)
    if broad_text.strip():
        raw_chunks.append(f"[General overview: {topic}]\n{broad_text}")
        sources.append(f"General overview: {topic}")

    return raw_chunks, sources
