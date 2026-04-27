"""
All tools available to research agents.

Each @tool-decorated function becomes a callable that agents can invoke via
LangGraph's tool-call mechanism. The tool docstring is what the LLM sees when
deciding which tool to use — keep them precise.
"""
from __future__ import annotations

import os
from typing import Any

import requests
from bs4 import BeautifulSoup
from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.tools import tool
import dotenv
dotenv.load_dotenv()
from langchain_experimental.tools import PythonREPLTool

try:
    import arxiv as _arxiv
    _ARXIV_AVAILABLE = True
except ImportError:
    _ARXIV_AVAILABLE = False

# ── Pre-built LangChain tools ─────────────────────────────────────────────────

tavily_search = TavilySearchResults(max_results=5)
ddg_search = DuckDuckGoSearchRun()
wiki_search = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=3))
python_repl = PythonREPLTool()


# ── Custom tools ──────────────────────────────────────────────────────────────

@tool
def scrape_webpage(url: str) -> str:
    """Fetch and extract the readable text content of a webpage. Returns up to 4000 characters."""
    try:
        response = requests.get(
            url,
            timeout=10,
            headers={"User-Agent": "ResearchBot/1.0 (+https://github.com/research-agent)"},
        )
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()
        text = soup.get_text(separator=" ", strip=True)
        return text[:4000]
    except Exception as exc:
        return f"Error scraping {url}: {exc}"


@tool
def arxiv_search(query: str, max_results: int = 5) -> list[dict[str, Any]]:
    """Search arXiv for academic papers. Returns title, authors, abstract snippet, and URL."""
    if not _ARXIV_AVAILABLE:
        return [{"error": "arxiv package not installed"}]
    try:
        client = _arxiv.Client()
        search = _arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=_arxiv.SortCriterion.Relevance,
        )
        return [
            {
                "title": r.title,
                "authors": [a.name for a in r.authors[:3]],
                "abstract": r.summary[:500],
                "url": r.entry_id,
                "published": str(r.published.date()) if r.published else "",
            }
            for r in client.results(search)
        ]
    except Exception as exc:
        return [{"error": str(exc)}]


# ── Tool collections returned to agents ──────────────────────────────────────

def get_search_tools(use_tavily: bool = True) -> list:
    """Return search tools, preferring Tavily when an API key is available."""
    tools: list = []
    if use_tavily and os.getenv("TAVILY_API_KEY"):
        tools.append(tavily_search)
    tools.extend([ddg_search, arxiv_search, wiki_search, scrape_webpage])
    return tools


def get_analysis_tools() -> list:
    """Return tools available to the Analyzer agent."""
    # NOTE: PythonREPLTool executes arbitrary code. In production, sandbox it.
    return [python_repl]
