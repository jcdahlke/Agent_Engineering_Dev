"""
Research tools for the Microsoft Agent Framework demo.

agent-framework uses its own @tool decorator. Tool parameters use
Annotated[type, "description string"] — the description becomes what
the LLM sees when deciding how to call the tool.

@tool(approval_mode="always_require") forces a human confirmation
before the tool executes — the framework's built-in HITL mechanism,
demonstrated on publish_report.
"""
from __future__ import annotations

import os
from typing import Annotated

import requests
from bs4 import BeautifulSoup
from agent_framework import tool

try:
    import arxiv as _arxiv
    _ARXIV_AVAILABLE = True
except ImportError:
    _ARXIV_AVAILABLE = False

try:
    from ddgs import DDGS
    _DDG_AVAILABLE = True
except ImportError:
    _DDG_AVAILABLE = False

try:
    import wikipedia as _wiki
    _WIKI_AVAILABLE = True
except ImportError:
    _WIKI_AVAILABLE = False


# ── Research tools ────────────────────────────────────────────────────────────

@tool
def web_search(
    query: Annotated[str, "The search query to look up on the web"],
) -> str:
    """Search the web for information about a topic. Returns top 5 results with
    titles, URLs, and snippets. Use for current events and general information."""
    tavily_key = os.getenv("TAVILY_API_KEY", "")
    if tavily_key:
        try:
            resp = requests.post(
                "https://api.tavily.com/search",
                json={"api_key": tavily_key, "query": query, "max_results": 5},
                timeout=15,
            )
            data = resp.json()
            results = data.get("results", [])
            lines = [
                f"[{i+1}] {r.get('title', '')}\n    {r.get('url', '')}\n    {r.get('content', '')[:200]}"
                for i, r in enumerate(results)
            ]
            return "\n\n".join(lines) or "No Tavily results found."
        except Exception:
            pass  # fall through to DuckDuckGo

    if _DDG_AVAILABLE:
        try:
            with DDGS() as ddg:
                results = list(ddg.text(query, max_results=5))
            lines = [
                f"[{i+1}] {r.get('title', '')}\n    {r.get('href', '')}\n    {r.get('body', '')[:200]}"
                for i, r in enumerate(results)
            ]
            return "\n\n".join(lines) or "No DuckDuckGo results found."
        except Exception as exc:
            return f"Web search error: {exc}"

    return "No search backend available. Install duckduckgo-search or set TAVILY_API_KEY."


@tool
def arxiv_search(
    query: Annotated[str, "Academic search query for arXiv papers"],
    max_results: Annotated[int, "Maximum number of papers to return (1-10)"] = 5,
) -> str:
    """Search arXiv for academic papers. Returns title, authors, abstract snippet,
    and URL. Use for technical and scientific topics."""
    if not _ARXIV_AVAILABLE:
        return "arxiv package not installed. Run: pip install arxiv"
    try:
        client = _arxiv.Client()
        search = _arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=_arxiv.SortCriterion.Relevance,
        )
        papers = []
        for r in client.results(search):
            authors = ", ".join(a.name for a in r.authors[:3])
            papers.append(
                f"Title: {r.title}\n"
                f"Authors: {authors}\n"
                f"Published: {str(r.published.date()) if r.published else 'unknown'}\n"
                f"Abstract: {r.summary[:400]}...\n"
                f"URL: {r.entry_id}"
            )
        return "\n\n---\n\n".join(papers) if papers else f"No arXiv papers found for: {query}"
    except Exception as exc:
        return f"arXiv search error: {exc}"


@tool
def wikipedia_search(
    topic: Annotated[str, "The topic or article title to look up on Wikipedia"],
) -> str:
    """Look up a topic on Wikipedia. Returns the article summary (first ~6 sentences).
    Good for background definitions, historical context, and overview information."""
    if not _WIKI_AVAILABLE:
        return "wikipedia package not installed. Run: pip install wikipedia"
    try:
        _wiki.set_lang("en")
        summary = _wiki.summary(topic, sentences=6, auto_suggest=True)
        return summary
    except Exception as exc:
        return f"Wikipedia lookup error for '{topic}': {exc}"


@tool
def scrape_webpage(
    url: Annotated[str, "Full URL of the web page to scrape (must start with http:// or https://)"],
) -> str:
    """Fetch and extract the readable text content of a web page.
    Removes navigation, scripts, and boilerplate. Returns up to 4000 characters.
    Use this to read the full content of a specific page found via web_search."""
    try:
        response = requests.get(
            url,
            timeout=10,
            headers={"User-Agent": "ResearchBot/1.0"},
        )
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()
        text = soup.get_text(separator=" ", strip=True)
        return text[:4000] if text else "No readable text found on this page."
    except Exception as exc:
        return f"Error scraping {url}: {exc}"


# ── Analysis tool ─────────────────────────────────────────────────────────────

@tool
def extract_key_facts(
    content: Annotated[str, "The text content to extract key facts from"],
    topic: Annotated[str, "The research topic to focus the extraction on"],
) -> str:
    """Extract and structure key facts, statistics, and findings from a block of text.
    Returns a bulleted list of the most important factual claims relevant to the topic."""
    lines = [line.strip() for line in content.split("\n") if line.strip()]
    candidates = [l for l in lines if any(c.isdigit() for c in l) or len(l) > 40]
    preview = "\n".join(f"- {l}" for l in candidates[:20])
    return (
        f"Key fact candidates extracted from content about '{topic}':\n\n"
        f"{preview or '(No structured facts found — use raw content for analysis)'}"
    )


# ── HITL tool (requires human approval before execution) ─────────────────────

@tool(approval_mode="always_require")
def publish_report(
    report: Annotated[str, "The complete final research report in Markdown format"],
    topic: Annotated[str, "The research topic this report covers"],
) -> str:
    """Publish the final research report. This action REQUIRES human approval
    before proceeding — the framework pauses and prompts for confirmation.
    Only call this when the report is ready for delivery."""
    char_count = len(report)
    return (
        f"Report on '{topic}' published successfully.\n"
        f"Length: {char_count} characters (~{char_count // 5} words)."
    )


# ── Tool collections ──────────────────────────────────────────────────────────

def get_researcher_tools() -> list:
    """Tools available to the Researcher agent."""
    return [web_search, arxiv_search, wikipedia_search, scrape_webpage]


def get_analyzer_tools() -> list:
    """Tools available to the Analyzer agent."""
    return [extract_key_facts]


def get_writer_tools() -> list:
    """Tools available to the Writer agent (publish_report is HITL-gated)."""
    return [publish_report]
