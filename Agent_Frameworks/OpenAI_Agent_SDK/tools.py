"""
All @function_tool definitions shared across research agents.

OpenAI Agents SDK patterns demonstrated:
  - @function_tool: auto-generates JSON schema from type annotations + docstring
  - RunContextWrapper[ResearchContext]: first param is injected by the runner;
    tools read/write shared ResearchContext without any global state
  - Pydantic-style type hints: the SDK validates inputs before calling the function
"""
from __future__ import annotations

import requests
from bs4 import BeautifulSoup

import arxiv
from agents import RunContextWrapper, function_tool
from ddgs import DDGS

from context import ResearchContext


# ── Web Search ────────────────────────────────────────────────────────────────

@function_tool
def web_search(
    ctx: RunContextWrapper[ResearchContext],
    query: str,
    max_results: int = 5,
) -> str:
    """Search the web using DuckDuckGo and return formatted results with titles, URLs, and snippets."""
    try:
        with DDGS() as ddgs:
            raw = list(ddgs.text(query, max_results=max_results))
        if not raw:
            return "No results found for this query."
        lines: list[str] = []
        for i, r in enumerate(raw, 1):
            url = r.get("href", "")
            title = r.get("title", "No title")
            snippet = r.get("body", "")[:350]
            lines.append(f"[{i}] {title}\n    URL: {url}\n    {snippet}...")
            if url and url not in ctx.context.source_urls:
                ctx.context.source_urls.append(url)
        ctx.context.log(f"web_search('{query}') → {len(raw)} results")
        return "\n\n".join(lines)
    except Exception as exc:
        return f"Search error: {exc}"


# ── Webpage Fetcher ───────────────────────────────────────────────────────────

@function_tool
def fetch_webpage(
    ctx: RunContextWrapper[ResearchContext],
    url: str,
    max_chars: int = 4000,
) -> str:
    """Fetch a webpage and return its readable text content, stripped of HTML markup."""
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        }
        resp = requests.get(url, headers=headers, timeout=12)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
        text = "\n".join(line for line in text.splitlines() if line.strip())
        ctx.context.log(f"fetch_webpage({url}) → {len(text)} chars")
        return text[:max_chars] if text else "Page had no readable text."
    except Exception as exc:
        return f"Failed to fetch {url}: {exc}"


# ── ArXiv Academic Search ─────────────────────────────────────────────────────

@function_tool
def arxiv_search(
    ctx: RunContextWrapper[ResearchContext],
    query: str,
    max_results: int = 4,
) -> str:
    """Search academic papers on ArXiv for peer-reviewed research relevant to the topic."""
    try:
        client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance,
        )
        papers = list(client.results(search))
        if not papers:
            return "No academic papers found."
        entries: list[str] = []
        for p in papers:
            authors = ", ".join(str(a) for a in p.authors[:3])
            if len(p.authors) > 3:
                authors += " et al."
            entries.append(
                f"**{p.title}**\n"
                f"Authors: {authors}\n"
                f"Published: {p.published.strftime('%Y-%m-%d')}\n"
                f"Abstract: {p.summary[:400]}...\n"
                f"PDF: {p.pdf_url}"
            )
            if p.pdf_url not in ctx.context.source_urls:
                ctx.context.source_urls.append(p.pdf_url)
        ctx.context.log(f"arxiv_search('{query}') → {len(papers)} papers")
        return "\n\n---\n\n".join(entries)
    except Exception as exc:
        return f"ArXiv error: {exc}"


# ── Research Note Store ───────────────────────────────────────────────────────

@function_tool
def save_research_notes(
    ctx: RunContextWrapper[ResearchContext],
    key: str,
    content: str,
) -> str:
    """Save a chunk of research notes under a named key so the Analyzer can retrieve it later."""
    ctx.context.raw_research[key] = content
    ctx.context.log(f"save_research_notes('{key}') → {len(content)} chars")
    return f"Saved notes under key '{key}'."


@function_tool
def load_research_notes(
    ctx: RunContextWrapper[ResearchContext],
    key: str,
) -> str:
    """Retrieve previously saved research notes by their key name."""
    content = ctx.context.raw_research.get(key)
    if content is None:
        available = list(ctx.context.raw_research.keys())
        return f"No notes for '{key}'. Available keys: {available}"
    return content


@function_tool
def list_research_keys(ctx: RunContextWrapper[ResearchContext]) -> str:
    """List all keys currently in the research note store."""
    keys = list(ctx.context.raw_research.keys())
    return f"Stored keys: {keys}" if keys else "No research notes saved yet."


# ── Analysis Persistence ──────────────────────────────────────────────────────

@function_tool
def save_analysis(
    ctx: RunContextWrapper[ResearchContext],
    analysis_text: str,
) -> str:
    """Persist the Analyzer's synthesized findings so the Writer can build the report from them."""
    ctx.context.analysis = analysis_text
    ctx.context.log(f"save_analysis() → {len(analysis_text)} chars")
    return "Analysis saved to context."


@function_tool
def get_analysis(ctx: RunContextWrapper[ResearchContext]) -> str:
    """Retrieve the analysis text saved by the Analyzer agent."""
    return ctx.context.analysis or "No analysis has been saved yet."


# ── Report Drafting ───────────────────────────────────────────────────────────

@function_tool
def save_draft_report(
    ctx: RunContextWrapper[ResearchContext],
    report: str,
) -> str:
    """Save the current report draft and increment the revision counter."""
    ctx.context.draft_report = report
    ctx.context.revision_count += 1
    ctx.context.log(f"save_draft_report() → revision #{ctx.context.revision_count}")
    return f"Draft saved (revision #{ctx.context.revision_count})."


@function_tool
def get_draft_report(ctx: RunContextWrapper[ResearchContext]) -> str:
    """Retrieve the latest draft report written by the Writer agent."""
    return ctx.context.draft_report or "No draft has been saved yet."


@function_tool
def save_final_report(
    ctx: RunContextWrapper[ResearchContext],
    report: str,
    quality_score: float,
    critique_summary: str,
) -> str:
    """Mark the report as finalized. quality_score must be 0.0–10.0."""
    ctx.context.final_report = report
    ctx.context.quality_score = quality_score
    ctx.context.critique = critique_summary
    ctx.context.completed = True
    ctx.context.log(f"save_final_report() → score={quality_score:.1f}/10 ✓")
    return f"Report finalized. Quality score: {quality_score:.1f}/10."
