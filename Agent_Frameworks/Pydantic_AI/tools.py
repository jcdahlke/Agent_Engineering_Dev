"""
tools.py — Shared async tool implementations.

These are plain async functions, NOT decorated with @agent.tool here.
Each agent module wraps them with @agent.tool so they receive RunContext.

The scrape_webpage function demonstrates dependency injection:
it uses ctx.deps.http_client (the injected httpx.AsyncClient) instead
of creating a new requests.Session — no global state, fully testable.
"""
from __future__ import annotations

import asyncio

from pydantic_ai import RunContext

from deps import ResearchDependencies


async def web_search(
    ctx: RunContext[ResearchDependencies],
    query: str,
    max_results: int = 5,
) -> str:
    """Search the web using DuckDuckGo. Returns formatted titles, URLs, and snippets."""
    try:
        from ddgs import DDGS

        def _search() -> list[dict]:
            with DDGS() as ddgs:
                return list(ddgs.text(query, max_results=max_results))

        results = await asyncio.to_thread(_search)
        lines: list[str] = []
        for i, r in enumerate(results, 1):
            url = r.get("href", "")
            if url and url not in ctx.deps.sources:
                ctx.deps.sources.append(url)
            snippet = r.get("body", "")[:300]
            lines.append(f"[{i}] {r.get('title', '')}\n    URL: {url}\n    {snippet}")
        return "\n\n".join(lines) if lines else "No results found."
    except Exception as exc:
        return f"Search error: {exc}"


async def arxiv_search(
    ctx: RunContext[ResearchDependencies],
    query: str,
    max_results: int = 4,
) -> str:
    """Search academic papers on arXiv for peer-reviewed research."""
    try:
        import arxiv

        def _search() -> list:
            client = arxiv.Client()
            search = arxiv.Search(query=query, max_results=max_results)
            return list(client.results(search))

        papers = await asyncio.to_thread(_search)
        lines: list[str] = []
        for p in papers:
            url = p.entry_id
            if url and url not in ctx.deps.sources:
                ctx.deps.sources.append(url)
            lines.append(
                f"Title: {p.title}\n"
                f"Authors: {', '.join(str(a) for a in p.authors[:3])}\n"
                f"Published: {p.published.strftime('%Y-%m-%d') if p.published else 'N/A'}\n"
                f"URL: {url}\n"
                f"Abstract: {p.summary[:400]}"
            )
        return "\n\n---\n\n".join(lines) if lines else "No arXiv papers found."
    except Exception as exc:
        return f"arXiv error: {exc}"


async def wikipedia_search(
    ctx: RunContext[ResearchDependencies],
    query: str,
) -> str:
    """Look up a topic on Wikipedia for background context and definitions."""
    try:
        import wikipedia

        def _search() -> str:
            try:
                page = wikipedia.page(query, auto_suggest=True)
                url = page.url
                if url and url not in ctx.deps.sources:
                    ctx.deps.sources.append(url)
                return f"Title: {page.title}\nURL: {url}\n\n{page.summary[:1200]}"
            except wikipedia.DisambiguationError as e:
                page = wikipedia.page(e.options[0])
                return f"Title: {page.title}\nURL: {page.url}\n\n{page.summary[:1200]}"

        return await asyncio.to_thread(_search)
    except Exception as exc:
        return f"Wikipedia error: {exc}"


async def scrape_webpage(
    ctx: RunContext[ResearchDependencies],
    url: str,
) -> str:
    """Fetch a webpage and return its readable text content.

    Uses ctx.deps.http_client — the injected httpx.AsyncClient — instead of
    creating a new client. This is Pydantic AI's dependency injection in action.
    """
    try:
        from bs4 import BeautifulSoup

        resp = await ctx.deps.http_client.get(url, follow_redirects=True)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
        lines = [ln.strip() for ln in text.splitlines() if len(ln.strip()) > 40]
        content = "\n".join(lines[:120])
        if url not in ctx.deps.sources:
            ctx.deps.sources.append(url)
        return content or "No readable content found."
    except Exception as exc:
        return f"Scrape error for {url}: {exc}"
