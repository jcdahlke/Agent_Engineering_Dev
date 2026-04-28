"""
Tools for the CrewAI research system.

Two categories:
  1. Built-in crewai-tools: SerperDevTool (Google Search), WebsiteSearchTool
  2. Custom @tool functions: arxiv_search, scrape_webpage, analyze_text

CrewAI's @tool decorator requires functions to return str — the LLM receives
the return value as a plain-text observation. The docstring IS the tool
description shown to the LLM when it decides which tool to call.
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING

import requests
from bs4 import BeautifulSoup
from crewai.tools import tool

import arxiv as _arxiv

from config import settings

if TYPE_CHECKING:
    pass


# ── Custom tools ───────────────────────────────────────────────────────────────

@tool("ArXiv Academic Search")
def arxiv_search_tool(query: str) -> str:
    """Search arXiv for academic papers on a topic and return the top 5 results.
    Returns each paper's title, authors, publication date, abstract summary, and URL.
    Use this to find peer-reviewed research and technical papers.
    Input: a search query string (e.g. 'transformer attention mechanism')."""
    try:
        client = _arxiv.Client()
        search = _arxiv.Search(
            query=query,
            max_results=5,
            sort_by=_arxiv.SortCriterion.Relevance,
        )
        results = []
        for paper in client.results(search):
            authors = ", ".join(a.name for a in paper.authors[:3])
            if len(paper.authors) > 3:
                authors += " et al."
            abstract = paper.summary[:300].replace("\n", " ") + "..."
            results.append(
                f"Title: {paper.title}\n"
                f"Authors: {authors}\n"
                f"Published: {paper.published.strftime('%Y-%m-%d')}\n"
                f"Abstract: {abstract}\n"
                f"URL: {paper.entry_id}\n"
            )
        if not results:
            return f"No arXiv papers found for query: {query}"
        return f"Found {len(results)} papers on '{query}':\n\n" + "\n---\n".join(results)
    except Exception as exc:
        return f"ArXiv search failed: {exc}"


@tool("Web Page Text Extractor")
def scrape_webpage_tool(url: str) -> str:
    """Fetch and extract the readable text content from a web page URL.
    Removes navigation menus, scripts, ads, and styling. Returns up to 4000 characters.
    Use this to get the full content of a specific web page you want to read.
    Input: a complete URL starting with http:// or https://"""
    try:
        headers = {"User-Agent": "Mozilla/5.0 (research-bot/1.0)"}
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
            tag.decompose()
        text = soup.get_text(separator=" ", strip=True)
        # Collapse whitespace
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) > 4000:
            text = text[:4000] + "\n\n[...truncated at 4000 chars]"
        return text or "No readable text found on this page."
    except Exception as exc:
        return f"Failed to scrape {url}: {exc}"


@tool("Text Statistics Analyzer")
def analyze_text_tool(text: str) -> str:
    """Compute word count, sentence count, paragraph count, and extract all numeric
    values from a block of text. Useful for quantifying and verifying research content.
    Returns a structured summary of text statistics.
    Input: a plain text string to analyze."""
    if not text or not text.strip():
        return "Input text is empty."

    words = text.split()
    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    # Extract numeric values (integers and decimals)
    numbers = re.findall(r"\b\d+(?:\.\d+)?%?\b", text)
    unique_numbers = list(dict.fromkeys(numbers))[:20]

    avg_sentence_len = len(words) / max(len(sentences), 1)

    return (
        f"Text statistics:\n"
        f"  Word count       : {len(words)}\n"
        f"  Sentence count   : {len(sentences)}\n"
        f"  Paragraph count  : {len(paragraphs)}\n"
        f"  Avg sentence len : {avg_sentence_len:.1f} words\n"
        f"  Numeric values   : {', '.join(unique_numbers) if unique_numbers else 'none found'}\n"
        f"  Character count  : {len(text)}\n"
    )


# ── Built-in crewai-tools helpers ──────────────────────────────────────────────
# Instantiated inside functions (not at module level) to guard against a missing
# SERPER_API_KEY — crewai-tools raises ValueError at instantiation if the key
# is absent, which would crash the import of this module.

def get_researcher_tools() -> list:
    """Return the tool list for the Senior Researcher agent."""
    tools: list = [arxiv_search_tool, scrape_webpage_tool]

    try:
        from crewai_tools import WebsiteSearchTool
        tools.append(WebsiteSearchTool())
    except Exception:
        pass

    if settings.serper_api_key:
        try:
            from crewai_tools import SerperDevTool
            tools.insert(0, SerperDevTool())
        except Exception:
            pass

    return tools


def get_analyst_tools() -> list:
    """Return the tool list for the Data Analyst agent."""
    return [analyze_text_tool, scrape_webpage_tool]
