"""
Standalone tool functions used by the Web Researcher agent.
Wrapped as FunctionTool objects in agents/web_researcher.py.
"""

import re
import textwrap

import requests
import wikipedia
from bs4 import BeautifulSoup
from ddgs import DDGS


def duckduckgo_search(query: str, max_results: int = 5) -> str:
    """Search the web for current information using DuckDuckGo."""
    try:
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append(f"Title: {r['title']}\nURL: {r['href']}\nSnippet: {r['body']}\n")
        return "\n---\n".join(results) if results else "No results found."
    except Exception as e:
        return f"Search error: {e}"


def wikipedia_search(query: str, sentences: int = 5) -> str:
    """Retrieve a Wikipedia summary for a topic."""
    try:
        wikipedia.set_lang("en")
        return wikipedia.summary(query, sentences=sentences, auto_suggest=True)
    except wikipedia.exceptions.DisambiguationError as e:
        try:
            return wikipedia.summary(e.options[0], sentences=sentences)
        except Exception:
            return f"Disambiguation error for '{query}'. Options: {e.options[:5]}"
    except wikipedia.exceptions.PageError:
        return f"No Wikipedia page found for '{query}'."
    except Exception as e:
        return f"Wikipedia error: {e}"


def scrape_webpage(url: str, max_chars: int = 3000) -> str:
    """Fetch a webpage and extract its main text content."""
    try:
        headers = {"User-Agent": "Mozilla/5.0 (research-agent/1.0)"}
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()

        text = soup.get_text(separator=" ", strip=True)
        text = re.sub(r"\s+", " ", text).strip()
        return textwrap.shorten(text, width=max_chars, placeholder="... [truncated]")
    except Exception as e:
        return f"Scrape error for {url}: {e}"
