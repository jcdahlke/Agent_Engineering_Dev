"""
Researcher agent — gathers information using a manual ReAct loop.

LangGraph patterns demonstrated:
  - Manual ReAct loop: we implement the think→act→observe cycle ourselves rather
    than using create_react_agent. This makes the control flow explicit and
    educational — students see exactly how tool calls are built and parsed.
  - StreamWriter: emits structured progress events consumed by stream_mode="custom"
    in runner.py, giving real-time visibility into what the agent is doing.
"""
from __future__ import annotations

import os

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.types import StreamWriter

from config import settings
from state import ResearchState
from tools import get_search_tools

_SYSTEM_TEMPLATE = """\
You are a Research Agent. Gather comprehensive information about the topic using your tools.

Topic       : {topic}
Depth       : {depth}
Sources so far: {sources}

Depth guidelines:
  quick    → 2–3 searches, skim results
  standard → 4–6 searches, scrape 2–3 pages
  deep     → 8+ searches, scrape key pages, search arXiv for academic coverage

Strategy:
  1. Start with a broad web search to get an overview.
  2. Follow up with specific sub-topic searches.
  3. Search arXiv for technical/academic topics.
  4. Use Wikipedia for background definitions.
  5. Scrape the most relevant pages for full content.
  6. Stop when you have sufficient coverage — do not over-research.

Use your tools systematically. When done, respond with a plain text summary of what you found."""


def researcher_node(state: ResearchState, writer: StreamWriter = lambda _: None) -> dict:
    use_tavily = bool(os.getenv("TAVILY_API_KEY"))
    search_tools = get_search_tools(use_tavily=use_tavily)
    tool_map = {t.name: t for t in search_tools}

    llm = ChatOpenAI(
        model=settings.researcher_model,
        temperature=0.3,
        api_key=settings.openai_api_key,
    ).bind_tools(search_tools)

    writer({"event": "researcher_start", "topic": state["research_topic"]})

    system_msg = SystemMessage(
        content=_SYSTEM_TEMPLATE.format(
            topic=state["research_topic"],
            depth=state.get("research_depth", "standard"),
            sources=len(state.get("sources_found", [])),
        )
    )
    human_msg = HumanMessage(content=f"Research this topic: {state['research_topic']}")
    loop_messages = [system_msg, human_msg]

    new_sources: list[str] = []
    new_content: list[str] = []
    all_new_messages: list = []
    max_tool_rounds = 12

    for _ in range(max_tool_rounds):
        response: AIMessage = llm.invoke(loop_messages)
        loop_messages.append(response)
        all_new_messages.append(response)

        if not response.tool_calls:
            # LLM decided it has enough information
            break

        for tc in response.tool_calls:
            tool_name = tc["name"]
            tool_args = tc["args"]
            writer({"event": "tool_call", "tool": tool_name})

            raw_result: Any = f"Tool '{tool_name}' not found."
            if tool_name in tool_map:
                try:
                    raw_result = tool_map[tool_name].invoke(tool_args)
                except Exception as exc:
                    raw_result = f"Tool error: {exc}"

            # Extract URLs and text from structured search results
            if isinstance(raw_result, list):
                for item in raw_result:
                    if isinstance(item, dict):
                        url = item.get("url") or item.get("href") or item.get("entry_id", "")
                        if url:
                            new_sources.append(str(url))
                        snippet = (
                            item.get("content")
                            or item.get("abstract")
                            or item.get("snippet")
                            or ""
                        )
                        if snippet:
                            new_content.append(str(snippet)[:1000])
                serialised = str(raw_result)
            elif isinstance(raw_result, str):
                # A URL returned directly (e.g., scrape result is the page text)
                if raw_result.startswith("http"):
                    new_sources.append(raw_result)
                elif len(raw_result) > 100:
                    new_content.append(raw_result[:2000])
                serialised = raw_result
            else:
                serialised = str(raw_result)

            tool_msg = ToolMessage(
                content=serialised[:3000],
                tool_call_id=tc["id"],
            )
            loop_messages.append(tool_msg)
            all_new_messages.append(tool_msg)

    writer({
        "event": "researcher_done",
        "new_sources": len(new_sources),
        "new_content_chunks": len(new_content),
    })

    return {
        "messages": all_new_messages,
        "sources_found": new_sources,
        "raw_content": new_content,
        "current_agent": "researcher",
    }
