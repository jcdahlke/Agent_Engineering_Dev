"""
Analyzer agent — extracts structured insights from raw research content.

LangGraph patterns demonstrated:
  - with_structured_output: forces the LLM to return a typed Pydantic model
    instead of free-form text. Ideal for downstream agents that need reliable
    structured data.
  - Tool loop before structured extraction: the analyzer can run Python code
    (via PythonREPLTool) to compute statistics before producing its summary.
"""
from __future__ import annotations

from typing import Any, Callable

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from config import settings
from state import ResearchState
from tools import get_analysis_tools

_SYSTEM_TEMPLATE = """\
You are a Research Analyst. Analyze the gathered content and extract structured insights.

Topic        : {topic}
Content chunks: {content_count}
Sources found : {sources_count}

Your tasks:
  1. Identify the 5–15 most important findings and key facts.
  2. Extract any structured data (statistics, comparisons, timelines) as dict lists.
  3. Run Python code if you need to compute, convert, or visualize anything.
  4. Assess whether the research is sufficient for a complete report.

Be thorough. Vague findings like "there are benefits" are not useful — quantify and specify."""

_EXTRACTION_SYSTEM = """\
Extract a structured analysis from this research content.
Be specific, comprehensive, and honest about confidence level."""


class AnalysisResult(BaseModel):
    key_findings: list[str] = Field(
        description="Key findings as specific, factual bullet points (aim for 5–12)"
    )
    data_tables: list[dict] = Field(
        default_factory=list,
        description="Structured data extracted (e.g. comparisons, statistics) as list of dicts",
    )
    code_outputs: list[str] = Field(
        default_factory=list,
        description="Results from any Python code execution",
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Confidence in analysis quality based on source coverage (0–1)",
    )
    needs_more_research: bool = Field(
        description="True if critical information is missing and more research is warranted",
    )
    analysis_summary: str = Field(
        description="2–3 sentence summary of the analytical conclusions",
    )


def analyzer_node(state: ResearchState, writer: Callable[[Any], None]) -> dict:
    analysis_tools = get_analysis_tools()
    tool_map = {t.name: t for t in analysis_tools}

    writer({"event": "analyzer_start", "content_chunks": len(state.get("raw_content", []))})

    content_blob = "\n\n---\n\n".join(state.get("raw_content", [])[:12])
    if not content_blob.strip():
        content_blob = "No raw content available. Base analysis on the topic alone."

    # Optional tool loop: let the LLM run Python code before final structured extraction
    loop_llm = ChatOpenAI(
        model=settings.analyst_model,
        temperature=0.2,
        api_key=settings.openai_api_key,
    ).bind_tools(analysis_tools)

    system_msg = SystemMessage(
        content=_SYSTEM_TEMPLATE.format(
            topic=state["research_topic"],
            content_count=len(state.get("raw_content", [])),
            sources_count=len(state.get("sources_found", [])),
        )
    )
    human_msg = HumanMessage(
        content=(
            f"Analyze this research content about '{state['research_topic']}':\n\n"
            f"{content_blob[:8000]}"
        )
    )

    loop_messages = [system_msg, human_msg]
    all_new_messages: list = []
    new_code_outputs: list[str] = []

    for _ in range(5):
        response: AIMessage = loop_llm.invoke(loop_messages)
        loop_messages.append(response)
        all_new_messages.append(response)

        if not response.tool_calls:
            break

        for tc in response.tool_calls:
            result_str = f"Tool '{tc['name']}' not available."
            if tc["name"] in tool_map:
                try:
                    result_str = str(tool_map[tc["name"]].invoke(tc["args"]))
                    new_code_outputs.append(result_str[:500])
                    writer({"event": "code_executed", "output_preview": result_str[:100]})
                except Exception as exc:
                    result_str = f"Error: {exc}"

            tool_msg = ToolMessage(content=result_str[:2000], tool_call_id=tc["id"])
            loop_messages.append(tool_msg)
            all_new_messages.append(tool_msg)

    # Structured extraction pass — uses a fresh LLM call with typed output
    extraction_llm = ChatOpenAI(
        model=settings.analyst_model,
        temperature=0.1,
        api_key=settings.openai_api_key,
    ).with_structured_output(AnalysisResult)

    analysis: AnalysisResult = extraction_llm.invoke([
        SystemMessage(content=_EXTRACTION_SYSTEM),
        HumanMessage(
            content=(
                f"Topic: {state['research_topic']}\n\n"
                f"Content:\n{content_blob[:6000]}\n\n"
                f"Prior analysis notes:\n{str([m.content for m in all_new_messages if hasattr(m, 'content')])[:2000]}"
            )
        ),
    ])

    writer({
        "event": "analyzer_done",
        "findings": len(analysis.key_findings),
        "confidence": analysis.confidence,
        "needs_more_research": analysis.needs_more_research,
    })

    return {
        "messages": all_new_messages,
        "key_findings": analysis.key_findings,
        "data_tables": analysis.data_tables,
        "code_outputs": new_code_outputs or analysis.code_outputs,
        "current_agent": "analyzer",
    }
