"""
Writer agent — synthesizes research findings into a structured report.

LangGraph patterns demonstrated:
  - with_structured_output (Pydantic): the LLM must return a typed ResearchReport
    object. The writer then renders it to Markdown — clean separation between
    LLM output and presentation format.
  - StreamWriter: emits progress events so the runner can show real-time status.
"""
from __future__ import annotations

from typing import Any, Callable

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from config import settings
from state import ResearchState

_SYSTEM_TEMPLATE = """\
You are a Research Writer. Produce a comprehensive, well-structured research report.

Topic        : {topic}
Key findings : {findings_count} available
Sources      : {sources_count}
Critique to address: {critique}

Requirements:
  - Clear, academic-yet-accessible prose
  - Introduction + 3–5 thematic body sections + Conclusion
  - Cite sources inline where relevant
  - Address specific critique points from previous review cycles
  - Use Markdown formatting (## headings, **bold** key terms, bullet lists)
  - Aim for depth appropriate to the research gathered"""


class ReportSection(BaseModel):
    title: str = Field(description="Section heading")
    content: str = Field(description="Section body in Markdown prose")


class ResearchReport(BaseModel):
    title: str = Field(description="Descriptive report title")
    executive_summary: str = Field(description="2–3 sentence executive summary")
    sections: list[ReportSection] = Field(description="Main body sections (3–5)")
    conclusion: str = Field(description="Concluding paragraph")
    citations: list[str] = Field(default_factory=list, description="Source URLs referenced")
    word_count_estimate: int = Field(description="Rough word count of the full report")


def writer_node(state: ResearchState, writer: Callable[[Any], None]) -> dict:
    llm = ChatOpenAI(
        model=settings.writer_model,
        temperature=0.7,
        api_key=settings.openai_api_key,
    ).with_structured_output(ResearchReport)

    writer({"event": "writer_start", "topic": state["research_topic"]})

    findings_text = "\n".join(f"- {f}" for f in state.get("key_findings", []))
    sources_text = "\n".join(state.get("sources_found", [])[:20])
    critique = state.get("critique", "")
    human_feedback = state.get("human_feedback", "")
    raw_preview = "\n\n".join(state.get("raw_content", [])[:5])[:4000]

    critique_block = f"\n\nPREVIOUS CRITIQUE TO ADDRESS:\n{critique}" if critique else ""
    feedback_block = f"\n\nHUMAN FEEDBACK:\n{human_feedback}" if human_feedback else ""

    context = (
        f"Write a comprehensive research report on: {state['research_topic']}\n\n"
        f"KEY FINDINGS:\n{findings_text or 'Use your knowledge of the topic.'}\n\n"
        f"SOURCES:\n{sources_text or 'No explicit sources recorded.'}"
        f"{critique_block}"
        f"{feedback_block}\n\n"
        f"RAW CONTENT REFERENCE:\n{raw_preview}"
    )

    report: ResearchReport = llm.invoke([
        SystemMessage(
            content=_SYSTEM_TEMPLATE.format(
                topic=state["research_topic"],
                findings_count=len(state.get("key_findings", [])),
                sources_count=len(state.get("sources_found", [])),
                critique=critique or "none — this is the first draft",
            )
        ),
        HumanMessage(content=context),
    ])

    # Render structured report to Markdown
    md = f"# {report.title}\n\n"
    md += f"**Executive Summary:** {report.executive_summary}\n\n"
    md += "---\n\n"
    for section in report.sections:
        md += f"## {section.title}\n\n{section.content}\n\n"
    md += f"## Conclusion\n\n{report.conclusion}\n\n"
    if report.citations:
        md += "## Sources\n\n" + "\n".join(f"- {c}" for c in report.citations) + "\n"

    writer({
        "event": "writer_done",
        "sections": len(report.sections),
        "word_estimate": report.word_count_estimate,
    })

    return {
        "messages": [
            AIMessage(
                content=(
                    f"Draft complete: '{report.title}' "
                    f"(~{report.word_count_estimate} words, {len(report.sections)} sections)"
                )
            )
        ],
        "report_draft": md,
        "report_sections": [s.title for s in report.sections],
        "current_agent": "writer",
    }
