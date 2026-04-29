"""
Report Writer step — produces a structured Pydantic report.

Demonstrates LlamaIndex's structured LLM output:
- llm.as_structured_llm(output_cls=ResearchReport) binds a Pydantic model
  to an LLM instance, ensuring every response matches the schema.
- ChatMemoryBuffer is read here so the writer has full conversation history
  as context before composing the final report.
"""

from __future__ import annotations

from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.callbacks import CallbackManager
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.llms.openai import OpenAI

from config import settings
from workflow import ResearchReport, ReportSection, SynthesisDoneEvent

WRITER_SYSTEM = """\
You are an expert research writer. Produce a comprehensive, well-structured
research report based on the findings provided. The report must be factual,
well-organized, and suitable for an academic or professional audience.
"""


async def run_report_writer(
    ev: SynthesisDoneEvent,
    memory: ChatMemoryBuffer,
    callback_manager: CallbackManager,
) -> ResearchReport:
    llm = OpenAI(
        model=settings.writer_model,
        temperature=0.7,
        callback_manager=callback_manager,
    )

    # Retrieve conversation history from the shared memory buffer
    history = memory.get()
    history_text = ""
    if history:
        history_text = "\n".join(
            f"{m.role.value}: {m.content}" for m in history[-6:]
        )

    # Build the synthesis summary for the prompt
    findings_text = "\n\n".join(
        f"Finding {i+1}: {f}" for i, f in enumerate(ev.merged_findings)
    )
    themes_text = ", ".join(ev.key_themes)
    sources_text = "\n".join(f"- {s}" for s in ev.sources[:10])

    user_prompt = f"""\
Research Topic: {ev.topic}

Key Themes Identified: {themes_text}

Synthesized Findings:
{findings_text}

Sources Consulted:
{sources_text}

Recent Context:
{history_text}

Please write a comprehensive research report covering:
1. An executive summary (2-3 paragraphs)
2. Methodology section describing the AI-driven RAG research approach
3. 3-4 substantive sections covering the key findings
4. A conclusion with actionable insights
5. Citations list

Assess the overall quality of the research on a 0.0-1.0 scale.
Estimate the total word count of the report.
Report how many RAG queries were run: {ev.router_query_count}.
"""

    # as_structured_llm binds the Pydantic model to the LLM — the response
    # is automatically parsed and validated against ResearchReport's schema.
    structured_llm = llm.as_structured_llm(output_cls=ResearchReport)

    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content=WRITER_SYSTEM),
        ChatMessage(role=MessageRole.USER, content=user_prompt),
    ]

    try:
        response = await structured_llm.achat(messages)
        report: ResearchReport = response.raw

        # Update the shared memory buffer with this exchange
        memory.put(ChatMessage(role=MessageRole.USER, content=user_prompt[:500]))
        memory.put(ChatMessage(
            role=MessageRole.ASSISTANT,
            content=f"Report generated: {report.title}",
        ))

        return report

    except Exception as e:
        # Fallback: construct a minimal report if structured output fails
        return ResearchReport(
            title=f"Research Report: {ev.topic}",
            executive_summary="\n\n".join(ev.merged_findings[:2]),
            methodology="Multi-agent RAG pipeline using LlamaIndex Workflow, VectorStoreIndex, and SubQuestionQueryEngine.",
            sections=[
                ReportSection(
                    title=theme,
                    content=finding,
                    supporting_sources=ev.sources[:2],
                )
                for theme, finding in zip(ev.key_themes, ev.merged_findings)
            ],
            conclusion="See synthesized findings above.",
            citations=ev.sources,
            quality_score=0.5,
            rag_queries_run=ev.router_query_count,
            word_count_estimate=sum(len(f.split()) for f in ev.merged_findings),
        )
