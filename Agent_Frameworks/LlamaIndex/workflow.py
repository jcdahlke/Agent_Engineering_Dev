"""
ResearchWorkflow — the central orchestration class.

Uses LlamaIndex's event-driven Workflow system: each @step method
subscribes to one typed Event and emits the next. The Workflow engine
routes automatically via Python type annotations — no explicit routing
table required (unlike LangGraph conditional edges or CrewAI @router).

Event chain:
  StartEvent → PlanReadyEvent → WebResearchDoneEvent
    → RagAnalysisDoneEvent → SynthesisDoneEvent → StopEvent
"""

from __future__ import annotations

from typing import Any

from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.workflow import (
    Context,
    Event,
    StartEvent,
    StopEvent,
    Workflow,
    step,
)
from pydantic import BaseModel, Field


# ── Event payloads ────────────────────────────────────────────────────────────

class PlanReadyEvent(Event):
    """Emitted by the Orchestrator after decomposing the research topic."""
    topic: str
    sub_questions: list[str]
    depth: str


class WebResearchDoneEvent(Event):
    """Emitted by the Web Researcher after gathering raw content from the web."""
    topic: str
    sub_questions: list[str]
    raw_chunks: list[str]   # raw text chunks for indexing
    sources: list[str]       # URLs / page titles collected


class SubAnswer(BaseModel):
    question: str
    answer: str
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)


class RagAnalysisDoneEvent(Event):
    """Emitted by the RAG Analyst after answering sub-questions via VectorStoreIndex."""
    topic: str
    sub_questions: list[str]
    answers: list[SubAnswer]
    sources: list[str]
    raw_chunks: list[str]   # passed through for the Synthesizer's RouterQueryEngine


class SynthesisDoneEvent(Event):
    """Emitted by the Synthesizer after routing thematic queries across indexes."""
    topic: str
    merged_findings: list[str]
    key_themes: list[str]
    sources: list[str]
    router_query_count: int


class ProgressEvent(Event):
    """Emitted at key milestones so stream mode can surface live updates."""
    step: str
    message: str
    data: dict[str, Any] = Field(default_factory=dict)


# ── Structured output model (final deliverable) ───────────────────────────────

class ReportSection(BaseModel):
    title: str
    content: str
    supporting_sources: list[str] = Field(default_factory=list)


class ResearchReport(BaseModel):
    title: str
    executive_summary: str
    methodology: str
    sections: list[ReportSection]
    conclusion: str
    citations: list[str]
    quality_score: float = Field(ge=0.0, le=1.0)
    rag_queries_run: int
    word_count_estimate: int


# ── Workflow ──────────────────────────────────────────────────────────────────

class ResearchWorkflow(Workflow):
    """
    Five-step research pipeline built on LlamaIndex Workflow.

    The workflow is async throughout. Steps communicate exclusively via
    typed Event objects — the engine dispatches each event to the step
    whose parameter annotation matches the event type.
    """

    def __init__(self, debug: bool = False, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.debug = debug

        # LlamaDebugHandler: intercepts every LLM call, embedding call,
        # and retrieval event. Unique to LlamaIndex — provides token-level
        # observability that LangGraph/CrewAI do not offer out of the box.
        self.debug_handler = LlamaDebugHandler(print_trace_on_end=debug)
        self.callback_manager = CallbackManager([self.debug_handler])

        # ChatMemoryBuffer: token-limited conversation history shared
        # across all async steps via the workflow Context store.
        self.memory = ChatMemoryBuffer.from_defaults(token_limit=8192)

    # ── Step 1: Orchestrator ──────────────────────────────────────────────────

    @step
    async def orchestrator_step(
        self, ctx: Context, ev: StartEvent
    ) -> PlanReadyEvent:
        from agents.orchestrator import run_orchestrator

        topic: str = ev.get("topic", "")
        depth: str = ev.get("depth", "standard")

        # Store shared context so all downstream steps can read it
        await ctx.store.set("topic", topic)
        await ctx.store.set("depth", depth)
        await ctx.store.set("memory", self.memory)
        await ctx.store.set("callback_manager", self.callback_manager)

        sub_questions = await run_orchestrator(topic, depth, self.callback_manager)

        ctx.write_event_to_stream(ProgressEvent(
            step="orchestrator",
            message=f"Research plan ready — {len(sub_questions)} sub-questions",
            data={"sub_questions": sub_questions},
        ))

        return PlanReadyEvent(topic=topic, sub_questions=sub_questions, depth=depth)

    # ── Step 2: Web Researcher ────────────────────────────────────────────────

    @step
    async def web_research_step(
        self, ctx: Context, ev: PlanReadyEvent
    ) -> WebResearchDoneEvent:
        from agents.web_researcher import run_web_researcher

        callback_manager = await ctx.store.get("callback_manager")
        raw_chunks, sources = await run_web_researcher(
            ev.topic, ev.sub_questions, ev.depth, callback_manager
        )

        ctx.write_event_to_stream(ProgressEvent(
            step="web_researcher",
            message=f"Gathered {len(raw_chunks)} text chunks from {len(sources)} sources",
            data={"source_count": len(sources), "chunk_count": len(raw_chunks)},
        ))

        return WebResearchDoneEvent(
            topic=ev.topic,
            sub_questions=ev.sub_questions,
            raw_chunks=raw_chunks,
            sources=sources,
        )

    # ── Step 3: RAG Analyst ───────────────────────────────────────────────────

    @step
    async def rag_analysis_step(
        self, ctx: Context, ev: WebResearchDoneEvent
    ) -> RagAnalysisDoneEvent:
        from agents.rag_analyst import run_rag_analyst

        callback_manager = await ctx.store.get("callback_manager")
        answers = await run_rag_analyst(
            ev.topic, ev.sub_questions, ev.raw_chunks, callback_manager
        )

        ctx.write_event_to_stream(ProgressEvent(
            step="rag_analyst",
            message=f"VectorStoreIndex built — answered {len(answers)} sub-questions",
            data={"answers": [a.question for a in answers]},
        ))

        return RagAnalysisDoneEvent(
            topic=ev.topic,
            sub_questions=ev.sub_questions,
            answers=answers,
            sources=ev.sources,
            raw_chunks=ev.raw_chunks,
        )

    # ── Step 4: Synthesizer ───────────────────────────────────────────────────

    @step
    async def synthesizer_step(
        self, ctx: Context, ev: RagAnalysisDoneEvent
    ) -> SynthesisDoneEvent:
        from agents.synthesizer import run_synthesizer

        callback_manager = await ctx.store.get("callback_manager")
        merged_findings, key_themes, query_count = await run_synthesizer(
            ev.topic, ev.answers, ev.raw_chunks, callback_manager
        )

        ctx.write_event_to_stream(ProgressEvent(
            step="synthesizer",
            message=f"RouterQueryEngine ran {query_count} thematic queries",
            data={"themes": key_themes},
        ))

        return SynthesisDoneEvent(
            topic=ev.topic,
            merged_findings=merged_findings,
            key_themes=key_themes,
            sources=ev.sources,
            router_query_count=query_count,
        )

    # ── Step 5: Report Writer ─────────────────────────────────────────────────

    @step
    async def report_writer_step(
        self, ctx: Context, ev: SynthesisDoneEvent
    ) -> StopEvent:
        from agents.report_writer import run_report_writer

        memory: ChatMemoryBuffer = await ctx.store.get("memory")
        callback_manager = await ctx.store.get("callback_manager")

        report = await run_report_writer(ev, memory, callback_manager)

        ctx.write_event_to_stream(ProgressEvent(
            step="report_writer",
            message="Research report complete",
            data={"quality_score": report.quality_score, "sections": len(report.sections)},
        ))

        return StopEvent(result=report)
