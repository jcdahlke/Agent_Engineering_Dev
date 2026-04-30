"""
pipeline.py — Multi-agent orchestration using Pydantic AI's composition model.

Key Pydantic AI pattern: agents are called like regular async Python functions.
No graph framework, no message bus — just await agent.run(..., deps=deps).

Context flows via deps= (shared ResearchDependencies dataclass).
History chains with message_history=result.new_messages() between agents.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass

import httpx
from rich.console import Console

from agents.analyzer import analyzer_agent
from agents.critic import critic_agent
from agents.researcher import researcher_agent
from agents.supervisor import supervisor_agent
from agents.writer import writer_agent
from config import config
from deps import ResearchDependencies
from models import AnalysisResult, CritiqueResult, ResearchFindings, WrittenReport

console = Console()


@dataclass
class PipelineResult:
    topic: str
    research: ResearchFindings | None
    analysis: AnalysisResult | None
    report: WrittenReport | None
    critique: CritiqueResult | None
    final_markdown: str
    revision_cycles: int
    approved: bool


async def run_research(topic: str, depth: str | None = None) -> PipelineResult:
    """Execute the full 5-agent research pipeline.

    Agent call order:
      1. supervisor  → research plan (str)
      2. researcher  → ResearchFindings  (uses web/arxiv/wiki tools)
      3. analyzer    → AnalysisResult    (message_history chains from researcher)
      4. writer      → WrittenReport     (message_history chains from analyzer)
      5. critic      → CritiqueResult    (output_validator enforces quality)
      Repeat 4-5 up to MAX_REVISIONS times if not approved.
    """
    effective_depth = depth or config.research_depth

    async with httpx.AsyncClient(
        timeout=45.0,
        headers={"User-Agent": "PydanticAI-Research/1.0"},
        follow_redirects=True,
    ) as http:
        deps = ResearchDependencies(http_client=http, config=config)

        console.print(
            f"\n[bold #e92063]Session {deps.session_id}[/] | "
            f"Topic: [cyan]{topic}[/] | Depth: [yellow]{effective_depth}[/]\n"
        )

        # ── Step 1: Supervisor plans the research ────────────────────────────
        console.print("[dim]▸ Step 1/5  Supervisor planning research strategy...[/]")
        sup_result = await supervisor_agent.run(
            f"Create a research plan for this topic: {topic}",
            deps=deps,
        )
        research_plan: str = sup_result.output
        console.print(f"[dim]  Plan preview: {research_plan[:160].strip()}...[/]\n")

        # ── Step 2: Researcher gathers data ──────────────────────────────────
        console.print("[dim]▸ Step 2/5  Researcher gathering data (may take a moment)...[/]")
        res_result = await researcher_agent.run(
            f"Research topic: {topic}\n\nResearch plan to follow:\n{research_plan}\n\nDepth: {effective_depth}",
            deps=deps,
        )
        findings: ResearchFindings = res_result.output
        console.print(
            f"[dim]  Found {len(findings.sources)} sources, "
            f"{len(findings.raw_content)} content chunks. "
            f"Queries run: {findings.search_queries_run}[/]\n"
        )

        # ── Step 3: Analyzer extracts structured insights ────────────────────
        console.print("[dim]▸ Step 3/5  Analyzer extracting structured insights...[/]")
        content_blob = "\n\n---\n\n".join(findings.raw_content[:12])
        ana_result = await analyzer_agent.run(
            f"Analyze this research content about '{topic}':\n\n"
            f"Sources found: {findings.sources[:12]}\n\n"
            f"Content:\n{content_blob[:7000]}",
            deps=deps,
            message_history=res_result.new_messages(),
        )
        analysis: AnalysisResult = ana_result.output
        console.print(
            f"[dim]  {len(analysis.key_findings)} key findings | "
            f"confidence={analysis.confidence:.2f} | "
            f"needs_more={analysis.needs_more_research}[/]\n"
        )

        # ── Steps 4–5: Writer → Critic → revision loop ───────────────────────
        revision_cycles = 0
        prior_critique_text = ""
        wri_result = None
        cri_result = None

        while True:
            console.print(
                f"[dim]▸ Step 4/5  Writer drafting report "
                f"(cycle {revision_cycles + 1}/{config.max_revisions + 1})...[/]"
            )
            findings_list = "\n".join(f"  - {f}" for f in analysis.key_findings)
            critique_block = (
                f"\n\nPREVIOUS CRITIQUE — address these issues:\n{prior_critique_text}"
                if prior_critique_text
                else ""
            )

            wri_result = await writer_agent.run(
                f"Write a comprehensive research report on: {topic}\n\n"
                f"Key findings to cover:\n{findings_list}\n\n"
                f"Sources available: {', '.join(findings.sources[:12])}"
                f"{critique_block}",
                deps=deps,
                message_history=ana_result.new_messages(),
            )
            report: WrittenReport = wri_result.output
            console.print(
                f"[dim]  Draft: '{report.title}' "
                f"(~{report.word_count_estimate} words, {len(report.sections)} sections)[/]\n"
            )

            console.print("[dim]▸ Step 5/5  Critic evaluating report quality...[/]")
            report_md = _render_markdown(report)
            cri_result = await critic_agent.run(
                f"Review this research report on '{topic}':\n\n{report_md[:9000]}",
                deps=deps,
            )
            critique: CritiqueResult = cri_result.output
            status_color = "green" if critique.approved else "yellow"
            console.print(
                f"[dim]  Score: [{status_color}]{critique.score:.2f}[/] | "
                f"Approved: [{status_color}]{critique.approved}[/][/]\n"
            )

            if critique.approved or revision_cycles >= config.max_revisions:
                break

            prior_critique_text = critique.revision_instructions
            revision_cycles += 1
            console.print(
                f"[yellow]  Revision {revision_cycles}/{config.max_revisions} requested. "
                f"Rewriting...[/]\n"
            )

        final_md = _render_markdown(report) if wri_result else ""
        return PipelineResult(
            topic=topic,
            research=findings,
            analysis=analysis,
            report=report if wri_result else None,
            critique=critique if cri_result else None,
            final_markdown=final_md,
            revision_cycles=revision_cycles,
            approved=critique.approved if cri_result else False,
        )


def _render_markdown(report: WrittenReport) -> str:
    """Convert a WrittenReport to a Markdown string."""
    md = f"# {report.title}\n\n"
    md += f"**Executive Summary:** {report.executive_summary}\n\n---\n\n"
    for section in report.sections:
        md += f"## {section.title}\n\n{section.content}\n\n"
    md += f"## Conclusion\n\n{report.conclusion}\n\n"
    if report.citations:
        md += "## Sources\n\n" + "\n".join(f"- {c}" for c in report.citations)
    return md


if __name__ == "__main__":
    import sys

    topic = " ".join(sys.argv[1:]) or "large language models"
    result = asyncio.run(run_research(topic))
    console.print(result.final_markdown)
