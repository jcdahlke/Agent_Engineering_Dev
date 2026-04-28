"""
Task definitions for the CrewAI research system.

Key CrewAI task concepts demonstrated here:

  output_pydantic : Forces the agent to produce structured output matching a
                    Pydantic model. Access via task.output.pydantic after the crew runs.

  context         : List of upstream tasks whose outputs are injected as context
                    into this task's prompt. This is how CrewAI passes data between
                    agents — no shared state dict needed.

  callback        : Python function called after a task completes.
                    step_callback (on the Crew) fires after every ReAct step.

  {topic}/{depth} : Template variables filled from crew.kickoff(inputs={...}).
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from crewai import Task

from agents import data_analyst, editor_critic, senior_researcher, technical_writer


# ── Pydantic output models ─────────────────────────────────────────────────────
# These define the structured schema each task must produce.
# When output_pydantic= is set, CrewAI instructs the LLM to output valid JSON
# matching the model, then parses and validates it automatically.

class ResearchGathering(BaseModel):
    """Structured output from the research task."""
    sources: list[str] = Field(description="List of URLs or source identifiers found")
    raw_summaries: list[str] = Field(description="Key content summaries from each source")
    search_queries_used: list[str] = Field(description="All search queries that were run")
    coverage_assessment: str = Field(description="Analyst assessment of research sufficiency")


class AnalysisResult(BaseModel):
    """Structured output from the analysis task."""
    key_findings: list[str] = Field(
        description="Specific, factual bullet-point findings — cite statistics, dates, names (5-12 items)"
    )
    statistics: list[str] = Field(description="Quantitative data points extracted from the research")
    knowledge_gaps: list[str] = Field(description="Areas where the research coverage is insufficient")
    confidence_score: float = Field(
        ge=0.0, le=1.0, description="Analyst's confidence in coverage completeness (0.0–1.0)"
    )


class ReportSection(BaseModel):
    """A single section of the research report."""
    heading: str = Field(description="Section heading/title")
    content: str = Field(description="Section body text (100-250 words)")


class ResearchReport(BaseModel):
    """Structured output from the writing task."""
    title: str = Field(description="Report title")
    executive_summary: str = Field(description="Standalone 2-3 paragraph overview of all key points")
    sections: list[ReportSection] = Field(
        description="List of 3-5 report sections, each with a heading and content"
    )
    conclusion: str = Field(description="Final synthesis and implications")
    citations: list[str] = Field(description="All sources cited, formatted as readable strings")
    word_count_estimate: int = Field(description="Approximate total word count of the report body")


class EditorialReview(BaseModel):
    """Structured output from the editor/critic task."""
    quality_score: float = Field(
        ge=0.0, le=1.0,
        description="Editorial quality score: 0.9+ publication-ready, 0.7-0.89 good, below 0.7 needs revision"
    )
    approved: bool = Field(description="True if quality_score >= 0.7 and report is acceptable")
    strengths: list[str] = Field(description="Specific strengths of the report (3-5 items)")
    weaknesses: list[str] = Field(description="Specific weaknesses or gaps found (0-5 items)")
    revision_instructions: str = Field(
        description="Specific, actionable revision instructions (empty string if approved without changes)"
    )
    requires_revision: bool = Field(description="True if substantive revisions are required")


# ── Callbacks ──────────────────────────────────────────────────────────────────
# task_callback: attached per-task via callback= parameter, fires on completion.
# step_callback: attached at the Crew level, fires after every agent ReAct step.

def task_callback(output) -> None:
    """Called by CrewAI after each task completes. output is a TaskOutput object."""
    try:
        from rich.console import Console
        from rich.panel import Panel
        console = Console()
        agent_name = getattr(output, "agent", "unknown agent")
        raw_preview = str(getattr(output, "raw", ""))[:200]
        pydantic_obj = getattr(output, "pydantic", None)
        type_label = type(pydantic_obj).__name__ if pydantic_obj else "str"
        console.print(Panel(
            f"[green]Agent:[/] {agent_name}\n"
            f"[green]Output type:[/] {type_label}\n"
            f"[dim]{raw_preview}{'...' if len(str(getattr(output, 'raw', ''))) > 200 else ''}[/]",
            title="[bold cyan]Task Complete[/]",
            border_style="cyan",
        ))
    except Exception:
        pass


def step_callback(agent_action) -> None:
    """Called by CrewAI after every ReAct step. agent_action has .tool, .tool_input, .log."""
    try:
        from rich.console import Console
        console = Console()
        tool_name = getattr(agent_action, "tool", None)
        tool_input = getattr(agent_action, "tool_input", "")
        if tool_name:
            input_preview = str(tool_input)[:80]
            console.print(f"  [dim cyan]→ {tool_name}[/] [dim]({input_preview})[/]")
    except Exception:
        pass


# ── Tasks ──────────────────────────────────────────────────────────────────────

research_task = Task(
    description=(
        "Conduct thorough research on the topic: **{topic}**\n\n"
        "Research depth: {depth}\n\n"
        "Instructions:\n"
        "- Use SerperDevTool or WebsiteSearchTool to search for recent web articles\n"
        "- Use arxiv_search_tool to find peer-reviewed academic papers\n"
        "- Use scrape_webpage_tool to extract full content from the 2-3 most valuable URLs\n"
        "- For 'deep' depth: find at least 3 academic papers from arXiv\n"
        "- For 'quick' depth: 3-4 sources are sufficient\n"
        "- Document every URL and search query used\n"
        "- Note the publication date of each source"
    ),
    expected_output=(
        "A structured research summary with: all source URLs found, "
        "key content summaries from each source (2-4 sentences each), "
        "all search queries that were run, and an honest assessment of "
        "whether coverage is sufficient for a comprehensive report."
    ),
    agent=senior_researcher,
    output_pydantic=ResearchGathering,
    callback=task_callback,
)

analysis_task = Task(
    description=(
        "Analyze all the research gathered about **{topic}**.\n\n"
        "Instructions:\n"
        "- Extract 5-12 specific, factual findings — not vague summaries. "
        "  Each finding should cite a specific statistic, date, name, or measurement.\n"
        "- Use analyze_text_tool on the raw summaries to extract all numeric values\n"
        "- List all quantitative data points (percentages, dollar amounts, counts, dates)\n"
        "- Identify knowledge gaps: what important questions remain unanswered?\n"
        "- Rate your confidence in the coverage completeness (0.0 to 1.0)"
    ),
    expected_output=(
        "A structured analysis with: 5-12 specific key findings as bullet points, "
        "a list of all quantitative statistics extracted, identified knowledge gaps, "
        "and a confidence score between 0.0 and 1.0."
    ),
    agent=data_analyst,
    context=[research_task],
    output_pydantic=AnalysisResult,
    callback=task_callback,
)

writing_task = Task(
    description=(
        "Write a comprehensive research report on **{topic}**.\n\n"
        "Report structure:\n"
        "  1. Executive Summary (standalone, 2-3 paragraphs)\n"
        "  2. Introduction (context and scope)\n"
        "  3. 3-5 thematic sections based on the key findings\n"
        "  4. Conclusion (synthesis and implications)\n"
        "  5. Citations (all sources used)\n\n"
        "Requirements:\n"
        "- Cite sources inline (e.g., [Source: URL])\n"
        "- Incorporate specific statistics from the analysis\n"
        "- Target 800-1200 words total\n"
        "- Use clear professional prose, no jargon without explanation"
    ),
    expected_output=(
        "A complete structured research report with: title, executive summary, "
        "3-5 labeled sections with substantive content, conclusion, and full citation list. "
        "Each section should be 100-250 words. All key findings from the analysis must appear."
    ),
    agent=technical_writer,
    context=[research_task, analysis_task],
    output_pydantic=ResearchReport,
    callback=task_callback,
)

editing_task = Task(
    description=(
        "Review the research report on **{topic}** produced by the Technical Writer.\n\n"
        "Evaluation rubric:\n"
        "  0.9–1.0 : Publication-ready — comprehensive, well-cited, excellent prose\n"
        "  0.7–0.89 : Good — minor issues only, publishable with light edits\n"
        "  0.5–0.69 : Adequate — notable gaps or structural issues\n"
        "  0.3–0.49 : Major issues — significant rework needed\n"
        "  0.0–0.29 : Fundamental problems — near-complete rewrite required\n\n"
        "Set approved=True only if quality_score >= 0.7.\n"
        "Provide specific revision_instructions if not approved (reference exact sections/sentences)."
    ),
    expected_output=(
        "An editorial review with: a quality score (0.0-1.0), approved flag (True/False), "
        "3-5 specific strengths, 0-5 specific weaknesses, and detailed revision instructions "
        "if the score is below 0.7."
    ),
    agent=editor_critic,
    context=[writing_task],
    output_pydantic=EditorialReview,
    callback=task_callback,
)

final_summary_task = Task(
    description=(
        "Produce the final deliverable for the research on **{topic}**.\n\n"
        "Check the editorial review:\n"
        "- If approved=True: format and present the final polished report in clean Markdown. "
        "  Incorporate any light suggestions from the editor's revision_instructions.\n"
        "- If approved=False (requires_revision=True): carefully apply the editor's specific "
        "  revision_instructions to the report before presenting it.\n\n"
        "Format requirements:\n"
        "- Use proper Markdown: # for title, ## for sections, **bold** for key terms\n"
        "- End with a line: '---\\n*Quality score: X.XX | Reviewed by: Senior Editor*'"
    ),
    expected_output=(
        "The complete, final research report in clean Markdown format. "
        "Must include all sections, citations, and a quality certification footer. "
        "If revisions were requested, the final text must reflect those changes."
    ),
    agent=technical_writer,
    context=[writing_task, editing_task],
    # No output_pydantic — demonstrates plain string TaskOutput alongside typed outputs
    callback=task_callback,
)
