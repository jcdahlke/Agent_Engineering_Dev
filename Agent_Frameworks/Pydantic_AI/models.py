from pydantic import BaseModel, Field


class ResearchFindings(BaseModel):
    """Structured output returned by the Researcher agent."""

    sources: list[str] = Field(description="URLs of pages found during research")
    raw_content: list[str] = Field(description="Extracted text chunks from sources")
    search_queries_run: int = Field(description="Number of search queries executed")
    coverage_summary: str = Field(description="1-2 sentence summary of what was found")


class AnalysisResult(BaseModel):
    """Structured output returned by the Analyzer agent."""

    key_findings: list[str] = Field(description="5-12 specific, factual findings")
    themes: list[str] = Field(description="Overarching themes across the research")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in the analysis (0-1)")
    needs_more_research: bool = Field(description="Whether additional research would help")
    analysis_summary: str = Field(description="Concise narrative summary of the analysis")


class ReportSection(BaseModel):
    """A single section within the written report."""

    title: str
    content: str = Field(description="Markdown-formatted prose for this section")


class WrittenReport(BaseModel):
    """Structured output returned by the Writer agent."""

    title: str
    executive_summary: str = Field(description="2-3 sentence high-level summary")
    sections: list[ReportSection] = Field(description="3-5 thematic sections")
    conclusion: str = Field(description="Synthesizing conclusion paragraph")
    citations: list[str] = Field(description="Source URLs cited in the report")
    word_count_estimate: int = Field(description="Approximate word count of the report")


class CritiqueResult(BaseModel):
    """Structured output returned by the Critic agent."""

    score: float = Field(ge=0.0, le=1.0, description="Quality score (0-1)")
    approved: bool = Field(description="True if score >= 0.7 and report is publication-ready")
    strengths: list[str] = Field(description="Specific strengths of the report")
    weaknesses: list[str] = Field(description="Specific weaknesses or gaps")
    revision_instructions: str = Field(
        description="Concrete instructions for improvement (empty string if approved)"
    )
