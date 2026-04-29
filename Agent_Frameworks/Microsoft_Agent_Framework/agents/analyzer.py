"""
Analyzer agent — extracts structured insights from raw research content.

Microsoft Agent Framework patterns demonstrated:
  - agent.run() returns an AgentResult; .text gives the response string
  - The analyzer receives raw content in its prompt and returns structured text
  - extract_key_facts tool assists with structured extraction
  - Structured output is achieved through prompt engineering (explicit output
    format requirements) rather than with_structured_output — this is the
    Agent Framework approach for portable structured responses.
"""
from __future__ import annotations

from agent_framework.openai import OpenAIChatClient

from config import settings
from tools import get_analyzer_tools

_ANALYZER_INSTRUCTIONS = """\
You are a Research Analyst. Extract structured insights from the provided content.

Your tasks:
  1. Identify the 5-15 most important findings and key facts.
  2. Extract any statistics, comparisons, or quantitative data.
  3. Assess the overall confidence in the research coverage (0.0–1.0).
  4. Note any gaps that need more research.

Output format — always structure your response EXACTLY as:

KEY FINDINGS:
- [specific factual bullet 1]
- [specific factual bullet 2]
...

DATA POINTS:
- [any statistics, numbers, or comparisons found]

CONFIDENCE: [0.0-1.0] — [brief justification]

GAPS: [list any missing information that would improve the report]

ANALYSIS SUMMARY: [2-3 sentence conclusion synthesizing the above]

Be specific and quantitative. Vague findings like "there are many benefits" are not useful.
"""


def build_analyzer():
    """Build and return the Analyzer agent."""
    client = OpenAIChatClient(
        model=settings.analyst_model,
        api_key=settings.openai_api_key,
    )
    return client.as_agent(
        name="Analyzer",
        instructions=_ANALYZER_INSTRUCTIONS,
        tools=get_analyzer_tools(),
    )
