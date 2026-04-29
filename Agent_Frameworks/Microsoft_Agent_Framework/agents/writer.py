"""
Writer agent — synthesizes research findings into a structured report.

Microsoft Agent Framework patterns demonstrated:
  - publish_report tool has approval_mode="always_require" — the framework
    pauses before executing it and prompts for human confirmation.
  - This is the primary HITL mechanism in Agent Framework: tool-level approval
    rather than graph-level interrupt() as in LangGraph.
  - The writer is instructed to call publish_report when done, which triggers
    the HITL gate automatically — no special runner logic required.

Contrast with LangGraph:
  - LangGraph: interrupt() pauses the entire graph mid-node; the graph state
    is preserved and Command(resume=...) resumes it.
  - Agent Framework: the framework pauses only for the specific tool call.
    The agent's internal state (conversation) is preserved automatically.
"""
from __future__ import annotations

from agent_framework.openai import OpenAIChatClient

from config import settings
from tools import get_writer_tools

_WRITER_INSTRUCTIONS = """\
You are a Research Writer. Produce a comprehensive, well-structured research report.

Requirements:
  - Clear, academic-yet-accessible prose
  - Introduction + 3-5 thematic body sections + Conclusion
  - Use Markdown formatting (## headings, **bold** key terms, bullet lists)
  - Cite sources inline where relevant (include URLs or author names)
  - Address any specific critique points from previous review cycles
  - Aim for 800-1500 words

Report structure:
  # [Report Title]

  **Executive Summary:** [2-3 sentence overview]

  ## Introduction
  [Context and scope of the research]

  ## [Theme Section 1]
  [Body content with evidence]

  ## [Theme Section 2-4]
  [Body content with evidence]

  ## Conclusion
  [Synthesis, implications, and future directions]

  ## Sources
  - [URL or reference 1]
  - [URL or reference 2]

After writing the complete report, call publish_report with the full Markdown text.
The publish_report tool requires human approval — this is intentional and demonstrates
Agent Framework's tool-level HITL capability.
"""


def build_writer():
    """Build and return the Writer agent."""
    client = OpenAIChatClient(
        model=settings.writer_model,
        api_key=settings.openai_api_key,
    )
    return client.as_agent(
        name="Writer",
        instructions=_WRITER_INSTRUCTIONS,
        tools=get_writer_tools(),
    )
