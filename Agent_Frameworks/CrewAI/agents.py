"""
Agent definitions for the CrewAI research system.

Each Agent is defined declaratively with three core fields:
  - role      : the agent's job title / expertise label
  - goal      : what this agent is trying to accomplish (injected into system prompt)
  - backstory : personality and experience context (shapes the LLM's persona)

{topic} and {depth} are template variables filled at crew.kickoff(inputs={...}).

Additional parameters:
  - tools       : list of @tool functions or crewai_tools instances the agent may call
  - llm         : LLM object (wraps LiteLLM — "gpt-4o-mini" etc. work directly)
  - memory      : True enables short-term in-run memory for this agent
  - max_iter    : caps the ReAct loop iterations to prevent runaway agents
  - max_retry_limit : retries if structured output validation fails
  - verbose     : surface all agent reasoning to terminal when True
"""
from __future__ import annotations

from crewai import Agent, LLM

from config import settings
from tools import get_analyst_tools, get_researcher_tools

# ── LLM instances ──────────────────────────────────────────────────────────────
# CrewAI wraps LiteLLM; plain model name strings like "gpt-4o-mini" resolve to
# OpenAI by default. temperature=0.0–0.3 for factual agents, higher for writers.

_researcher_llm = LLM(model=settings.researcher_model, temperature=0.3)
_analyst_llm    = LLM(model=settings.analyst_model,    temperature=0.2)
_writer_llm     = LLM(model=settings.writer_model,     temperature=0.7)
_editor_llm     = LLM(model=settings.editor_model,     temperature=0.1)


# ── Agents ─────────────────────────────────────────────────────────────────────

senior_researcher = Agent(
    role="Senior Research Scientist",
    goal=(
        "Conduct comprehensive, multi-source research on {topic} at depth level {depth}. "
        "Search the web for recent news and articles, query arXiv for academic papers, "
        "and scrape key pages for full content. Prioritize authoritative, recent sources. "
        "Aim for at least 5 diverse sources; for 'deep' depth find 3+ academic papers."
    ),
    backstory=(
        "You are a veteran research scientist with 20 years of experience synthesizing "
        "information across disciplines. You know how to identify high-quality sources, "
        "cross-reference claims, and recognize when coverage is sufficient for a thorough "
        "report. You are rigorous about citing sources and noting the date of each finding."
    ),
    tools=get_researcher_tools(),
    llm=_researcher_llm,
    memory=True,
    verbose=settings.verbose,
    max_iter=15,
    max_retry_limit=3,
)

data_analyst = Agent(
    role="Data Analyst and Synthesizer",
    goal=(
        "Extract structured insights, key findings, statistics, and patterns from the "
        "research content gathered about {topic}. Quantify wherever possible — pull out "
        "specific numbers, percentages, dates, and named entities. Identify gaps in coverage."
    ),
    backstory=(
        "You are a meticulous data analyst who specializes in distilling large volumes "
        "of unstructured research into clean, actionable insights. You always ask: what "
        "is the evidence, what does it mean quantitatively, and what is missing? You never "
        "accept vague summaries — you demand specifics."
    ),
    tools=get_analyst_tools(),
    llm=_analyst_llm,
    memory=True,
    verbose=settings.verbose,
    max_iter=8,
    max_retry_limit=2,
)

technical_writer = Agent(
    role="Technical Research Writer",
    goal=(
        "Produce a well-structured, comprehensive research report on {topic} based on the "
        "analyzed findings. The report must have an executive summary, clearly labeled "
        "sections, inline citations, and a conclusion. Target 800-1200 words."
    ),
    backstory=(
        "You are an award-winning science communicator who translates complex research "
        "into clear, compelling prose for professional audiences. You structure reports "
        "rigorously — every claim is backed by the provided research, every section has "
        "a clear purpose, and the executive summary stands alone as a complete overview."
    ),
    tools=[],
    llm=_writer_llm,
    memory=True,
    verbose=settings.verbose,
    max_iter=5,
    max_retry_limit=2,
)

editor_critic = Agent(
    role="Senior Editor and Fact-Checker",
    goal=(
        "Review the research report on {topic} for accuracy, completeness, logical "
        "consistency, and writing quality. Score it on a 0.0–1.0 scale using the "
        "editorial rubric. Provide specific, actionable revision instructions if the "
        "score is below 0.7."
    ),
    backstory=(
        "You are a rigorous senior editor with a reputation for catching errors others "
        "miss. You evaluate content against a strict rubric, score it numerically, and "
        "give specific revision instructions rather than vague suggestions. You approve "
        "only what meets publication standards. You distinguish between minor polish "
        "issues and fundamental structural problems."
    ),
    tools=[],
    llm=_editor_llm,
    memory=True,
    verbose=settings.verbose,
    max_iter=5,
    max_retry_limit=2,
)
