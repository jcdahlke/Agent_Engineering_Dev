"""
Orchestrator step — decomposes the research topic into focused sub-questions.

Uses the LLM directly (no tools) to reason about what sub-questions will
best cover the topic. The output drives every subsequent step.
"""

from __future__ import annotations

import re

from llama_index.core.callbacks import CallbackManager
from llama_index.llms.openai import OpenAI

from config import settings

PLAN_PROMPT = """\
You are a research planning expert. Your job is to decompose a research topic
into {count} specific, focused sub-questions that together provide comprehensive
coverage of the topic.

Topic: {topic}
Research depth: {depth}

Guidelines:
- quick:    3 sub-questions, broad coverage
- standard: 4 sub-questions, balanced coverage
- deep:     5 sub-questions, thorough coverage

Return ONLY the sub-questions as a numbered list, one per line.
No preamble, no explanations — just the numbered questions.
"""

DEPTH_TO_COUNT = {"quick": 3, "standard": 4, "deep": 5}


def _parse_sub_questions(text: str) -> list[str]:
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    questions = []
    for line in lines:
        # Strip leading numbering like "1.", "1)", "•", "-"
        cleaned = re.sub(r"^[\d]+[.)]\s*|^[-•*]\s*", "", line).strip()
        if cleaned and "?" in cleaned:
            questions.append(cleaned)
    # Fallback: take any non-empty line if no "?" found
    if not questions:
        questions = [re.sub(r"^[\d]+[.)]\s*|^[-•*]\s*", "", l).strip() for l in lines if l.strip()]
    return questions[:settings.max_sub_questions]


async def run_orchestrator(
    topic: str,
    depth: str,
    callback_manager: CallbackManager,
) -> list[str]:
    count = DEPTH_TO_COUNT.get(depth, 4)

    llm = OpenAI(
        model=settings.orchestrator_model,
        temperature=0,
        callback_manager=callback_manager,
    )

    prompt = PLAN_PROMPT.format(topic=topic, depth=depth, count=count)
    response = await llm.acomplete(prompt)
    return _parse_sub_questions(response.text)
