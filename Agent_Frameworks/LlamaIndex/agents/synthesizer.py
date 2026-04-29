"""
Synthesizer step — merges findings using LlamaIndex's RouterQueryEngine.

Demonstrates RouterQueryEngine with LLMSingleSelector:
- Builds two indexes from the same content: VectorStoreIndex (dense retrieval)
  and SummaryIndex (sequential summarization).
- RouterQueryEngine uses an LLM to select the best engine per query.
  Dense retrieval for specific facts; Summary for broad thematic synthesis.
- This is uniquely LlamaIndex — no other framework has a first-class
  LLM-routing selector between index types.
"""

from __future__ import annotations

from llama_index.core import Document, Settings, SummaryIndex, VectorStoreIndex
from llama_index.core.callbacks import CallbackManager
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector
from llama_index.core.tools import QueryEngineTool
from llama_index.llms.openai import OpenAI

from config import settings
from workflow import SubAnswer

THEMATIC_QUERIES = [
    "What are the key findings and main conclusions from all the research?",
    "What are the most significant trends, patterns, or developments in this area?",
    "What are the limitations, challenges, or open questions identified?",
    "What is the practical significance and real-world impact of these findings?",
]


async def run_synthesizer(
    topic: str,
    answers: list[SubAnswer],
    raw_chunks: list[str],
    callback_manager: CallbackManager,
) -> tuple[list[str], list[str], int]:
    llm = OpenAI(
        model=settings.synthesizer_model,
        temperature=0.2,
        callback_manager=callback_manager,
    )
    Settings.llm = llm
    Settings.callback_manager = callback_manager

    # Combine sub-question answers + raw web chunks into one document set
    combined_texts = [f"Q: {a.question}\nA: {a.answer}" for a in answers]
    combined_texts.extend(raw_chunks)

    docs = [Document(text=t) for t in combined_texts if t.strip()]

    if not docs:
        return (
            ["Insufficient data for synthesis."],
            ["No themes identified"],
            0,
        )

    splitter = SentenceSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    # Build two indexes from the same content
    vector_index = VectorStoreIndex.from_documents(
        docs, transformations=[splitter], show_progress=False
    )
    summary_index = SummaryIndex.from_documents(docs, show_progress=False)

    # Wrap both as QueryEngineTool with descriptions that guide the selector
    vector_tool = QueryEngineTool.from_defaults(
        query_engine=vector_index.as_query_engine(
            similarity_top_k=settings.similarity_top_k
        ),
        name="vector_retrieval",
        description=(
            "Useful for specific factual questions, precise evidence retrieval, "
            "and queries about particular details or data points."
        ),
    )
    summary_tool = QueryEngineTool.from_defaults(
        query_engine=summary_index.as_query_engine(
            response_mode="tree_summarize"
        ),
        name="summary_synthesis",
        description=(
            "Useful for broad thematic questions, overall summaries, trend analysis, "
            "and questions requiring a high-level synthesis of the full body of research."
        ),
    )

    # RouterQueryEngine: LLMSingleSelector picks the best tool per query.
    # The selector's prompt asks the LLM to choose based on the tool descriptions.
    router = RouterQueryEngine(
        selector=LLMSingleSelector.from_defaults(llm=llm),
        query_engine_tools=[vector_tool, summary_tool],
        verbose=False,
    )

    merged_findings: list[str] = []
    query_count = 0

    for query in THEMATIC_QUERIES:
        try:
            response = await router.aquery(query)
            text = str(response).strip()
            if text:
                merged_findings.append(text)
                query_count += 1
        except Exception as e:
            merged_findings.append(f"[Synthesis error: {e}]")

    # Extract key themes from the merged findings
    key_themes = await _extract_themes(topic, merged_findings, llm)

    return merged_findings, key_themes, query_count


async def _extract_themes(
    topic: str, findings: list[str], llm: OpenAI
) -> list[str]:
    if not findings:
        return []

    combined = "\n\n".join(findings[:3])
    prompt = (
        f"Based on the following research findings about '{topic}', "
        "list 3-5 key themes as short phrases (one per line, no numbering):\n\n"
        f"{combined[:2000]}"
    )
    try:
        response = await llm.acomplete(prompt)
        lines = [l.strip("•-* ") for l in response.text.strip().splitlines() if l.strip()]
        return lines[:5]
    except Exception:
        return ["Key findings", "Main trends", "Research implications"]
