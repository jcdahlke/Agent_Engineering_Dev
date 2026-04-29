"""
RAG Analyst step — the heart of LlamaIndex's unique capabilities.

Demonstrates three LlamaIndex-specific RAG primitives:
1. VectorStoreIndex: builds an in-memory vector index from web-gathered text
2. QueryEngineTool: wraps the index as a tool an agent or query engine can call
3. SubQuestionQueryEngine: given a complex question, automatically decomposes it
   into sub-questions, queries the vector index for each, and synthesizes a
   unified answer. There is no equivalent primitive in LangGraph or CrewAI.

Uses OpenAI text-embedding-3-small for dense retrieval.
"""

from __future__ import annotations

from llama_index.core import Document, Settings, VectorStoreIndex
from llama_index.core.callbacks import CallbackManager
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.question_gen import LLMQuestionGenerator
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.core.tools import QueryEngineTool
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

from config import settings
from workflow import SubAnswer


async def run_rag_analyst(
    topic: str,
    sub_questions: list[str],
    raw_chunks: list[str],
    callback_manager: CallbackManager,
) -> list[SubAnswer]:
    # Configure LlamaIndex global Settings before building any index.
    # This must be done here (not at import time) so the API key is loaded.
    llm = OpenAI(
        model=settings.analyst_model,
        temperature=0,
        callback_manager=callback_manager,
    )
    embed_model = OpenAIEmbedding(
        model=settings.embedding_model,
        callback_manager=callback_manager,
    )
    Settings.llm = llm
    Settings.embed_model = embed_model
    Settings.chunk_size = settings.chunk_size
    Settings.chunk_overlap = settings.chunk_overlap
    Settings.callback_manager = callback_manager

    # Build Documents from raw text chunks
    docs = [
        Document(text=chunk, metadata={"source_index": i})
        for i, chunk in enumerate(raw_chunks)
        if chunk.strip()
    ]

    if not docs:
        return [
            SubAnswer(
                question=q,
                answer="Insufficient data gathered for RAG analysis.",
                confidence=0.1,
            )
            for q in sub_questions
        ]

    # VectorStoreIndex: uses SimpleVectorStore (pure-Python, bundled in
    # llama-index-core) with real OpenAI embeddings. No FAISS C++ dependency.
    splitter = SentenceSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    index = VectorStoreIndex.from_documents(
        docs,
        transformations=[splitter],
        show_progress=False,
    )

    # Wrap the vector index as a QueryEngineTool — this is how LlamaIndex
    # exposes indexes to agents and composite query engines.
    vector_tool = QueryEngineTool.from_defaults(
        query_engine=index.as_query_engine(
            similarity_top_k=settings.similarity_top_k,
            streaming=False,
        ),
        name="research_knowledge_base",
        description=(
            f"Contains web-gathered research about '{topic}'. "
            "Use for specific factual questions and direct evidence retrieval."
        ),
    )

    # SubQuestionQueryEngine: LlamaIndex's flagship query engine.
    # It automatically decomposes each incoming question into sub-queries,
    # runs each against the underlying tool, then synthesizes a final answer.
    # LLMQuestionGenerator (bundled in llama-index-core) is passed explicitly
    # to avoid a hard dependency on the optional llama-index-question-gen-openai package.
    question_gen = LLMQuestionGenerator.from_defaults(llm=llm)
    sub_question_engine = SubQuestionQueryEngine.from_defaults(
        query_engine_tools=[vector_tool],
        question_gen=question_gen,
        llm=llm,
        verbose=False,
        use_async=True,
    )

    answers: list[SubAnswer] = []
    for question in sub_questions:
        try:
            response = await sub_question_engine.aquery(question)
            answer_text = str(response).strip()
            confidence = 0.85 if len(answer_text) > 100 else 0.5
        except Exception as e:
            answer_text = f"Analysis error: {e}"
            confidence = 0.1

        answers.append(SubAnswer(
            question=question,
            answer=answer_text,
            confidence=confidence,
        ))

    return answers
