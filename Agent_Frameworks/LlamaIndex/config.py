import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings

# Load .env into os.environ so the openai SDK and LlamaIndex internals
# can read OPENAI_API_KEY directly from the environment.
load_dotenv(Path(__file__).parent / ".env")


class Settings(BaseSettings):
    openai_api_key: str

    # Model assignments per agent (cheaper models for faster/cheaper steps)
    orchestrator_model: str = "gpt-4o-mini"
    researcher_model: str = "gpt-4o-mini"
    analyst_model: str = "gpt-4o"
    synthesizer_model: str = "gpt-4o"
    writer_model: str = "gpt-4o"

    # Embedding
    embedding_model: str = "text-embedding-3-small"

    # RAG tuning
    similarity_top_k: int = 4
    chunk_size: int = 512
    chunk_overlap: int = 64

    # Workflow tuning
    max_sub_questions: int = 5
    workflow_timeout: int = 300

    model_config = {
        "env_file": Path(__file__).parent / ".env",
        "env_file_encoding": "utf-8",
    }


settings = Settings()
