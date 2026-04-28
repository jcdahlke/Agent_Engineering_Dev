from pathlib import Path

# Must load dotenv BEFORE any crewai imports — CrewAI reads OPENAI_API_KEY at import time
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    openai_api_key: str = ""
    # SerperDevTool reads this at instantiation; empty string = tool disabled
    serper_api_key: str = ""

    researcher_model: str = "gpt-4o-mini"
    analyst_model: str = "gpt-4o-mini"
    writer_model: str = "gpt-4o"
    editor_model: str = "gpt-4o-mini"
    manager_model: str = "gpt-4o"

    max_rpm: int = 10
    # CrewAI long-term memory writes SQLite + vector files here at runtime
    memory_dir: str = ".crew_memory"
    verbose: bool = True

    model_config = {
        "env_file": Path(__file__).parent / ".env",
        "env_file_encoding": "utf-8",
    }


settings = Settings()
