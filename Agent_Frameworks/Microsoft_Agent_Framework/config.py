from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings

# Agent Framework does NOT auto-load .env — load explicitly before BaseSettings reads env vars
load_dotenv(Path(__file__).parent / ".env")


class Settings(BaseSettings):
    openai_api_key: str
    tavily_api_key: str = ""

    orchestrator_model: str = "gpt-4o"
    researcher_model: str   = "gpt-4o-mini"
    analyst_model: str      = "gpt-4o"
    writer_model: str       = "gpt-4o"
    critic_model: str       = "gpt-4o-mini"

    max_iterations: int  = 5
    research_depth: str  = "standard"   # quick | standard | deep

    model_config = {
        "env_file": Path(__file__).parent / ".env",
        "env_file_encoding": "utf-8",
    }


settings = Settings()
