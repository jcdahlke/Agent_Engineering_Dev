"""
Configuration — loaded once at import time.

IMPORTANT: load_dotenv() is called here, before any `from agents import ...`
statement, so the OpenAI SDK can find OPENAI_API_KEY in the environment when it
initialises its default HTTP client.
"""
from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv(Path(__file__).parent / ".env")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(Path(__file__).parent / ".env"),
        extra="ignore",
        case_sensitive=False,
    )

    # ── Required ──────────────────────────────────────────────────────────────
    openai_api_key: str = Field(...)

    # ── Per-agent model selection ─────────────────────────────────────────────
    supervisor_model: str = Field(default="gpt-4o-mini")
    researcher_model: str = Field(default="gpt-4o-mini")
    analyst_model: str = Field(default="gpt-4o")
    writer_model: str = Field(default="gpt-4o")
    critic_model: str = Field(default="gpt-4o-mini")
    guardrail_model: str = Field(default="gpt-4o-mini")

    # ── Research behavior ─────────────────────────────────────────────────────
    max_search_results: int = Field(default=5)
    max_revisions: int = Field(default=2)
    research_depth: str = Field(default="standard")  # minimal | standard | deep
    max_turns: int = Field(default=50)  # each tool call + each agent response = 1 turn


settings = Settings()
