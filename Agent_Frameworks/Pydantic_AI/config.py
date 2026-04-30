from pathlib import Path
from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv(Path(__file__).parent / ".env")

_ENV_FILE = str(Path(__file__).parent / ".env")


class ResearchConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_ENV_FILE,
        extra="ignore",
        case_sensitive=False,
    )

    openai_api_key: str = Field(...)

    supervisor_model: str = Field(default="gpt-4o-mini")
    researcher_model: str = Field(default="gpt-4o-mini")
    analyst_model: str = Field(default="gpt-4o")
    writer_model: str = Field(default="gpt-4o")
    critic_model: str = Field(default="gpt-4o-mini")

    max_revisions: int = Field(default=2)
    research_depth: str = Field(default="standard")


config = ResearchConfig()
