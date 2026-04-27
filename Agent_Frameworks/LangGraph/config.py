from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    openai_api_key: str
    tavily_api_key: str = ""

    supervisor_model: str = "gpt-4o-mini"
    researcher_model: str = "gpt-4o-mini"
    analyst_model: str = "gpt-4o"
    writer_model: str = "gpt-4o"
    critic_model: str = "gpt-4o-mini"

    max_iterations: int = 5
    research_depth: str = "standard"

    model_config = {
        "env_file": Path(__file__).parent / ".env",
        "env_file_encoding": "utf-8",
    }


settings = Settings()
