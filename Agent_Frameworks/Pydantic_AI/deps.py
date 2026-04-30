"""
deps.py — Dependency injection container for Pydantic AI agents.

Pydantic AI's core pattern: a typed dataclass is created once in pipeline.py,
then passed to every agent.run(..., deps=deps) call. Inside tools and system
prompts, ctx: RunContext[ResearchDependencies] gives type-safe access via ctx.deps.

Benefits over global state:
  - IDE autocomplete on ctx.deps.http_client, ctx.deps.config, etc.
  - Easy to mock in tests (just swap the dataclass fields)
  - No hidden coupling — every agent's dependencies are explicit
"""
import uuid
from dataclasses import dataclass, field

import httpx

from config import ResearchConfig


@dataclass
class ResearchDependencies:
    http_client: httpx.AsyncClient
    config: ResearchConfig
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    sources: list[str] = field(default_factory=list)
    research_notes: dict[str, str] = field(default_factory=dict)
