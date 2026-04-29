"""
ResearchContext — the single shared mutable state object that flows through the
entire agent pipeline via RunContextWrapper[ResearchContext].

Every @function_tool receives ctx: RunContextWrapper[ResearchContext] as its
first parameter and mutates ctx.context directly. This is how the OpenAI Agents
SDK carries state across agent handoffs without needing an external database.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ResearchContext:
    # ── Input (set before run starts) ─────────────────────────────────────────
    topic: str = ""
    depth: str = "standard"            # minimal | standard | deep

    # ── Pipeline state (mutated by tools) ─────────────────────────────────────
    research_plan: str = ""
    raw_research: dict[str, str] = field(default_factory=dict)   # key → content
    source_urls: list[str] = field(default_factory=list)

    analysis: str = ""                 # prose analysis saved by Analyzer
    analysis_structured: dict = field(default_factory=dict)      # Pydantic dump

    draft_report: str = ""             # latest Writer draft
    critique: str = ""                 # latest Critic feedback
    quality_score: float = 0.0         # 0.0–10.0 from Critic

    # ── Control ───────────────────────────────────────────────────────────────
    revision_count: int = 0
    max_revisions: int = 2
    final_report: str = ""
    completed: bool = False

    # ── Metadata ──────────────────────────────────────────────────────────────
    agents_visited: list[str] = field(default_factory=list)
    agent_log: list[str] = field(default_factory=list)

    def log(self, message: str) -> None:
        """Append a timestamped log entry (used by tools for traceability)."""
        self.agent_log.append(message)

    def visit(self, agent_name: str) -> None:
        """Record that an agent was invoked (called from hooks)."""
        if not self.agents_visited or self.agents_visited[-1] != agent_name:
            self.agents_visited.append(agent_name)
