"""Multi-agent handoff protocol — typed contracts for agent communication.

Every agent-to-agent message uses these dataclasses. No agent talks
directly to another — all communication flows through Ora.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AgentRequest:
    """What Ora sends to an agent."""

    from_agent: str = "ora"
    to_agent: str = ""
    action: str = ""
    context: dict = field(default_factory=dict)
    expectations: list[str] = field(default_factory=list)


@dataclass
class AgentResponse:
    """What an agent returns to Ora."""

    from_agent: str = ""
    status: str = "completed"  # completed | partial | failed
    result: dict = field(default_factory=dict)
    unresolved: list[str] = field(default_factory=list)
    confidence: float = 0.0
    reasoning: str = ""


@dataclass
class ValidationResult:
    """Result of Ora's validation at each handoff."""

    passed: bool = False
    issues: list[str] = field(default_factory=list)
    checks: list[dict] = field(default_factory=list)


@dataclass
class QueryDecomposition:
    """Ora's decomposition of a user query into logical parts."""

    parts: list[dict] = field(default_factory=list)
    # Each part: {"id": "A", "description": "...", "target_domain": "...", "target_source": "..."}
    entities_to_resolve: list[str] = field(default_factory=list)
    calculations_needed: list[str] = field(default_factory=list)
    comparison_type: str = ""
    raw_query: str = ""
    is_cross_source: bool = False  # True if parts target different data sources
    data_gaps: list[dict] = field(default_factory=list)  # schema gaps detected per part

    # ── Plan fields (Phase 0 — Ora's execution plan) ──────────────
    plan_understanding: str = ""        # what the user actually wants
    plan_approach: str = ""             # how Ora will answer using its agents
    plan_tables: list[str] = field(default_factory=list)  # which tables to use
    plan_filters_needed: bool = True    # whether entity filters are expected
    plan_limitations: str = ""          # what can't be fully answered and why
    plan_steps: list[dict] = field(default_factory=list)  # expected output per agent step
