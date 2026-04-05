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
    # Each part: {"id": "A", "description": "...", "needs_calculation": "...", "depends_on": []}
    entities_to_resolve: list[str] = field(default_factory=list)
    calculations_needed: list[str] = field(default_factory=list)
    comparison_type: str = ""
    raw_query: str = ""
