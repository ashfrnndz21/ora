"""Ora ReAct Controller — thin wrapper calling the OraOrchestrator.

The orchestration logic lives in sqlagent/agents/orchestrator.py.
This file provides the AgentTrace class and the ora_react entry point
that LangGraph calls.
"""

from __future__ import annotations

import time
from typing import Any

from sqlagent.graph.state import QueryState


class AgentTrace:
    """Dynamic trace collector — records agent interactions as they happen."""

    def __init__(self):
        self.events: list[dict] = []
        self._start = time.monotonic()

    def record(self, agent: str, action: str, status: str = "completed",
               input_context: str = "", output: str = "", reasoning: str = "",
               latency_ms: int = 0, tokens: int = 0, **extra):
        self.events.append({
            "agent": agent,
            "action": action,
            "status": status,
            "input_context": input_context[:300],
            "output": output[:300],
            "reasoning": reasoning[:300],
            "latency_ms": latency_ms,
            "tokens": tokens,
            "timestamp_ms": int((time.monotonic() - self._start) * 1000),
            # Legacy compat — UI reads "node" and "summary"
            "node": agent,
            "summary": f"{action}: {output[:80]}" if output else action,
            **{k: v for k, v in extra.items() if k not in ("agent", "action", "status")},
        })

    def to_trace_events(self) -> list[dict]:
        return self.events


async def ora_react(state: QueryState, services: Any) -> dict:
    """Entry point called by LangGraph. Delegates to OraOrchestrator."""
    from sqlagent.agents.orchestrator import OraOrchestrator

    trace = AgentTrace()
    orchestrator = OraOrchestrator(services, trace)
    return await orchestrator.run(state)
