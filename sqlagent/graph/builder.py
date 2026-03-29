"""Compile the LangGraph StateGraph for query orchestration.

This is the heart of sqlagent. The graph routes queries through:
- Simple path: understand → prune → retrieve → plan → generate → execute → respond → learn
- Cross-source: understand → decompose → fan_out → synthesize → respond → learn
- Correction: execute → correct → execute (retry, up to max_corrections)
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import StateGraph, END

from sqlagent.graph.state import QueryState
from sqlagent.graph.nodes import (
    make_understand_node,
    make_prune_node,
    make_retrieve_node,
    make_plan_node,
    make_generate_node,
    make_execute_node,
    make_correct_node,
    make_respond_node,
    make_learn_node,
    make_decompose_node,
    make_fan_out_node,
    make_synthesize_node,
)


# ═══════════════════════════════════════════════════════════════════════════════
# CONDITIONAL EDGE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def route_after_understand(state: QueryState) -> str:
    """After understanding: route to simple pipeline or cross-source decomposition."""
    if state.get("is_cross_source"):
        return "decompose"
    return "prune"


def route_after_execute(state: QueryState) -> str:
    """After execution: success → respond, error → correct (if budget allows)."""
    error = state.get("execution_error", "")
    if not error:
        return "respond"

    # Check if we can still correct
    correction_round = state.get("correction_round", 0)
    max_corrections = state.get("max_corrections", 3)
    if correction_round < max_corrections:
        return "correct"

    # Max corrections exceeded — respond with error
    return "respond"


def route_after_correct(state: QueryState) -> str:
    """After correction: always retry execution."""
    return "execute"


# ═══════════════════════════════════════════════════════════════════════════════
# GRAPH COMPILATION
# ═══════════════════════════════════════════════════════════════════════════════

def compile_query_graph(services: Any) -> Any:
    """Compile the full query orchestration graph.

    Args:
        services: PipelineServices with llm, connectors, ensemble, policy, etc.

    Returns:
        A compiled LangGraph that can be invoked with:
            result = await graph.ainvoke(initial_state)
    """
    graph = StateGraph(QueryState)

    # ── Add all nodes ─────────────────────────────────────────────────────────
    graph.add_node("understand", make_understand_node(services))
    graph.add_node("prune", make_prune_node(services))
    graph.add_node("retrieve", make_retrieve_node(services))
    graph.add_node("plan", make_plan_node(services))
    graph.add_node("generate", make_generate_node(services))
    graph.add_node("execute", make_execute_node(services))
    graph.add_node("correct", make_correct_node(services))
    graph.add_node("respond", make_respond_node(services))
    graph.add_node("learn", make_learn_node(services))
    graph.add_node("decompose", make_decompose_node(services))
    graph.add_node("fan_out", make_fan_out_node(services))
    graph.add_node("synthesize", make_synthesize_node(services))

    # ── Entry point ───────────────────────────────────────────────────────────
    graph.set_entry_point("understand")

    # ── Conditional: understand → simple or cross-source ──────────────────────
    graph.add_conditional_edges("understand", route_after_understand, {
        "prune": "prune",
        "decompose": "decompose",
    })

    # ── Simple path (linear) ──────────────────────────────────────────────────
    graph.add_edge("prune", "retrieve")
    graph.add_edge("retrieve", "plan")
    graph.add_edge("plan", "generate")
    graph.add_edge("generate", "execute")

    # ── After execute: success → respond, error → correct ─────────────────────
    graph.add_conditional_edges("execute", route_after_execute, {
        "respond": "respond",
        "correct": "correct",
    })

    # ── After correct: retry execution ────────────────────────────────────────
    graph.add_conditional_edges("correct", route_after_correct, {
        "execute": "execute",
    })

    # ── Cross-source path ─────────────────────────────────────────────────────
    graph.add_edge("decompose", "fan_out")
    graph.add_edge("fan_out", "synthesize")
    graph.add_edge("synthesize", "respond")

    # ── Terminal ──────────────────────────────────────────────────────────────
    graph.add_edge("respond", "learn")
    graph.add_edge("learn", END)

    return graph.compile()
