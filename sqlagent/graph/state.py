"""QueryState — the TypedDict that flows through the LangGraph orchestrator.

Every graph node reads from this state and returns a partial update.
LangGraph merges the partial update into the full state automatically.
"""

from __future__ import annotations

from typing import Any, TypedDict


class SubQueryResult(TypedDict, total=False):
    """Result from a single sub-query in cross-source decomposition."""

    sub_query_id: str
    source_id: str
    nl_description: str
    sql: str
    rows: list[dict]
    columns: list[str]
    row_count: int
    latency_ms: int
    succeeded: bool
    error: str


class QueryState(TypedDict, total=False):
    """Full state for a single query flowing through the LangGraph.

    Every field is optional (total=False) so nodes can return partial updates.
    LangGraph merges partials into the accumulated state.
    """

    # ── Input ─────────────────────────────────────────────────────────────────
    nl_query: str
    session_id: str
    user_id: str
    workspace_id: str
    source_ids: list[str]  # available source_ids in this workspace

    # ── Routing (understand_node) ─────────────────────────────────────────────
    is_cross_source: bool
    is_compound_query: bool  # single-source query with multiple distinct analytical questions
    target_sources: list[str]  # which sources this query needs
    complexity: str  # "simple" | "moderate" | "complex"
    routing_reasoning: str

    # ── Schema pruning (prune_node) ───────────────────────────────────────────
    full_schema: dict[str, Any]  # SchemaSnapshot(s) as dict
    pruned_schema: dict[str, Any]  # After CHESS LSH
    columns_before: int
    columns_after: int
    selected_tables: list[str]
    pruning_reasoning: str
    soul_context: str  # SOUL enrichment injected into pruning

    # ── Example retrieval (retrieve_node) ─────────────────────────────────────
    similar_examples: list[dict]  # TrainingExample dicts with similarity
    example_count: int

    # ── Planning (plan_node) ──────────────────────────────────────────────────
    plan_strategy: str  # "direct", "join_and_aggregate", "decompose"
    plan_reasoning: str
    planned_tables: list[str]
    planned_joins: list[str]
    planned_filters: list[str]

    # ── Generation (generate_node) ────────────────────────────────────────────
    candidates: list[dict]  # Candidate dicts from parallel generators
    winner: dict | None  # Selected Candidate dict
    winner_generator: str
    selection_reasoning: str
    generation_tokens: int
    generation_cost_usd: float

    # ── Execution (execute_node) ──────────────────────────────────────────────
    sql: str  # Final SQL to execute / that was executed
    rows: list[dict]  # Result rows as dicts
    columns: list[str]  # Column names
    row_count: int
    execution_latency_ms: int
    execution_error: str  # Empty string if success

    # ── Correction (correct_node) ─────────────────────────────────────────────
    correction_round: int  # 0 = no correction yet
    max_corrections: int  # from config, default 3
    correction_stage: str  # "error_aware", "schema_aware", "db_confirmed"
    correction_error: str  # Error from failed correction

    # ── Cross-source decomposition ────────────────────────────────────────────
    entity_filters: list[str]  # Entities extracted by understand_node (e.g. ["Malaysia", "Vietnam"])
    decomposition_plan: dict | None  # DecompositionPlan as dict
    sub_queries: list[dict]  # SubProblem dicts
    sub_results: list[SubQueryResult]  # Results from parallel sub-queries
    synthesis_sql: str  # DuckDB synthesis query
    synthesis_result: dict | None  # Final synthesized result

    # ── Response (respond_node) ───────────────────────────────────────────────
    nl_response: str
    follow_ups: list[str]
    chart_config: dict | None  # {"type": "bar", "x": "col", "y": "col", ...}

    # ── Budget ────────────────────────────────────────────────────────────────
    tokens_used: int
    cost_usd: float
    budget_exhausted: bool

    # ── Trace events (accumulated by all nodes) ───────────────────────────────
    trace_events: list[dict]  # TraceNode dicts emitted by @traced_node

    # ── Semantic reasoning (semantic_resolve_node) ───────────────────────────────
    semantic_resolution: dict | None  # SemanticResolution as dict (cache hit or agent output)
    semantic_cache_hit: bool          # True if resolution came from cache

    # ── Learned context (from user corrections — injected into every generation) ─
    data_context_notes: list[str]  # Workspace-level semantic lessons learned

    # ── Schema pruning query ──────────────────────────────────────────────────
    nl_query_for_pruning: str  # Query used for LSH pruning (may differ from display)

    # ── Conversation history (multi-turn, passed from client) ─────────────────
    conversation_history: list[dict]  # [{role, text, sql, nl_response}, ...] last N turns
    intent: str  # "data_query" | "cross_source" | "conversational"
    conversational_answer: str  # Set when intent=conversational, skips SQL pipeline

    # ── Ora orchestrator ─────────────────────────────────────────────────────
    analytical_intent: str      # "correlation"|"comparison"|"trend"|"ranking"|"aggregate"|"describe"
    data_warnings: list[str]    # entities/concepts not found in any source — surfaced to user
    ora_reasoning: str          # Ora's full chain-of-thought (for trace)

    # ── Meta ──────────────────────────────────────────────────────────────────
    query_id: str
    succeeded: bool
    error: str
    started_at: str  # ISO timestamp
    completed_at: str  # ISO timestamp
    display_nl_query: str  # Clean user question (strips multi-turn context prefix)
