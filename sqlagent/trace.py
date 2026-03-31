"""Trace system — collects execution traces from LangGraph runs.

TraceCollector: builds a trace tree from pipeline events + OTel spans
TraceStore: persists traces to SQLite for the Tasks view
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

import structlog

from sqlagent.models import Trace, TraceNode, TraceStatus

logger = structlog.get_logger()


# ═══════════════════════════════════════════════════════════════════════════════
# TRACE COLLECTOR
# ═══════════════════════════════════════════════════════════════════════════════


class TraceCollector:
    """Builds a Trace tree from pipeline trace_events accumulated in QueryState.

    After a LangGraph run, call build_trace(final_state) to construct
    the trace tree from the trace_events list in the state.
    """

    @staticmethod
    def build_trace(state: dict, workspace_id: str = "", user_id: str = "") -> Trace:
        """Build a Trace from the final QueryState after graph execution."""
        trace_events = state.get("trace_events", [])
        query_id = state.get("query_id", str(uuid.uuid4())[:12])

        # Build root node with children from trace events
        children = []
        for i, event in enumerate(trace_events):
            node = TraceNode(
                node_id=f"n{i + 1}",
                name=_node_name(event.get("node", ""), event),
                agent=_agent_for_node(event.get("node", "")),
                status=TraceStatus.COMPLETED
                if event.get("status") == "completed"
                else TraceStatus.FAILED,
                latency_ms=event.get("latency_ms", 0),
                tokens=event.get("tokens", 0),
                summary=event.get("summary", ""),
                detail=event,
            )

            # Handle nested children (e.g., fan_out with sub-queries)
            if "children" in event:
                for j, child_evt in enumerate(event["children"]):
                    node.children.append(
                        TraceNode(
                            node_id=f"n{i + 1}.{j + 1}",
                            name=child_evt.get("node", f"sub_{j}"),
                            status=TraceStatus.COMPLETED
                            if child_evt.get("status") == "completed"
                            else TraceStatus.FAILED,
                            summary=child_evt.get("summary", ""),
                            parent_id=node.node_id,
                        )
                    )

            children.append(node)

        # Use display_nl_query if set — it's the clean user question without multi-turn prefix
        display_query = state.get("display_nl_query") or state.get("nl_query", "")

        root = TraceNode(
            node_id="root",
            name=f"Query: {display_query[:50]}",
            agent="orchestrator",
            status=TraceStatus.COMPLETED if state.get("succeeded") else TraceStatus.FAILED,
            children=children,
            summary=f"{len(children)} steps",
        )

        # Calculate totals
        total_latency = sum(c.latency_ms for c in children)
        total_tokens = state.get("tokens_used", 0)
        total_cost = state.get("cost_usd", 0.0)

        return Trace(
            trace_id=f"qry_{query_id}",
            workspace_id=workspace_id,
            user_id=user_id,
            nl_query=display_query,
            root=root,
            status=TraceStatus.COMPLETED if state.get("succeeded") else TraceStatus.FAILED,
            started_at=datetime.fromisoformat(state["started_at"])
            if state.get("started_at")
            else datetime.now(timezone.utc),
            completed_at=datetime.fromisoformat(state["completed_at"])
            if state.get("completed_at")
            else datetime.now(timezone.utc),
            total_latency_ms=total_latency,
            total_tokens=total_tokens,
            total_cost_usd=total_cost,
            succeeded=state.get("succeeded", False),
            winner_generator=state.get("winner_generator", ""),
            correction_rounds=state.get("correction_round", 0),
            sql=state.get("sql", ""),
            row_count=state.get("row_count", 0),
            error=state.get("execution_error", "") or state.get("error", ""),
        )


def _node_name(node_key: str, event: dict | None = None) -> str:
    """Generate a natural, context-aware display name for a trace node.

    Uses the event summary to vary the label based on what actually happened,
    so users see meaningful, human phrasing — not the same static text every time.
    """
    import random as _random

    summary = (event or {}).get("summary", "")

    if node_key == "understand":
        if "cross" in summary.lower() or "multi" in summary.lower():
            return "Spans multiple datasets"
        opts = [
            "Figuring out what you need",
            "Reading your question",
            "Got it, thinking through this",
        ]
        return _random.choice(opts)

    if node_key == "prune":
        cols_after = (event or {}).get("columns_after", 0)
        tables = (event or {}).get("selected_tables", [])
        if cols_after and cols_after < 20:
            return f"Focused on {cols_after} relevant columns"
        if tables:
            return f"Narrowed down to {len(tables)} table{'s' if len(tables) != 1 else ''}"
        opts = ["Scanning the schema", "Finding what's relevant", "Cutting through the noise"]
        return _random.choice(opts)

    if node_key == "retrieve":
        count = (event or {}).get("example_count") or (
            int(summary.split()[0]) if summary and summary[0].isdigit() else 0
        )
        if count == 0:
            return "No past examples — reasoning from scratch"
        if count == 1:
            return "Found a useful reference query"
        return f"Found {count} helpful past {'queries' if count > 1 else 'query'}"

    if node_key == "plan":
        strategy = (event or {}).get("strategy", "")
        if "join" in strategy or "join" in summary.lower():
            return "Mapping out the joins"
        if "subquery" in strategy or "window" in strategy:
            return "Planning a multi-step approach"
        if "direct" in strategy or "simple" in summary.lower():
            return "Straightforward — going direct"
        opts = [
            "Thinking through the approach",
            "Planning how to answer this",
            "Mapping out the SQL",
        ]
        return _random.choice(opts)

    if node_key == "generate":
        if "3 candidates" in summary or "candidates" in summary:
            n = summary.split()[0] if summary[0].isdigit() else "multiple"
            return f"Wrote {n} SQL versions, picking the best"
        opts = ["Writing the SQL", "Crafting the query", "Building the SQL"]
        return _random.choice(opts)

    if node_key == "execute":
        rows = (event or {}).get("row_count", 0)
        if "error" in summary.lower() or "fail" in summary.lower():
            return "Hit an error — handing off to correction"
        if rows:
            return f"Got {rows:,} row{'s' if rows != 1 else ''} back"
        opts = ["Running the query", "Executing against the database", "Querying the data"]
        return _random.choice(opts)

    if node_key == "correct":
        stage = (event or {}).get("stage", "")
        if "schema" in stage:
            return "Rethinking with the full schema"
        if "db_confirmed" in stage:
            return "Trying a database-confirmed fix"
        opts = ["Something was off — adjusting", "Fixing and retrying", "Revising the query"]
        return _random.choice(opts)

    if node_key == "respond":
        if "context" in summary.lower():
            return "Answering from our conversation"
        if "chart" in summary.lower():
            return "Writing the answer + chart"
        opts = [
            "Putting the answer together",
            "Writing up the findings",
            "Summarising what I found",
        ]
        return _random.choice(opts)

    if node_key == "learn":
        if "skipped" in summary.lower():
            return "Nothing to save this time"
        return "Saved to memory"

    if node_key == "decompose":
        n = (event or {}).get("sub_query_count", 0)
        if n:
            return f"Split into {n} parallel queries"
        opts = ["Breaking this down", "Splitting across sources", "Decomposing the question"]
        return _random.choice(opts)

    if node_key == "fan_out":
        n = (event or {}).get("sub_query_count", 0)
        if n:
            return f"Running {n} queries in parallel"
        return "Running all sources at once"

    if node_key == "synthesize":
        rows = (event or {}).get("row_count", 0)
        if rows:
            return f"Joined results — {rows:,} rows"
        opts = ["Bringing the results together", "Merging across sources", "Joining everything up"]
        return _random.choice(opts)

    return node_key.replace("_", " ").title()


def _agent_for_node(node_key: str) -> str:
    """Map node to a short, readable agent label."""
    agents = {
        "understand": "",  # no badge needed — it's trivial routing
        "prune": "schema",
        "retrieve": "memory",
        "plan": "planner",
        "generate": "SQL agent",
        "execute": "",  # no badge — mechanical
        "correct": "self-heal",
        "respond": "writer",
        "learn": "memory",
        "decompose": "planner",
        "fan_out": "",
        "synthesize": "DuckDB",
    }
    return agents.get(node_key, "")


# ═══════════════════════════════════════════════════════════════════════════════
# TRACE STORE (SQLite persistence)
# ═══════════════════════════════════════════════════════════════════════════════


class TraceStore:
    """Persists traces to SQLite for the Tasks view."""

    def __init__(self, db_path: str = ""):
        import os

        if not db_path:
            db_path = os.path.join(os.path.expanduser("~"), ".sqlagent", "traces.db")
        self._db_path = db_path
        self._initialized = False

    async def init(self) -> None:
        if self._initialized:
            return
        import os
        import aiosqlite

        os.makedirs(os.path.dirname(self._db_path), exist_ok=True)
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS traces (
                    trace_id TEXT PRIMARY KEY,
                    workspace_id TEXT,
                    user_id TEXT,
                    nl_query TEXT,
                    sql TEXT,
                    succeeded INTEGER,
                    row_count INTEGER,
                    total_latency_ms INTEGER,
                    total_tokens INTEGER,
                    total_cost_usd REAL,
                    winner_generator TEXT,
                    correction_rounds INTEGER,
                    error TEXT,
                    trace_json TEXT,
                    created_at TEXT,
                    model_id TEXT DEFAULT '',
                    tokens_input INTEGER DEFAULT 0,
                    tokens_output INTEGER DEFAULT 0
                )
            """)
            # Migration: add new columns to existing databases that predate them
            for col_def in [
                "model_id TEXT DEFAULT ''",
                "tokens_input INTEGER DEFAULT 0",
                "tokens_output INTEGER DEFAULT 0",
            ]:
                try:
                    await db.execute(f"ALTER TABLE traces ADD COLUMN {col_def}")
                except Exception as exc:
                    logger.debug("trace.operation_failed", error=str(exc))
                    pass  # Already exists — ignore
            await db.commit()
        self._initialized = True

    async def save(self, trace: Trace) -> None:
        await self.init()
        import aiosqlite

        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                "INSERT OR REPLACE INTO traces VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    trace.trace_id,
                    trace.workspace_id,
                    trace.user_id,
                    trace.nl_query,
                    trace.sql,
                    int(trace.succeeded),
                    trace.row_count,
                    trace.total_latency_ms,
                    trace.total_tokens,
                    trace.total_cost_usd,
                    trace.winner_generator,
                    trace.correction_rounds,
                    trace.error,
                    json.dumps(trace.to_dict()),
                    trace.started_at.isoformat(),
                    trace.model_id,
                    trace.tokens_input,
                    trace.tokens_output,
                ),
            )
            await db.commit()

    async def get(self, trace_id: str) -> Trace | None:
        await self.init()
        import aiosqlite

        async with aiosqlite.connect(self._db_path) as db:
            cursor = await db.execute(
                "SELECT trace_json FROM traces WHERE trace_id = ?", (trace_id,)
            )
            row = await cursor.fetchone()
            if not row:
                return None
            return _trace_from_json(json.loads(row[0]))

    async def list_for_workspace(
        self,
        workspace_id: str,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict]:
        """List traces as summary dicts (for Tasks view table)."""
        await self.init()
        import aiosqlite

        async with aiosqlite.connect(self._db_path) as db:
            cursor = await db.execute(
                "SELECT trace_id, nl_query, sql, succeeded, row_count, "
                "total_latency_ms, total_cost_usd, winner_generator, "
                "correction_rounds, error, created_at, total_tokens, "
                "model_id, tokens_input, tokens_output "
                "FROM traces WHERE workspace_id = ? "
                "ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (workspace_id, limit, offset),
            )
            rows = await cursor.fetchall()
            return [
                {
                    "trace_id": r[0],
                    "nl_query": r[1],
                    "sql": r[2],
                    "succeeded": bool(r[3]),
                    "row_count": r[4],
                    "total_latency_ms": r[5],
                    "total_cost_usd": r[6],
                    "winner_generator": r[7],
                    "correction_rounds": r[8],
                    "error": r[9],
                    "created_at": r[10],
                    "total_tokens": r[11] if len(r) > 11 else 0,
                    "model_id": r[12] if len(r) > 12 else "",
                    "tokens_input": r[13] if len(r) > 13 else 0,
                    "tokens_output": r[14] if len(r) > 14 else 0,
                }
                for r in rows
            ]

    async def count(self, workspace_id: str) -> int:
        await self.init()
        import aiosqlite

        async with aiosqlite.connect(self._db_path) as db:
            cursor = await db.execute(
                "SELECT COUNT(*) FROM traces WHERE workspace_id = ?", (workspace_id,)
            )
            row = await cursor.fetchone()
            return row[0] if row else 0


def _trace_from_json(data: dict) -> Trace:
    """Reconstruct a Trace from its JSON dict."""
    root_data = data.get("root")
    root = _node_from_json(root_data) if root_data else None

    return Trace(
        trace_id=data.get("trace_id", ""),
        workspace_id=data.get("workspace_id", ""),
        user_id=data.get("user_id", ""),
        nl_query=data.get("nl_query", ""),
        root=root,
        status=TraceStatus(data.get("status", "completed")),
        total_latency_ms=data.get("total_latency_ms", 0),
        total_tokens=data.get("total_tokens", 0),
        total_cost_usd=data.get("total_cost_usd", 0.0),
        succeeded=data.get("succeeded", False),
        winner_generator=data.get("winner_generator", ""),
        correction_rounds=data.get("correction_rounds", 0),
        sql=data.get("sql", ""),
        row_count=data.get("row_count", 0),
        error=data.get("error", ""),
    )


def _node_from_json(data: dict) -> TraceNode:
    """Reconstruct a TraceNode from JSON."""
    return TraceNode(
        node_id=data.get("node_id", ""),
        name=data.get("name", ""),
        agent=data.get("agent", ""),
        status=TraceStatus(data.get("status", "completed")),
        latency_ms=data.get("latency_ms", 0),
        tokens=data.get("tokens", 0),
        cost_usd=data.get("cost_usd", 0.0),
        summary=data.get("summary", ""),
        detail=data.get("detail", {}),
        children=[_node_from_json(c) for c in data.get("children", [])],
    )
