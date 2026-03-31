"""OpenTelemetry + Prometheus + Audit — all observability in one file.

This is NOT optional. Every query gets:
- OTel span tree (exportable to Jaeger, Langfuse, etc.)
- Prometheus metrics (queries, latency, corrections, cost)
- SQLite audit log (immutable per-query record)
"""

from __future__ import annotations

import collections
import functools
import json as _json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable

import structlog

logger = structlog.get_logger()

# ═══════════════════════════════════════════════════════════════════════════════
# OPENTELEMETRY SETUP
# ═══════════════════════════════════════════════════════════════════════════════

_tracer = None
_meter = None
_trace_buffer: collections.deque = collections.deque(maxlen=200)


def setup_telemetry(config: Any) -> None:
    """Initialize OTel TracerProvider + MeterProvider.

    Call once at server startup. Exports to OTLP endpoint if configured,
    otherwise falls back to console (dev mode).
    """
    global _tracer, _meter

    from opentelemetry import trace as otel_trace, metrics as otel_metrics
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import (
        BatchSpanProcessor,
        ConsoleSpanExporter,
    )
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.resources import Resource

    resource = Resource.create({"service.name": "sqlagent", "service.version": "2.0.0"})

    # Tracer
    provider = TracerProvider(resource=resource)
    if config.otel_endpoint:
        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

            exporter = OTLPSpanExporter(endpoint=config.otel_endpoint)
            provider.add_span_processor(BatchSpanProcessor(exporter))
        except Exception as e:
            logger.warn("OTLP exporter failed, falling back to console", error=str(e))
            provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
    else:
        # Dev mode: just log to console (or nowhere for tests)
        pass

    otel_trace.set_tracer_provider(provider)
    _tracer = otel_trace.get_tracer("sqlagent", "2.0.0")

    # Meter
    meter_provider = MeterProvider(resource=resource)
    otel_metrics.set_meter_provider(meter_provider)
    _meter = otel_metrics.get_meter("sqlagent", "2.0.0")

    logger.info("telemetry.initialized", endpoint=config.otel_endpoint or "console")


def get_tracer():
    """Get the sqlagent OTel tracer. Returns a no-op tracer if not initialized."""
    global _tracer
    if _tracer is None:
        from opentelemetry import trace as otel_trace

        _tracer = otel_trace.get_tracer("sqlagent", "2.0.0")
    return _tracer


# ═══════════════════════════════════════════════════════════════════════════════
# @traced_node DECORATOR
# ═══════════════════════════════════════════════════════════════════════════════


def traced_node(node_name: str):
    """Decorator for LangGraph node functions.

    Wraps with:
    1. OTel span (child of current trace context)
    2. Timing
    3. Buffer entry for /debug/traces

    Usage:
        @traced_node("prune")
        async def prune_node(state):
            ...
    """

    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        async def wrapper(state, *args, **kwargs):
            tracer = get_tracer()
            with tracer.start_as_current_span(f"sqlagent.{node_name}") as span:
                started = time.monotonic()
                try:
                    result = await fn(state, *args, **kwargs)

                    latency_ms = int((time.monotonic() - started) * 1000)
                    span.set_attribute("sqlagent.node", node_name)
                    span.set_attribute("sqlagent.latency_ms", latency_ms)

                    # Add any trace_events from the result to span attributes
                    events = result.get("trace_events", [])
                    if events:
                        last = events[-1]
                        for k, v in last.items():
                            if isinstance(v, (str, int, float, bool)):
                                span.set_attribute(f"sqlagent.{k}", v)

                    # Buffer for /debug/traces
                    _trace_buffer.append(
                        {
                            "node": node_name,
                            "latency_ms": latency_ms,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "query_id": state.get("query_id", ""),
                        }
                    )

                    return result

                except Exception as e:
                    from opentelemetry.trace import StatusCode

                    span.set_status(StatusCode.ERROR, str(e))
                    span.record_exception(e)
                    _trace_buffer.append(
                        {
                            "node": node_name,
                            "error": str(e),
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "query_id": state.get("query_id", ""),
                        }
                    )
                    raise

        return wrapper

    return decorator


def get_recent_traces(limit: int = 50) -> list[dict]:
    """Return recent trace events from the in-memory buffer."""
    return list(_trace_buffer)[-limit:]


# ═══════════════════════════════════════════════════════════════════════════════
# PROMETHEUS METRICS
# ═══════════════════════════════════════════════════════════════════════════════

try:
    from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

    QUERY_TOTAL = Counter(
        "sqlagent_queries_total",
        "Total queries executed",
        ["source_id", "succeeded", "generator"],
    )
    QUERY_LATENCY = Histogram(
        "sqlagent_query_latency_seconds",
        "Query latency in seconds",
        ["source_id"],
        buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0],
    )
    CORRECTION_TOTAL = Counter(
        "sqlagent_corrections_total",
        "Total correction attempts",
        ["stage"],
    )
    COST_TOTAL = Counter(
        "sqlagent_cost_usd_total",
        "Total LLM cost in USD",
        ["source_id", "model"],
    )
    TRAINING_PAIRS = Gauge(
        "sqlagent_training_pairs_total",
        "Total training pairs in vector store",
        ["source_id"],
    )
    ACTIVE_SESSIONS = Gauge(
        "sqlagent_active_sessions",
        "Currently active query sessions",
    )

    def record_query_metrics(
        source_id: str,
        succeeded: bool,
        generator: str,
        latency_s: float,
        cost_usd: float,
        model: str,
        correction_stages: list[str] | None = None,
    ) -> None:
        """Record metrics for a completed query."""
        QUERY_TOTAL.labels(
            source_id=source_id,
            succeeded=str(succeeded).lower(),
            generator=generator,
        ).inc()
        QUERY_LATENCY.labels(source_id=source_id).observe(latency_s)
        COST_TOTAL.labels(source_id=source_id, model=model).inc(cost_usd)
        for stage in correction_stages or []:
            CORRECTION_TOTAL.labels(stage=stage).inc()

    def get_prometheus_metrics() -> bytes:
        """Generate Prometheus text format output."""
        return generate_latest()

    def get_prometheus_content_type() -> str:
        return CONTENT_TYPE_LATEST

except ImportError:
    # Prometheus not installed — metrics are no-ops
    def record_query_metrics(**kwargs) -> None:
        pass

    def get_prometheus_metrics() -> bytes:
        return b"# prometheus-client not installed\n"

    def get_prometheus_content_type() -> str:
        return "text/plain"


# ═══════════════════════════════════════════════════════════════════════════════
# AUDIT LOG (SQLite, immutable)
# ═══════════════════════════════════════════════════════════════════════════════


class AuditLog:
    """Append-only SQLite audit trail for every query."""

    def __init__(self, db_path: str = ""):
        import os

        if not db_path:
            db_path = os.path.join(os.path.expanduser("~"), ".sqlagent", "audit.db")
        self._db_path = db_path

    async def init(self) -> None:
        """Create audit table if not exists."""
        import aiosqlite
        import os

        os.makedirs(os.path.dirname(self._db_path), exist_ok=True)
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS audit (
                    query_id TEXT PRIMARY KEY,
                    workspace_id TEXT,
                    user_id TEXT,
                    nl_query TEXT,
                    sql TEXT,
                    succeeded INTEGER,
                    row_count INTEGER,
                    cost_usd REAL,
                    latency_ms INTEGER,
                    corrections INTEGER,
                    generator TEXT,
                    error TEXT,
                    created_at TEXT
                )
            """)
            await db.commit()

    async def record(
        self,
        query_id: str,
        workspace_id: str = "",
        user_id: str = "",
        nl_query: str = "",
        sql: str = "",
        succeeded: bool = False,
        row_count: int = 0,
        cost_usd: float = 0.0,
        latency_ms: int = 0,
        corrections: int = 0,
        generator: str = "",
        error: str = "",
    ) -> None:
        """Write an immutable audit record."""
        import aiosqlite

        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                "INSERT OR REPLACE INTO audit VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    query_id,
                    workspace_id,
                    user_id,
                    nl_query,
                    sql,
                    int(succeeded),
                    row_count,
                    cost_usd,
                    latency_ms,
                    corrections,
                    generator,
                    error,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            await db.commit()

    async def recent(self, limit: int = 50) -> list[dict]:
        """Get recent audit records."""
        import aiosqlite

        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM audit ORDER BY created_at DESC LIMIT ?", (limit,)
            )
            rows = await cursor.fetchall()
            return [dict(r) for r in rows]


# ═══════════════════════════════════════════════════════════════════════════════
# LESSON STORE (SQLite — persists Learn Agent correction records)
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class LessonRecord:
    """One user correction processed by the Learn Agent.

    Created in POST /train/sql (correction path) after the user reviews
    and approves the rewritten SQL.  Persisted to SQLite so it survives
    server restarts and is shown on the Learn page as the 'Correction Log'.
    """

    lesson_id: str
    workspace_id: str
    user_id: str
    nl_query: str
    original_sql: str
    corrected_sql: str
    domain_insight: str  # what the agent discovered about the data domain
    context_rule: str  # the extracted general business rule (also in data_context_notes)
    what_changed: str  # technical one-liner explaining the SQL fix
    rows_preview: str  # JSON-serialised list[dict] (max 5 rows)
    columns: str  # JSON-serialised list[str]
    row_count: int
    failed_stage: str  # schema | retrieval | planning | generation | filtering
    failed_node: str  # prune | retrieve | plan | generate | execute | correct
    pair_id: str  # vector-store training pair ID (if saved)
    tokens_used: int
    cost_usd: float
    created_at: str  # ISO timestamp


class LessonStore:
    """Append-only SQLite store for Learn Agent lesson records.

    Same pattern as AuditLog — one table, async aiosqlite reads/writes.
    Default path: ~/.sqlagent/lessons.db
    """

    def __init__(self, db_path: str = ""):
        import os

        if not db_path:
            db_path = os.path.join(os.path.expanduser("~"), ".sqlagent", "lessons.db")
        self._db_path = db_path

    async def init(self) -> None:
        import aiosqlite
        import os

        os.makedirs(os.path.dirname(self._db_path), exist_ok=True)
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS lessons (
                    lesson_id    TEXT PRIMARY KEY,
                    workspace_id TEXT,
                    user_id      TEXT,
                    nl_query     TEXT,
                    original_sql TEXT,
                    corrected_sql TEXT,
                    domain_insight TEXT,
                    context_rule TEXT,
                    what_changed TEXT,
                    rows_preview TEXT,
                    columns      TEXT,
                    row_count    INTEGER,
                    failed_stage TEXT,
                    failed_node  TEXT,
                    pair_id      TEXT,
                    tokens_used  INTEGER,
                    cost_usd     REAL,
                    created_at   TEXT
                )
            """)
            await db.commit()

    async def save(self, record: LessonRecord) -> None:
        import aiosqlite

        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                "INSERT OR REPLACE INTO lessons VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    record.lesson_id,
                    record.workspace_id,
                    record.user_id,
                    record.nl_query,
                    record.original_sql,
                    record.corrected_sql,
                    record.domain_insight,
                    record.context_rule,
                    record.what_changed,
                    record.rows_preview,
                    record.columns,
                    record.row_count,
                    record.failed_stage,
                    record.failed_node,
                    record.pair_id,
                    record.tokens_used,
                    record.cost_usd,
                    record.created_at,
                ),
            )
            await db.commit()

    async def list_for_workspace(self, workspace_id: str = "", limit: int = 20) -> list[dict]:
        import aiosqlite

        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            if workspace_id:
                cursor = await db.execute(
                    "SELECT * FROM lessons WHERE workspace_id=? ORDER BY created_at DESC LIMIT ?",
                    (workspace_id, limit),
                )
            else:
                cursor = await db.execute(
                    "SELECT * FROM lessons ORDER BY created_at DESC LIMIT ?", (limit,)
                )
            rows = await cursor.fetchall()

        result = []
        for r in rows:
            d = dict(r)
            try:
                d["rows_preview"] = _json.loads(d.get("rows_preview") or "[]")
            except Exception as exc:
                logger.debug("telemetry.operation_failed", error=str(exc))
                d["rows_preview"] = []
            try:
                d["columns"] = _json.loads(d.get("columns") or "[]")
            except Exception as exc:
                logger.debug("telemetry.operation_failed", error=str(exc))
                d["columns"] = []
            result.append(d)
        return result

    async def delete(self, lesson_id: str) -> None:
        import aiosqlite

        async with aiosqlite.connect(self._db_path) as db:
            await db.execute("DELETE FROM lessons WHERE lesson_id=?", (lesson_id,))
            await db.commit()

    async def count(self, workspace_id: str = "") -> int:
        import aiosqlite

        async with aiosqlite.connect(self._db_path) as db:
            if workspace_id:
                cursor = await db.execute(
                    "SELECT COUNT(*) FROM lessons WHERE workspace_id=?", (workspace_id,)
                )
            else:
                cursor = await db.execute("SELECT COUNT(*) FROM lessons")
            row = await cursor.fetchone()
            return row[0] if row else 0


# ═══════════════════════════════════════════════════════════════════════════════
# LANGFUSE INTEGRATION (optional)
# ═══════════════════════════════════════════════════════════════════════════════


def get_langfuse_handler(config: Any):
    """Return a LangChain CallbackHandler for Langfuse tracing.

    Passed to graph.ainvoke(config={"callbacks": [handler]}).
    Returns None if Langfuse is not configured.
    """
    if not getattr(config, "langfuse_public_key", ""):
        return None
    try:
        from langfuse.callback import CallbackHandler

        return CallbackHandler(
            public_key=config.langfuse_public_key,
            secret_key=config.langfuse_secret_key,
            host=getattr(config, "langfuse_host", "https://cloud.langfuse.com"),
        )
    except ImportError:
        logger.warn("langfuse not installed — pip install sqlagent[langfuse]")
        return None
