"""SQLAgent — the main entry point. Ties everything together.

Usage:
    from sqlagent import SQLAgent
    agent = SQLAgent(db="sqlite:///northwind.db", llm_model="gpt-4o")
    result = await agent.query("top 10 customers by revenue")

Or the one-liner:
    from sqlagent import ask
    result = ask("top 10 customers", db="sqlite:///northwind.db")
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import structlog

from sqlagent.config import AgentConfig
from sqlagent.models import PipelineResult, Trace
from sqlagent.exceptions import ConfigurationError

logger = structlog.get_logger()


# ═══════════════════════════════════════════════════════════════════════════════
# PIPELINE SERVICES (dependency injection container)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class PipelineServices:
    """All services needed by graph nodes. Injected at compile time."""
    config: AgentConfig = field(default_factory=AgentConfig)
    llm: Any = None
    embedder: Any = None
    connectors: dict = field(default_factory=dict)        # source_id → Connector
    schema_selector: Any = None
    example_store: Any = None
    ensemble: Any = None
    policy: Any = None
    memory_manager: Any = None
    soul: Any = None
    trace_store: Any = None
    audit_log: Any = None
    lesson_store: Any = None   # LessonStore — persists Learn Agent correction records


# ═══════════════════════════════════════════════════════════════════════════════
# SQL AGENT
# ═══════════════════════════════════════════════════════════════════════════════

class SQLAgent:
    """The main sqlagent class. Compiles a LangGraph and runs queries through it.

    Lazy initialization: services are built on first query, not on __init__.
    """

    def __init__(
        self,
        db: str = "",
        llm_model: str = "",
        config: AgentConfig | None = None,
    ):
        self._config = config or AgentConfig()
        if llm_model:
            self._config = AgentConfig(
                **{**self._config.__dict__, "llm_model": llm_model}
            )
        self._db_url = db
        self._services: PipelineServices | None = None
        self._graph = None
        self._ready = False
        # Learn Agent state
        self._learn_activity: list[dict] = []   # recent auto-learn events (capped at 50)
        self._auto_learns_total: int = 0
        self._soul_evolutions: int = 0
        self._auto_learn: bool = True           # can be overridden by server settings
        # Workspace semantic context — lessons learned from user corrections.
        # Injected into EVERY SQL generation prompt so the LLM never repeats the same mistake.
        self._data_context_notes: list[str] = []
        # Learn graph (compiled alongside query graph, used by /learn/regenerate)
        self._learn_graph = None

    async def _ensure_ready(self) -> None:
        """Lazy-init all services and compile the graph."""
        if self._ready:
            return

        services = PipelineServices(config=self._config)

        # LLM
        from sqlagent.llm import LiteLLMProvider
        services.llm = LiteLLMProvider(
            model=self._config.llm_model,
            temperature=self._config.temperature,
            max_tokens=self._config.max_tokens,
        )

        # Embedder
        from sqlagent.llm import FastEmbedEmbedder
        services.embedder = FastEmbedEmbedder(model=self._config.embedding_model)

        # Connector
        if self._db_url:
            from sqlagent.connectors import ConnectorRegistry
            source_id = f"src_{uuid.uuid4().hex[:8]}"
            conn = ConnectorRegistry.from_url(source_id, self._db_url)
            await conn.connect()
            services.connectors[source_id] = conn

        # Schema selector
        from sqlagent.schema import SchemaSelector
        services.schema_selector = SchemaSelector(
            embedder=services.embedder,
            top_k=self._config.schema_pruning_top_k,
        )

        # Vector store + example store
        from sqlagent.retrieval import QdrantVectorStore, ExampleStore
        vector_store = QdrantVectorStore()
        await vector_store.ensure_collection(dimensions=self._config.embedding_dimensions)
        services.example_store = ExampleStore(vector_store=vector_store, embedder=services.embedder)

        # Generators + ensemble
        from sqlagent.generators import FewshotGenerator, PlanGenerator, DecomposeGenerator, GeneratorEnsemble
        generators = []
        for gen_name in self._config.generators:
            if gen_name == "fewshot":
                generators.append(FewshotGenerator(services.llm))
            elif gen_name == "plan":
                generators.append(PlanGenerator(services.llm))
            elif gen_name == "decompose":
                generators.append(DecomposeGenerator(services.llm))
        services.ensemble = GeneratorEnsemble(generators=generators, llm=services.llm)

        # Policy
        from sqlagent.runtime import PolicyGateway
        services.policy = PolicyGateway(config=self._config)

        # Memory
        from sqlagent.runtime import MemoryManager, WorkingMemory, EpisodicMemory
        services.memory_manager = MemoryManager(
            working=WorkingMemory(),
            episodic=EpisodicMemory(),
        )

        # SOUL
        from sqlagent.soul import UserSOUL
        services.soul = UserSOUL(llm=services.llm)

        # Trace store
        from sqlagent.trace import TraceStore
        services.trace_store = TraceStore()

        # Audit log
        from sqlagent.telemetry import AuditLog
        services.audit_log = AuditLog()
        await services.audit_log.init()

        # Lesson store (persists Learn Agent correction records to SQLite)
        from sqlagent.telemetry import LessonStore
        services.lesson_store = LessonStore()
        await services.lesson_store.init()

        self._services = services

        # Compile the query orchestration LangGraph
        from sqlagent.graph.builder import compile_query_graph
        self._graph = compile_query_graph(services)

        # Compile the Learn Agent correction LangGraph (separate graph)
        from sqlagent.graph.learn_graph import compile_learn_graph
        self._learn_graph = compile_learn_graph(services)

        self._ready = True
        logger.info("agent.ready", model=self._config.llm_model, sources=list(services.connectors.keys()))

    async def query(
        self,
        nl_query: str,
        user_id: str = "default",
        workspace_id: str = "",
        session_id: str = "",
    ) -> PipelineResult:
        """Run a natural language query through the full agentic pipeline."""
        await self._ensure_ready()

        query_id = str(uuid.uuid4())[:12]

        # Build initial state
        from sqlagent.graph.state import QueryState
        initial_state: QueryState = {
            "nl_query": nl_query,
            "query_id": query_id,
            "user_id": user_id,
            "workspace_id": workspace_id,
            "session_id": session_id,
            "source_ids": list(self._services.connectors.keys()),
            "trace_events": [],
            "tokens_used": 0,
            "cost_usd": 0.0,
            "budget_exhausted": False,
            "correction_round": 0,
            "max_corrections": self._config.max_corrections,
        }

        # Optional Langfuse callback
        config = {}
        from sqlagent.telemetry import get_langfuse_handler
        handler = get_langfuse_handler(self._config)
        if handler:
            config["callbacks"] = [handler]

        # Run the graph
        try:
            result_state = await self._graph.ainvoke(initial_state, config=config)
        except Exception as e:
            logger.error("agent.query_failed", error=str(e), query=nl_query[:50])
            return PipelineResult(
                query_id=query_id, nl_query=nl_query,
                succeeded=False, error=str(e),
            )

        # Build PipelineResult
        from sqlagent.trace import TraceCollector
        trace = TraceCollector.build_trace(result_state, workspace_id=workspace_id, user_id=user_id)

        # Persist trace
        if self._services.trace_store:
            try:
                await self._services.trace_store.save(trace)
            except Exception as exc:
                logger.debug("agent.operation_failed", error=str(exc))

        # Audit log
        if self._services.audit_log:
            try:
                await self._services.audit_log.record(
                    query_id=query_id, workspace_id=workspace_id, user_id=user_id,
                    nl_query=nl_query, sql=result_state.get("sql", ""),
                    succeeded=result_state.get("succeeded", False),
                    row_count=result_state.get("row_count", 0),
                    cost_usd=result_state.get("cost_usd", 0.0),
                    latency_ms=trace.total_latency_ms,
                    corrections=result_state.get("correction_round", 0),
                    generator=result_state.get("winner_generator", ""),
                    error=result_state.get("execution_error", ""),
                )
            except Exception as exc:
                logger.debug("agent.operation_failed", error=str(exc))

        # Prometheus metrics
        from sqlagent.telemetry import record_query_metrics
        record_query_metrics(
            source_id=result_state.get("target_sources", [""])[0] if result_state.get("target_sources") else "",
            succeeded=result_state.get("succeeded", False),
            generator=result_state.get("winner_generator", ""),
            latency_s=trace.total_latency_ms / 1000.0,
            cost_usd=result_state.get("cost_usd", 0.0),
            model=self._config.llm_model,
        )

        return PipelineResult(
            query_id=query_id,
            nl_query=nl_query,
            sql=result_state.get("sql", ""),
            succeeded=result_state.get("succeeded", False),
            error=result_state.get("execution_error", "") or result_state.get("error", ""),
            rows=result_state.get("rows", []),
            columns=result_state.get("columns", []),
            row_count=result_state.get("row_count", 0),
            nl_response=result_state.get("nl_response", ""),
            follow_ups=result_state.get("follow_ups", []),
            chart_config=result_state.get("chart_config"),
            total_tokens=result_state.get("tokens_used", 0),
            total_cost_usd=result_state.get("cost_usd", 0.0),
            latency_ms=trace.total_latency_ms,
            winner_generator=result_state.get("winner_generator", ""),
            correction_rounds=result_state.get("correction_round", 0),
            trace=trace,
        )

    async def query_stream(
        self,
        nl_query: str,
        user_id: str = "default",
        workspace_id: str = "",
        session_id: str = "",
        display_nl_query: str = "",
    ):
        """Run query with streaming — yields trace events as each node completes.

        Yields dicts: {"type": "trace.node.completed"|"query.result", "data": {...}}
        Uses LangGraph astream("values") to get full state after each node.
        """
        await self._ensure_ready()
        import time
        started = time.monotonic()
        query_id = str(uuid.uuid4())[:12]

        # Reset per-query token accumulators on the LLM provider
        if hasattr(self._services, 'llm') and hasattr(self._services.llm, 'reset_session_tokens'):
            self._services.llm.reset_session_tokens()

        from sqlagent.graph.state import QueryState
        initial_state: QueryState = {
            "nl_query": nl_query,  # full context-enriched query for LLM reasoning
            "display_nl_query": display_nl_query if display_nl_query else nl_query,  # clean question for display
            "query_id": query_id,
            "user_id": user_id,
            "workspace_id": workspace_id,
            "session_id": session_id,
            "source_ids": list(self._services.connectors.keys()),
            "trace_events": [],
            "tokens_used": 0,
            "cost_usd": 0.0,
            "budget_exhausted": False,
            "correction_round": 0,
            "max_corrections": self._config.max_corrections,
            # Workspace-specific semantic lessons from user corrections
            # These inject into every SQL generation so the LLM never repeats known mistakes
            "data_context_notes": list(self._data_context_notes),
        }

        config = {}
        from sqlagent.telemetry import get_langfuse_handler
        handler = get_langfuse_handler(self._config)
        if handler:
            config["callbacks"] = [handler]

        # Stream: yields accumulated state after each node completes
        prev_trace_count = 0
        final_state = initial_state
        try:
            async for state in self._graph.astream(initial_state, config=config, stream_mode="values"):
                final_state = state
                new_events = state.get("trace_events", [])
                # Yield only NEW trace events since last yield
                if len(new_events) > prev_trace_count:
                    for evt in new_events[prev_trace_count:]:
                        yield {"type": "trace.node.completed", "data": evt}
                    prev_trace_count = len(new_events)
        except Exception as e:
            logger.error("agent.stream_failed", error=str(e))
            yield {"type": "query.error", "data": {"error": str(e)}}
            return

        # Collect per-query token breakdown from LLM provider session counters
        _llm = self._services.llm if hasattr(self._services, 'llm') else None
        tokens_input_session = getattr(_llm, '_session_tokens_input', 0)
        tokens_output_session = getattr(_llm, '_session_tokens_output', 0)
        active_model_id = getattr(_llm, 'model', self._config.llm_model)

        # Build trace + persist — wrapped so a serialization bug never blocks query.result
        trace = None
        try:
            from sqlagent.trace import TraceCollector
            trace = TraceCollector.build_trace(final_state, workspace_id=workspace_id, user_id=user_id)
            trace.model_id = active_model_id
            trace.tokens_input = tokens_input_session
            trace.tokens_output = tokens_output_session
            if self._services.trace_store:
                try:
                    await self._services.trace_store.save(trace)
                except Exception as exc:
                    logger.debug("agent.trace_save_failed", error=str(exc))
        except Exception as exc:
            logger.debug("agent.trace_build_failed", error=str(exc))

        # ── Learn Agent: auto-learn + SOUL observation (background, never blocks) ──
        _nl_for_learn = final_state.get("display_nl_query") or nl_query
        _sql_for_learn = final_state.get("sql", "")
        _succeeded_q = final_state.get("succeeded", False)
        # auto_learn can be disabled by server setting _auto_learn attribute before query
        _auto_learn_enabled = getattr(self, '_auto_learn', True) is not False

        learn_event = {"action": None, "soul_evolved": False}

        async def _background_learn():
            # 1. Auto-learn: save successful query as training pair
            if _succeeded_q and _auto_learn_enabled and _sql_for_learn and self._services.example_store:
                try:
                    from sqlagent.agents import LearningLoop
                    loop = LearningLoop(self._services.example_store)
                    await loop.on_thumbs_up(_nl_for_learn, _sql_for_learn)
                    self._auto_learns_total += 1
                    _activity = {
                        "action": "auto_learned",
                        "nl_query": _nl_for_learn[:80],
                        "generator": final_state.get("winner_generator", ""),
                        "ts": datetime.now(timezone.utc).isoformat(),
                    }
                    self._learn_activity.insert(0, _activity)
                    if len(self._learn_activity) > 50:
                        self._learn_activity.pop()
                    learn_event["action"] = "auto_learned"
                except Exception as exc:
                    logger.debug("agent.operation_failed", error=str(exc))

            # 2. SOUL observation: runs after every query (tracks patterns)
            if self._services.soul:
                try:
                    tables = final_state.get("pruned_tables") or []
                    prev_evolutions = self._soul_evolutions
                    await self._services.soul.observe(
                        user_id=user_id or "local",
                        nl_query=_nl_for_learn,
                        tables_used=tables,
                    )
                    # Check if SOUL evolved (query_count hit a 20-multiple)
                    soul_profile = self._services.soul._profiles.get(user_id or "local")
                    if soul_profile and soul_profile.query_count > 0 and soul_profile.query_count % 20 == 0:
                        self._soul_evolutions += 1
                        learn_event["soul_evolved"] = True
                        _soul_act = {
                            "action": "soul_evolved",
                            "nl_query": _nl_for_learn[:80],
                            "query_count": soul_profile.query_count,
                            "ts": datetime.now(timezone.utc).isoformat(),
                        }
                        self._learn_activity.insert(0, _soul_act)
                        if len(self._learn_activity) > 50:
                            self._learn_activity.pop()
                except Exception as exc:
                    logger.debug("agent.operation_failed", error=str(exc))

        import asyncio as _asyncio
        _asyncio.create_task(_background_learn())

        # Audit + metrics
        total_ms = int((time.monotonic() - started) * 1000)
        if self._services.audit_log:
            try:
                await self._services.audit_log.record(
                    query_id=query_id, workspace_id=workspace_id, user_id=user_id,
                    nl_query=final_state.get("display_nl_query") or nl_query, sql=final_state.get("sql", ""),
                    succeeded=final_state.get("succeeded", False),
                    row_count=final_state.get("row_count", 0),
                    cost_usd=final_state.get("cost_usd", 0.0),
                    latency_ms=total_ms,
                    corrections=final_state.get("correction_round", 0),
                    generator=final_state.get("winner_generator", ""),
                    error=final_state.get("execution_error", ""),
                )
            except Exception as exc:
                logger.debug("agent.operation_failed", error=str(exc))

        try:
            from sqlagent.telemetry import record_query_metrics
            record_query_metrics(
                source_id=final_state.get("target_sources", [""])[0] if final_state.get("target_sources") else "",
                succeeded=final_state.get("succeeded", False),
                generator=final_state.get("winner_generator", ""),
                latency_s=total_ms / 1000.0,
                cost_usd=final_state.get("cost_usd", 0.0),
                model=self._config.llm_model,
            )
        except Exception as exc:
            logger.debug("agent.metrics_failed", error=str(exc))

        # ── Final result event ────────────────────────────────────────────────
        # Build every field defensively — any failure here would silently drop
        # the query.result SSE event and show "No result received" in the UI.

        # Sanitize rows one final time: NaN/Inf → None, non-serializable → str.
        # execute_node already cleans, but cross-source or unusual dtypes can slip through.
        import math as _math
        def _safe_val(v):
            if isinstance(v, float) and (_math.isnan(v) or _math.isinf(v)):
                return None
            try:
                import json as _json
                _json.dumps(v)
                return v
            except (TypeError, ValueError):
                return str(v)

        raw_rows = final_state.get("rows", [])
        safe_rows = [{k: _safe_val(v) for k, v in row.items()} for row in raw_rows] if raw_rows else []

        # trace.to_dict() wrapped — a serialization bug in the trace must never
        # prevent the result from reaching the browser.
        trace_dict = None
        if trace is not None:
            try:
                trace_dict = trace.to_dict()
            except Exception as exc:
                logger.debug("agent.trace_dict_failed", error=str(exc))

        yield {
            "type": "query.result",
            "data": {
                "query_id": query_id,
                "sql": final_state.get("sql", ""),
                "succeeded": final_state.get("succeeded", False),
                "error": final_state.get("execution_error", "") or final_state.get("error", ""),
                "rows": safe_rows,
                "columns": final_state.get("columns", []),
                "row_count": final_state.get("row_count", 0),
                "nl_response": final_state.get("nl_response", ""),
                "follow_ups": final_state.get("follow_ups", []),
                "chart_config": final_state.get("chart_config"),
                "total_tokens": final_state.get("tokens_used", 0),
                "tokens_input": tokens_input_session,
                "tokens_output": tokens_output_session,
                "model_id": active_model_id,
                "total_cost_usd": final_state.get("cost_usd", 0.0),
                "latency_ms": total_ms,
                "winner_generator": final_state.get("winner_generator", ""),
                "correction_rounds": final_state.get("correction_round", 0),
                "trace": trace_dict,
                "learn_event": learn_event,
            },
        }

    async def add_connector(self, source_id: str, db_url: str) -> None:
        """Add an additional data source."""
        await self._ensure_ready()
        from sqlagent.connectors import ConnectorRegistry
        conn = ConnectorRegistry.from_url(source_id, db_url)
        await conn.connect()
        self._services.connectors[source_id] = conn

    async def train_sql(self, nl_query: str, sql: str, source_id: str = "") -> str:
        """Add a verified NL→SQL training pair."""
        await self._ensure_ready()
        return await self._services.example_store.add(
            nl_query=nl_query, sql=sql, source_id=source_id,
            generator="user_trained", verified=True,
        )

    async def train_docs(self, text: str, source: str = "docs") -> None:
        """Store documentation for enriching the semantic layer."""
        # TODO: parse docs and extract column descriptions
        logger.info("agent.train_docs", source=source, length=len(text))

    @property
    def services(self) -> PipelineServices:
        return self._services


# ═══════════════════════════════════════════════════════════════════════════════
# QUICK API — one-liner functions
# ═══════════════════════════════════════════════════════════════════════════════

def ask(
    nl_query: str,
    db: str = "",
    model: str = "gpt-4o",
    **kwargs,
) -> PipelineResult:
    """One-liner synchronous query.

    Usage:
        from sqlagent import ask
        result = ask("top 10 customers", db="sqlite:///northwind.db")
        print(result.dataframe)
    """
    agent = SQLAgent(db=db, llm_model=model)
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(agent.query(nl_query, **kwargs))
    finally:
        loop.close()


class _MultiSourceContext:
    """Context manager for multi-source queries."""

    def __init__(self, sources: dict[str, str], model: str = "gpt-4o"):
        self._sources = sources
        self._model = model
        self._agent: SQLAgent | None = None

    async def __aenter__(self):
        self._agent = SQLAgent(llm_model=self._model)
        await self._agent._ensure_ready()
        for name, url in self._sources.items():
            await self._agent.add_connector(name, url)
        return self._agent

    async def __aexit__(self, *args):
        pass

    def __enter__(self):
        loop = asyncio.new_event_loop()
        self._agent = SQLAgent(llm_model=self._model)
        loop.run_until_complete(self._agent._ensure_ready())
        for name, url in self._sources.items():
            loop.run_until_complete(self._agent.add_connector(name, url))
        self._loop = loop
        return self._agent

    def __exit__(self, *args):
        pass


def connect(**sources: str) -> _MultiSourceContext:
    """Multi-source context manager.

    Usage:
        with connect(sales="postgresql://...", staff="data/headcount.csv") as agent:
            result = agent.query("revenue per employee")
    """
    return _MultiSourceContext(sources)
