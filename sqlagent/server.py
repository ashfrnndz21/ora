"""FastAPI server — all routes in one file.

Endpoints: auth, workspaces, queries (SSE streaming), schema, tasks, training, setup, observability.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import uuid
from datetime import datetime, timezone
from typing import Any

import structlog
from fastapi import FastAPI, HTTPException, Depends, Request, Response, UploadFile, File, Body, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

logger = structlog.get_logger()

# ═══════════════════════════════════════════════════════════════════════════════
# APP STATE (initialized at startup)
# ═══════════════════════════════════════════════════════════════════════════════

_state: dict[str, Any] = {}


def _sanitize_for_json(rows: list[dict]) -> list[dict]:
    """Replace NaN/Inf with None for JSON serialization."""
    clean = []
    for row in rows:
        clean_row = {}
        for k, v in row.items():
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                clean_row[k] = None
            else:
                clean_row[k] = v
        clean.append(clean_row)
    return clean


# ── Pydantic request models — must be module-level for FastAPI/Pydantic v2 ──
class MagicLinkRequest(BaseModel):
    email: str


class MagicLinkVerify(BaseModel):
    email: str
    code: str


class CreateWorkspace(BaseModel):
    name: str
    description: str = ""


class QueryRequest(BaseModel):
    query: str
    workspace_id: str = ""
    session_id: str = ""


class TrainRequest(BaseModel):
    nl_query: str
    sql: str
    source_id: str = ""


class InstallPackRequest(BaseModel):
    pack_name: str


class FeedbackRequest(BaseModel):
    query_id: str
    nl_query: str
    sql: str
    feedback: str  # "thumbs_up" | "thumbs_down"
    corrected_sql: str = ""


class SetupChatRequest(BaseModel):
    message: str


def create_app(config: Any = None, default_db: str = "") -> FastAPI:
    """Create the FastAPI application."""
    from sqlagent.config import AgentConfig

    cfg = config or AgentConfig()
    _state["default_db"] = default_db

    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def lifespan(app_instance):
        # Startup
        from sqlagent.auth import AuthStore
        from sqlagent.workspace import WorkspaceStore
        from sqlagent.trace import TraceStore
        from sqlagent.telemetry import AuditLog, LessonStore, setup_telemetry

        if cfg.otel_enabled:
            setup_telemetry(cfg)

        _state["config"] = cfg
        _state["auth_store"] = AuthStore()
        await _state["auth_store"].init()
        _state["workspace_store"] = WorkspaceStore()
        await _state["workspace_store"].init()
        _state["trace_store"] = TraceStore()
        await _state["trace_store"].init()
        _state["audit_log"] = AuditLog()
        await _state["audit_log"].init()
        _state["lesson_store"] = LessonStore()
        await _state["lesson_store"].init()
        _state["agents"] = {}

        # Load persisted user settings (API keys, model) — survive restarts
        import json as _json
        import os as _os

        _settings_path = _os.path.join(_os.path.expanduser("~"), ".sqlagent", "user_settings.json")
        if _os.path.exists(_settings_path):
            try:
                with open(_settings_path) as _f:
                    saved = _json.load(_f)
                _state["user_settings"] = saved
                # Re-apply API keys to env
                if saved.get("openai_key"):
                    _os.environ["OPENAI_API_KEY"] = saved["openai_key"]
                if saved.get("anthropic_key"):
                    _os.environ["ANTHROPIC_API_KEY"] = saved["anthropic_key"]
                import structlog as _sl

                _sl.get_logger().info(
                    "settings.loaded",
                    model=saved.get("model", ""),
                    has_openai=bool(saved.get("openai_key")),
                    has_anthropic=bool(saved.get("anthropic_key")),
                )
            except Exception as _e:
                import structlog as _sl

                _sl.get_logger().warn("settings.load_failed", error=str(_e))

        yield
        # Shutdown (cleanup if needed)

    app = FastAPI(
        title="ora",
        description="The NL2SQL Agentic Runtime — connect any database, ask in plain English.",
        version="2.0.0",
        lifespan=lifespan,
        docs_url=None,  # We serve a custom dark-themed /docs
        redoc_url=None,  # Replaced by /docs
        openapi_tags=[
            {"name": "query", "description": "Run natural language queries against databases"},
            {"name": "schema", "description": "Schema introspection, pruning, and knowledge graph"},
            {"name": "train", "description": "Training feedback and learning"},
            {"name": "workspaces", "description": "Workspace management"},
            {"name": "auth", "description": "Authentication"},
            {"name": "tasks", "description": "Query execution history and traces"},
            {"name": "hub", "description": "QueryHub community training packs"},
            {"name": "soul", "description": "SOUL user mental model"},
            {"name": "observability", "description": "Health, metrics, and debug"},
            {"name": "settings", "description": "Configuration and API keys"},
        ],
    )

    # CORS — env var overrides config; wildcard + credentials is an XSS vector so
    # we disallow that combination and warn loudly if someone tries.
    _cors_origins = cfg.cors_origins or []
    _env_origins = os.environ.get("SQLAGENT_CORS_ORIGINS", "")
    if _env_origins:
        _cors_origins = [o.strip() for o in _env_origins.split(",") if o.strip()]
    _allow_credentials = bool(_cors_origins and "*" not in _cors_origins)
    if "*" in _cors_origins and cfg.auth_enabled:
        import warnings

        warnings.warn(
            "CORS wildcard ('*') with auth_enabled=True is a security risk. "
            "Set SQLAGENT_CORS_ORIGINS to specific origins instead.",
            stacklevel=2,
        )
        _cors_origins = []  # deny rather than allow unsafe combination
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins if _cors_origins else [],
        allow_credentials=_allow_credentials,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "X-Requested-With"],
    )

    # OTel instrumentation
    if cfg.otel_enabled:
        try:
            from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

            FastAPIInstrumentor.instrument_app(app)
        except Exception as exc:
            logger.debug("otel.instrumentation_skipped", error=str(exc))

    # ── Auth dependency ───────────────────────────────────────────────────────

    async def get_current_user(request: Request):
        if not cfg.auth_enabled:
            from sqlagent.auth import LOCAL_USER

            return LOCAL_USER

        token = request.headers.get("Authorization", "").replace("Bearer ", "")
        if not token:
            token = request.cookies.get("sqlagent_token", "")
        if not token:
            raise HTTPException(401, "Not authenticated")

        from sqlagent.auth import verify_token
        from sqlagent.exceptions import InvalidToken

        try:
            payload = verify_token(token, cfg.auth_jwt_secret)
            user = await _state["auth_store"].get_by_id(payload["user_id"])
            if not user:
                raise HTTPException(401, "User not found")
            return user
        except InvalidToken:
            raise HTTPException(401, "Invalid or expired token")

    # ── Auth routes ───────────────────────────────────────────────────────────

    @app.post("/auth/magic-link", tags=["auth"])
    async def auth_magic_link(request: Request):
        body = await request.json()
        email = body.get("email", "").strip()
        if not email:
            raise HTTPException(400, "email required")
        from sqlagent.auth import send_magic_link

        send_magic_link(email)
        # NOTE: code is NOT returned in the response — it must be delivered via email.
        # For local dev without an email provider, check server logs at DEBUG level.
        return {"message": "If that address is registered, a code is on its way."}

    @app.post("/auth/magic-link/verify", tags=["auth"])
    async def auth_verify_magic(request: Request):
        body = await request.json()
        email = body.get("email", "").strip()
        code = body.get("code", "").strip()
        from sqlagent.auth import verify_magic_link, create_token

        if not verify_magic_link(email, code):
            raise HTTPException(401, "Invalid or expired code")
        user = await _state["auth_store"].get_or_create(email)
        token = create_token(user.user_id, cfg.auth_jwt_secret)
        return {
            "token": token,
            "user": {
                "user_id": user.user_id,
                "email": user.email,
                "display_name": user.display_name,
            },
        }

    @app.get("/auth/me", tags=["auth"])
    async def auth_me(user=Depends(get_current_user)):
        return {
            "user_id": user.user_id,
            "email": user.email,
            "display_name": user.display_name,
            "avatar_url": user.avatar_url,
        }

    # ── Workspace routes ──────────────────────────────────────────────────────

    @app.post("/workspaces", tags=["workspaces"])
    async def create_workspace(
        name: str = Body(...), description: str = Body(""), user=Depends(get_current_user)
    ):
        ws = await _state["workspace_store"].create(name, user.user_id, description)
        return {"workspace_id": ws.workspace_id, "name": ws.name, "status": ws.status.value}

    @app.get("/workspaces", tags=["workspaces"])
    async def list_workspaces(user=Depends(get_current_user)):
        workspaces = await _state["workspace_store"].list_for_user(user.user_id)
        return [
            {
                "workspace_id": w.workspace_id,
                "name": w.name,
                "status": w.status.value,
                "sources": w.sources,
                "query_count": w.query_count,
                "created_at": w.created_at.isoformat()
                if hasattr(w.created_at, "isoformat")
                else str(w.created_at),
            }
            for w in workspaces
        ]

    @app.get("/workspaces/{workspace_id}", tags=["workspaces"])
    async def get_workspace(workspace_id: str, user=Depends(get_current_user)):
        from sqlagent.exceptions import WorkspaceNotFound

        try:
            ws = await _state["workspace_store"].get(workspace_id)
        except WorkspaceNotFound:
            raise HTTPException(404, "Workspace not found")
        return {
            "workspace_id": ws.workspace_id,
            "name": ws.name,
            "status": ws.status.value,
            "sources": ws.sources,
            "query_count": ws.query_count,
        }

    @app.delete("/workspaces/{workspace_id}", tags=["workspaces"])
    async def delete_workspace(workspace_id: str, user=Depends(get_current_user)):
        await _state["workspace_store"].delete(workspace_id)
        # Clean ALL cached state for this workspace
        _state.get("agents", {}).pop(workspace_id, None)
        _state.get("setup_agents", {}).pop(workspace_id, None)
        _state.pop(f"kg_{workspace_id}", None)
        _state.pop("kg_default", None)  # Also clear default KG
        # Clean uploaded files
        import shutil

        uploads_dir = os.path.join(os.path.expanduser("~"), ".sqlagent", "uploads", workspace_id)
        if os.path.exists(uploads_dir):
            shutil.rmtree(uploads_dir, ignore_errors=True)
        return {"deleted": workspace_id}

    @app.put("/workspaces/{workspace_id}", tags=["workspaces"])
    async def update_workspace(workspace_id: str, request: Request, user=Depends(get_current_user)):
        body = await request.json()
        ws = await _state["workspace_store"].update(workspace_id, **body)
        return {"workspace_id": ws.workspace_id, "name": ws.name, "status": ws.status.value}

    # ── Query routes ──────────────────────────────────────────────────────────

    @app.post("/query", tags=["query"])
    async def run_query(request: Request, user=Depends(get_current_user)):
        body = await request.json()
        query = body.get("query", "")
        workspace_id = body.get("workspace_id", "")
        session_id = body.get("session_id", "")
        agent = await _get_or_create_agent(workspace_id, user.user_id)
        result = await agent.query(
            nl_query=query,
            user_id=user.user_id,
            workspace_id=workspace_id,
            session_id=session_id,
        )
        return {
            "query_id": result.query_id,
            "sql": result.sql,
            "succeeded": result.succeeded,
            "error": result.error,
            "rows": _sanitize_for_json(result.rows[:100]),
            "columns": result.columns,
            "row_count": result.row_count,
            "nl_response": result.nl_response,
            "follow_ups": result.follow_ups,
            "chart_config": result.chart_config,
            "total_tokens": result.total_tokens,
            "total_cost_usd": result.total_cost_usd,
            "latency_ms": result.latency_ms,
            "winner_generator": result.winner_generator,
            "correction_rounds": result.correction_rounds,
            "trace": result.trace.to_dict() if result.trace else None,
        }

    @app.post("/query/stream", tags=["query"])
    async def stream_query(request: Request, user=Depends(get_current_user)):
        """True SSE streaming — trace nodes appear one-by-one as each LangGraph node completes."""
        body = await request.json()
        query = body.get("query", "")
        workspace_id = body.get("workspace_id", "")
        context = body.get("context", [])
        agent = await _get_or_create_agent(workspace_id, user.user_id)

        # ── Apply user's selected model (Settings override) ──────────────────
        # User settings are saved to _state["user_settings"] by POST /api/settings.
        # The agent's LLM provider is updated here before each query so the correct
        # model is always used regardless of the startup config value.
        _user_settings = _state.get("user_settings", {})
        _selected_model = _user_settings.get("model", "")
        if (
            _selected_model
            and hasattr(agent, "_services")
            and agent._services
            and hasattr(agent._services, "llm")
        ):
            agent._services.llm.model = _selected_model
        # Sync auto_learn setting to agent
        if hasattr(agent, "_auto_learn"):
            agent._auto_learn = _user_settings.get("auto_learn", True) is not False

        # Keep original query clean for display; pass conversation_history as structured state
        original_query = query
        # Normalize context turns to the conversation_history schema
        conversation_history = []
        for turn in (context or [])[-10:]:
            role = turn.get("role", "")
            if role in ("user", "assistant"):
                conversation_history.append(
                    {
                        "role": role,
                        "text": turn.get("text", ""),
                        "sql": turn.get("sql", ""),
                        "nl_response": turn.get("nl_response") or turn.get("response", ""),
                    }
                )

        async def event_stream():
            import asyncio as _aio
            import math as _math

            def _sse_json(obj):
                """JSON encoder that converts NaN/Infinity → null and non-serializable → str.
                NaN from pandas DataFrames is NOT valid JSON; browsers silently drop the event."""
                if isinstance(obj, float) and (_math.isnan(obj) or _math.isinf(obj)):
                    return None
                return str(obj)

            # Headers that disable proxy/browser buffering
            yield f"event: query.started\ndata: {json.dumps({'query': original_query})}\n\n"

            # Use query_stream for true per-node streaming
            # Pass original_query as display_nl_query so task records store the clean question
            async for event in agent.query_stream(
                nl_query=query,
                user_id=user.user_id,
                workspace_id=workspace_id,
                display_nl_query=original_query,
                conversation_history=conversation_history,
            ):
                event_type = event["type"]
                event_data = event["data"]
                yield f"event: {event_type}\ndata: {json.dumps(event_data, default=_sse_json)}\n\n"
                # Yield a comment to flush buffers between events
                yield ": keep-alive\n\n"
                await _aio.sleep(0)  # Yield control to event loop so data is flushed

        headers = {
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
            "Connection": "keep-alive",
        }
        return StreamingResponse(event_stream(), media_type="text/event-stream", headers=headers)

    # ── Schema routes ─────────────────────────────────────────────────────────

    @app.get("/schema", tags=["schema"])
    async def get_schema(workspace_id: str = "", user=Depends(get_current_user)):
        agent = await _get_or_create_agent(workspace_id, user.user_id)
        if not agent.services:
            return {"tables": []}
        schemas = []
        for sid, conn in agent.services.connectors.items():
            try:
                snap = await conn.introspect()
                schemas.append(
                    {
                        "source_id": sid,
                        "dialect": snap.dialect,
                        "tables": [
                            {
                                "name": t.name,
                                "row_count": t.row_count_estimate,
                                "columns": [
                                    {
                                        "name": c.name,
                                        "data_type": c.data_type,
                                        "is_pk": c.is_primary_key,
                                        "is_fk": c.is_foreign_key,
                                    }
                                    for c in t.columns
                                ],
                            }
                            for t in snap.tables
                        ],
                    }
                )
            except Exception as exc:
                logger.debug("server.operation_failed", error=str(exc))
        return {"sources": schemas}

    @app.get("/schema/graph", tags=["schema"])
    async def get_schema_graph(workspace_id: str = "", user=Depends(get_current_user)):
        """Return the knowledge graph if it exists, otherwise return raw schema with FK edges."""
        agent = await _get_or_create_agent(workspace_id, user.user_id)
        if not agent.services:
            return {"nodes": [], "edges": [], "layers": [], "analyzed": False}

        # Check in-memory cache first, then disk
        kg_key = f"kg_{workspace_id or 'default'}"
        if kg_key in _state:
            return {**_state[kg_key].to_dict(), "analyzed": True}

        # Try loading from disk (survives server restarts)
        import json as _json
        import pathlib as _pl
        import os as _os2

        _kg_file = (
            _pl.Path(_os2.path.expanduser("~/.sqlagent/kg")) / f"{workspace_id or 'default'}.json"
        )
        if _kg_file.exists():
            try:
                return _json.loads(_kg_file.read_text())
            except Exception as exc:
                logger.debug("server.operation_failed", error=str(exc))

        # Build basic graph from raw introspection (no LLM)
        nodes = []
        edges = []
        for sid, conn in agent.services.connectors.items():
            try:
                snap = await conn.introspect()
                for t in snap.tables:
                    nodes.append(
                        {
                            "id": f"tbl:{t.name}",
                            "type": "table",
                            "name": t.name,
                            "source_id": sid,
                            "x": 0,
                            "y": 0,
                            "properties": {
                                "row_count": t.row_count_estimate,
                                "column_count": len(t.columns),
                                "columns": [
                                    {
                                        "name": c.name,
                                        "data_type": c.data_type,
                                        "is_pk": c.is_primary_key,
                                        "is_fk": c.is_foreign_key,
                                        "fk_ref": c.foreign_key_ref,
                                    }
                                    for c in t.columns
                                ],
                            },
                        }
                    )
                for fk in snap.foreign_keys:
                    edges.append(
                        {
                            "id": f"fk:{fk.from_table}.{fk.from_column}->{fk.to_table}.{fk.to_column}",
                            "source": f"tbl:{fk.from_table}",
                            "target": f"tbl:{fk.to_table}",
                            "type": "declared_fk",
                            "label": f"{fk.from_column} → {fk.to_column}",
                            "description": f"{fk.from_table} belongs to {fk.to_table} (via {fk.from_column})",
                            "from_table": fk.from_table,
                            "to_table": fk.to_table,
                            "from_column": fk.from_column,
                            "to_column": fk.to_column,
                            "confidence": 1.0,
                        }
                    )
            except Exception as exc:
                logger.debug("server.operation_failed", error=str(exc))

        return {"nodes": nodes, "edges": edges, "layers": [], "analyzed": False}

    @app.post("/schema/analyze", tags=["schema"])
    async def run_schema_analysis(
        request: Request, workspace_id: str = "", user=Depends(get_current_user)
    ):
        """Trigger the SchemaAgent LLM analysis pipeline. Returns the knowledge graph."""
        try:
            body = await request.json()
            workspace_id = body.get("workspace_id", workspace_id)
        except Exception as exc:
            logger.debug(
                "server.operation_failed", error=str(exc)
            )  # No body is fine — workspace_id comes from query param
        agent = await _get_or_create_agent(workspace_id, user.user_id)
        if not agent.services:
            return {"error": "No data sources connected"}

        # Collect snapshots + connectors
        snapshots = {}
        connectors = {}
        for sid, conn in agent.services.connectors.items():
            try:
                snap = await conn.introspect()
                snapshots[sid] = snap
                connectors[sid] = conn
            except Exception as exc:
                logger.debug("server.operation_failed", error=str(exc))

        if not snapshots:
            return {"error": "No schemas to analyze"}

        # Run the SchemaAgent
        from sqlagent.agents import SchemaAgent

        schema_agent = SchemaAgent(llm=agent.services.llm, embedder=agent.services.embedder)
        kg = await schema_agent.analyze(
            workspace_id=workspace_id or "default",
            snapshots=snapshots,
            connectors=connectors,
        )

        # Cache the knowledge graph in memory + persist to disk
        kg_key = f"kg_{workspace_id or 'default'}"
        _state[kg_key] = kg

        # Cache enriched snapshots (with SchemaColumn.examples) on the agent's services
        # so prune_node can use them for future queries without re-sampling
        if hasattr(agent, "services") and agent.services:
            agent.services._enriched_snapshots = snapshots

        # Persist to disk so server restarts don't lose analysis
        import json as _json
        import pathlib as _pl
        import os as _os2

        _kg_dir = _pl.Path(_os2.path.expanduser("~/.sqlagent/kg"))
        _kg_dir.mkdir(parents=True, exist_ok=True)
        _kg_file = _kg_dir / f"{workspace_id or 'default'}.json"
        try:
            _kg_file.write_text(_json.dumps({**kg.to_dict(), "analyzed": True}, default=str))
        except Exception as exc:
            logger.debug("server.operation_failed", error=str(exc))

        return {**kg.to_dict(), "analyzed": True}

    @app.post("/schema/chat", tags=["schema"])
    async def schema_chat(request: Request, workspace_id: str = "", user=Depends(get_current_user)):
        """Chat with the Schema Agent about the data schema. Q&As feed into SQL Agent context."""
        import re as _re

        body = await request.json()
        workspace_id = body.get("workspace_id", workspace_id) or ""
        message = body.get("message", "").strip()
        if not message:
            return {"answer": "Please provide a question."}

        agent = await _get_or_create_agent(workspace_id, user.user_id)
        if not agent.services:
            return {"answer": "No data sources connected. Please connect a database first."}

        # Build schema context from the cached KG (rich) or raw introspection (basic)
        kg_key = f"kg_{workspace_id or 'default'}"
        schema_context_parts = []
        kg_obj = _state.get(kg_key)

        # Try disk cache if not in memory
        if not kg_obj:
            import json as _json2
            import pathlib as _pl2
            import os as _os3

            _kg_f = (
                _pl2.Path(_os3.path.expanduser("~/.sqlagent/kg"))
                / f"{workspace_id or 'default'}.json"
            )
            if _kg_f.exists():
                try:
                    raw = _json2.loads(_kg_f.read_text())
                    # Build quick context from raw dict
                    schema_context_parts.append("## Tables")
                    for n in raw.get("nodes", []):
                        if n.get("type") == "table":
                            p = n.get("properties", {})
                            cols = ", ".join(
                                f"{c['name']} ({c.get('data_type', '')})"
                                for c in p.get("columns", [])[:12]
                            )
                            schema_context_parts.append(
                                f"**{n['name']}** ({p.get('row_count', 0):,} rows): {cols}"
                            )
                    for layer in raw.get("layers", []):
                        schema_context_parts.append(
                            f"\nEntity Group **{layer['name']}**: {layer.get('description', '')}"
                        )
                    for edge in raw.get("edges", [])[:20]:
                        schema_context_parts.append(f"Relationship: {edge.get('description', '')}")
                    for entry in raw.get("glossary", []):
                        schema_context_parts.append(
                            f"Glossary — **{entry['term']}**: {entry.get('definition', '')}"
                        )
                except Exception as exc:
                    logger.debug("server.operation_failed", error=str(exc))

        if kg_obj:
            schema_context_parts.append("## Tables")
            for n in kg_obj.nodes:
                if n.type == "table":
                    cols_info = n.properties.get("columns", [])
                    col_str = ", ".join(
                        f"{c['name']} ({c.get('data_type', '')})" for c in cols_info[:12]
                    )
                    if not col_str:
                        # fall back to column-type KGNodes
                        col_nodes = [
                            x
                            for x in kg_obj.nodes
                            if x.type == "column"
                            and x.id.startswith(f"col:{n.source_id}.{n.name}.")
                        ]
                        col_str = ", ".join(
                            f"{x.name} ({x.properties.get('data_type', '')})"
                            for x in col_nodes[:12]
                        )
                    schema_context_parts.append(
                        f"**{n.name}** ({n.properties.get('row_count', 0):,} rows): {col_str}"
                    )
            for layer in kg_obj.layers:
                schema_context_parts.append(f"\nEntity Group **{layer.name}**: {layer.description}")
                schema_context_parts.append(
                    f"  Tables: {', '.join(t.split('.')[-1] for t in layer.tables)}"
                )
            for edge in kg_obj.edges[:20]:
                schema_context_parts.append(
                    f"Relationship: {edge.description} ({int(edge.confidence * 100)}% confidence)"
                )
            if kg_obj.glossary:
                schema_context_parts.append("\n## Existing Glossary")
                for entry in kg_obj.glossary:
                    schema_context_parts.append(f"- **{entry.term}**: {entry.definition}")

        if not schema_context_parts:
            # Bare-metal fallback: introspect live
            for sid, conn in agent.services.connectors.items():
                try:
                    snap = await conn.introspect()
                    schema_context_parts.append(f"\nDatabase: {sid}")
                    for t in snap.tables[:20]:
                        cols = ", ".join(f"{c.name} ({c.data_type})" for c in t.columns[:10])
                        schema_context_parts.append(f"  {t.name}: {cols}")
                except Exception as exc:
                    logger.debug("server.operation_failed", error=str(exc))

        if not schema_context_parts:
            return {"answer": "No schema available. Please run schema analysis first."}

        schema_context = "\n".join(schema_context_parts)

        system_prompt = (
            "You are a Schema Intelligence assistant embedded in a data analytics platform.\n"
            "Your role is to help users deeply understand their database schema, relationships, "
            "and business concepts — so they can ask better natural language questions.\n\n"
            f"{schema_context}\n\n"
            "Guidelines:\n"
            "- Be specific: reference exact table/column names\n"
            "- Explain relationships in plain English (avoid raw SQL unless asked)\n"
            "- If the user defines or confirms a business term, extract it in this format on its own line:\n"
            "  GLOSSARY: <term> = <definition>\n"
            "- If the user asks how to query something, describe which tables/joins are needed\n"
            "- Keep answers concise (3–6 sentences unless more detail is needed)"
        )

        try:
            response = await agent.services.llm.complete(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": message},
                ]
            )
            raw_answer = response.content if hasattr(response, "content") else str(response)

            # Extract GLOSSARY: lines
            glossary_updates = []
            for match in _re.finditer(r"GLOSSARY:\s*([^=\n]+?)\s*=\s*([^\n]+)", raw_answer):
                glossary_updates.append(
                    {"term": match.group(1).strip(), "definition": match.group(2).strip()}
                )
            clean_answer = _re.sub(r"\nGLOSSARY:.*", "", raw_answer).strip()

            # Persist new glossary entries to the KG
            if glossary_updates:
                from sqlagent.models import SemanticEntry
                import json as _json3
                import pathlib as _pl3
                import os as _os4

                target_kg = _state.get(kg_key)
                if target_kg:
                    existing = {g.term.lower() for g in target_kg.glossary}
                    for gu in glossary_updates:
                        if gu["term"].lower() not in existing:
                            target_kg.glossary.append(
                                SemanticEntry(term=gu["term"], definition=gu["definition"])
                            )
                            existing.add(gu["term"].lower())
                    # Persist to disk
                    _kg_dir2 = _pl3.Path(_os4.path.expanduser("~/.sqlagent/kg"))
                    _kg_dir2.mkdir(parents=True, exist_ok=True)
                    try:
                        (_kg_dir2 / f"{workspace_id or 'default'}.json").write_text(
                            _json3.dumps({**target_kg.to_dict(), "analyzed": True}, default=str)
                        )
                    except Exception as exc:
                        logger.debug("server.operation_failed", error=str(exc))
                # Also store as schema insight on agent services semantic layer
                if (
                    hasattr(agent, "services")
                    and agent.services
                    and hasattr(agent.services, "_semantic_layer")
                ):
                    sl = agent.services._semantic_layer
                    if hasattr(sl, "glossary") and isinstance(sl.glossary, list):
                        from sqlagent.schema.semantic_layer import GlossaryEntry

                        for gu in glossary_updates:
                            sl.glossary.append(
                                GlossaryEntry(term=gu["term"], definition=gu["definition"])
                            )

            return {"answer": clean_answer, "glossary_updates": glossary_updates}
        except Exception as exc:
            return {"answer": f"Schema Agent error: {exc}", "glossary_updates": []}

    # ── Task routes ───────────────────────────────────────────────────────────

    @app.get("/workspaces/{workspace_id}/tasks", tags=["tasks"])
    async def list_tasks(workspace_id: str, limit: int = 50, user=Depends(get_current_user)):
        traces = await _state["trace_store"].list_for_workspace(workspace_id, limit=limit)
        return {"tasks": traces, "total": len(traces)}

    @app.get("/workspaces/{workspace_id}/tasks/{trace_id}", tags=["tasks"])
    async def get_task(workspace_id: str, trace_id: str, user=Depends(get_current_user)):
        trace = await _state["trace_store"].get(trace_id)
        if not trace:
            raise HTTPException(404, "Task not found")
        return trace.to_dict()

    # ── Training routes ───────────────────────────────────────────────────────

    @app.post("/train/sql", tags=["train"])
    async def train_sql(request: Request, user=Depends(get_current_user)):
        body = await request.json()
        workspace_id = body.get("workspace_id", "")
        agent = await _get_or_create_agent(workspace_id, user.user_id)
        nl = body.get("nl_query", "")
        sql = body.get("sql", "")
        source = body.get("source", body.get("source_id", ""))
        correction_note = body.get("correction_note", "")
        original_sql = body.get("original_sql", "")
        failure_type = body.get("failure_type", "")
        failed_node = body.get("failed_node", "")
        trace_events = body.get("trace_events", [])

        from sqlagent.agents import LearningLoop

        loop = LearningLoop(agent.services.example_store)

        if source == "correction":
            # ── Correction path: save training pair + LessonRecord ────────────
            extracted_lesson = body.get("extracted_lesson", "")
            domain_insight = body.get("domain_insight", "")
            what_changed = body.get("what_changed", "")
            rows_preview = body.get("rows_preview", [])
            columns_preview = body.get("columns_preview", [])
            row_count_preview = body.get("row_count_preview", 0)
            tokens_used_regen = body.get("tokens_used", 0)
            cost_usd_regen = body.get("cost_usd", 0.0)

            result = await loop.on_correction(
                nl_query=nl,
                corrected_sql=sql,
                original_sql=original_sql,
                trace_events=trace_events,
                failure_type=failure_type,
                failed_node=failed_node,
                correction_note=correction_note,
            )

            # ── Persist semantic lesson to workspace context (in-memory + note) ─
            lesson_to_save = extracted_lesson or correction_note
            if lesson_to_save and hasattr(agent, "_data_context_notes"):
                if lesson_to_save not in agent._data_context_notes:
                    agent._data_context_notes.append(lesson_to_save)
                    if len(agent._data_context_notes) > 20:
                        agent._data_context_notes = agent._data_context_notes[-20:]
                    logger.info("learn.context_note_saved", lesson=lesson_to_save[:80])

            # ── Persist LessonRecord to SQLite ────────────────────────────────
            # This is the full chain: original → domain insight → rule → corrected SQL
            import uuid as _uuid
            import json as _json
            from datetime import datetime, timezone as _tz
            from sqlagent.telemetry import LessonRecord

            lesson_record = LessonRecord(
                lesson_id="lesson_" + _uuid.uuid4().hex[:12],
                workspace_id=workspace_id or "",
                user_id=user.user_id,
                nl_query=nl,
                original_sql=original_sql,
                corrected_sql=sql,
                domain_insight=domain_insight,
                context_rule=lesson_to_save or "",
                what_changed=what_changed,
                rows_preview=_json.dumps(rows_preview[:5]),
                columns=_json.dumps(columns_preview),
                row_count=row_count_preview,
                failed_stage=result.get("failed_stage", ""),
                failed_node=failed_node,
                pair_id=result.get("pair_id", ""),
                tokens_used=tokens_used_regen,
                cost_usd=cost_usd_regen,
                created_at=datetime.now(_tz.utc).isoformat(),
            )
            try:
                await _state["lesson_store"].save(lesson_record)
            except Exception as _e:
                logger.warn("learn.lesson_store_save_failed", error=str(_e))

            # Push to agent activity feed (in-memory ring buffer)
            if hasattr(agent, "_learn_activity"):
                agent._learn_activity.insert(
                    0,
                    {
                        "action": "correction",
                        "nl_query": nl[:60],
                        "failed_stage": result.get("failed_stage", ""),
                        "lesson": lesson_to_save[:80] if lesson_to_save else "",
                        "ts": datetime.now(_tz.utc).isoformat(),
                    },
                )
                agent._learn_activity = agent._learn_activity[:50]

            logger.info(
                "learning.correction",
                failed_stage=result.get("failed_stage"),
                has_lesson=bool(lesson_to_save),
                lesson_id=lesson_record.lesson_id,
            )
            return {
                "example_id": result.get("pair_id", ""),
                "lesson_id": lesson_record.lesson_id,
                "message": result.get("message", "Correction saved"),
                "failed_stage": result.get("failed_stage", ""),
                "lesson_saved": lesson_to_save[:80] if lesson_to_save else "",
            }
        else:
            # Thumbs-up / manual training pair
            example_id = await agent.train_sql(nl, sql, source)
            logger.info("training.saved", source=source)
            return {"example_id": example_id, "message": "Training pair saved"}

    @app.post("/learn/regenerate", tags=["train"])
    async def learn_regenerate(request: Request, user=Depends(get_current_user)):
        """
        Learn Agent — runs the correction through the LearnGraph (LangGraph).

        Pipeline (each node has an OTel span):
          understand_correction → analyze_schema → rewrite_sql → execute_corrected → extract_lesson

        Returns the full LearnState so the user can review the rewritten SQL,
        domain insight, extracted lesson, and actual query results BEFORE saving.
        Saving happens via POST /train/sql with source="correction".
        """
        body = await request.json()
        workspace_id = body.get("workspace_id", "")
        agent = await _get_or_create_agent(workspace_id, user.user_id)

        from sqlagent.graph.learn_graph import LearnState

        initial_state: LearnState = {
            "nl_query": body.get("nl_query", ""),
            "original_sql": body.get("original_sql", ""),
            "failure_type": body.get("failure_type", ""),
            "failed_node": body.get("failed_node", ""),
            "correction_note": body.get("correction_note", ""),
            "trace_events": body.get("trace_events", []),
            "workspace_id": workspace_id,
            "user_id": user.user_id,
            "source_id": body.get("source_id", ""),
            "tokens_used": 0,
            "cost_usd": 0.0,
            "learn_trace_events": [],
        }

        # Run the LearnGraph — 5 OTel-traced nodes
        try:
            learn_graph = getattr(agent, "_learn_graph", None)
            if learn_graph is None:
                raise HTTPException(
                    status_code=501, detail="Learn Agent not initialized for this workspace"
                )
            final_state = await learn_graph.ainvoke(initial_state)
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Learn Agent failed: {exc}")

        logger.info(
            "learn.regenerated",
            workspace_id=workspace_id,
            failed_stage=final_state.get("failed_stage", ""),
            has_results=bool(final_state.get("rows")),
            has_lesson=bool(final_state.get("extracted_lesson")),
            tokens=final_state.get("tokens_used", 0),
        )

        return {
            "regenerated_sql": final_state.get("rewritten_sql", ""),
            "what_changed": final_state.get("what_changed", ""),
            "domain_insight": final_state.get("domain_insight", ""),
            "extracted_lesson": final_state.get("extracted_lesson", ""),
            "rows": final_state.get("rows", []),
            "columns": final_state.get("columns", []),
            "row_count": final_state.get("row_count", 0),
            "exec_error": final_state.get("exec_error", ""),
            "tokens_used": final_state.get("tokens_used", 0),
            "learn_trace_events": final_state.get("learn_trace_events", []),
        }

    @app.get("/hub/packs", tags=["hub"])
    async def list_hub_packs():
        from sqlagent.hub import list_packs

        return {"packs": list_packs()}

    @app.post("/hub/install", tags=["hub"])
    async def install_hub_pack(request: Request, user=Depends(get_current_user)):
        body = await request.json()
        agent = await _get_or_create_agent("", user.user_id)
        from sqlagent.hub import install_pack

        pack_name = body.get("pack_name", "")
        count = await install_pack(pack_name, agent.services.example_store)
        return {"installed": count, "pack": pack_name}

    # ── Feedback ──────────────────────────────────────────────────────────────

    @app.post("/api/feedback", tags=["train"])
    async def submit_feedback(request: Request, user=Depends(get_current_user)):
        body = await request.json()
        agent = await _get_or_create_agent("", user.user_id)
        from sqlagent.agents import LearningLoop

        loop = LearningLoop(agent.services.example_store)
        if body.get("feedback") == "thumbs_up":
            await loop.on_thumbs_up(body.get("nl_query", ""), body.get("sql", ""))
            return {"message": "Training pair saved from positive feedback"}
        elif body.get("feedback") == "thumbs_down" and body.get("corrected_sql"):
            await loop.on_correction(body.get("nl_query", ""), body.get("corrected_sql", ""))
            return {"message": "Corrected training pair saved"}
        return {"message": "Feedback recorded"}

    @app.get("/api/learn/status", tags=["train"])
    async def get_learn_status(workspace_id: str = "", user=Depends(get_current_user)):
        """Live Learn Agent status: training pairs, auto-learn activity, SOUL progress."""
        agent = None
        try:
            agent = await _get_or_create_agent(workspace_id, user.user_id)
        except Exception as exc:
            logger.debug("server.operation_failed", error=str(exc))

        # Training pair count from vector store
        training_pairs = 0
        if agent and agent.services.example_store:
            try:
                training_pairs = await agent.services.example_store.count()
            except Exception as exc:
                logger.debug("server.operation_failed", error=str(exc))
                training_pairs = 0

        # SOUL profile
        soul_profile = None
        query_count = 0
        soul_evolutions = getattr(agent, "_soul_evolutions", 0) if agent else 0
        if agent and agent.services.soul:
            try:
                sp = agent.services.soul._profiles.get(user.user_id or "local")
                if sp:
                    soul_profile = {
                        "query_count": sp.query_count,
                        "accountability_unit": sp.accountability_unit,
                        "time_reference": sp.time_reference,
                        "vocabulary_map": sp.vocabulary_map,
                    }
                    query_count = sp.query_count
            except Exception as exc:
                logger.debug("server.operation_failed", error=str(exc))

        # Recent activity from agent (in-memory ring buffer)
        recent_activity = getattr(agent, "_learn_activity", [])[:10] if agent else []

        # SOUL progress toward next evolution (every 20 queries)
        soul_next_in = max(0, 20 - (query_count % 20)) if query_count > 0 else 20

        # Auto-learn enabled?
        if agent and hasattr(agent, "_auto_learn"):
            auto_learn_enabled = bool(agent._auto_learn)
        else:
            raw = _state.get("user_settings", {}).get("auto_learn", True)
            auto_learn_enabled = bool(raw) if raw is not None else True

        # Workspace semantic context notes (from user corrections — in-memory)
        data_context_notes = getattr(agent, "_data_context_notes", []) if agent else []

        # Persistent correction count from LessonStore (survives server restarts)
        correction_count = 0
        lesson_records: list[dict] = []
        workspace_id_for_lessons = workspace_id or ""
        try:
            correction_count = await _state["lesson_store"].count(workspace_id_for_lessons)
            lesson_records = await _state["lesson_store"].list_for_workspace(
                workspace_id_for_lessons, limit=10
            )
        except Exception as exc:
            logger.debug("server.operation_failed", error=str(exc))

        return {
            "status": "active" if auto_learn_enabled else "paused",
            "auto_learn_enabled": auto_learn_enabled,
            # Real counts — no fake accuracy stat
            "training_pairs": training_pairs,  # from vector store
            "correction_count": correction_count,  # from LessonStore (SQLite)
            "context_rules_count": len(data_context_notes),
            "soul_query_count": query_count,
            "soul_next_in": soul_next_in,
            "soul_progress_pct": min(100, round((20 - soul_next_in) / 20 * 100)),
            "soul_evolutions": soul_evolutions,
            "recent_activity": recent_activity,
            "soul_profile": soul_profile,
            "data_context_notes": data_context_notes,
            "lesson_records": lesson_records,  # full chain, shown as Lesson Cards in UI
        }

    @app.get("/api/learning/stats", tags=["train"])
    async def api_learning_stats(workspace_id: str = "", user=Depends(get_current_user)):
        """Alias at /api/learning/stats — matches the path expected by the Home dashboard UI."""
        return await get_learn_status(workspace_id=workspace_id, user=user)

    # ── Flat task endpoints (workspace-agnostic, for UI convenience) ──────────

    @app.get("/api/tasks", tags=["tasks"])
    async def list_all_tasks(
        limit: int = 50, workspace_id: str = "", user=Depends(get_current_user)
    ):
        """Flat task list across all workspaces (or filtered by workspace_id query param).
        Convenience wrapper around /workspaces/{id}/tasks for the Tasks view."""
        try:
            if workspace_id:
                traces = await _state["trace_store"].list_for_workspace(workspace_id, limit=limit)
            else:
                # Collect from all user workspaces
                workspaces = await _state["workspace_store"].list_for_user(user.user_id)
                traces = []
                for ws in workspaces:
                    ws_traces = await _state["trace_store"].list_for_workspace(
                        ws.workspace_id, limit=limit
                    )
                    traces.extend(ws_traces)
                traces = sorted(
                    traces,
                    key=lambda t: t.get("created_at") or t.get("started_at") or "",
                    reverse=True,
                )[:limit]
        except Exception as exc:
            logger.debug("api_tasks.failed", error=str(exc))
            traces = []
        return {"tasks": traces, "total": len(traces)}

    @app.get("/api/tasks/{trace_id}", tags=["tasks"])
    async def get_task_by_id(trace_id: str, user=Depends(get_current_user)):
        """Get a single task by trace_id (workspace-agnostic). Used by TaskDetail view."""
        trace = await _state["trace_store"].get(trace_id)
        if not trace:
            raise HTTPException(404, "Task not found")
        return trace.to_dict() if hasattr(trace, "to_dict") else trace

    # ── Learn Agent — lesson management ───────────────────────────────────────

    @app.get("/api/learn/lessons", tags=["train"])
    async def list_lessons(workspace_id: str = "", user=Depends(get_current_user)):
        """List all lesson records for a workspace (the Correction Log in UI)."""
        records = await _state["lesson_store"].list_for_workspace(workspace_id, limit=50)
        return {"lessons": records, "total": len(records)}

    @app.delete("/api/learn/lessons/{lesson_id}", tags=["train"])
    async def delete_lesson(lesson_id: str, user=Depends(get_current_user)):
        """Remove a lesson record (does NOT remove the training pair from vector store)."""
        await _state["lesson_store"].delete(lesson_id)
        return {"deleted": lesson_id}

    @app.delete("/api/learn/context-notes/{idx}", tags=["train"])
    async def delete_context_note(idx: int, workspace_id: str = "", user=Depends(get_current_user)):
        """Remove a learned context rule by index."""
        try:
            agent = await _get_or_create_agent(workspace_id, user.user_id)
        except Exception as exc:
            logger.debug("server.operation_failed", error=str(exc))
            raise HTTPException(status_code=404, detail="Agent not found")
        notes = getattr(agent, "_data_context_notes", [])
        if idx < 0 or idx >= len(notes):
            raise HTTPException(status_code=404, detail="Rule index out of range")
        agent._data_context_notes = [n for i, n in enumerate(notes) if i != idx]
        return {"removed": notes[idx], "remaining": len(agent._data_context_notes)}

    # ── Streaming greeting — completely isolated from the setup conversation ──
    _GREET_SYSTEM = (
        "You are ora, a data tool that lets people query any database in plain English.\n\n"
        "Write ONE short welcome message addressed to the user by name.\n\n"
        "STRICT OUTPUT RULES:\n"
        "- Output ONLY the message text. No labels, no headers, no numbering, no markdown.\n"
        "- No 'Opening Message', no '#', no '---', no '1.', no bullet dashes.\n"
        "- Do NOT generate multiple versions. ONE message only. Pure flowing prose.\n"
        "- Under 100 words.\n\n"
        "Weave in ONE capability (vary each time):\n"
        "- querying multiple databases at once with zero SQL\n"
        "- self-correcting SQL up to 3 times automatically\n"
        "- a live trace showing the agent reasoning step by step\n"
        "- learning your team vocabulary after a few queries\n\n"
        "Include a brief vivid metaphor for how plain English becomes SQL.\n"
        "End with one sentence inviting them to ask their first question.\n"
        "Every message must feel completely different in tone, metaphor, phrasing."
    )

    @app.post("/workspaces/{workspace_id}/setup/greet", tags=["workspaces"])
    async def setup_greet(workspace_id: str, request: Request, user=Depends(get_current_user)):
        """Stream an LLM-generated greeting — always uses a fast model for low TTFT."""
        body = await request.json()
        user_name = body.get("user_name", "there")
        user_settings = _state.get("user_settings", {})
        configured_model = user_settings.get("model") or cfg.llm_model

        # Always use the fastest available model for greetings regardless of selected query model
        # This keeps TTFT under ~300ms. Query model is used for actual SQL work.
        anthropic_key = user_settings.get("anthropic_key") or os.environ.get(
            "ANTHROPIC_API_KEY", ""
        )
        openai_key = user_settings.get("openai_key") or os.environ.get("OPENAI_API_KEY", "")

        if anthropic_key:
            greet_model = (
                configured_model if "claude" in configured_model.lower() else "claude-haiku-4-5"
            )
            api_key = anthropic_key
        elif openai_key:
            greet_model = "gpt-4o-mini"
            api_key = openai_key
        else:
            # Fall back to configured model if no key override
            greet_model = configured_model
            is_anthropic = "claude" in configured_model or "anthropic" in configured_model
            api_key = anthropic_key if is_anthropic else openai_key

        async def _generate():
            try:
                import litellm

                kwargs: dict = {
                    "model": greet_model,
                    "messages": [
                        {"role": "system", "content": _GREET_SYSTEM},
                        {
                            "role": "user",
                            "content": f"User name: {user_name}. Write the greeting now.",
                        },
                    ],
                    "stream": True,
                    "max_tokens": 130,
                }
                if api_key:
                    kwargs["api_key"] = api_key

                response = await litellm.acompletion(**kwargs)
                async for chunk in response:
                    delta = chunk.choices[0].delta.content or ""
                    if delta:
                        yield f"data: {json.dumps({'text': delta})}\n\n"
            except Exception as exc:
                yield f"data: {json.dumps({'error': str(exc)[:120]})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            _generate(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/workspaces/{workspace_id}/setup/chat", tags=["workspaces"])
    async def setup_chat(workspace_id: str, request: Request, user=Depends(get_current_user)):
        body = await request.json()
        # Persist SetupAgent per workspace so conversation history is maintained
        if "setup_agents" not in _state:
            _state["setup_agents"] = {}
        user_settings = _state.get("user_settings", {})
        model = user_settings.get("model") or cfg.llm_model

        if workspace_id not in _state["setup_agents"]:
            from sqlagent.agents import SetupAgent
            from sqlagent.llm import LiteLLMProvider

            llm = LiteLLMProvider(model=model)
            _state["setup_agents"][workspace_id] = SetupAgent(llm=llm)

        agent = _state["setup_agents"][workspace_id]
        # Always sync the model in case settings changed since the agent was created
        if hasattr(agent._llm, "model") and agent._llm.model != model:
            agent._llm.model = model

        message = body.get("message", "")

        events = []
        async for event in agent.chat(workspace_id, message):
            events.append({"type": event.event_type, "data": event.data})

        # If workspace_ready, clean up the setup agent
        for ev in events:
            if ev.get("type") == "workspace_ready":
                _state["setup_agents"].pop(workspace_id, None)
                break

        return {"events": events}

    # ── File Upload ────────────────────────────────────────────────────────────

    @app.post("/api/upload", tags=["settings"])
    async def upload_file(
        file: UploadFile = File(...), workspace_id: str = Form(""), user=Depends(get_current_user)
    ):
        """Upload CSV/XLSX/Parquet file → save to workspace uploads dir, return metadata."""
        import shutil

        upload_dir = os.path.join(
            os.path.expanduser("~"), ".sqlagent", "uploads", workspace_id or "default"
        )
        os.makedirs(upload_dir, exist_ok=True)

        # Sanitize filename to prevent path traversal
        import re as _re

        safe_name = os.path.basename(file.filename or "upload")
        safe_name = _re.sub(r"[^\w.\-]", "_", safe_name)
        if not safe_name or safe_name.startswith("."):
            safe_name = f"upload_{uuid.uuid4().hex[:8]}"
        # Save file
        file_path = os.path.join(upload_dir, safe_name)
        # Verify final path is within upload_dir (defense in depth)
        if not os.path.abspath(file_path).startswith(os.path.abspath(upload_dir)):
            raise HTTPException(400, "Invalid filename")
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # Introspect via DuckDB/file connector
        try:
            from sqlagent.connectors import ConnectorRegistry
            from sqlagent.connectors.file_connector import FileConnector
            from sqlagent.data_profiler import profile_file

            source_id = f"file_{safe_name.rsplit('.', 1)[0].lower().replace(' ', '_')}"

            # ── LLM data profiling (one call, result stored on the connector) ──
            # Runs before connect() so the connector uses the correct read args.
            # Gracefully skipped for non-CSV formats or when no LLM key is set.
            read_config = None
            ext = os.path.splitext(safe_name)[1].lower()
            if ext in (".csv", ".tsv", ".txt", ".json"):
                try:
                    llm = None
                    llm_model = ""
                    try:
                        # Reuse the workspace agent's LLM if one is already live
                        agent_tmp = _state.get(f"agent_{workspace_id or 'default'}_{user.user_id}")
                        if agent_tmp and hasattr(agent_tmp, "_services") and agent_tmp._services:
                            llm = agent_tmp._services.llm
                            llm_model = getattr(llm, "_model", "")
                    except Exception:
                        pass

                    read_config = await profile_file(
                        file_path=file_path,
                        source_id=source_id,
                        llm=llm,
                        model=llm_model,
                    )
                    logger.info(
                        "upload.profiled",
                        file=file.filename,
                        null_strings=read_config.extra_null_strings,
                        duckdb_args=read_config.duckdb_args,
                        casts=list(read_config.cast_exprs.keys()),
                    )
                except Exception as profile_exc:
                    logger.warning(
                        "upload.profile_failed", file=file.filename, error=str(profile_exc)
                    )
                    read_config = None

            # Build connector — pass read_config if it's a file we can profile
            if read_config is not None:
                conn = FileConnector(
                    source_id=source_id, file_path=file_path, read_config=read_config
                )
            else:
                conn = ConnectorRegistry.from_url(source_id, file_path)

            await conn.connect()
            snap = await conn.introspect()

            # Register with workspace agent
            agent = await _get_or_create_agent(workspace_id, user.user_id)
            await agent._ensure_ready()
            agent.services.connectors[source_id] = conn

            # Persist source to workspace store so it survives agent recreation
            if workspace_id:
                try:
                    await _state["workspace_store"].add_source(
                        workspace_id,
                        {
                            "source_id": source_id,
                            "type": "file",
                            "file_path": file_path,
                            "filename": safe_name,
                            "dialect": snap.dialect,
                        },
                    )
                    logger.info("upload.source_persisted", workspace=workspace_id, source=source_id)
                except Exception as e:
                    logger.error(
                        "upload.source_persist_failed", workspace=workspace_id, error=str(e)
                    )

            # Invalidate cached knowledge graph + agent (new data = rebuild everything)
            kg_key = f"kg_{workspace_id or 'default'}"
            _state.pop(kg_key, None)
            # Don't invalidate agent here — the connector is already registered on it above
            # But if the agent was cached from a previous session without this file, force refresh
            # The connector is live on the current agent instance already

            # Build profiler summary for the UI ("2 null encodings fixed, 1 currency cast")
            profiler_summary = None
            if read_config is not None:
                fixes = []
                if read_config.extra_null_strings:
                    fixes.append(
                        f"{len(read_config.extra_null_strings)} null encoding(s) normalised"
                    )
                if read_config.cast_exprs:
                    fixes.append(f"{len(read_config.cast_exprs)} column type(s) fixed")
                if read_config.duckdb_args.get("dateformat"):
                    fixes.append(f"date format detected ({read_config.duckdb_args['dateformat']})")
                profiler_summary = "; ".join(fixes) if fixes else "no issues found"

            return {
                "filename": safe_name,
                "file_path": file_path,
                "source_id": source_id,
                "data_quality": profiler_summary,
                "dialect": snap.dialect,
                "table_count": snap.table_count,
                "column_count": snap.column_count,
                "tables": [
                    {"name": t.name, "row_count": t.row_count_estimate, "columns": len(t.columns)}
                    for t in snap.tables
                ],
            }
        except Exception as e:
            return {"filename": safe_name, "file_path": file_path, "error": str(e)}

    # ── Settings ───────────────────────────────────────────────────────────────

    @app.get("/api/settings", tags=["settings"])
    async def get_settings(user=Depends(get_current_user)):
        import os as _os

        stored = dict(_state.get("user_settings", {}))
        # Tell the UI which env vars are active so it can show helpful hints
        stored["_env_anthropic"] = bool(_os.environ.get("ANTHROPIC_API_KEY", "").strip())
        stored["_env_openai"] = bool(_os.environ.get("OPENAI_API_KEY", "").strip())
        return stored

    @app.post("/api/settings", tags=["settings"])
    async def save_settings(request: Request, user=Depends(get_current_user)):
        body = await request.json()
        # Merge into existing settings so partial saves don't wipe unrelated keys
        current = dict(_state.get("user_settings", {}))
        current.update({k: v for k, v in body.items() if v is not None and v != ""})
        _state["user_settings"] = current
        body = current
        # Apply API keys to environment
        import os
        import json as _json

        if body.get("openai_key"):
            os.environ["OPENAI_API_KEY"] = body["openai_key"]
        if body.get("anthropic_key"):
            os.environ["ANTHROPIC_API_KEY"] = body["anthropic_key"]
        # Persist to disk so settings survive server restarts
        try:
            _settings_path = os.path.join(
                os.path.expanduser("~"), ".sqlagent", "user_settings.json"
            )
            os.makedirs(os.path.dirname(_settings_path), exist_ok=True)
            with open(_settings_path, "w") as _f:
                _json.dump(body, _f, indent=2)
        except Exception as exc:
            logger.debug("server.operation_failed", error=str(exc))
        # Apply model and auto_learn to all cached agents immediately
        selected_model = body.get("model", "")
        auto_learn = body.get("auto_learn", None)
        for agent_obj in _state.get("agents", {}).values():
            if (
                selected_model
                and hasattr(agent_obj, "_services")
                and agent_obj._services
                and hasattr(agent_obj._services, "llm")
            ):
                agent_obj._services.llm.model = selected_model
            if auto_learn is not None and hasattr(agent_obj, "_auto_learn"):
                agent_obj._auto_learn = auto_learn is not False
        # Full invalidation when API keys or model change — agents must be recreated
        if body.get("openai_key") or body.get("anthropic_key") or body.get("model"):
            _state["agents"] = {}
            _state["setup_agents"] = {}  # setup agents also cache the LLM provider
        return {"saved": True, "model": selected_model}

    @app.post("/api/test-key", tags=["settings"])
    async def test_api_key(request: Request):
        """Validate an API key by making a minimal LLM call. No auth required for onboarding."""
        body = await request.json()
        key = body.get("key", "").strip()
        provider = body.get("provider", "openai").lower()
        if not key:
            return {"valid": False, "error": "No key provided"}
        try:
            import litellm

            model_id = "claude-haiku-4-5" if provider == "anthropic" else "gpt-4o-mini"
            # litellm.completion is sync — run in a thread so we don't block the event loop
            await asyncio.to_thread(
                litellm.completion,
                model=model_id,
                messages=[{"role": "user", "content": "Hi"}],
                api_key=key,
                max_tokens=5,
            )
            return {"valid": True}
        except Exception as exc:
            err_str = str(exc).lower()
            if "auth" in err_str or "key" in err_str or "invalid" in err_str or "401" in err_str:
                return {"valid": False, "error": "Key rejected by provider"}
            # Unexpected error (rate limit, etc.) — still return valid=True as the key likely works
            return {"valid": True, "warning": str(exc)[:80]}

    @app.get("/api/current-model", tags=["settings"])
    async def get_current_model(workspace_id: str = "", user=Depends(get_current_user)):
        """Return the currently active model for this workspace."""
        user_settings = _state.get("user_settings", {})
        cfg = _state.get("config")
        startup_model = cfg.llm_model if cfg else "gpt-4o"
        model = user_settings.get("model") or startup_model
        provider = user_settings.get("model_provider", "")
        if not provider:
            if "claude" in model.lower() or "anthropic" in model.lower():
                provider = "anthropic"
            elif "gpt" in model.lower() or "o1" in model.lower():
                provider = "openai"
            else:
                provider = "local"
        return {"model": model, "provider": provider, "startup_model": startup_model}

    @app.get("/api/model-stats", tags=["settings"])
    async def get_model_stats(workspace_id: str = "", user=Depends(get_current_user)):
        """Per-model observability: query counts, tokens, cost, avg latency."""
        traces = await _state["trace_store"].list_for_workspace(workspace_id or "", limit=2000)
        by_model: dict = {}
        for t in traces:
            mid = t.get("model_id") or "unknown"
            if mid not in by_model:
                by_model[mid] = {
                    "model": mid,
                    "queries": 0,
                    "succeeded": 0,
                    "total_tokens": 0,
                    "tokens_input": 0,
                    "tokens_output": 0,
                    "total_cost_usd": 0.0,
                    "latencies": [],
                }
            s = by_model[mid]
            s["queries"] += 1
            if t.get("succeeded"):
                s["succeeded"] += 1
            s["total_tokens"] += t.get("total_tokens", 0)
            s["tokens_input"] += t.get("tokens_input", 0)
            s["tokens_output"] += t.get("tokens_output", 0)
            s["total_cost_usd"] += t.get("total_cost_usd", 0.0)
            if t.get("total_latency_ms"):
                s["latencies"].append(t["total_latency_ms"])
        result = []
        for s in by_model.values():
            lats = s.pop("latencies")
            s["avg_latency_ms"] = round(sum(lats) / len(lats)) if lats else 0
            s["p95_latency_ms"] = (
                round(sorted(lats)[int(len(lats) * 0.95)])
                if len(lats) >= 5
                else s["avg_latency_ms"]
            )
            s["success_rate"] = round(s["succeeded"] / s["queries"] * 100, 1) if s["queries"] else 0
            s["total_cost_usd"] = round(s["total_cost_usd"], 6)
            result.append(s)
        result.sort(key=lambda x: -x["queries"])
        return {"models": result}

    @app.post("/api/settings/verify-key", tags=["settings"])
    async def verify_api_key(request: Request, user=Depends(get_current_user)):
        import os as _os

        body = await request.json()
        provider = body.get("provider")  # "anthropic" or "openai"
        api_key = body.get("api_key", "").strip()

        # If no key in request, fall back to environment variable
        if not api_key:
            if provider == "anthropic":
                api_key = _os.environ.get("ANTHROPIC_API_KEY", "")
            elif provider == "openai":
                api_key = _os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            return {
                "valid": False,
                "error": "No API key provided. Paste your key above and click Verify Key.",
            }

        try:
            import litellm

            if provider == "anthropic":
                model = "claude-haiku-4-5"
                resp = await asyncio.to_thread(
                    litellm.completion,
                    model=model,
                    messages=[{"role": "user", "content": "hi"}],
                    max_tokens=1,
                    api_key=api_key,
                )
                model_used = resp.model or model
            elif provider == "openai":
                model = "gpt-4o-mini"
                resp = await asyncio.to_thread(
                    litellm.completion,
                    model=model,
                    messages=[{"role": "user", "content": "hi"}],
                    max_tokens=1,
                    api_key=api_key,
                )
                model_used = resp.model or model
            else:
                return {"valid": False, "error": "Unknown provider"}
            return {"valid": True, "model": model_used}
        except Exception as ex:
            err = str(ex)
            logger.warning(
                "verify_key.failed",
                provider=provider,
                error_type=type(ex).__name__,
                error=err[:200],
            )
            if (
                "401" in err
                or "invalid_api_key" in err.lower()
                or "authentication" in err.lower()
                or "x-api-key" in err.lower()
            ):
                return {
                    "valid": False,
                    "error": "Key rejected — authentication failed. Paste a fresh key.",
                }
            elif "not_found" in err.lower() or "model_not_found" in err.lower():
                return {
                    "valid": False,
                    "error": f"Model not found. Try a different model. ({err[:80]})",
                }
            elif "429" in err or "rate_limit" in err.lower():
                return {"valid": True, "warning": "Key is valid (rate limited)"}
            elif "connect" in err.lower() or "network" in err.lower() or "timeout" in err.lower():
                return {"valid": False, "error": "Network error — check your internet connection"}
            else:
                return {"valid": False, "error": f"{type(ex).__name__}: {err[:150]}"}

    # ── Observability ─────────────────────────────────────────────────────────

    @app.get("/health", tags=["observability"])
    async def health():
        # Check whether any LLM key is available (env or saved settings)
        user_settings = _state.get("user_settings", {})
        _has_anthropic = bool(
            user_settings.get("anthropic_key") or os.environ.get("ANTHROPIC_API_KEY", "")
        )
        _has_openai = bool(user_settings.get("openai_key") or os.environ.get("OPENAI_API_KEY", ""))
        return {
            "status": "ok",
            "version": "2.0.0",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "auth_required": cfg.auth_enabled,  # False by default — frontend skips login
            "has_llm_key": _has_anthropic or _has_openai,
            "llm_providers": {
                "anthropic": _has_anthropic,
                "openai": _has_openai,
            },
        }

    @app.get("/ready", tags=["observability"])
    async def ready():
        return {"ready": True}

    @app.get("/api/status", tags=["observability"])
    async def api_status(user=Depends(get_current_user)):
        """System status summary used by the sidebar and Home dashboard."""
        user_settings = _state.get("user_settings", {})
        model = user_settings.get("model", cfg.llm_model)
        agents = _state.get("agents", {})
        return {
            "status": "ok",
            "version": "2.0.0",
            "model": model,
            "agent_ready": bool(agents),
            "auth_required": cfg.auth_enabled,
        }

    @app.get("/api/connections", tags=["observability"])
    async def api_connections(user=Depends(get_current_user)):
        """Return all workspaces as connection entries (used by Home dashboard)."""
        try:
            workspaces = await _state["workspace_store"].list_for_user(user.user_id)
            return [
                {
                    "workspace_id": ws.workspace_id,
                    "name": ws.name,
                    "status": ws.status.value if hasattr(ws.status, "value") else str(ws.status),
                    "source_count": len(ws.sources) if hasattr(ws, "sources") and ws.sources else 0,
                }
                for ws in workspaces
            ]
        except Exception as exc:
            logger.debug("api_connections.failed", error=str(exc))
            return []

    @app.get("/metrics", tags=["observability"])
    async def metrics():
        from sqlagent.telemetry import get_prometheus_metrics, get_prometheus_content_type

        return Response(content=get_prometheus_metrics(), media_type=get_prometheus_content_type())

    @app.get("/debug/traces", tags=["observability"])
    async def debug_traces(user=Depends(get_current_user)):
        from sqlagent.telemetry import get_recent_traces

        return {"traces": get_recent_traces()}

    @app.get("/debug/audit", tags=["observability"])
    async def debug_audit(user=Depends(get_current_user)):
        records = await _state["audit_log"].recent(50)
        return {"records": records}

    # ── Aggregate history stats (Home dashboard KPIs) ─────────────────────────

    @app.get("/api/history/stats", tags=["observability"])
    async def api_history_stats(workspace_id: str = "", user=Depends(get_current_user)):
        """Aggregate query stats for the Home dashboard KPI cards."""
        from datetime import timedelta

        try:
            traces = await _state["trace_store"].list_for_workspace(workspace_id or "", limit=5000)
        except Exception:
            traces = []

        now = datetime.now(timezone.utc)
        week_ago = now - timedelta(days=7)

        total = len(traces)
        succeeded = sum(1 for t in traces if t.get("succeeded"))
        total_cost = sum(t.get("total_cost_usd", 0.0) for t in traces)
        latencies = [t["total_latency_ms"] for t in traces if t.get("total_latency_ms")]

        # Queries this week
        queries_this_week = 0
        for t in traces:
            ts = t.get("created_at") or t.get("started_at")
            if ts:
                try:
                    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                    if dt >= week_ago:
                        queries_this_week += 1
                except Exception:
                    pass

        accuracy_pct = round(succeeded / total * 100, 1) if total else 0
        avg_latency_ms = round(sum(latencies) / len(latencies)) if latencies else 0

        return {
            "queries_this_week": queries_this_week,
            "total_queries": total,
            "accuracy_pct": accuracy_pct,
            "total_cost_usd": round(total_cost, 4),
            "avg_latency_ms": avg_latency_ms,
        }

    # ── Chat sessions ──────────────────────────────────────────────────────────

    @app.get("/api/chat/sessions", tags=["settings"])
    async def list_chat_sessions(
        workspace_id: str = "", limit: int = 20, user=Depends(get_current_user)
    ):
        """Return recent sessions (one per unique session_id in trace store)."""
        try:
            traces = await _state["trace_store"].list_for_workspace(workspace_id or "", limit=500)
        except Exception:
            traces = []

        seen: dict = {}
        for t in traces:
            sid = t.get("session_id") or t.get("trace_id", "")
            if not sid or sid in seen:
                continue
            seen[sid] = {
                "session_id": sid,
                "workspace_id": workspace_id,
                "preview": (t.get("nl_query") or "")[:80],
                "query_count": 1,
                "last_query_at": t.get("created_at") or t.get("started_at") or "",
                "succeeded": t.get("succeeded", False),
            }

        sessions = list(seen.values())[:limit]
        return {"sessions": sessions, "total": len(sessions)}

    @app.get("/api/chat/messages", tags=["settings"])
    async def get_chat_messages(
        session_id: str = "", workspace_id: str = "", user=Depends(get_current_user)
    ):
        """Return all trace entries for a given session (used to restore chat history)."""
        if not session_id:
            return {"messages": []}
        try:
            traces = await _state["trace_store"].list_for_workspace(workspace_id or "", limit=500)
        except Exception:
            traces = []

        messages = []
        for t in traces:
            if t.get("session_id") != session_id and t.get("trace_id") != session_id:
                continue
            messages.append(
                {
                    "role": "user",
                    "content": t.get("nl_query", ""),
                    "timestamp": t.get("created_at") or t.get("started_at") or "",
                }
            )
            if t.get("response") or t.get("answer"):
                messages.append(
                    {
                        "role": "assistant",
                        "content": t.get("response") or t.get("answer") or "",
                        "sql": t.get("sql") or t.get("final_sql") or "",
                        "row_count": t.get("row_count", 0),
                        "succeeded": t.get("succeeded", False),
                        "timestamp": t.get("created_at") or t.get("started_at") or "",
                    }
                )
        return {"messages": messages}

    # ── Schema refresh / validate-link / diff ─────────────────────────────────

    @app.post("/schema/refresh", tags=["schema"])
    async def schema_refresh(workspace_id: str = "", user=Depends(get_current_user)):
        """Force re-introspect all connectors for a workspace."""
        try:
            agent = await _get_or_create_agent(workspace_id, user.user_id)
            snapshots = []
            for sid, conn in agent.services.connectors.items():
                snap = await conn.introspect()
                agent.services.schema_cache[sid] = snap
                snapshots.append({"source_id": sid, "table_count": len(snap.tables)})
            # Bust cached knowledge graph
            kg_path = os.path.join(
                os.path.expanduser("~"), ".sqlagent", "kg", f"{workspace_id or 'default'}.json"
            )
            if os.path.exists(kg_path):
                os.remove(kg_path)
            return {"status": "refreshed", "sources": snapshots}
        except Exception as exc:
            raise HTTPException(500, f"Schema refresh failed: {exc}")

    @app.post("/schema/validate-link", tags=["schema"])
    async def schema_validate_link(request: Request, user=Depends(get_current_user)):
        """Confirm or reject an inferred FK relationship."""
        body = await request.json()
        # Store the validation decision in workspace metadata for future use
        workspace_id = body.get("workspace_id", "")
        link = {
            "from_table": body.get("from_table"),
            "from_col": body.get("from_col"),
            "to_table": body.get("to_table"),
            "to_col": body.get("to_col"),
            "confirmed": body.get("confirmed", True),
        }
        try:
            ws = await _state["workspace_store"].get(workspace_id)
            validated = ws.metadata.get("validated_links", []) if ws.metadata else []
            # Remove existing entry for same pair
            validated = [
                v
                for v in validated
                if not (
                    v.get("from_table") == link["from_table"]
                    and v.get("from_col") == link["from_col"]
                    and v.get("to_table") == link["to_table"]
                )
            ]
            validated.append(link)
            if not ws.metadata:
                ws.metadata = {}
            ws.metadata["validated_links"] = validated
            await _state["workspace_store"].update(workspace_id, {"metadata": ws.metadata})
        except Exception as exc:
            logger.debug("validate_link.store_failed", error=str(exc))
        return {"status": "saved", "link": link}

    @app.get("/schema/diff", tags=["schema"])
    async def schema_diff(workspace_id: str = "", user=Depends(get_current_user)):
        """Return schema changes since last introspect (placeholder — returns empty diff if no prior snapshot)."""
        try:
            agent = await _get_or_create_agent(workspace_id, user.user_id)
            diffs = []
            for sid, conn in agent.services.connectors.items():
                current = await conn.introspect()
                cached = agent.services.schema_cache.get(sid)
                if not cached:
                    agent.services.schema_cache[sid] = current
                    continue
                cached_tables = {t.name for t in cached.tables}
                current_tables = {t.name for t in current.tables}
                added = current_tables - cached_tables
                removed = cached_tables - current_tables
                if added or removed:
                    diffs.append(
                        {
                            "source_id": sid,
                            "tables_added": list(added),
                            "tables_removed": list(removed),
                        }
                    )
            return {"diffs": diffs, "has_changes": bool(diffs)}
        except Exception as exc:
            return {"diffs": [], "has_changes": False, "error": str(exc)}

    # ── SOUL ──────────────────────────────────────────────────────────────────

    @app.get("/soul/{user_id}", tags=["soul"])
    async def get_soul(user_id: str):
        from sqlagent.soul import UserSOUL

        soul = UserSOUL()
        profile = soul.get_profile(user_id)
        return {
            "user_id": profile.user_id,
            "accountability_unit": profile.accountability_unit,
            "time_reference": profile.time_reference,
            "vocabulary_map": profile.vocabulary_map,
            "query_count": profile.query_count,
            "version": profile.version,
        }

    @app.get("/soul/{user_id}/md", tags=["soul"])
    async def get_soul_md(user_id: str):
        from sqlagent.soul import UserSOUL

        soul = UserSOUL()
        profile = soul.get_profile(user_id)
        return Response(content=profile.to_markdown(), media_type="text/markdown")

    # ── Static files (UI) ─────────────────────────────────────────────────────

    _ui_dir = os.path.join(os.path.dirname(__file__), "ui")

    @app.get("/")
    async def serve_landing():
        """Landing page — the main home for all users."""
        bp_path = os.path.join(_ui_dir, "preview_bp.html")
        if os.path.exists(bp_path):
            return FileResponse(
                bp_path,
                headers={
                    "Cache-Control": "no-cache, no-store, must-revalidate",
                    "Pragma": "no-cache",
                    "Expires": "0",
                },
            )
        html_path = os.path.join(_ui_dir, "app.html")
        if os.path.exists(html_path):
            return FileResponse(html_path, headers={"Cache-Control": "no-cache"})
        return {"message": "sqlagent API", "docs": "/docs", "guide": "/guide", "version": "2.0.0"}

    @app.get("/app")
    async def serve_workspace_app():
        """Workspace app — entered via + New Workspace on the landing page."""
        html_path = os.path.join(_ui_dir, "app.html")
        if os.path.exists(html_path):
            return FileResponse(
                html_path,
                headers={
                    "Cache-Control": "no-cache, no-store, must-revalidate",
                    "Pragma": "no-cache",
                    "Expires": "0",
                },
            )
        return {"message": "Workspace app not found"}

    @app.get("/guide")
    async def serve_docs():
        """Serve the built-in docs/guide page. Use /guide to avoid shadowing FastAPI's /docs OpenAPI UI."""
        docs_path = os.path.join(_ui_dir, "docs.html")
        if os.path.exists(docs_path):
            return FileResponse(
                docs_path,
                headers={
                    "Cache-Control": "no-cache, no-store, must-revalidate",
                    "Pragma": "no-cache",
                    "Expires": "0",
                },
            )
        raise HTTPException(404, "Docs page not found")

    if os.path.exists(_ui_dir):
        app.mount("/ui", StaticFiles(directory=_ui_dir), name="ui")

    # ── Custom API docs — cream + sketch aesthetic (matches landing page) ─────
    @app.get("/docs", include_in_schema=False)
    async def custom_swagger_ui():
        from fastapi.responses import HTMLResponse

        html = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>ora — API Reference</title>
  <link rel="preconnect" href="https://fonts.googleapis.com"/>
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin/>
  <link href="https://fonts.googleapis.com/css2?family=Caveat:wght@400;500;600;700&family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet"/>
  <link rel="stylesheet" href="https://unpkg.com/swagger-ui-dist@5.11.0/swagger-ui.css"/>
  <style>
    /* ── Fonts ───────────────────────────────────────────────── */
    @font-face {
      font-family: 'DynaPuff';
      src: url('/ui/DynaPuff-Variable.ttf') format('truetype');
      font-weight: 100 900; font-style: normal; font-display: swap;
    }
    @font-face {
      font-family: 'DynaPuff';
      src: url('/ui/DynaPuff-Regular.ttf') format('truetype');
      font-weight: 400; font-style: normal; font-display: swap;
    }

    /* ── Reset & Canvas ──────────────────────────────────────── */
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    html, body {
      background: #f5f2ec;
      color: #1e160a;
      font-family: 'Inter', -apple-system, sans-serif;
      -webkit-font-smoothing: antialiased;
      min-height: 100vh;
    }
    /* Subtle paper texture via repeating gradient */
    body::before {
      content: '';
      position: fixed; inset: 0; pointer-events: none; z-index: 0;
      background-image:
        radial-gradient(ellipse at 20% 10%, rgba(91,138,240,.05) 0%, transparent 55%),
        radial-gradient(ellipse at 80% 90%, rgba(59,95,192,.03) 0%, transparent 45%);
    }

    /* ── Scrollbar ───────────────────────────────────────────── */
    ::-webkit-scrollbar { width: 7px; height: 7px; }
    ::-webkit-scrollbar-track { background: #ede9e0; }
    ::-webkit-scrollbar-thumb { background: rgba(20,14,4,.18); border-radius: 4px; }
    ::-webkit-scrollbar-thumb:hover { background: rgba(20,14,4,.30); }

    /* ── Top nav ─────────────────────────────────────────────── */
    #ora-nav {
      position: sticky; top: 0; z-index: 200;
      display: flex; align-items: center; justify-content: space-between;
      padding: 0 36px;
      height: 52px;
      background: rgba(245,242,236,.96);
      border-bottom: 1.5px solid rgba(20,14,4,.14);
      backdrop-filter: blur(10px);
    }
    .nav-brand {
      display: flex; align-items: center; gap: 10px;
      text-decoration: none;
    }
    .nav-wordmark {
      font-family: 'DynaPuff', 'JetBrains Mono', monospace;
      font-size: 22px; font-weight: 400; letter-spacing: .02em;
      background: linear-gradient(135deg, #1a2d6b 0%, #3b5fc0 55%, #1e160a 100%);
      -webkit-background-clip: text; -webkit-text-fill-color: transparent;
      background-clip: text;
      filter: drop-shadow(0 1px 2px rgba(59,95,192,.10));
    }
    .nav-cursor {
      display: inline-block; width: 2px; height: .7em;
      background: #3b5fc0; border-radius: 1px; margin-left: 1px;
      animation: blink .8s ease-in-out infinite; vertical-align: middle;
    }
    @keyframes blink { 0%,100%{opacity:1} 50%{opacity:0} }
    .nav-badge-ref {
      font-size: 9px; font-weight: 600; letter-spacing: .12em;
      color: rgba(59,95,192,.78); border: 1px solid rgba(59,95,192,.28);
      border-radius: 3px; padding: 2px 7px;
      font-family: 'JetBrains Mono', monospace;
      background: rgba(59,95,192,.06); text-transform: uppercase;
    }
    .nav-links { display: flex; align-items: center; gap: 2px; }
    .nav-links a {
      font-size: 12.5px; font-weight: 500; color: rgba(20,14,4,.45);
      text-decoration: none; padding: 5px 12px; border-radius: 6px;
      transition: all .12s ease; font-family: 'Inter', sans-serif;
    }
    .nav-links a:hover { color: rgba(20,14,4,.80); background: rgba(20,14,4,.05); }
    .nav-links a.active {
      color: rgba(59,95,192,.90);
      background: rgba(59,95,192,.08);
      border: 1px solid rgba(59,95,192,.18);
    }

    /* ── Page layout ─────────────────────────────────────────── */
    #page-wrap {
      position: relative; z-index: 1;
      max-width: 1060px; margin: 0 auto;
      padding: 36px 32px 80px;
    }

    /* ── Hero header ─────────────────────────────────────────── */
    .docs-hero {
      border: 1.5px solid rgba(20,14,4,.24);
      border-radius: 12px;
      padding: 28px 32px 24px;
      background: rgba(255,252,242,.60);
      position: relative;
      box-shadow: 3px 4px 0 rgba(20,14,4,.06), 6px 8px 0 rgba(20,14,4,.03);
      margin-bottom: 28px;
    }
    /* Corner sketch marks */
    .docs-hero::after {
      content: ''; position: absolute; top: 7px; left: 7px;
      width: 14px; height: 14px;
      border-top: 1.5px solid rgba(20,14,4,.22);
      border-left: 1.5px solid rgba(20,14,4,.22);
    }
    .docs-hero::before {
      content: ''; position: absolute; bottom: 7px; right: 7px;
      width: 14px; height: 14px;
      border-bottom: 1.5px solid rgba(20,14,4,.22);
      border-right: 1.5px solid rgba(20,14,4,.22);
    }
    .hero-title {
      font-family: 'Caveat', cursive;
      font-size: 34px; font-weight: 600; letter-spacing: -.2px;
      color: #1e160a; line-height: 1.1; margin-bottom: 8px;
    }
    .hero-title .hl {
      color: #3b5fc0;
      text-decoration: underline; text-decoration-color: rgba(59,95,192,.35);
      text-underline-offset: 3px;
    }
    .hero-desc {
      font-size: 13.5px; color: rgba(20,14,4,.52); line-height: 1.6;
      max-width: 560px; margin-bottom: 16px;
    }
    .hero-pills { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
    .hero-pill {
      font-size: 10.5px; font-family: 'JetBrains Mono', monospace;
      padding: 4px 12px; border-radius: 100px;
      background: rgba(255,252,244,.80); border: 1px solid rgba(20,14,4,.18);
      color: rgba(20,14,4,.52);
    }
    .hero-pill.blue {
      background: rgba(59,95,192,.08); border-color: rgba(59,95,192,.22);
      color: rgba(40,72,180,.78);
    }
    .hero-pill.green {
      background: rgba(34,164,74,.08); border-color: rgba(34,164,74,.22);
      color: rgba(20,130,55,.80);
    }

    /* ── Swagger UI global resets ────────────────────────────── */
    .swagger-ui {
      font-family: 'Inter', -apple-system, sans-serif !important;
      color: #1e160a !important;
    }
    .swagger-ui *, .swagger-ui *::before, .swagger-ui *::after {
      font-family: 'Inter', -apple-system, sans-serif;
    }
    /* Hide swagger default topbar & info block (we have our own header) */
    .swagger-ui .topbar { display: none !important; }
    .swagger-ui .information-container { display: none !important; }
    .swagger-ui .scheme-container {
      background: rgba(255,252,244,.70) !important;
      border: 1.5px solid rgba(20,14,4,.18) !important;
      border-radius: 9px; padding: 10px 18px; margin: 0 0 20px;
      box-shadow: 2px 3px 0 rgba(20,14,4,.05);
    }
    .swagger-ui .scheme-container .schemes > label {
      color: rgba(20,14,4,.45) !important; font-size: 11px; font-weight: 600;
      letter-spacing: .06em; text-transform: uppercase;
    }
    .swagger-ui select {
      background: rgba(255,252,244,.90) !important;
      border: 1px solid rgba(20,14,4,.20) !important;
      color: rgba(20,14,4,.75) !important;
      border-radius: 6px; padding: 5px 10px;
      font-family: 'JetBrains Mono', monospace; font-size: 12px;
      box-shadow: 1px 2px 0 rgba(20,14,4,.05);
    }

    /* ── Tag / group headers ─────────────────────────────────── */
    .swagger-ui .opblock-tag {
      border-bottom: 1.5px solid rgba(20,14,4,.18) !important;
      color: #1e160a !important;
      font-size: 15px; font-weight: 600;
      font-family: 'Caveat', cursive !important;
      letter-spacing: .01em;
      padding: 14px 0 10px !important;
      margin-top: 12px;
    }
    .swagger-ui .opblock-tag:hover { background: transparent !important; }
    .swagger-ui .opblock-tag a, .swagger-ui .opblock-tag a:visited {
      color: #1e160a !important;
      font-family: 'Caveat', cursive !important;
      font-size: 18px; font-weight: 600;
    }
    .swagger-ui .opblock-tag small {
      color: rgba(20,14,4,.42) !important; font-size: 12px; font-weight: 400;
      font-family: 'Inter', sans-serif !important;
    }
    .swagger-ui .opblock-tag svg { fill: rgba(20,14,4,.30) !important; }

    /* ── Endpoint blocks ─────────────────────────────────────── */
    .swagger-ui .opblock {
      border-radius: 9px !important;
      border: 1.5px solid rgba(20,14,4,.20) !important;
      background: rgba(255,252,242,.55) !important;
      margin: 5px 0 !important;
      box-shadow: 2px 3px 0 rgba(20,14,4,.05), 4px 6px 0 rgba(20,14,4,.025) !important;
      overflow: hidden;
      transition: box-shadow .12s ease, transform .12s ease;
    }
    .swagger-ui .opblock:hover {
      border-color: rgba(20,14,4,.32) !important;
      box-shadow: 3px 5px 0 rgba(20,14,4,.08), 6px 10px 0 rgba(20,14,4,.03) !important;
    }

    /* Method colors — left accent bars */
    .swagger-ui .opblock.opblock-get  { border-left: 3px solid #22a44a !important; }
    .swagger-ui .opblock.opblock-post { border-left: 3px solid #3b5fc0 !important; }
    .swagger-ui .opblock.opblock-put  { border-left: 3px solid #c04a00 !important; }
    .swagger-ui .opblock.opblock-delete { border-left: 3px solid #b91c1c !important; }
    .swagger-ui .opblock.opblock-patch  { border-left: 3px solid #7c3aed !important; }

    /* Summary row */
    .swagger-ui .opblock .opblock-summary {
      background: transparent !important;
      border: none !important; padding: 9px 14px !important;
    }
    .swagger-ui .opblock .opblock-summary:hover {
      background: rgba(20,14,4,.03) !important;
    }
    .swagger-ui .opblock-summary-method {
      border-radius: 4px !important;
      font-family: 'JetBrains Mono', monospace !important;
      font-size: 10.5px !important; font-weight: 700 !important;
      min-width: 58px !important; text-align: center !important;
      padding: 3px 7px !important; letter-spacing: .04em;
    }
    .swagger-ui .opblock.opblock-get .opblock-summary-method
      { background: rgba(34,164,74,.12) !important; color: #1a7a3a !important; }
    .swagger-ui .opblock.opblock-post .opblock-summary-method
      { background: rgba(59,95,192,.12) !important; color: #2040a0 !important; }
    .swagger-ui .opblock.opblock-put .opblock-summary-method
      { background: rgba(192,74,0,.10) !important; color: #a03800 !important; }
    .swagger-ui .opblock.opblock-delete .opblock-summary-method
      { background: rgba(185,28,28,.10) !important; color: #991b1b !important; }
    .swagger-ui .opblock.opblock-patch .opblock-summary-method
      { background: rgba(124,58,237,.10) !important; color: #5b21b6 !important; }

    .swagger-ui .opblock-summary-path {
      color: rgba(20,14,4,.82) !important;
      font-family: 'JetBrains Mono', monospace !important;
      font-size: 13px !important; font-weight: 500 !important;
    }
    .swagger-ui .opblock-summary-path .nostyle { color: rgba(20,14,4,.82) !important; }
    .swagger-ui .opblock-summary-path b { color: #3b5fc0 !important; font-weight: 600; }
    .swagger-ui .opblock-summary-description {
      color: rgba(20,14,4,.45) !important; font-size: 12.5px !important;
    }
    .swagger-ui .opblock-summary-control svg { fill: rgba(20,14,4,.28) !important; }
    .swagger-ui .opblock-summary-control:hover svg { fill: rgba(20,14,4,.55) !important; }

    /* ── Expanded body ───────────────────────────────────────── */
    .swagger-ui .opblock-body {
      background: rgba(245,240,228,.50) !important;
      border-top: 1.5px solid rgba(20,14,4,.12) !important;
    }
    .swagger-ui .opblock-section-header {
      background: rgba(228,222,210,.55) !important;
      border-bottom: 1px solid rgba(20,14,4,.10) !important;
    }
    .swagger-ui .opblock-section-header h4 {
      color: rgba(20,14,4,.45) !important; font-size: 10.5px !important;
      font-weight: 700 !important; letter-spacing: .55px !important;
      text-transform: uppercase !important; font-family: 'JetBrains Mono', monospace !important;
    }
    .swagger-ui .opblock-description-wrapper p {
      color: rgba(20,14,4,.55) !important; font-size: 13px;
    }

    /* ── Parameter tables ────────────────────────────────────── */
    .swagger-ui table { background: transparent !important; width: 100%; border-collapse: collapse; }
    .swagger-ui table thead tr th {
      background: rgba(228,222,210,.55) !important;
      border-bottom: 1.5px solid rgba(20,14,4,.12) !important;
      color: rgba(20,14,4,.40) !important; font-size: 10.5px !important;
      font-weight: 700 !important; letter-spacing: .5px; text-transform: uppercase;
      padding: 9px 16px !important; font-family: 'JetBrains Mono', monospace !important;
    }
    .swagger-ui table tbody tr td {
      border-bottom: 1px solid rgba(20,14,4,.07) !important;
      padding: 9px 16px !important; color: rgba(20,14,4,.80) !important;
      background: transparent !important;
    }
    .swagger-ui table tbody tr:last-child td { border-bottom: none !important; }
    .swagger-ui .parameter__name {
      color: rgba(20,14,4,.82) !important; font-family: 'JetBrains Mono', monospace;
      font-size: 12.5px; font-weight: 500;
    }
    .swagger-ui .parameter__name.required::after { color: #b91c1c; }
    .swagger-ui .parameter__type { color: #3b5fc0 !important; font-family: 'JetBrains Mono', monospace; font-size: 11.5px; }
    .swagger-ui .parameter__in { color: rgba(20,14,4,.35) !important; font-size: 10.5px; font-family: 'JetBrains Mono', monospace; }

    /* ── Inputs ──────────────────────────────────────────────── */
    .swagger-ui input[type=text],
    .swagger-ui input[type=password],
    .swagger-ui input[type=email],
    .swagger-ui textarea {
      background: rgba(255,252,244,.90) !important;
      border: 1.5px solid rgba(20,14,4,.22) !important;
      border-radius: 7px !important;
      color: rgba(20,14,4,.82) !important;
      font-family: 'JetBrains Mono', monospace !important;
      font-size: 12.5px !important; padding: 7px 11px !important;
      box-shadow: 1px 2px 0 rgba(20,14,4,.05);
    }
    .swagger-ui input[type=text]:focus,
    .swagger-ui textarea:focus {
      border-color: rgba(59,95,192,.50) !important;
      outline: none !important;
      box-shadow: 0 0 0 3px rgba(59,95,192,.08), 1px 2px 0 rgba(20,14,4,.04) !important;
    }
    .swagger-ui input[type=text]::placeholder,
    .swagger-ui textarea::placeholder { color: rgba(20,14,4,.28) !important; }

    /* ── Buttons ─────────────────────────────────────────────── */
    .swagger-ui .btn {
      border-radius: 7px !important;
      font-family: 'Inter', sans-serif !important;
      font-size: 12.5px !important; font-weight: 600 !important;
      padding: 7px 15px !important; cursor: pointer;
      transition: all .12s ease;
      box-shadow: 1px 2px 0 rgba(20,14,4,.10) !important;
    }
    .swagger-ui .btn.execute {
      background: #3b5fc0 !important;
      border: 1.5px solid #2a4aaa !important;
      color: #fff !important;
      box-shadow: 2px 3px 0 rgba(20,14,4,.15) !important;
    }
    .swagger-ui .btn.execute:hover {
      background: #2a4aaa !important;
      box-shadow: 3px 4px 0 rgba(20,14,4,.18) !important;
      transform: translateY(-1px);
    }
    .swagger-ui .btn.btn-clear {
      background: rgba(255,252,244,.90) !important;
      border: 1.5px solid rgba(20,14,4,.22) !important;
      color: rgba(20,14,4,.55) !important;
    }
    .swagger-ui .btn.btn-clear:hover {
      background: rgba(255,252,244,1) !important; color: rgba(20,14,4,.80) !important;
    }
    .swagger-ui .btn.authorize {
      background: rgba(34,164,74,.10) !important;
      border: 1.5px solid rgba(34,164,74,.28) !important;
      color: #1a7a3a !important;
    }
    .swagger-ui .btn.authorize svg { fill: #1a7a3a !important; }
    .swagger-ui .try-out__btn {
      background: rgba(59,95,192,.10) !important;
      border: 1.5px solid rgba(59,95,192,.28) !important;
      color: #2040a0 !important;
    }

    /* ── Responses ───────────────────────────────────────────── */
    .swagger-ui .responses-wrapper { padding: 14px !important; }
    .swagger-ui .response-col_status {
      color: rgba(20,14,4,.80) !important; font-family: 'JetBrains Mono', monospace; font-size: 12.5px;
    }
    .swagger-ui .response-col_description { color: rgba(20,14,4,.50) !important; font-size: 12.5px; }
    .swagger-ui .responses-inner h4, .swagger-ui .responses-inner h5 {
      color: rgba(20,14,4,.40) !important; font-size: 10.5px; font-weight: 700;
      letter-spacing: .5px; text-transform: uppercase;
      font-family: 'JetBrains Mono', monospace !important;
    }

    /* ── Code / JSON ─────────────────────────────────────────── */
    .swagger-ui .highlight-code, .swagger-ui pre.microlight,
    .swagger-ui pre, .swagger-ui code {
      background: rgba(228,222,210,.65) !important;
      border: 1.5px solid rgba(20,14,4,.16) !important;
      border-radius: 7px !important;
      font-family: 'JetBrains Mono', monospace !important;
      font-size: 11.5px !important;
      color: rgba(20,14,4,.78) !important;
      padding: 10px 14px !important;
      box-shadow: 1px 2px 0 rgba(20,14,4,.04) !important;
    }
    .swagger-ui .microlight .number { color: #7c3aed !important; }
    .swagger-ui .microlight .string { color: #1a7a3a !important; }
    .swagger-ui .microlight .key    { color: #2040a0 !important; }
    .swagger-ui .microlight .boolean { color: #c04a00 !important; }
    .swagger-ui .microlight .null   { color: #b91c1c !important; }
    .swagger-ui .copy-to-clipboard {
      background: rgba(255,252,244,.85) !important; border-radius: 5px !important;
      border: 1px solid rgba(20,14,4,.16) !important;
    }

    /* ── Schema / Models ─────────────────────────────────────── */
    .swagger-ui section.models {
      border: 1.5px solid rgba(20,14,4,.20) !important;
      border-radius: 10px !important; overflow: hidden;
      background: rgba(255,252,242,.55) !important;
      margin-top: 28px;
      box-shadow: 2px 3px 0 rgba(20,14,4,.05), 4px 6px 0 rgba(20,14,4,.025);
    }
    .swagger-ui section.models h4 {
      color: rgba(20,14,4,.80) !important; font-size: 15px; font-weight: 600;
      font-family: 'Caveat', cursive !important;
      padding: 14px 20px; margin: 0;
      border-bottom: 1.5px solid rgba(20,14,4,.12);
      background: rgba(228,222,210,.45);
    }
    .swagger-ui .model-container {
      background: rgba(255,252,244,.70) !important;
      border-radius: 7px !important; margin: 8px;
      border: 1px solid rgba(20,14,4,.14) !important;
    }
    .swagger-ui .model-title { color: rgba(20,14,4,.80) !important; font-size: 13px; }
    .swagger-ui .model { color: rgba(20,14,4,.60) !important; font-family: 'JetBrains Mono', monospace; font-size: 11.5px; }
    .swagger-ui .model .property { color: #2040a0 !important; }
    .swagger-ui .model .property.primitive { color: #1a7a3a !important; }
    .swagger-ui .prop-type { color: #7c3aed !important; }
    .swagger-ui .prop-format { color: #c04a00 !important; }

    /* ── Auth modal ──────────────────────────────────────────── */
    .swagger-ui .dialog-ux .modal-ux {
      background: #f5f2ec !important;
      border: 1.5px solid rgba(20,14,4,.24) !important;
      border-radius: 12px !important;
      box-shadow: 4px 8px 0 rgba(20,14,4,.10), 8px 16px 0 rgba(20,14,4,.05) !important;
    }
    .swagger-ui .dialog-ux .modal-ux-header {
      background: rgba(228,222,210,.55) !important;
      border-bottom: 1.5px solid rgba(20,14,4,.14) !important;
    }
    .swagger-ui .dialog-ux .modal-ux-header h3 {
      color: #1e160a !important; font-family: 'Caveat', cursive !important;
      font-size: 20px; font-weight: 600;
    }
    .swagger-ui .dialog-ux .modal-ux-content p { color: rgba(20,14,4,.58) !important; font-size: 13px; }
    .swagger-ui .dialog-ux .modal-ux-content label {
      color: rgba(20,14,4,.45) !important; font-size: 10.5px; font-weight: 700;
      text-transform: uppercase; letter-spacing: .5px;
      font-family: 'JetBrains Mono', monospace !important;
    }
    .swagger-ui .dialog-ux .close-modal { color: rgba(20,14,4,.28) !important; }
    .swagger-ui .dialog-ux .close-modal:hover { color: rgba(20,14,4,.60) !important; }

    /* ── Links ───────────────────────────────────────────────── */
    .swagger-ui a, .swagger-ui a:visited { color: #3b5fc0 !important; }
    .swagger-ui a:hover { color: #2040a0 !important; }

    /* ── Markdown in descriptions ────────────────────────────── */
    .swagger-ui .renderedMarkdown p { color: rgba(20,14,4,.55) !important; font-size: 13px; line-height: 1.6; }
    .swagger-ui .renderedMarkdown code {
      background: rgba(59,95,192,.09) !important;
      color: #2040a0 !important; border: none !important;
      padding: 2px 6px !important; border-radius: 4px; font-size: 11.5px;
      font-family: 'JetBrains Mono', monospace !important;
    }

    /* ── SVGs / icons ────────────────────────────────────────── */
    .swagger-ui svg.arrow { fill: rgba(20,14,4,.28) !important; }
    .swagger-ui .expand-methods svg, .swagger-ui .expand-operation svg { fill: rgba(20,14,4,.30) !important; }

    /* ── Filter bar ──────────────────────────────────────────── */
    .swagger-ui .filter .operation-filter-input {
      background: rgba(255,252,244,.90) !important;
      border: 1.5px solid rgba(20,14,4,.20) !important;
      border-radius: 7px !important; color: rgba(20,14,4,.82) !important;
      box-shadow: 1px 2px 0 rgba(20,14,4,.05);
    }

    /* ── Required ────────────────────────────────────────────── */
    .swagger-ui .required { color: #b91c1c !important; }

    /* ── Misc ────────────────────────────────────────────────── */
    .swagger-ui hr { border-color: rgba(20,14,4,.12) !important; }
    .swagger-ui .wrapper { max-width: none !important; padding: 0 !important; }
    .swagger-ui .block { max-width: none !important; }
    .swagger-ui h2, .swagger-ui h3 { color: #1e160a !important; }
    .swagger-ui h4, .swagger-ui h5 { color: rgba(20,14,4,.55) !important; }
    .swagger-ui .request-url {
      background: rgba(228,222,210,.65) !important; border-radius: 7px;
      padding: 7px 12px; border: 1px solid rgba(20,14,4,.14) !important;
    }
    .swagger-ui .request-url span { color: rgba(20,14,4,.60) !important; font-family: 'JetBrains Mono', monospace; font-size: 12px; }
    .swagger-ui .loading-container { background: transparent !important; }
    .swagger-ui .loading-container .loading::after { border-top-color: #3b5fc0 !important; }
  </style>
</head>
<body>
  <!-- ── Nav ────────────────────────────────────────────────── -->
  <nav id="ora-nav">
    <a href="/" class="nav-brand">
      <span class="nav-wordmark">ora<span class="nav-cursor"></span></span>
      <span class="nav-badge-ref">API REF</span>
    </a>
    <div class="nav-links">
      <a href="/guide">Guide</a>
      <a href="/app">Workspace</a>
      <a href="/docs" class="active">API Docs</a>
    </div>
  </nav>

  <div id="page-wrap">
    <!-- ── Hero ───────────────────────────────────────────────── -->
    <div class="docs-hero">
      <div class="hero-title">
        ora <span class="hl">API Reference</span>
      </div>
      <p class="hero-desc">
        Connect any database. Ask in plain English. Full REST API for queries, schema, training, auth, and more.
      </p>
      <div class="hero-pills">
        <span class="hero-pill blue">v2.0.0</span>
        <span class="hero-pill">REST + SSE</span>
        <span class="hero-pill green">60+ endpoints</span>
        <span class="hero-pill">OpenAPI 3.1</span>
        <span class="hero-pill">Bearer Auth</span>
      </div>
    </div>

    <!-- ── Swagger UI ─────────────────────────────────────────── -->
    <div id="swagger-ui"></div>
  </div>

  <script src="https://unpkg.com/swagger-ui-dist@5.11.0/swagger-ui-bundle.js"></script>
  <script src="https://unpkg.com/swagger-ui-dist@5.11.0/swagger-ui-standalone-preset.js"></script>
  <script>
    window.onload = function() {
      SwaggerUIBundle({
        url: "/openapi.json",
        dom_id: '#swagger-ui',
        deepLinking: true,
        presets: [SwaggerUIBundle.presets.apis, SwaggerUIStandalonePreset],
        layout: "BaseLayout",
        tryItOutEnabled: true,
        persistAuthorization: true,
        displayRequestDuration: true,
        defaultModelsExpandDepth: 0,
        defaultModelExpandDepth: 2,
        docExpansion: "list",
        filter: true,
        syntaxHighlight: { activate: true, theme: "idea" }
      })
    }
  </script>
</body>
</html>"""
        return HTMLResponse(content=html)

    return app


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════


async def _get_or_create_agent(workspace_id: str, user_id: str) -> Any:
    """Get or create a SQLAgent for a workspace.

    Loads all sources from: default_db CLI flag + workspace persisted sources.
    """
    key = workspace_id or "default"
    if key in _state.get("agents", {}):
        return _state["agents"][key]

    from sqlagent.agent import SQLAgent

    cfg = _state.get("config")
    default_db = _state.get("default_db", "")

    # Check if workspace has its own sources — if so, don't use the default_db
    ws_sources = []
    if workspace_id:
        try:
            ws = await _state["workspace_store"].get(workspace_id)
            ws_sources = ws.sources or []
        except Exception as exc:
            logger.debug("server.operation_failed", error=str(exc))

    # Only use default_db if there's NO workspace (direct API use) or workspace has sources
    # New workspaces start EMPTY — they should not inherit the CLI default_db
    db_to_use = ""
    if not workspace_id:
        db_to_use = default_db  # Direct API mode (no workspace)
    elif ws_sources:
        db_to_use = ""  # Workspace has its own sources, don't use default
    agent = SQLAgent(db=db_to_use, config=cfg)
    await agent._ensure_ready()

    # Load workspace sources (uploaded files + database connections)
    for source in ws_sources:
        sid = source.get("source_id", "")
        if sid and sid in agent.services.connectors:
            continue
        url = source.get("file_path") or source.get("connection_string", "")
        if url:
            try:
                from sqlagent.connectors import ConnectorRegistry

                conn = ConnectorRegistry.from_url(
                    sid or f"src_{len(agent.services.connectors)}", url
                )
                await conn.connect()
                agent.services.connectors[sid] = conn
            except Exception as e:
                import structlog

                structlog.get_logger().warn("agent.source_load_failed", source=sid, error=str(e))

    _state.setdefault("agents", {})[key] = agent
    return agent


# Module-level app instance for uvicorn/gunicorn: 'uvicorn sqlagent.server:app'
app = create_app()
