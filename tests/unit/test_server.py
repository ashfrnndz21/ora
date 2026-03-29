"""Test FastAPI server endpoints."""

import pytest
from httpx import AsyncClient, ASGITransport
from sqlagent.server import create_app, _state
from sqlagent.config import AgentConfig


@pytest.fixture
async def app(tmp_path):
    config = AgentConfig(auth_enabled=False, otel_enabled=False)
    application = create_app(config)
    # Manually run lifespan startup since ASGITransport doesn't trigger it
    from sqlagent.auth import AuthStore
    from sqlagent.workspace import WorkspaceStore
    from sqlagent.trace import TraceStore
    from sqlagent.telemetry import AuditLog
    _state["config"] = config
    _state["auth_store"] = AuthStore(db_path=str(tmp_path / "auth.db"))
    await _state["auth_store"].init()
    _state["workspace_store"] = WorkspaceStore(db_path=str(tmp_path / "ws.db"))
    await _state["workspace_store"].init()
    _state["trace_store"] = TraceStore(db_path=str(tmp_path / "traces.db"))
    await _state["trace_store"].init()
    _state["audit_log"] = AuditLog(db_path=str(tmp_path / "audit.db"))
    await _state["audit_log"].init()
    _state["agents"] = {}
    return application


@pytest.mark.asyncio
async def test_health(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["version"] == "2.0.0"


@pytest.mark.asyncio
async def test_ready(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/ready")
        assert resp.status_code == 200
        assert resp.json()["ready"] is True


@pytest.mark.asyncio
async def test_auth_me_no_auth(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/auth/me")
        assert resp.status_code == 200
        data = resp.json()
        assert data["user_id"] == "local"
        assert data["email"] == "local@localhost"


@pytest.mark.asyncio
async def test_list_workspaces_empty(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/workspaces")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)


@pytest.mark.asyncio
async def test_create_and_get_workspace(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Create
        resp = await client.post("/workspaces", json={"name": "Test Analytics"}, headers={"Content-Type": "application/json"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "Test Analytics"
        ws_id = data["workspace_id"]

        # Get
        resp = await client.get(f"/workspaces/{ws_id}")
        assert resp.status_code == 200
        assert resp.json()["name"] == "Test Analytics"

        # List
        resp = await client.get("/workspaces")
        assert resp.status_code == 200
        assert len(resp.json()) >= 1


@pytest.mark.asyncio
async def test_hub_packs(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/hub/packs")
        assert resp.status_code == 200
        packs = resp.json()["packs"]
        assert len(packs) >= 3
        names = [p["name"] for p in packs]
        assert "retail-asean" in names


@pytest.mark.asyncio
async def test_metrics(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/metrics")
        assert resp.status_code == 200


@pytest.mark.asyncio
async def test_debug_traces(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/debug/traces")
        assert resp.status_code == 200
        assert "traces" in resp.json()


@pytest.mark.asyncio
async def test_soul_default(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/soul/test_user")
        assert resp.status_code == 200
        data = resp.json()
        assert data["user_id"] == "test_user"


@pytest.mark.asyncio
async def test_schema_empty(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/schema")
        assert resp.status_code == 200
