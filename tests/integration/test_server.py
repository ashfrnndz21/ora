"""Integration tests for the sqlagent FastAPI server.

These tests spin up the real application and hit HTTP endpoints.
No real LLM calls are made — LLM is mocked at the litellm layer.
"""
from __future__ import annotations

import pytest
from httpx import AsyncClient, ASGITransport

from sqlagent.config import AgentConfig
from sqlagent.server import create_app


@pytest.fixture
def cfg():
    return AgentConfig(
        auth_enabled=False,
        otel_enabled=False,
        prometheus_enabled=False,
    )


@pytest.fixture
async def client(cfg):
    app = create_app(cfg)
    async with app.router.lifespan_context(app):
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as ac:
            yield ac


# ── Health ────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_health_returns_ok(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "version" in body
    assert "auth_required" in body
    assert "has_llm_key" in body
    assert "llm_providers" in body
    assert isinstance(body["llm_providers"], dict)


@pytest.mark.asyncio
async def test_health_auth_required_false_when_disabled(client):
    resp = await client.get("/health")
    assert resp.json()["auth_required"] is False


@pytest.mark.asyncio
async def test_health_has_llm_key_field(client):
    """has_llm_key must be a boolean so frontend can gate API key setup UI."""
    body = (await client.get("/health")).json()
    assert isinstance(body["has_llm_key"], bool)


# ── Auth ──────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_magic_link_does_not_return_code(client):
    """Security: the HTTP response must never include the magic code."""
    resp = await client.post("/auth/magic-link", json={"email": "test@example.com"})
    assert resp.status_code == 200
    body = resp.json()
    assert "code" not in body, "Magic link code must not be in the HTTP response"
    assert "message" in body


@pytest.mark.asyncio
async def test_magic_link_missing_email_returns_400(client):
    resp = await client.post("/auth/magic-link", json={})
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_auth_me_local_mode(client):
    """When auth is disabled, /auth/me returns the local user."""
    resp = await client.get("/auth/me")
    assert resp.status_code == 200
    body = resp.json()
    assert body["user_id"] == "local"


# ── Workspaces ────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_list_workspaces_returns_list(client):
    resp = await client.get("/workspaces")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


@pytest.mark.asyncio
async def test_create_and_get_workspace(client):
    resp = await client.post("/workspaces", json={"name": "Test WS", "description": "CI test"})
    assert resp.status_code == 200
    body = resp.json()
    ws_id = body["workspace_id"]
    assert body["name"] == "Test WS"

    # Get it back
    get_resp = await client.get(f"/workspaces/{ws_id}")
    assert get_resp.status_code == 200
    assert get_resp.json()["workspace_id"] == ws_id


@pytest.mark.asyncio
async def test_create_and_delete_workspace(client):
    resp = await client.post("/workspaces", json={"name": "Delete Me"})
    ws_id = resp.json()["workspace_id"]

    del_resp = await client.delete(f"/workspaces/{ws_id}")
    assert del_resp.status_code == 200

    get_resp = await client.get(f"/workspaces/{ws_id}")
    assert get_resp.status_code == 404


# ── Status / API endpoints ────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_api_status(client):
    resp = await client.get("/api/status")
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_api_connections(client):
    resp = await client.get("/api/connections")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


# ── CORS headers ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_cors_not_wildcard_by_default(client):
    """Security: CORS wildcard must not be set in default config."""
    resp = await client.options(
        "/health",
        headers={"Origin": "https://evil.example.com", "Access-Control-Request-Method": "GET"},
    )
    # With no cors_origins configured, cross-origin requests should be denied
    acao = resp.headers.get("access-control-allow-origin", "")
    assert acao != "*", "CORS wildcard must not be enabled by default"


# ── UI ────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_root_serves_html(client):
    resp = await client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    # Must set no-cache headers
    cc = resp.headers.get("cache-control", "")
    assert "no-cache" in cc or "no-store" in cc
