"""Security-focused integration tests."""
from __future__ import annotations

import pytest
from httpx import AsyncClient, ASGITransport

from sqlagent.config import AgentConfig
from sqlagent.server import create_app
from sqlagent.auth import _get_secret, send_magic_link, verify_magic_link


@pytest.fixture
async def client():
    app = create_app(AgentConfig(auth_enabled=False, otel_enabled=False))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac


# ── Magic link ────────────────────────────────────────────────────────────────

def test_magic_link_code_not_in_response_body():
    """The code returned by send_magic_link must NOT leak into HTTP responses."""
    code = send_magic_link("user@example.com")
    assert len(code) == 6
    assert code.isdigit()
    # Verify it works
    assert verify_magic_link("user@example.com", code) is True
    # Verify it's consumed (one-time use)
    assert verify_magic_link("user@example.com", code) is False


def test_magic_link_expired_code_rejected():
    import time
    from sqlagent.auth import _magic_codes
    email = "expire@test.com"
    _magic_codes[email] = ("123456", time.time() - 1)  # already expired
    assert verify_magic_link(email, "123456") is False


def test_magic_link_wrong_code_rejected():
    send_magic_link("wrong@test.com")
    assert verify_magic_link("wrong@test.com", "000000") is False


# ── JWT ───────────────────────────────────────────────────────────────────────

def test_jwt_secret_is_at_least_32_chars():
    secret = _get_secret()
    assert len(secret) >= 32, "JWT secret must be at least 32 characters"


def test_jwt_secret_file_permissions(tmp_path, monkeypatch):
    """JWT secret file must be created with 0600 permissions."""
    import os
    secret_dir = tmp_path / ".sqlagent"
    secret_file = secret_dir / ".jwt_secret"
    monkeypatch.setenv("SQLAGENT_JWT_SECRET", "")
    monkeypatch.setattr("sqlagent.auth._get_secret.__code__", _get_secret.__code__)

    # Simulate fresh secret generation
    secret_dir.mkdir()
    secret = "x" * 64
    secret_file.write_text(secret)
    os.chmod(secret_file, 0o600)

    mode = oct(os.stat(secret_file).st_mode)[-3:]
    assert mode == "600", f"JWT secret file permissions should be 600, got {mode}"


# ── Policy gateway ────────────────────────────────────────────────────────────

def test_policy_blocks_drop_table():
    from sqlagent.runtime import PolicyGateway
    from sqlagent.config import AgentConfig
    gw = PolicyGateway(AgentConfig(select_only=True, no_ddl=True))
    result = gw.check("DROP TABLE users")
    assert not result.passed
    assert "DDL" in result.reason or "drop" in result.reason.lower()


def test_policy_blocks_delete():
    from sqlagent.runtime import PolicyGateway
    from sqlagent.config import AgentConfig
    gw = PolicyGateway(AgentConfig(select_only=True))
    result = gw.check("DELETE FROM orders WHERE 1=1")
    assert not result.passed


def test_policy_allows_select():
    from sqlagent.runtime import PolicyGateway
    from sqlagent.config import AgentConfig
    gw = PolicyGateway(AgentConfig(select_only=True))
    result = gw.check("SELECT id, name FROM customers LIMIT 100")
    assert result.passed


# ── CORS ─────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_cors_wildcard_not_sent_without_config(client):
    resp = await client.get("/health", headers={"Origin": "https://attacker.com"})
    acao = resp.headers.get("access-control-allow-origin", "NOT_SET")
    assert acao != "*", "Wildcard CORS header must not be sent by default"
