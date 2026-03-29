"""Test PolicyGateway — deterministic SQL policy enforcement."""

import pytest
from sqlagent.runtime import PolicyGateway, PolicyResult, QuerySession, TokenBudget


class MockConfig:
    no_ddl = True
    select_only = True
    row_limit = 1000
    cost_ceiling_usd = 5.0
    pii_columns = ["email", "phone", "national_id"]


@pytest.fixture
def policy():
    return PolicyGateway(MockConfig())


# ── DDL blocking ──────────────────────────────────────────────────────────────

def test_blocks_drop(policy):
    r = policy.check("DROP TABLE customers")
    assert not r.passed
    assert r.rule_id == "no_ddl"


def test_blocks_delete(policy):
    r = policy.check("DELETE FROM orders WHERE id = 1")
    assert not r.passed
    assert r.rule_id == "no_ddl"


def test_blocks_update(policy):
    r = policy.check("UPDATE orders SET total = 0")
    assert not r.passed
    assert r.rule_id == "no_ddl"


def test_blocks_insert(policy):
    r = policy.check("INSERT INTO customers VALUES ('X', 'Y')")
    assert not r.passed
    assert r.rule_id == "no_ddl"


def test_blocks_truncate(policy):
    r = policy.check("TRUNCATE TABLE orders")
    assert not r.passed


# ── SELECT only ───────────────────────────────────────────────────────────────

def test_allows_select(policy):
    r = policy.check("SELECT * FROM customers")
    assert r.passed


def test_allows_with(policy):
    r = policy.check("WITH cte AS (SELECT 1) SELECT * FROM cte")
    assert r.passed


# ── Row limit ─────────────────────────────────────────────────────────────────

def test_adds_limit_when_missing(policy):
    r = policy.check("SELECT * FROM customers")
    assert r.passed
    assert "LIMIT 1000" in r.modified_sql


def test_does_not_add_limit_when_present(policy):
    r = policy.check("SELECT * FROM customers LIMIT 10")
    assert r.passed
    assert r.modified_sql == ""  # no modification needed


# ── PII columns ───────────────────────────────────────────────────────────────

def test_blocks_pii_column(policy):
    r = policy.check("SELECT email, name FROM customers")
    assert not r.passed
    assert r.rule_id == "pii_columns"


def test_blocks_pii_phone(policy):
    r = policy.check("SELECT phone FROM staff")
    assert not r.passed
    assert r.rule_id == "pii_columns"


def test_allows_non_pii(policy):
    r = policy.check("SELECT name, country FROM customers")
    assert r.passed


# ── Cost ceiling ──────────────────────────────────────────────────────────────

def test_blocks_when_budget_exhausted(policy):
    r = policy.check("SELECT 1", {"cost_usd": 6.0})
    assert not r.passed
    assert r.rule_id == "cost_ceiling"


def test_allows_within_budget(policy):
    r = policy.check("SELECT 1", {"cost_usd": 1.0})
    assert r.passed


# ── Session ───────────────────────────────────────────────────────────────────

def test_session_create():
    s = QuerySession.create(user_id="u1", workspace_id="ws1")
    assert s.session_id
    assert s.user_id == "u1"
    assert s.status == "active"


def test_token_budget():
    b = TokenBudget(max_tokens=1000, max_cost_usd=1.0)
    assert not b.exhausted
    b.spend(tokens=500, cost=0.5)
    assert b.tokens_remaining == 500
    assert not b.exhausted
    b.spend(tokens=500, cost=0.5)
    assert b.exhausted
