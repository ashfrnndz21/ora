"""Runtime: Policy + Sessions + Memory + Checkpoint — all in one file.

PolicyGateway: deterministic SQL policy enforcement (NO LLM calls)
QuerySession: multi-turn conversation state with token budget
Memory: 3-tier (working + episodic + manager)
Checkpoint: session serialization to SQLite
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import structlog

logger = structlog.get_logger()


# ═══════════════════════════════════════════════════════════════════════════════
# POLICY GATEWAY (deterministic — NO LLM, cannot be hallucinated around)
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class PolicyResult:
    passed: bool = True
    rule_id: str = ""
    reason: str = ""
    action: str = ""  # "block" | "warn" | "redact" | "modify"
    modified_sql: str = ""  # Non-empty if policy modified the SQL (e.g., added LIMIT)


class PolicyGateway:
    """Deterministic SQL policy enforcement. Runs before EVERY database call.

    Rules:
    - no_ddl: blocks DROP, TRUNCATE, ALTER, CREATE, DELETE, UPDATE, INSERT
    - select_only: blocks anything that isn't SELECT or WITH
    - row_limit: auto-adds LIMIT if missing
    - cost_ceiling: blocks if session budget exhausted
    - pii_columns: blocks if query references PII columns
    """

    _DDL_PATTERN = re.compile(
        r"\b(DROP|TRUNCATE|ALTER|CREATE|DELETE|UPDATE|INSERT|GRANT|REVOKE)\b",
        re.IGNORECASE,
    )
    _SELECT_PATTERN = re.compile(
        r"^\s*(SELECT|WITH)\b",
        re.IGNORECASE,
    )

    def __init__(self, config: Any = None):
        self._config = config
        self._no_ddl = getattr(config, "no_ddl", True) if config else True
        self._select_only = getattr(config, "select_only", True) if config else True
        self._row_limit = getattr(config, "row_limit", 10_000) if config else 10_000
        self._cost_ceiling = getattr(config, "cost_ceiling_usd", 10.0) if config else 10.0
        self._pii_columns = set(getattr(config, "pii_columns", []) if config else [])

    def check(self, sql: str, state: dict | None = None) -> PolicyResult:
        """Check SQL against all policy rules. Returns PolicyResult."""
        state = state or {}
        sql_upper = sql.strip().upper()

        # Rule: no_ddl
        if self._no_ddl and self._DDL_PATTERN.search(sql):
            match = self._DDL_PATTERN.search(sql)
            keyword = match.group(1) if match else "DDL"
            return PolicyResult(
                passed=False,
                rule_id="no_ddl",
                reason=f"{keyword} statements are not allowed",
                action="block",
            )

        # Rule: select_only
        if self._select_only and not self._SELECT_PATTERN.match(sql.strip()):
            return PolicyResult(
                passed=False,
                rule_id="select_only",
                reason="Only SELECT/WITH queries are allowed",
                action="block",
            )

        # Rule: cost_ceiling
        session_cost = state.get("cost_usd", 0.0)
        if session_cost >= self._cost_ceiling:
            return PolicyResult(
                passed=False,
                rule_id="cost_ceiling",
                reason=f"Session cost ${session_cost:.4f} exceeds ceiling ${self._cost_ceiling:.2f}",
                action="block",
            )

        # Rule: pii_columns
        if self._pii_columns:
            sql_lower = sql.lower()
            for pii_col in self._pii_columns:
                if pii_col.lower() in sql_lower:
                    return PolicyResult(
                        passed=False,
                        rule_id="pii_columns",
                        reason=f"Query references PII column: {pii_col}",
                        action="block",
                    )

        # Rule: row_limit (modify, not block)
        modified_sql = ""
        if self._row_limit and "LIMIT" not in sql_upper:
            modified_sql = sql.rstrip().rstrip(";") + f" LIMIT {self._row_limit}"

        return PolicyResult(passed=True, modified_sql=modified_sql)


# ═══════════════════════════════════════════════════════════════════════════════
# QUERY SESSION
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class TokenBudget:
    """Tracks token/cost spending per session."""

    max_tokens: int = 100_000
    max_cost_usd: float = 10.0
    tokens_used: int = 0
    cost_used: float = 0.0

    @property
    def tokens_remaining(self) -> int:
        return max(0, self.max_tokens - self.tokens_used)

    @property
    def cost_remaining(self) -> float:
        return max(0.0, self.max_cost_usd - self.cost_used)

    @property
    def exhausted(self) -> bool:
        return self.tokens_used >= self.max_tokens or self.cost_used >= self.max_cost_usd

    def spend(self, tokens: int = 0, cost: float = 0.0) -> None:
        self.tokens_used += tokens
        self.cost_used += cost


@dataclass
class Turn:
    """A single turn in a multi-turn conversation."""

    turn_id: str = ""
    role: str = ""  # "user" | "assistant"
    nl_query: str = ""
    sql: str = ""
    nl_response: str = ""
    succeeded: bool = False
    tokens: int = 0
    cost_usd: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class QuerySession:
    """Multi-turn query session with conversation memory and budget tracking."""

    session_id: str = ""
    user_id: str = ""
    workspace_id: str = ""
    db_targets: list[str] = field(default_factory=list)
    conversation: list[Turn] = field(default_factory=list)
    token_budget: TokenBudget = field(default_factory=TokenBudget)
    status: str = "active"  # "active" | "suspended" | "closed"
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def add_turn(self, turn: Turn) -> None:
        self.conversation.append(turn)
        self.token_budget.spend(tokens=turn.tokens, cost=turn.cost_usd)

    def recent_context(self, n: int = 5) -> list[Turn]:
        """Get the last n turns for conversation context."""
        return self.conversation[-n:]

    def to_messages(self, n: int = 5) -> list[dict]:
        """Convert recent turns to LLM message format."""
        messages = []
        for turn in self.recent_context(n):
            if turn.role == "user":
                messages.append({"role": "user", "content": turn.nl_query})
            elif turn.role == "assistant":
                messages.append({"role": "assistant", "content": turn.nl_response or turn.sql})
        return messages

    @staticmethod
    def create(user_id: str = "", workspace_id: str = "") -> "QuerySession":
        return QuerySession(
            session_id=str(uuid.uuid4()),
            user_id=user_id,
            workspace_id=workspace_id,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# MEMORY (3-tier)
# ═══════════════════════════════════════════════════════════════════════════════


class WorkingMemory:
    """Tier 1: Session-scoped in-memory store. Dies when session ends."""

    def __init__(self):
        self._store: dict[str, Any] = {}

    def set(self, key: str, value: Any) -> None:
        self._store[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        return self._store.get(key, default)

    def get_recent_tables(self) -> list[str]:
        return self._store.get("recent_tables", [])

    def record_tables(self, tables: list[str]) -> None:
        existing = self._store.get("recent_tables", [])
        updated = list(dict.fromkeys(existing + tables))  # preserve order, dedup
        self._store["recent_tables"] = updated[-20:]  # keep last 20


class EpisodicMemory:
    """Tier 2: Per-user persistent memory in SQLite.

    Tracks: which tables the user queries, which generators win,
    query patterns, and corrections. Survives across sessions.
    """

    def __init__(self, db_path: str = ""):
        import os

        if not db_path:
            db_path = os.path.join(os.path.expanduser("~"), ".sqlagent", "episodic.db")
        self._db_path = db_path
        self._initialized = False

    async def init(self) -> None:
        if self._initialized:
            return
        import os
        import aiosqlite

        os.makedirs(os.path.dirname(self._db_path), exist_ok=True)
        async with aiosqlite.connect(self._db_path) as db:
            await db.executescript("""
                CREATE TABLE IF NOT EXISTS query_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT, nl_query TEXT, sql TEXT,
                    tables_used TEXT, generator TEXT, succeeded INTEGER,
                    created_at TEXT
                );
                CREATE TABLE IF NOT EXISTS table_frequency (
                    user_id TEXT, table_name TEXT, count INTEGER DEFAULT 1,
                    PRIMARY KEY (user_id, table_name)
                );
                CREATE TABLE IF NOT EXISTS generator_wins (
                    user_id TEXT, generator TEXT, count INTEGER DEFAULT 1,
                    PRIMARY KEY (user_id, generator)
                );
            """)
            await db.commit()
        self._initialized = True

    async def record_query(
        self,
        user_id: str,
        nl_query: str,
        sql: str,
        tables: list[str],
        generator: str,
        succeeded: bool,
    ) -> None:
        await self.init()
        import aiosqlite

        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                "INSERT INTO query_log (user_id, nl_query, sql, tables_used, generator, succeeded, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    user_id,
                    nl_query,
                    sql,
                    ",".join(tables),
                    generator,
                    int(succeeded),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            for table in tables:
                await db.execute(
                    "INSERT INTO table_frequency (user_id, table_name, count) VALUES (?, ?, 1) "
                    "ON CONFLICT(user_id, table_name) DO UPDATE SET count = count + 1",
                    (user_id, table),
                )
            if generator and succeeded:
                await db.execute(
                    "INSERT INTO generator_wins (user_id, generator, count) VALUES (?, ?, 1) "
                    "ON CONFLICT(user_id, generator) DO UPDATE SET count = count + 1",
                    (user_id, generator),
                )
            await db.commit()

    async def get_top_tables(self, user_id: str, limit: int = 10) -> list[tuple[str, int]]:
        await self.init()
        import aiosqlite

        async with aiosqlite.connect(self._db_path) as db:
            cursor = await db.execute(
                "SELECT table_name, count FROM table_frequency "
                "WHERE user_id = ? ORDER BY count DESC LIMIT ?",
                (user_id, limit),
            )
            return [(r[0], r[1]) for r in await cursor.fetchall()]

    async def get_query_count(self, user_id: str) -> int:
        await self.init()
        import aiosqlite

        async with aiosqlite.connect(self._db_path) as db:
            cursor = await db.execute(
                "SELECT COUNT(*) FROM query_log WHERE user_id = ?", (user_id,)
            )
            row = await cursor.fetchone()
            return row[0] if row else 0


class MemoryManager:
    """Coordinates all memory tiers."""

    def __init__(
        self, working: WorkingMemory | None = None, episodic: EpisodicMemory | None = None
    ):
        self.working = working or WorkingMemory()
        self.episodic = episodic or EpisodicMemory()

    async def record_query(
        self,
        user_id: str,
        nl_query: str,
        sql: str,
        tables: list[str],
        generator: str,
        succeeded: bool,
    ) -> None:
        # Working memory
        self.working.record_tables(tables)

        # Episodic memory
        try:
            await self.episodic.record_query(
                user_id=user_id,
                nl_query=nl_query,
                sql=sql,
                tables=tables,
                generator=generator,
                succeeded=succeeded,
            )
        except Exception as e:
            logger.warn("episodic.write_failed", error=str(e))
