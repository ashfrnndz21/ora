"""ora — The NL2SQL Agentic Runtime.

Drop-in, zero-config interface:

    import ora

    # Single source
    db = ora.connect("postgresql://localhost/mydb")
    result = db.query("top 10 customers by revenue")
    print(result.dataframe)

    # Multi-source (cross-database synthesis)
    with ora.connect(
        sales="postgresql://prod/sales",
        inventory="snowflake://warehouse/inventory",
        staff="headcount.csv",
    ) as db:
        result = db.query("revenue per employee by store this quarter?")
        print(result.dataframe)

    # One-liner
    result = ora.ask("top 10 customers", db="sqlite:///northwind.db")
"""

from __future__ import annotations

__version__ = "2.0.0"

import asyncio
import concurrent.futures
from typing import Any

from sqlagent.exceptions import SQLAgentError, ConfigurationError

__all__ = [
    "__version__",
    "OraDB",
    "connect",
    "ask",
    "SQLAgentError",
    "ConfigurationError",
]


def _run_sync(coro):
    """Run a coroutine synchronously, safe from inside or outside an event loop."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # We're inside an async context (e.g. Jupyter, FastAPI test) — use a thread
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(asyncio.run, coro)
            return future.result()
    else:
        return asyncio.run(coro)


class OraDB:
    """Synchronous-friendly wrapper around SQLAgent.

    Returned by :func:`connect`. Supports both direct use and as a context manager::

        # Direct
        db = ora.connect("postgresql://localhost/mydb")
        result = db.query("top customers by revenue")

        # Context manager
        with ora.connect("postgresql://localhost/mydb") as db:
            result = db.query("top customers by revenue")
    """

    def __init__(self, agent: Any) -> None:
        self._agent = agent

    # ── Public API ────────────────────────────────────────────────────────────

    def query(self, nl_query: str, **kwargs) -> Any:
        """Run a natural language query synchronously and return a PipelineResult."""
        return _run_sync(self._agent.query(nl_query, **kwargs))

    async def async_query(self, nl_query: str, **kwargs) -> Any:
        """Async variant — use inside ``async def`` functions."""
        return await self._agent.query(nl_query, **kwargs)

    # ── Context manager support ───────────────────────────────────────────────

    def __enter__(self) -> "OraDB":
        return self

    def __exit__(self, *args) -> None:
        pass

    async def __aenter__(self) -> "OraDB":
        return self

    async def __aexit__(self, *args) -> None:
        pass

    # ── Passthrough to underlying agent ──────────────────────────────────────

    def __getattr__(self, name: str) -> Any:
        return getattr(self._agent, name)

    def __repr__(self) -> str:
        return f"<OraDB agent={self._agent!r}>"


class _MultiSourceContext:
    """Context manager for multi-source connections.

    Usage::

        with ora.connect(sales="postgresql://...", staff="headcount.csv") as db:
            result = db.query("revenue per employee?")
    """

    def __init__(self, sources: dict[str, str]) -> None:
        self._sources = sources

    def __enter__(self) -> OraDB:
        from sqlagent.agent import SQLAgent

        agent = SQLAgent()
        new_loop = asyncio.new_event_loop()
        try:
            new_loop.run_until_complete(agent._ensure_ready())
            for name, url in self._sources.items():
                new_loop.run_until_complete(agent.add_connector(name, url))
        finally:
            new_loop.close()
        return OraDB(agent)

    def __exit__(self, *args) -> None:
        pass

    async def __aenter__(self) -> OraDB:
        from sqlagent.agent import SQLAgent

        agent = SQLAgent()
        await agent._ensure_ready()
        for name, url in self._sources.items():
            await agent.add_connector(name, url)
        return OraDB(agent)

    async def __aexit__(self, *args) -> None:
        pass


def connect(url: str = "", /, **sources: str) -> "OraDB | _MultiSourceContext":
    """Connect to one or more data sources.

    **Single source** — returns an :class:`OraDB` directly (no ``with`` needed)::

        db = ora.connect("postgresql://localhost/mydb")
        result = db.query("top 10 customers by revenue")

    **Multi-source** — returns a context manager that synthesises results across sources::

        with ora.connect(
            sales="postgresql://prod/sales",
            staff="headcount.csv",
        ) as db:
            result = db.query("revenue per employee by store?")

    Args:
        url: A single database URL (positional).
        **sources: Named database URLs for multi-source queries.
    """
    if url and not sources:
        from sqlagent.agent import SQLAgent

        agent = SQLAgent(db=url)
        new_loop = asyncio.new_event_loop()
        try:
            new_loop.run_until_complete(agent._ensure_ready())
        finally:
            new_loop.close()
        return OraDB(agent)

    if sources:
        return _MultiSourceContext(sources)

    raise ValueError(
        "connect() requires either a database URL (positional) "
        "or named source keyword arguments.\n"
        "  ora.connect('postgresql://...')\n"
        "  ora.connect(sales='postgresql://...', staff='data.csv')"
    )


def ask(nl_query: str, db: str = "", **kwargs) -> Any:
    """One-liner synchronous query — the fastest way to get results.

    Usage::

        import ora
        result = ora.ask("top 10 customers by revenue", db="sqlite:///northwind.db")
        print(result.dataframe)

    Args:
        nl_query: Plain-English question about your data.
        db: Database connection URL.
        **kwargs: Passed to the underlying agent (model, user_id, etc.)
    """
    from sqlagent.agent import ask as _ask

    return _ask(nl_query, db=db, **kwargs)


# Lazy-import heavy modules only when accessed
def __getattr__(name: str) -> Any:
    if name == "SQLAgent":
        from sqlagent.agent import SQLAgent
        return SQLAgent
    raise AttributeError(f"module 'ora' has no attribute {name!r}")
