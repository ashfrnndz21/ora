"""sqlagent — The NL2SQL Agentic Runtime."""

__version__ = "2.0.0"

from sqlagent.exceptions import SQLAgentError, ConfigurationError

__all__ = [
    "__version__",
    "SQLAgent",
    "ask",
    "connect",
    "SQLAgentError",
    "ConfigurationError",
]

# Lazy imports for heavy modules — only loaded when accessed
def __getattr__(name: str):
    if name == "SQLAgent":
        from sqlagent.agent import SQLAgent
        return SQLAgent
    if name == "ask":
        from sqlagent.agent import ask
        return ask
    if name == "connect":
        from sqlagent.agent import connect
        return connect
    raise AttributeError(f"module 'sqlagent' has no attribute {name!r}")
