"""All sqlagent exceptions in one place."""

from __future__ import annotations


# ── Base ──────────────────────────────────────────────────────────────────────
class SQLAgentError(Exception):
    """Root exception for all sqlagent errors."""


# ── Configuration ─────────────────────────────────────────────────────────────
class ConfigurationError(SQLAgentError):
    """Invalid or missing configuration."""


class MissingAPIKeyError(ConfigurationError):
    """No API key for the requested LLM provider."""


# ── Connector ─────────────────────────────────────────────────────────────────
class ConnectorError(SQLAgentError):
    """Database connector error."""


class ConnectorNotFound(ConnectorError):
    """No connector registered for the given source_id or dialect."""


class ConnectionFailed(ConnectorError):
    """Could not establish database connection."""


class SchemaIntrospectionError(ConnectorError):
    """Failed to introspect database schema."""


# ── LLM ───────────────────────────────────────────────────────────────────────
class LLMError(SQLAgentError):
    """LLM call error."""


class LLMCallFailed(LLMError):
    """LLM API call failed (timeout, rate limit, auth, etc.)."""


class LLMParseError(LLMError):
    """Could not parse structured output from LLM response."""


class EmbeddingError(LLMError):
    """Embedding generation failed."""


# ── Pipeline ──────────────────────────────────────────────────────────────────
class PipelineError(SQLAgentError):
    """Pipeline execution error."""


class NoCandidatesSucceeded(PipelineError):
    """All SQL generators failed to produce valid SQL."""


class SchemaPruningError(PipelineError):
    """Schema pruning failed (no columns selected)."""


class GeneratorError(PipelineError):
    """Individual generator failure."""


class CorrectionExhausted(PipelineError):
    """All correction stages failed to fix the SQL."""


# ── Execution ─────────────────────────────────────────────────────────────────
class ExecutionError(SQLAgentError):
    """SQL execution error."""


class SQLExecutionFailed(ExecutionError):
    """SQL query failed against the database."""

    def __init__(self, sql: str, error: str, sql_state: str = ""):
        self.sql = sql
        self.error_message = error
        self.sql_state = sql_state
        super().__init__(f"SQL failed: {error}")


class ExecutionTimeout(ExecutionError):
    """SQL query exceeded timeout."""


# ── Policy ────────────────────────────────────────────────────────────────────
class PolicyViolation(SQLAgentError):
    """SQL blocked by policy gateway."""

    def __init__(self, rule_id: str, reason: str, sql: str = ""):
        self.rule_id = rule_id
        self.reason = reason
        self.sql = sql
        super().__init__(f"Policy violation [{rule_id}]: {reason}")


class BudgetExceeded(PolicyViolation):
    """Session cost/token budget exhausted."""


# ── Session ───────────────────────────────────────────────────────────────────
class SessionError(SQLAgentError):
    """Session management error."""


class SessionNotFound(SessionError):
    """Session ID does not exist."""


# ── Vector Store ──────────────────────────────────────────────────────────────
class VectorStoreError(SQLAgentError):
    """Vector store operation failed."""


# ── Synthesis ─────────────────────────────────────────────────────────────────
class SynthesisError(SQLAgentError):
    """Cross-source synthesis failed."""


class JoinKeyNotFound(SynthesisError):
    """No join key found between sub-query results."""


# ── Workspace ─────────────────────────────────────────────────────────────────
class WorkspaceError(SQLAgentError):
    """Workspace management error."""


class WorkspaceNotFound(WorkspaceError):
    """Workspace ID does not exist."""


# ── Auth ──────────────────────────────────────────────────────────────────────
class AuthError(SQLAgentError):
    """Authentication/authorization error."""


class InvalidToken(AuthError):
    """JWT token is invalid or expired."""
