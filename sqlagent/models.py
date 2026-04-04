"""All sqlagent data models in one file.

Covers: schema, knowledge graph, traces, workspaces, auth, events,
training examples, pipeline results, and SQL candidates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ═══════════════════════════════════════════════════════════════════════════════
# SCHEMA MODELS
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class SchemaColumn:
    name: str
    data_type: str
    nullable: bool = True
    is_primary_key: bool = False
    is_foreign_key: bool = False
    foreign_key_ref: dict | None = None  # {"table": ..., "column": ...}
    default_value: str | None = None
    column_position: int = 0
    description: str = ""
    examples: list[str] = field(default_factory=list)
    aliases: list[str] = field(default_factory=list)
    semantic_type: str = ""  # identifier, currency, email, timestamp, etc.
    is_pii: bool = False
    merkle_hash: str = ""


@dataclass
class SchemaTable:
    name: str
    schema_name: str = "public"
    columns: list[SchemaColumn] = field(default_factory=list)
    row_count_estimate: int = 0
    indexes: list[dict] = field(default_factory=list)
    constraints: list[dict] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    description: str = ""
    merkle_hash: str = ""


@dataclass
class ForeignKey:
    from_table: str
    from_column: str
    to_table: str
    to_column: str


@dataclass
class SchemaSnapshot:
    """Point-in-time capture of a database schema."""

    source_id: str
    dialect: str
    tables: list[SchemaTable] = field(default_factory=list)
    foreign_keys: list[ForeignKey] = field(default_factory=list)
    introspected_at: datetime = field(default_factory=_utcnow)
    merkle_root: str = ""

    @property
    def table_count(self) -> int:
        return len(self.tables)

    @property
    def column_count(self) -> int:
        return sum(len(t.columns) for t in self.tables)

    def get_table(self, name: str) -> SchemaTable | None:
        for t in self.tables:
            if t.name == name:
                return t
        return None


@dataclass
class ColumnStats:
    """Statistics for a single column from sampling."""

    distinct_count: int = 0
    null_count: int = 0
    min_value: Any = None
    max_value: Any = None
    mean_value: float | None = None
    sample_values: list[Any] = field(default_factory=list)
    pattern: str = ""  # regex pattern detected


@dataclass
class SampleData:
    """Sample rows + column stats for a table."""

    table: str
    sample_rows: list[dict] = field(default_factory=list)
    column_stats: dict[str, ColumnStats] = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════════════
# KNOWLEDGE GRAPH
# ═══════════════════════════════════════════════════════════════════════════════


class EdgeType(str, Enum):
    DECLARED_FK = "declared_fk"
    INFERRED = "inferred"
    CROSS_SOURCE = "cross_source"
    SEMANTIC = "semantic"


@dataclass
class KGNode:
    """A node in the knowledge graph (table or column)."""

    id: str  # "tbl:orders" or "col:orders.total_amount"
    type: str  # "table" | "column" | "source"
    source_id: str = ""
    name: str = ""
    properties: dict = field(default_factory=dict)
    # Table props: row_count, column_count, entity_group, description
    # Column props: data_type, is_pk, is_fk, semantic_type, description, pii, aliases, merkle_hash


@dataclass
class KGEdge:
    """An edge in the knowledge graph (relationship between nodes)."""

    id: str
    source: str  # node ID
    target: str  # node ID
    type: EdgeType = EdgeType.DECLARED_FK
    join_columns: dict = field(default_factory=dict)  # {"from": ..., "to": ...}
    confidence: float = 1.0
    cardinality: str = ""  # "one_to_one", "one_to_many", "many_to_one", "many_to_many"
    evidence: str = ""
    description: str = ""


@dataclass
class KGLayer:
    """An entity group layer in the knowledge graph."""

    id: str
    name: str
    description: str = ""
    tables: list[str] = field(default_factory=list)  # node IDs
    color: str = "#4f7df9"


@dataclass
class SemanticEntry:
    """A glossary term mapping business language to schema."""

    term: str
    maps_to: str = ""  # "table.column"
    definition: str = ""


@dataclass
class KnowledgeGraph:
    """The complete knowledge graph for a workspace."""

    graph_id: str = ""
    workspace_id: str = ""
    version: int = 1
    built_at: datetime = field(default_factory=_utcnow)

    sources: list[dict] = field(default_factory=list)  # source metadata
    nodes: list[KGNode] = field(default_factory=list)
    edges: list[KGEdge] = field(default_factory=list)
    layers: list[KGLayer] = field(default_factory=list)

    glossary: list[SemanticEntry] = field(default_factory=list)
    pii_columns: list[str] = field(default_factory=list)
    time_columns: list[dict] = field(default_factory=list)

    @property
    def table_count(self) -> int:
        return sum(1 for n in self.nodes if n.type == "table")

    @property
    def column_count(self) -> int:
        return sum(1 for n in self.nodes if n.type == "column")

    @property
    def edge_count(self) -> int:
        return len(self.edges)

    def to_dict(self) -> dict:
        """JSON-serializable dict for API responses."""
        return {
            "graph_id": self.graph_id,
            "workspace_id": self.workspace_id,
            "version": self.version,
            "built_at": self.built_at.isoformat(),
            "sources": self.sources,
            "nodes": [
                {
                    "id": n.id,
                    "type": n.type,
                    "source_id": n.source_id,
                    "name": n.name,
                    "properties": n.properties,
                }
                for n in self.nodes
            ],
            "edges": [
                {
                    "id": e.id,
                    "source": e.source,
                    "target": e.target,
                    "type": e.type.value,
                    "join_columns": e.join_columns,
                    "confidence": e.confidence,
                    "cardinality": e.cardinality,
                    "evidence": e.evidence,
                    "description": e.description,
                }
                for e in self.edges
            ],
            "layers": [
                {
                    "id": layer.id,
                    "name": layer.name,
                    "description": layer.description,
                    "tables": layer.tables,
                    "color": layer.color,
                }
                for layer in self.layers
            ],
            "glossary": [
                {"term": g.term, "maps_to": g.maps_to, "definition": g.definition}
                for g in self.glossary
            ],
            "pii_columns": self.pii_columns,
            "stats": {
                "tables": self.table_count,
                "columns": self.column_count,
                "edges": self.edge_count,
                "layers": len(self.layers),
                "glossary_terms": len(self.glossary),
            },
        }


# ═══════════════════════════════════════════════════════════════════════════════
# TRACE (EXECUTION TRACE TREE)
# ═══════════════════════════════════════════════════════════════════════════════


class TraceStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class TraceNode:
    """A single node in the execution trace tree."""

    node_id: str
    name: str
    agent: str = ""  # "orchestrator", "sql_agent", "schema_agent", etc.
    status: TraceStatus = TraceStatus.PENDING
    started_at: datetime | None = None
    completed_at: datetime | None = None
    latency_ms: int = 0
    tokens: int = 0
    cost_usd: float = 0.0
    summary: str = ""
    detail: dict = field(default_factory=dict)  # node-specific: sql, error, row_count, etc.
    children: list[TraceNode] = field(default_factory=list)
    parent_id: str | None = None

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "name": self.name,
            "agent": self.agent,
            "status": self.status.value,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "latency_ms": self.latency_ms,
            "tokens": self.tokens,
            "cost_usd": self.cost_usd,
            "summary": self.summary,
            "detail": self.detail,
            "children": [c.to_dict() for c in self.children],
        }


@dataclass
class Trace:
    """Complete execution trace for a query task."""

    trace_id: str
    workspace_id: str = ""
    user_id: str = ""
    nl_query: str = ""
    root: TraceNode | None = None
    status: TraceStatus = TraceStatus.PENDING
    started_at: datetime = field(default_factory=_utcnow)
    completed_at: datetime | None = None
    total_latency_ms: int = 0
    total_tokens: int = 0
    tokens_input: int = 0  # prompt / input tokens
    tokens_output: int = 0  # completion / output tokens
    total_cost_usd: float = 0.0
    succeeded: bool = False
    winner_generator: str = ""
    correction_rounds: int = 0
    sql: str = ""
    row_count: int = 0
    error: str = ""
    model_id: str = ""  # e.g. "claude-sonnet-4-5" — which model ran this query

    def to_dict(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "workspace_id": self.workspace_id,
            "user_id": self.user_id,
            "nl_query": self.nl_query,
            "root": self.root.to_dict() if self.root else None,
            "status": self.status.value,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "total_latency_ms": self.total_latency_ms,
            "total_tokens": self.total_tokens,
            "tokens_input": self.tokens_input,
            "tokens_output": self.tokens_output,
            "total_cost_usd": self.total_cost_usd,
            "succeeded": self.succeeded,
            "winner_generator": self.winner_generator,
            "correction_rounds": self.correction_rounds,
            "sql": self.sql,
            "row_count": self.row_count,
            "error": self.error,
            "model_id": self.model_id,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# WORKSPACE + AUTH
# ═══════════════════════════════════════════════════════════════════════════════


class WorkspaceStatus(str, Enum):
    SETUP = "setup"
    ANALYZING = "analyzing"
    READY = "ready"
    ERROR = "error"


@dataclass
class Workspace:
    workspace_id: str
    name: str
    owner_id: str = ""
    description: str = ""
    status: WorkspaceStatus = WorkspaceStatus.SETUP
    sources: list[dict] = field(default_factory=list)  # DataSourceConfig as dicts
    knowledge_graph_version: int = 0
    query_count: int = 0
    created_at: datetime = field(default_factory=_utcnow)
    updated_at: datetime = field(default_factory=_utcnow)


@dataclass
class User:
    user_id: str
    email: str
    display_name: str = ""
    avatar_url: str = ""
    provider: str = "email"  # "google" | "email"
    created_at: datetime = field(default_factory=_utcnow)
    last_login: datetime = field(default_factory=_utcnow)


# ═══════════════════════════════════════════════════════════════════════════════
# SQL GENERATION
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class Candidate:
    """Output from a SQL generator."""

    candidate_id: str = ""
    generator_id: str = ""  # "fewshot", "plan", "decompose"
    sql: str = ""
    reasoning: str = ""
    confidence: float = 0.0
    tokens_used: int = 0
    cost_usd: float = 0.0
    latency_ms: int = 0
    error: str = ""

    @property
    def succeeded(self) -> bool:
        return bool(self.sql) and not self.error


@dataclass
class TrainingExample:
    """An NL→SQL training pair stored in the vector store."""

    nl_query: str
    sql: str
    ddl: str = ""
    source_id: str = ""
    complexity: str = "simple"
    generator: str = ""
    verified: bool = False
    metadata: dict = field(default_factory=dict)


@dataclass
class SearchResult:
    """A training example with similarity score from vector search."""

    example: TrainingExample
    similarity: float = 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# PIPELINE RESULT
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class PipelineResult:
    """Final result from a query execution."""

    query_id: str = ""
    nl_query: str = ""
    sql: str = ""
    succeeded: bool = False
    error: str = ""

    # Data
    rows: list[dict] = field(default_factory=list)
    columns: list[str] = field(default_factory=list)
    row_count: int = 0

    # Response
    nl_response: str = ""
    follow_ups: list[str] = field(default_factory=list)
    chart_config: dict | None = None
    confidence: dict | None = None  # {"total": 85, "level": "high", "reasoning": "..."}

    # Metrics
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    latency_ms: int = 0
    winner_generator: str = ""
    correction_rounds: int = 0

    # Trace
    trace: Trace | None = None

    @property
    def dataframe(self):
        """Return results as a pandas DataFrame."""
        import pandas as pd

        return pd.DataFrame(self.rows, columns=self.columns) if self.rows else pd.DataFrame()


# ═══════════════════════════════════════════════════════════════════════════════
# EVENTS (emitted by pipeline nodes via EventBus)
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class BaseEvent:
    """Base for all pipeline events."""

    event_type: str = ""
    timestamp: datetime = field(default_factory=_utcnow)
    metadata: dict = field(default_factory=dict)


@dataclass
class QueryStarted(BaseEvent):
    event_type: str = "query.started"
    query_id: str = ""
    nl_query: str = ""
    user_id: str = ""
    workspace_id: str = ""


@dataclass
class SchemaRetrieved(BaseEvent):
    event_type: str = "schema.retrieved"
    source_id: str = ""
    table_count: int = 0
    column_count: int = 0


@dataclass
class SchemaPruned(BaseEvent):
    event_type: str = "schema.pruned"
    columns_before: int = 0
    columns_after: int = 0
    selected_tables: list[str] = field(default_factory=list)


@dataclass
class ExamplesRetrieved(BaseEvent):
    event_type: str = "examples.retrieved"
    example_count: int = 0
    top_similarity: float = 0.0


@dataclass
class PlanningCompleted(BaseEvent):
    event_type: str = "planning.completed"
    strategy: str = ""
    reasoning: str = ""


@dataclass
class CandidateGenerated(BaseEvent):
    event_type: str = "candidate.generated"
    generator_id: str = ""
    sql: str = ""
    confidence: float = 0.0
    tokens: int = 0
    latency_ms: int = 0
    succeeded: bool = True
    error: str = ""


@dataclass
class CandidateSelected(BaseEvent):
    event_type: str = "candidate.selected"
    winner_generator: str = ""
    sql: str = ""
    selection_method: str = ""  # "pairwise_llm", "majority_vote", "single"
    reasoning: str = ""


@dataclass
class ExecutionResult(BaseEvent):
    event_type: str = "execution.result"
    sql: str = ""
    succeeded: bool = False
    row_count: int = 0
    latency_ms: int = 0
    error: str = ""


@dataclass
class CorrectionStarted(BaseEvent):
    event_type: str = "correction.started"
    round: int = 0
    stage: str = ""  # "error_aware", "schema_aware", "db_confirmed"
    original_error: str = ""


@dataclass
class CorrectionResult(BaseEvent):
    event_type: str = "correction.result"
    round: int = 0
    stage: str = ""
    succeeded: bool = False
    new_sql: str = ""
    error: str = ""


@dataclass
class PolicyBlocked(BaseEvent):
    event_type: str = "policy.blocked"
    rule_id: str = ""
    reason: str = ""
    sql: str = ""


@dataclass
class DecompositionPlanned(BaseEvent):
    event_type: str = "decomposition.planned"
    sub_query_count: int = 0
    synthesis_strategy: str = ""


@dataclass
class SubQueryStarted(BaseEvent):
    event_type: str = "sub_query.started"
    sub_query_id: str = ""
    source_id: str = ""
    nl_description: str = ""


@dataclass
class SubQueryCompleted(BaseEvent):
    event_type: str = "sub_query.completed"
    sub_query_id: str = ""
    succeeded: bool = False
    row_count: int = 0
    latency_ms: int = 0


@dataclass
class SynthesisCompleted(BaseEvent):
    event_type: str = "synthesis.completed"
    strategy: str = ""
    row_count: int = 0
    latency_ms: int = 0


@dataclass
class ResponseGenerated(BaseEvent):
    event_type: str = "response.generated"
    nl_response: str = ""
    follow_up_count: int = 0
    chart_type: str = ""


@dataclass
class FinalResult(BaseEvent):
    event_type: str = "query.final"
    query_id: str = ""
    succeeded: bool = False
    sql: str = ""
    row_count: int = 0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    latency_ms: int = 0
    winner_generator: str = ""
    correction_rounds: int = 0


@dataclass
class ErrorEvent(BaseEvent):
    event_type: str = "query.error"
    query_id: str = ""
    error_type: str = ""
    error_message: str = ""
    stage: str = ""


# ═══════════════════════════════════════════════════════════════════════════════
# SCHEMA ANALYSIS (output from SchemaAgent LLM pass)
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class InferredRelationship:
    from_table: str
    from_column: str
    to_table: str
    to_column: str
    confidence: float = 0.0
    evidence: str = ""
    relationship_type: str = "many_to_one"


@dataclass
class EntityGrouping:
    name: str
    description: str = ""
    tables: list[str] = field(default_factory=list)
    color: str = "#4f7df9"


@dataclass
class ColumnSemantic:
    table: str
    column: str
    semantic_type: str = ""  # identifier, currency, email, timestamp, enum, etc.
    description: str = ""
    business_term: str = ""
    aliases: list[str] = field(default_factory=list)
    pii: bool = False


@dataclass
class DataQualityIssue:
    table: str
    column: str
    issue: str = ""
    severity: str = "info"  # info, warning, error


@dataclass
class SchemaAnalysis:
    """Complete LLM analysis of a database schema."""

    analysis_id: str = ""
    source_id: str = ""
    analyzed_at: datetime = field(default_factory=_utcnow)
    model_used: str = ""
    tokens_used: int = 0

    inferred_relationships: list[InferredRelationship] = field(default_factory=list)
    entity_groups: list[EntityGrouping] = field(default_factory=list)
    column_semantics: list[ColumnSemantic] = field(default_factory=list)
    data_quality: list[DataQualityIssue] = field(default_factory=list)
    suggested_joins: list[dict] = field(default_factory=list)
