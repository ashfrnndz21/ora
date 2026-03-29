"""All configuration dataclasses for sqlagent."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class AgentCoreConfig:
    """AWS AgentCore deployment configuration."""
    enabled: bool = False
    region: str = "us-east-1"
    bedrock_model_id: str = "anthropic.claude-sonnet-4-6"
    opensearch_endpoint: str = ""
    dynamodb_sessions_table: str = "sqlagent-sessions"
    cedar_policy_store_id: str = ""
    secrets_prefix: str = "sqlagent/"


@dataclass(frozen=True)
class DataSourceConfig:
    """Per-source connection configuration."""
    source_id: str
    type: str                          # sqlite, postgresql, mysql, duckdb, snowflake, bigquery, redshift, csv, xlsx
    connection_string: str = ""
    file_path: str = ""
    display_name: str = ""
    schema_name: str = ""
    warehouse: str = ""                # Snowflake
    dataset: str = ""                  # BigQuery
    read_only: bool = True


@dataclass(frozen=True)
class AgentConfig:
    """Master configuration for SQLAgent."""

    # LLM
    llm_provider: str = "litellm"
    llm_model: str = "gpt-4o"
    temperature: float = 0.0
    max_tokens: int = 4096

    # Embedding
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dimensions: int = 384

    # Generators
    generators: list[str] = field(default_factory=lambda: ["fewshot", "plan", "decompose"])
    max_candidates: int = 4
    generation_timeout_s: float = 30.0

    # Pipeline
    max_corrections: int = 3
    schema_pruning_top_k: int = 15
    example_retrieval_top_k: int = 3
    row_limit: int = 10_000
    query_timeout_s: float = 60.0

    # Memory
    vector_store_backend: str = "qdrant"   # qdrant | chroma
    episodic_backend: str = "sqlite"       # sqlite | dynamodb
    soul_evolution_interval: int = 20      # queries between SOUL evolution

    # Policy
    select_only: bool = True
    no_ddl: bool = True
    cost_ceiling_usd: float = 10.0
    pii_columns: list[str] = field(default_factory=list)

    # Auth
    auth_enabled: bool = False
    auth_jwt_secret: str = ""
    auth_google_client_id: str = ""
    auth_google_client_secret: str = ""

    # Telemetry
    otel_enabled: bool = True
    otel_endpoint: str = ""                # OTLP gRPC endpoint (empty = console)
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"
    prometheus_enabled: bool = True

    # AgentCore (AWS)
    agentcore: AgentCoreConfig = field(default_factory=AgentCoreConfig)

    # Server
    host: str = "0.0.0.0"
    port: int = 8080
    # CORS: empty list = deny all cross-origin requests (safe default).
    # For local dev set SQLAGENT_CORS_ORIGINS="http://localhost:3000,http://localhost:5173".
    # Never use ["*"] with allow_credentials=True — that is a known XSS vector.
    cors_origins: list[str] = field(default_factory=list)
