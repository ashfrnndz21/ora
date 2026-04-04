"""OraSpec — Universal Semantic Model for NL2SQL.

Vendor-neutral, YAML-serializable, LLM-injectable semantic model that encodes
business logic (metrics, dimensions, synonyms, join paths, verified queries,
fiscal calendars) across any database backend.

This is the single biggest accuracy lever for enterprise NL2SQL — more impactful
than model choice, prompt engineering, or pipeline complexity.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ═══════════════════════════════════════════════════════════════════════════════
# ENUMS
# ═══════════════════════════════════════════════════════════════════════════════


class DataType(str, Enum):
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    DATE = "date"
    TIMESTAMP = "timestamp"
    DECIMAL = "decimal"
    JSON = "json"


class AggregationType(str, Enum):
    SUM = "sum"
    COUNT = "count"
    COUNT_DISTINCT = "count_distinct"
    AVG = "avg"
    MIN = "min"
    MAX = "max"
    MEDIAN = "median"


class TimeGrain(str, Enum):
    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    QUARTER = "quarter"
    YEAR = "year"


class JoinType(str, Enum):
    INNER = "inner"
    LEFT = "left"
    RIGHT = "right"
    FULL = "full"


# ═══════════════════════════════════════════════════════════════════════════════
# COLUMN-LEVEL MODELS
# ═══════════════════════════════════════════════════════════════════════════════


class Synonym(BaseModel):
    """A business term that maps to a technical column/metric."""

    term: str  # e.g., "revenue", "sales", "top_line"
    canonical: str  # e.g., "total_revenue"


class Dimension(BaseModel):
    """A column used for grouping, filtering, or slicing data."""

    name: str  # Technical name: "store_region"
    display_name: str = ""  # Business name: "Store Region"
    description: str = ""  # "Geographic region where the store operates"
    expr: str = ""  # SQL expression: "store_region" or "UPPER(region_code)"
    data_type: DataType = DataType.STRING
    synonyms: list[str] = Field(default_factory=list)  # ["region", "area", "territory"]
    sample_values: list[str] = Field(default_factory=list)  # ["APAC", "EMEA", "AMER"]
    is_exhaustive: bool = False  # If True, sample_values is the full set
    is_unique: bool = False
    is_sensitive: bool = False  # PII flag — exclude from LLM context if True
    # Provenance — how this entry was created/evolved
    source: str = "auto"  # "auto" | "imported" | "user" | "learned"
    confidence: float = 0.5  # 0.0–1.0 Semantic Agent's certainty


class TimeDimension(BaseModel):
    """A time-typed dimension with grain awareness."""

    name: str
    display_name: str = ""
    description: str = ""
    expr: str = ""
    time_grain: TimeGrain = TimeGrain.DAY
    synonyms: list[str] = Field(default_factory=list)
    fiscal_year_start_month: int = 1  # 1=Jan, 4=Apr (for fiscal year resolution)
    source: str = "auto"
    confidence: float = 0.5


class Measure(BaseModel):
    """A numeric column that can be aggregated."""

    name: str  # "sale_amount"
    display_name: str = ""  # "Sale Amount"
    description: str = ""
    expr: str = ""  # SQL expression: "sale_amount" or "price * quantity"
    data_type: DataType = DataType.DECIMAL
    aggregation: AggregationType = AggregationType.SUM
    synonyms: list[str] = Field(default_factory=list)
    unit: Optional[str] = None  # "USD", "count", "percentage"
    is_sensitive: bool = False
    source: str = "auto"
    confidence: float = 0.5


class Metric(BaseModel):
    """A derived business metric composed of measures."""

    name: str  # "gross_margin"
    display_name: str = ""  # "Gross Margin %"
    description: str = ""  # "Revenue minus COGS divided by revenue"
    expr: str = ""  # SQL: "(SUM(revenue) - SUM(cogs)) / NULLIF(SUM(revenue), 0)"
    synonyms: list[str] = Field(default_factory=list)
    depends_on: list[str] = Field(default_factory=list)  # ["revenue", "cogs"]
    unit: Optional[str] = None
    source: str = "auto"
    confidence: float = 0.5


class Filter(BaseModel):
    """A named, reusable filter condition."""

    name: str  # "active_customers"
    display_name: str = ""  # "Active Customers"
    description: str = ""  # "Customers with purchase in last 12 months"
    expr: str = ""  # "last_purchase_date >= DATEADD(month, -12, CURRENT_DATE())"


class Relationship(BaseModel):
    """A join relationship between two logical tables."""

    from_table: str
    from_column: str
    to_table: str
    to_column: str
    join_type: JoinType = JoinType.INNER
    relationship_type: str = "many_to_one"  # "one_to_one", "many_to_one", "many_to_many"
    # Provenance
    source: str = "auto"  # "auto" | "fk" | "imported" | "user" | "inferred"
    confidence: float = 0.5  # Semantic Agent's certainty
    confirmed: bool = False  # User explicitly confirmed


class VerifiedQuery(BaseModel):
    """A pre-approved NL → SQL pair for few-shot learning."""

    question: str  # "What is the total revenue by region for Q3?"
    sql: str  # "SELECT region, SUM(revenue) FROM orders WHERE ..."
    tables_used: list[str] = Field(default_factory=list)
    verified_by: Optional[str] = None  # "analyst@company.com"
    verified_at: Optional[str] = None  # ISO timestamp
    tags: list[str] = Field(default_factory=list)  # ["revenue", "regional"]


class CustomInstruction(BaseModel):
    """Domain-specific instruction for SQL generation."""

    instruction: str  # "Always use fiscal year starting April 1"
    scope: str = "global"  # "global" | table name | metric name
    priority: int = 0  # Higher = more important


# ═══════════════════════════════════════════════════════════════════════════════
# TABLE-LEVEL MODEL
# ═══════════════════════════════════════════════════════════════════════════════


class LogicalTable(BaseModel):
    """A semantic table definition that maps to a physical table/view."""

    name: str  # Logical name: "orders"
    display_name: str = ""  # "Customer Orders"
    description: str = ""
    # Physical binding
    database: Optional[str] = None  # "analytics_db" — None = default connection
    schema_name: Optional[str] = None  # "public"
    table: str = ""  # Physical table: "fact_orders"
    backend: Optional[str] = None  # "snowflake", "postgres", "bigquery" — for federation
    # Columns
    dimensions: list[Dimension] = Field(default_factory=list)
    time_dimensions: list[TimeDimension] = Field(default_factory=list)
    measures: list[Measure] = Field(default_factory=list)
    # Pre-defined filters
    filters: list[Filter] = Field(default_factory=list)
    # Provenance
    source: str = "auto"
    confidence: float = 0.5

    def all_column_names(self) -> list[str]:
        """Return all column names across dims, time_dims, and measures."""
        names = [d.name for d in self.dimensions]
        names += [t.name for t in self.time_dimensions]
        names += [m.name for m in self.measures]
        return names


# ═══════════════════════════════════════════════════════════════════════════════
# ROOT SEMANTIC MODEL
# ═══════════════════════════════════════════════════════════════════════════════


class FiscalCalendar(BaseModel):
    """Fiscal calendar configuration."""

    fiscal_year_start_month: int = 1  # 1=Jan, 4=Apr
    week_start_day: str = "monday"


class OraSpec(BaseModel):
    """
    The root semantic model specification.
    One OraSpec per business domain (e.g., "sales", "inventory", "hr").

    The Semantic Agent is the sole owner of this model — it creates, evolves,
    and maintains it. Other components read it through the agent.
    """

    name: str  # "retail_analytics"
    version: str = "1.0"
    description: str = ""
    # Calendar
    fiscal_calendar: Optional[FiscalCalendar] = None
    # Schema
    tables: list[LogicalTable] = Field(default_factory=list)
    relationships: list[Relationship] = Field(default_factory=list)
    metrics: list[Metric] = Field(default_factory=list)
    # Knowledge
    verified_queries: list[VerifiedQuery] = Field(default_factory=list)
    custom_instructions: list[CustomInstruction] = Field(default_factory=list)
    # Global synonyms (cross-table)
    synonyms: list[Synonym] = Field(default_factory=list)

    # ── Lookups ──────────────────────────────────────────────────────────────

    def get_table(self, name: str) -> Optional[LogicalTable]:
        """Look up a logical table by name (case-insensitive)."""
        for t in self.tables:
            if t.name.lower() == name.lower():
                return t
        return None

    def get_all_synonyms(self) -> dict[str, str]:
        """Build a flat synonym → canonical mapping across all tables."""
        mapping: dict[str, str] = {}
        for syn in self.synonyms:
            mapping[syn.term.lower()] = syn.canonical
        for table in self.tables:
            for dim in table.dimensions:
                for s in dim.synonyms:
                    mapping[s.lower()] = f"{table.name}.{dim.name}"
            for td in table.time_dimensions:
                for s in td.synonyms:
                    mapping[s.lower()] = f"{table.name}.{td.name}"
            for meas in table.measures:
                for s in meas.synonyms:
                    mapping[s.lower()] = f"{table.name}.{meas.name}"
        for metric in self.metrics:
            for s in metric.synonyms:
                mapping[s.lower()] = metric.name
        return mapping

    def get_all_tables_for_backend(self, backend: str) -> list[LogicalTable]:
        """Filter tables by backend name."""
        return [t for t in self.tables if t.backend and t.backend.lower() == backend.lower()]

    def get_relationships_for_table(self, table_name: str) -> list[Relationship]:
        """Get all relationships involving a table."""
        return [
            r
            for r in self.relationships
            if r.from_table.lower() == table_name.lower()
            or r.to_table.lower() == table_name.lower()
        ]

    # ── Prompt Serialization ─────────────────────────────────────────────────

    def to_prompt_context(self, max_tables: int = 10) -> str:
        """Serialize to a compact string for LLM prompt injection.

        Excludes sensitive columns. Includes synonyms, sample values,
        metrics, relationships, and custom instructions.
        """
        lines = [f"# Semantic Model: {self.name}", f"# {self.description}", ""]

        if self.fiscal_calendar:
            lines.append(
                f"# Fiscal year starts month {self.fiscal_calendar.fiscal_year_start_month}, "
                f"week starts {self.fiscal_calendar.week_start_day}"
            )
            lines.append("")

        for table in self.tables[:max_tables]:
            backend_tag = f" [{table.backend}]" if table.backend else ""
            lines.append(f"## Table: {table.display_name or table.name} ({table.table}){backend_tag}")
            if table.description:
                lines.append(f"  Description: {table.description}")
            for dim in table.dimensions:
                if not dim.is_sensitive:
                    sv = f" | samples: {dim.sample_values[:5]}" if dim.sample_values else ""
                    syn = f" | synonyms: {dim.synonyms}" if dim.synonyms else ""
                    lines.append(
                        f"  - DIM {dim.name} ({dim.data_type.value}): "
                        f"{dim.description}{sv}{syn}"
                    )
            for td in table.time_dimensions:
                syn = f" | synonyms: {td.synonyms}" if td.synonyms else ""
                lines.append(
                    f"  - TIME {td.name} (grain={td.time_grain.value}): "
                    f"{td.description}{syn}"
                )
            for meas in table.measures:
                if not meas.is_sensitive:
                    syn = f" | synonyms: {meas.synonyms}" if meas.synonyms else ""
                    lines.append(
                        f"  - MEASURE {meas.name} ({meas.aggregation.value}): "
                        f"{meas.description} → {meas.expr}{syn}"
                    )
            for flt in table.filters:
                lines.append(f"  - FILTER {flt.name}: {flt.description} → {flt.expr}")
            lines.append("")

        if self.metrics:
            lines.append("## Derived Metrics")
            for m in self.metrics:
                syn = f" (also: {', '.join(m.synonyms)})" if m.synonyms else ""
                lines.append(f"  - {m.display_name or m.name}: {m.description} → {m.expr}{syn}")
            lines.append("")

        if self.relationships:
            lines.append("## Relationships")
            for r in self.relationships:
                conf = f" [{int(r.confidence * 100)}%]" if r.confidence < 1.0 else ""
                lines.append(
                    f"  - {r.from_table}.{r.from_column} → "
                    f"{r.to_table}.{r.to_column} ({r.join_type.value}){conf}"
                )
            lines.append("")

        if self.custom_instructions:
            lines.append("## Instructions")
            for ci in self.custom_instructions:
                lines.append(f"  - [{ci.scope}] {ci.instruction}")
            lines.append("")

        return "\n".join(lines)

    # ── Statistics ───────────────────────────────────────────────────────────

    def stats(self) -> dict:
        """Return summary statistics about this semantic model."""
        total_dims = sum(len(t.dimensions) for t in self.tables)
        total_time = sum(len(t.time_dimensions) for t in self.tables)
        total_meas = sum(len(t.measures) for t in self.tables)
        total_synonyms = len(self.get_all_synonyms())
        avg_confidence = 0.0
        conf_items = []
        for t in self.tables:
            conf_items.append(t.confidence)
            for d in t.dimensions:
                conf_items.append(d.confidence)
            for m in t.measures:
                conf_items.append(m.confidence)
        for r in self.relationships:
            conf_items.append(r.confidence)
        if conf_items:
            avg_confidence = sum(conf_items) / len(conf_items)

        return {
            "name": self.name,
            "version": self.version,
            "tables": len(self.tables),
            "dimensions": total_dims,
            "time_dimensions": total_time,
            "measures": total_meas,
            "metrics": len(self.metrics),
            "relationships": len(self.relationships),
            "verified_queries": len(self.verified_queries),
            "synonyms": total_synonyms,
            "avg_confidence": round(avg_confidence, 2),
            "backends": list({t.backend for t in self.tables if t.backend}),
        }
