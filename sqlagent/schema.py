"""Schema pruning (CHESS LSH) + M-Schema serializer + semantic layer.

CHESS LSH: reduces 500+ columns to ~8 relevant ones per query using
embedding similarity. This is the key to making enterprise schemas work.

M-Schema: serializes the pruned schema into the format LLMs understand best
(from XiYan-SQL, Alibaba 2024 — 3-5× token savings over raw DDL).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

import structlog

from sqlagent.models import SchemaTable, SchemaColumn

logger = structlog.get_logger()


# ═══════════════════════════════════════════════════════════════════════════════
# CHESS LSH SCHEMA SELECTOR
# ═══════════════════════════════════════════════════════════════════════════════

class SchemaSelector:
    """CHESS LSH schema pruning — embedding-based column selection.

    Algorithm:
    1. Embed the query and all column descriptions into the same vector space
    2. Compute cosine similarity between query and each column
    3. Keep top-K columns
    4. Always keep PKs and FKs of selected tables (join integrity)
    5. Boost columns matching SOUL context (user vocabulary)
    """

    def __init__(self, embedder: Any, top_k: int = 15):
        self._embedder = embedder
        self._top_k = top_k

    async def prune(
        self,
        query: str,
        tables: list[SchemaTable],
        soul_context: str = "",
    ) -> list[SchemaTable]:
        """Prune schema to most relevant tables/columns for this query."""
        if not tables:
            return []

        # Build column descriptions for embedding
        col_entries: list[tuple[str, str, SchemaColumn, SchemaTable]] = []
        for table in tables:
            for col in table.columns:
                # Rich description for better embedding match
                desc_parts = [
                    f"{table.name}.{col.name}",
                    col.data_type,
                ]
                if col.description:
                    desc_parts.append(col.description)
                if col.aliases:
                    desc_parts.extend(col.aliases)
                if col.semantic_type:
                    desc_parts.append(col.semantic_type)
                desc = " ".join(desc_parts)
                col_entries.append((desc, f"{table.name}.{col.name}", col, table))

        if not col_entries:
            return tables

        # Embed query + all column descriptions
        texts = [query + " " + soul_context] + [e[0] for e in col_entries]
        embeddings = await self._embedder.embed(texts)

        query_embedding = embeddings[0]
        col_embeddings = embeddings[1:]

        # Cosine similarity
        import numpy as np
        query_vec = np.array(query_embedding)
        scores = []
        for i, col_emb in enumerate(col_embeddings):
            col_vec = np.array(col_emb)
            sim = np.dot(query_vec, col_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(col_vec) + 1e-8)
            scores.append((float(sim), col_entries[i]))

        # Sort by similarity, take top-K
        scores.sort(key=lambda x: x[0], reverse=True)
        selected = scores[:self._top_k]

        # Collect selected tables with only their relevant columns
        table_columns: dict[str, list[SchemaColumn]] = {}
        table_objects: dict[str, SchemaTable] = {}
        for _, (desc, full_name, col, table) in selected:
            if table.name not in table_columns:
                table_columns[table.name] = []
                table_objects[table.name] = table
            table_columns[table.name].append(col)

        # Always include PKs and FKs of selected tables
        for tname, table in table_objects.items():
            existing_names = {c.name for c in table_columns[tname]}
            for col in table.columns:
                if (col.is_primary_key or col.is_foreign_key) and col.name not in existing_names:
                    table_columns[tname].append(col)

        # Build pruned tables
        result = []
        for tname, cols in table_columns.items():
            orig = table_objects[tname]
            result.append(SchemaTable(
                name=orig.name,
                schema_name=orig.schema_name,
                columns=cols,
                row_count_estimate=orig.row_count_estimate,
                tags=orig.tags,
                description=orig.description,
            ))

        return result


# ═══════════════════════════════════════════════════════════════════════════════
# M-SCHEMA SERIALIZER
# ═══════════════════════════════════════════════════════════════════════════════

class MSchemaSerializer:
    """Serialize schema to M-Schema format for LLM prompts.

    M-Schema (from XiYan-SQL) is significantly more token-efficient
    than raw DDL while preserving all information the LLM needs.

    Format:
    ```
    【DB: northwind】
    【Table: customers】
    · customer_id : TEXT [PK]
    · company_name : TEXT (Company legal name)
    · country : TEXT (e.g., "Germany", "Sweden")
    【Table: orders】
    · order_id : INTEGER [PK]
    · customer_id : TEXT [FK → customers.customer_id]
    · total_amount : REAL (Order total in USD, aliases: revenue, sales)
    ```
    """

    @staticmethod
    def serialize(tables: list[SchemaTable], dialect: str = "sql") -> str:
        lines = [f"【DB: {dialect}】"]

        for table in tables:
            row_info = f" ({table.row_count_estimate:,} rows)" if table.row_count_estimate else ""
            lines.append(f"【Table: {table.name}】{row_info}")

            for col in table.columns:
                parts = [f"· {col.name} : {col.data_type}"]

                badges = []
                if col.is_primary_key:
                    badges.append("PK")
                if col.is_foreign_key and col.foreign_key_ref:
                    ref = col.foreign_key_ref
                    badges.append(f"FK → {ref['table']}.{ref['column']}")
                elif col.is_foreign_key:
                    badges.append("FK")
                if badges:
                    parts.append(f"[{', '.join(badges)}]")

                notes = []
                if col.description:
                    notes.append(col.description)
                if col.aliases:
                    notes.append(f"aliases: {', '.join(col.aliases)}")
                if col.semantic_type:
                    notes.append(f"type: {col.semantic_type}")
                if col.examples:
                    # Show up to 8 example values so LLMs understand actual domain
                    ex_str = ", ".join(f'"{v}"' for v in col.examples[:8])
                    notes.append(f"values: {ex_str}")
                if notes:
                    parts.append(f"({'; '.join(notes)})")

                lines.append(" ".join(parts))

        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# SEMANTIC LAYER
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class GlossaryEntry:
    """Maps a business term to a schema column."""
    term: str                             # "revenue", "headcount"
    column: str                           # "orders.total_amount"
    definition: str = ""


@dataclass
class SemanticLayerState:
    """Accumulated semantic knowledge about the schema.

    Built from: schema analysis, dbt docs, catalog integrations,
    user corrections, SOUL observations.
    """
    glossary: list[GlossaryEntry] = field(default_factory=list)
    pii_columns: list[str] = field(default_factory=list)
    column_descriptions: dict[str, str] = field(default_factory=dict)  # "table.col" → description
    sensitivity_labels: dict[str, str] = field(default_factory=dict)

    def add_glossary(self, term: str, column: str, definition: str = "") -> None:
        # Avoid duplicates
        for g in self.glossary:
            if g.term == term and g.column == column:
                return
        self.glossary.append(GlossaryEntry(term=term, column=column, definition=definition))

    def resolve_term(self, term: str) -> str | None:
        """Look up a business term → schema column."""
        term_lower = term.lower()
        for g in self.glossary:
            if g.term.lower() == term_lower:
                return g.column
        return None

    def get_context_block(self) -> str:
        """Generate a context block for LLM prompts."""
        lines = []
        if self.glossary:
            lines.append("Business glossary:")
            for g in self.glossary:
                lines.append(f"  {g.term} → {g.column}" + (f" ({g.definition})" if g.definition else ""))
        if self.pii_columns:
            lines.append(f"PII columns (do not select): {', '.join(self.pii_columns)}")
        return "\n".join(lines)
