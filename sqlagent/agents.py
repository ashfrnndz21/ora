"""ALL agents in one file — Schema, Setup, Decompose, Synthesis, Response, Validator, Learning.

Every agent is REAL — calls actual LLMs, actual databases.
No mocks, no templates, no hardcoded responses.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, AsyncIterator

import structlog

from sqlagent.models import (
    SchemaSnapshot, SchemaAnalysis, InferredRelationship, EntityGrouping,
    ColumnSemantic, DataQualityIssue, KnowledgeGraph, KGNode, KGEdge, KGLayer,
    EdgeType, SemanticEntry, SampleData,
)

logger = structlog.get_logger()


def _salvage_truncated_json(raw: str) -> dict | None:
    """Attempt to salvage a truncated JSON object by closing open structures.

    When the LLM hits max_tokens mid-JSON, truncate each array to the last
    complete item and close all open brackets/braces.
    """
    # Find the last position where a complete object ends in the top-level arrays
    # Strategy: try progressively shorter truncations from the end
    for trunc in range(len(raw), max(len(raw) - 2000, 0), -1):
        candidate = raw[:trunc]
        # Count open braces/brackets to determine what needs closing
        depth_brace = candidate.count('{') - candidate.count('}')
        depth_bracket = candidate.count('[') - candidate.count(']')
        if depth_brace < 0 or depth_bracket < 0:
            continue
        # Close any open structures
        closing = ']' * depth_bracket + '}' * depth_brace
        try:
            return json.loads(candidate + closing)
        except json.JSONDecodeError:
            continue
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# SCHEMA AGENT — builds knowledge graph from schema introspection + LLM analysis
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SchemaAnalysisProgress:
    phase: str = ""            # structural, sampling, llm_analysis, cross_source, assembly
    progress_pct: float = 0.0
    message: str = ""
    node_count: int = 0
    edge_count: int = 0


class SchemaAgent:
    """Builds a KnowledgeGraph from database schemas.

    5-stage pipeline:
    1. Structural: extract FKs/PKs, infer FKs from naming patterns
    2. Sampling: 5 rows per table + column statistics
    3. LLM analysis: classify columns, infer relationships, entity groups
    4. Cross-source: detect links between different databases
    5. Assembly: build the final KnowledgeGraph JSON
    """

    def __init__(self, llm: Any, embedder: Any = None):
        self._llm = llm
        self._embedder = embedder

    async def analyze(
        self,
        workspace_id: str,
        snapshots: dict[str, SchemaSnapshot],
        connectors: dict[str, Any] | None = None,
        on_progress: Any = None,
    ) -> KnowledgeGraph:
        """Run the full analysis pipeline."""
        nodes: list[KGNode] = []
        edges: list[KGEdge] = []
        layers: list[KGLayer] = []
        glossary: list[SemanticEntry] = []
        pii_columns: list[str] = []
        sources_meta: list[dict] = []

        # Stage 1: Structural pass
        if on_progress:
            await on_progress(SchemaAnalysisProgress(phase="structural", progress_pct=0.1, message="Extracting schema structure..."))

        for source_id, snap in snapshots.items():
            sources_meta.append({
                "source_id": source_id, "dialect": snap.dialect,
                "table_count": snap.table_count, "column_count": snap.column_count,
            })
            # Add source node
            nodes.append(KGNode(
                id=f"src:{source_id}", type="source", source_id=source_id,
                name=source_id, properties={"dialect": snap.dialect, "table_count": snap.table_count},
            ))
            for table in snap.tables:
                # Add table node
                nodes.append(KGNode(
                    id=f"tbl:{source_id}.{table.name}", type="table", source_id=source_id,
                    name=table.name, properties={
                        "row_count": table.row_count_estimate,
                        "column_count": len(table.columns),
                        "columns": [
                            {
                                "name": c.name, "data_type": c.data_type,
                                "is_pk": c.is_primary_key, "is_fk": c.is_foreign_key,
                                "description": c.description or "",
                                "examples": c.examples or [],
                            }
                            for c in table.columns
                        ],
                    },
                ))
                for col in table.columns:
                    nodes.append(KGNode(
                        id=f"col:{source_id}.{table.name}.{col.name}", type="column",
                        source_id=source_id, name=col.name,
                        properties={
                            "data_type": col.data_type, "is_pk": col.is_primary_key,
                            "is_fk": col.is_foreign_key, "nullable": col.nullable,
                            "semantic_type": col.semantic_type, "description": col.description,
                        },
                    ))

            # Declared FKs → edges
            for fk in snap.foreign_keys:
                edges.append(KGEdge(
                    id=f"fk:{source_id}.{fk.from_table}.{fk.from_column}->{fk.to_table}.{fk.to_column}",
                    source=f"tbl:{source_id}.{fk.from_table}",
                    target=f"tbl:{source_id}.{fk.to_table}",
                    type=EdgeType.DECLARED_FK,
                    join_columns={"from": fk.from_column, "to": fk.to_column},
                    confidence=1.0,
                    description=f"{fk.from_table}.{fk.from_column} → {fk.to_table}.{fk.to_column}",
                ))

        # Stage 2: Sampling
        if on_progress:
            await on_progress(SchemaAnalysisProgress(phase="sampling", progress_pct=0.3, message="Sampling data..."))

        samples: dict[str, list[SampleData]] = {}
        if connectors:
            for source_id, conn in connectors.items():
                snap = snapshots.get(source_id)
                if not snap:
                    continue
                source_samples = []
                for table in snap.tables[:20]:  # Limit to 20 tables
                    try:
                        sample = await conn.sample(table.name, n=5)
                        source_samples.append(sample)
                    except Exception as exc:
                        logger.debug("agents.operation_failed", error=str(exc))
                samples[source_id] = source_samples

        # Stage 2b: Populate SchemaColumn.examples from sampled distinct values
        # This gives generators ground truth about what values columns actually contain
        # (e.g., Country column may contain "RestOfASEAN" — a regional bucket, not a country)
        for source_id, source_samples in samples.items():
            snap = snapshots.get(source_id)
            if not snap:
                continue
            # Build a lookup: table_name → {col_name → ColumnStats}
            stats_by_table: dict[str, dict] = {}
            for sd in source_samples:
                if sd.column_stats:
                    stats_by_table[sd.table] = sd.column_stats
            # Populate SchemaColumn.examples for low-cardinality columns
            for table in snap.tables:
                col_stats = stats_by_table.get(table.name, {})
                for col in table.columns:
                    stats = col_stats.get(col.name)
                    if stats and stats.sample_values and not col.examples:
                        # Convert to strings, truncate long values, limit to 15
                        col.examples = [
                            str(v)[:40] for v in stats.sample_values[:15]
                            if v is not None and str(v).strip()
                        ]

        # Stage 3: LLM analysis
        if on_progress:
            await on_progress(SchemaAnalysisProgress(phase="llm_analysis", progress_pct=0.5, message="Running LLM analysis..."))

        for source_id, snap in snapshots.items():
            try:
                analysis = await self._llm_analyze(snap, samples.get(source_id, []))

                # Inferred relationships → edges
                for rel in analysis.inferred_relationships:
                    edges.append(KGEdge(
                        id=f"inferred:{source_id}.{rel.from_table}.{rel.from_column}->{rel.to_table}.{rel.to_column}",
                        source=f"tbl:{source_id}.{rel.from_table}",
                        target=f"tbl:{source_id}.{rel.to_table}",
                        type=EdgeType.INFERRED,
                        join_columns={"from": rel.from_column, "to": rel.to_column},
                        confidence=rel.confidence,
                        evidence=rel.evidence,
                    ))

                # Entity groups → layers
                for eg in analysis.entity_groups:
                    layers.append(KGLayer(
                        id=f"layer:{source_id}.{eg.name}",
                        name=eg.name, description=eg.description,
                        tables=[f"tbl:{source_id}.{t}" for t in eg.tables],
                        color=eg.color,
                    ))

                # Column semantics → update node properties + glossary
                for cs in analysis.column_semantics:
                    node_id = f"col:{source_id}.{cs.table}.{cs.column}"
                    for node in nodes:
                        if node.id == node_id:
                            node.properties["semantic_type"] = cs.semantic_type
                            node.properties["description"] = cs.description
                            if cs.pii:
                                node.properties["pii"] = True
                                pii_columns.append(f"{cs.table}.{cs.column}")
                    if cs.business_term:
                        glossary.append(SemanticEntry(
                            term=cs.business_term,
                            maps_to=f"{cs.table}.{cs.column}",
                            definition=cs.description,
                        ))

            except Exception as e:
                logger.warn("schema_agent.llm_failed", source=source_id, error=str(e))

        # Stage 3b: Deterministic intra-source relationship detection
        # ONE edge per table-pair, listing ALL shared columns — no duplicates
        for source_id, snap in snapshots.items():
            tables = snap.tables
            # Build column→tables index
            col_to_tables: dict[str, list[str]] = {}
            for tbl in tables:
                for col in tbl.columns:
                    col_to_tables.setdefault(col.name.lower(), []).append(tbl.name)

            # Collect shared columns per table-pair
            SKIP_COLS = {"id", "name", "description", "created_at", "updated_at",
                         "value", "type", "status", "date", "unnamed: 0", "unnamed: 1",
                         "unnamed: 2", "index", "analytics_db", "row_number", "rank",
                         "flag", "notes", "comment", "comments", "label", "tag", "tags"}
            pair_cols: dict[tuple[str,str], list[str]] = {}
            for col_name, tbl_names in col_to_tables.items():
                # Skip single-char column names (noise), purely numeric names, and skip list
                if len(tbl_names) < 2 or col_name in SKIP_COLS or len(col_name) <= 1:
                    continue
                unique_tables = list(dict.fromkeys(tbl_names))  # deduplicate
                for i in range(len(unique_tables)):
                    for j in range(i + 1, len(unique_tables)):
                        ta, tb = unique_tables[i], unique_tables[j]
                        pair = (min(ta, tb), max(ta, tb))
                        pair_cols.setdefault(pair, [])
                        if col_name not in pair_cols[pair]:
                            pair_cols[pair].append(col_name)

            # One edge per pair (skip if LLM already added this pair)
            existing_pairs = {
                (e.source.replace(f"tbl:{source_id}.", ""),
                 e.target.replace(f"tbl:{source_id}.", ""))
                for e in edges
            }
            for (ta, tb), shared_cols in pair_cols.items():
                if (ta, tb) in existing_pairs or (tb, ta) in existing_pairs:
                    continue
                join_key = shared_cols[0]
                desc = f"{ta.split('_')[-1]} ↔ {tb.split('_')[-1]} via {', '.join(shared_cols[:3])}"
                edges.append(KGEdge(
                    id=f"inferred:{source_id}.{ta}↔{tb}",
                    source=f"tbl:{source_id}.{ta}",
                    target=f"tbl:{source_id}.{tb}",
                    type=EdgeType.INFERRED,
                    join_columns={"from": join_key, "to": join_key},
                    confidence=0.85,
                    evidence=f"Shared columns: {', '.join(shared_cols)}",
                    description=desc,
                ))

        # Stage 4: Cross-source link detection
        if len(snapshots) > 1:
            if on_progress:
                await on_progress(SchemaAnalysisProgress(phase="cross_source", progress_pct=0.8, message="Detecting cross-source links..."))

            source_list = list(snapshots.items())
            for i in range(len(source_list)):
                for j in range(i + 1, len(source_list)):
                    sid_a, snap_a = source_list[i]
                    sid_b, snap_b = source_list[j]
                    links = self._detect_cross_source_links(sid_a, snap_a, sid_b, snap_b)
                    edges.extend(links)

        # Stage 5: Assembly
        if on_progress:
            await on_progress(SchemaAnalysisProgress(
                phase="assembly", progress_pct=1.0, message="Knowledge graph built",
                node_count=len(nodes), edge_count=len(edges),
            ))

        return KnowledgeGraph(
            graph_id=f"kg_{workspace_id}_{uuid.uuid4().hex[:8]}",
            workspace_id=workspace_id,
            sources=sources_meta,
            nodes=nodes,
            edges=edges,
            layers=layers,
            glossary=glossary,
            pii_columns=pii_columns,
        )

    async def _llm_analyze(self, snap: SchemaSnapshot, samples: list[SampleData]) -> SchemaAnalysis:
        """Run LLM analysis on a single source."""
        # Build schema description
        schema_text = ""
        for table in snap.tables:
            cols = ", ".join(f"{c.name} {c.data_type}" for c in table.columns)
            schema_text += f"Table {table.name} ({table.row_count_estimate} rows): {cols}\n"

        # Add sample data
        sample_text = ""
        for sd in samples[:5]:
            if sd.sample_rows:
                sample_text += f"\nSample from {sd.table}:\n{json.dumps(sd.sample_rows[:3], default=str)}\n"

        prompt = (
            f"You are a data architect. Analyze this {snap.dialect} schema and infer the full data model.\n\n"
            f"Tables:\n{schema_text}\n"
            f"{sample_text}\n"
            f"Tasks:\n"
            f"1. inferred_relationships: Find ALL column-level join paths between tables. Look for:\n"
            f"   - Exact column name matches (e.g. Customer in tableA and Customer in tableB → joinable)\n"
            f"   - ID columns matching table names (e.g. customer_id → customer table)\n"
            f"   - Shared dimension columns (Country, Industry, Region etc)\n"
            f"   Format: [{{from_table, from_column, to_table, to_column, confidence (0-1), evidence: 'reason'}}]\n"
            f"2. entity_groups: Cluster tables by business domain.\n"
            f"   Format: [{{name, description, tables: [], color: '#hex'}}]\n"
            f"3. column_semantics: For key columns, explain business meaning.\n"
            f"   Format: [{{table, column, semantic_type, description, business_term, pii: bool}}]\n"
            f"4. data_quality: Flag obvious issues.\n"
            f"   Format: [{{table, column, issue, severity: 'low|medium|high'}}]\n\n"
            f"Return ONLY valid JSON. Be thorough — find every relationship."
        )

        resp = await self._llm.complete(
            [{"role": "user", "content": prompt}],
            json_mode=True,
            max_tokens=8192,
        )

        try:
            # Strip markdown fences — Claude wraps JSON in ```json...``` even in json_mode
            raw = resp.content.strip()
            if raw.startswith("```"):
                parts = raw.split("```")
                # Find the JSON part among the fence segments
                raw = next(
                    (p[4:].strip() if p.startswith("json") else p.strip()
                     for p in parts if p.startswith("json") or p.strip().startswith("{")),
                    raw,
                )
            # Try direct parse first
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                # Response may be truncated — salvage by truncating to last complete top-level array item
                data = _salvage_truncated_json(raw)
                if data is None:
                    raise
        except json.JSONDecodeError as e:
            logger.warn("schema_agent.json_parse_failed",
                        error=str(e),
                        content_start=resp.content[:80],
                        content_end=resp.content[-80:])
            return SchemaAnalysis(source_id=snap.source_id)
        except Exception as e:
            logger.warn("schema_agent.json_parse_failed",
                        error=str(e),
                        content_start=resp.content[:80])
            return SchemaAnalysis(source_id=snap.source_id)

        return SchemaAnalysis(
            analysis_id=str(uuid.uuid4())[:12],
            source_id=snap.source_id,
            model_used=resp.model,
            tokens_used=resp.tokens_input + resp.tokens_output,
            inferred_relationships=[
                InferredRelationship(**r) for r in data.get("inferred_relationships", [])
                if "from_table" in r and "to_table" in r
            ],
            entity_groups=[
                EntityGrouping(
                    name=g.get("name") or g.get("group_name", ""),
                    description=g.get("description", ""),
                    tables=g.get("tables", []),
                    color=g.get("color", "#4f7df9"),
                )
                for g in data.get("entity_groups", [])
                if g.get("name") or g.get("group_name")
            ],
            column_semantics=[
                ColumnSemantic(
                    table=c["table"],
                    column=c["column"],
                    semantic_type=c.get("semantic_type", ""),
                    description=c.get("description", ""),
                    business_term=c.get("business_term", ""),
                    aliases=c.get("aliases", []),
                    pii=bool(c.get("pii", False)),
                )
                for c in data.get("column_semantics", [])
                if "table" in c and "column" in c
            ],
            data_quality=[
                DataQualityIssue(
                    table=d["table"],
                    column=d.get("column", ""),
                    issue=d.get("issue", ""),
                    severity=d.get("severity", "low"),
                )
                for d in data.get("data_quality", [])
                if "table" in d
            ],
        )

    def _detect_cross_source_links(
        self, sid_a: str, snap_a: SchemaSnapshot, sid_b: str, snap_b: SchemaSnapshot,
    ) -> list[KGEdge]:
        """Detect join keys between two different sources (exact name match)."""
        links = []
        cols_a = {}
        for t in snap_a.tables:
            for c in t.columns:
                cols_a[c.name.lower()] = (t.name, c)

        for t in snap_b.tables:
            for c in t.columns:
                key = c.name.lower()
                if key in cols_a:
                    ta_name, ca = cols_a[key]
                    # Check type compatibility
                    if self._types_compatible(ca.data_type, c.data_type):
                        links.append(KGEdge(
                            id=f"cross:{sid_a}.{ta_name}.{ca.name}->{sid_b}.{t.name}.{c.name}",
                            source=f"tbl:{sid_a}.{ta_name}",
                            target=f"tbl:{sid_b}.{t.name}",
                            type=EdgeType.CROSS_SOURCE,
                            join_columns={"from": ca.name, "to": c.name},
                            confidence=1.0 if ca.name == c.name else 0.9,
                            evidence=f"Exact column name match: {ca.name}",
                        ))
        return links

    @staticmethod
    def _types_compatible(type_a: str, type_b: str) -> bool:
        """Check if two column types are compatible for joining."""
        text_types = {"text", "varchar", "char", "string", "nvarchar"}
        int_types = {"integer", "int", "bigint", "smallint", "int4", "int8"}
        a = type_a.lower().split("(")[0].strip()
        b = type_b.lower().split("(")[0].strip()
        if a == b:
            return True
        if a in text_types and b in text_types:
            return True
        if a in int_types and b in int_types:
            return True
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# SETUP AGENT — conversational workspace creation
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SetupEvent:
    event_type: str = ""       # message, action, source_added, workspace_ready, error
    data: dict = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


_SETUP_SYSTEM = """You know data, SQL, and how analysts think. Talk like a sharp colleague — short, direct, no filler.

No emojis. No markdown. No bullet points. Plain sentences only.
Never say "I'm here to help", "Great!", "Sure!", or any filler opening.

When the user gives you a database URL (postgresql://, mysql://, snowflake://, bigquery://, sqlite://, redshift://, duckdb://), output this on its own line:
{"tool": "connect_source", "type": "postgresql", "connection_string": "...url..."}

When you have a source connected and a name, output these on their own lines:
{"tool": "set_workspace_name", "name": "Short Name"}
{"tool": "finalize_setup"}

If files are already uploaded ([System: uploaded ...]), pick a name and finalize immediately — don't ask anything.
If the user confirms or sounds positive, finalize immediately.
Only call connect_source for real DB URLs — never for plain text."""


class SetupAgent:
    """Conversational agent for workspace creation.

    Instead of a form wizard, the user describes their data in natural language.
    The agent asks questions, collects connection details, and configures the workspace.
    """

    def __init__(self, llm: Any, connector_registry: Any = None):
        self._llm = llm
        self._registry = connector_registry
        self._conversations: dict[str, list[dict]] = {}  # workspace_id → messages

    async def chat(
        self,
        workspace_id: str,
        user_message: str,
        workspace_state: dict | None = None,
    ) -> AsyncIterator[SetupEvent]:
        """Process one user turn. Yields SetupEvent objects."""
        if workspace_id not in self._conversations:
            self._conversations[workspace_id] = [
                {"role": "system", "content": _SETUP_SYSTEM},
            ]

        self._conversations[workspace_id].append(
            {"role": "user", "content": user_message}
        )

        # Add workspace state context
        if workspace_state:
            state_msg = f"Current workspace state: {json.dumps(workspace_state)}"
            messages = self._conversations[workspace_id] + [
                {"role": "system", "content": state_msg}
            ]
        else:
            messages = self._conversations[workspace_id]

        resp = await self._llm.complete(messages)
        content = resp.content

        self._conversations[workspace_id].append(
            {"role": "assistant", "content": content}
        )

        # Known URL schemes for connect_source validation
        _VALID_URL_SCHEMES = (
            "postgresql://", "postgres://", "mysql://", "sqlite://",
            "snowflake://", "bigquery://", "redshift+psycopg2://",
            "duckdb://", "file://", "/", "./",
        )

        # Extract all JSON tool calls from the response (handles multiple per response)
        import re as _re
        tool_calls = []
        text_parts = []
        remaining = content
        for m in _re.finditer(r'\{[^{}]*"tool"[^{}]*\}', content):
            try:
                tool_calls.append(json.loads(m.group()))
            except json.JSONDecodeError:
                pass

        # Collect non-tool text
        text_only = _re.sub(r'\{[^{}]*"tool"[^{}]*\}', '', content).strip()
        if text_only:
            yield SetupEvent(event_type="message", data={"text": text_only})

        for tool_json in tool_calls:
            tool_name = tool_json.get("tool", "")

            if tool_name == "connect_source":
                conn_str = tool_json.get("connection_string", "")
                # Only proceed if the connection string looks like an actual URL or path
                if not any(conn_str.startswith(s) for s in _VALID_URL_SCHEMES):
                    # LLM hallucinated a non-URL — skip silently
                    continue
                yield SetupEvent(
                    event_type="action",
                    data={"action": "connecting", "type": tool_json.get("type"), "url": conn_str},
                )
                try:
                    from sqlagent.connectors import ConnectorRegistry
                    conn = ConnectorRegistry.from_url(
                        source_id=f"src_{tool_json.get('type', 'unknown')}",
                        url=conn_str,
                    )
                    snap = await conn.introspect()
                    yield SetupEvent(
                        event_type="source_added",
                        data={
                            "source_id": conn.source_id,
                            "dialect": conn.dialect,
                            "table_count": snap.table_count,
                            "column_count": snap.column_count,
                        },
                    )
                except Exception as e:
                    yield SetupEvent(event_type="error", data={"error": str(e)})

            elif tool_name == "set_workspace_name":
                yield SetupEvent(
                    event_type="action",
                    data={"action": "naming", "name": tool_json.get("name", "")},
                )

            elif tool_name == "finalize_setup":
                yield SetupEvent(event_type="workspace_ready", data={})


# ═══════════════════════════════════════════════════════════════════════════════
# DECOMPOSE AGENT — splits cross-source queries into sub-problems
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SubProblem:
    id: str
    nl_description: str
    target_source: str
    depends_on: list[str] = field(default_factory=list)
    expected_columns: list[str] = field(default_factory=list)
    priority: int = 0


@dataclass
class DecompositionPlan:
    sub_problems: list[SubProblem] = field(default_factory=list)
    synthesis_strategy: str = "join"     # join, union, merge, sequential, aggregate
    join_key: str = ""
    complexity_score: float = 0.5


class DecomposeAgent:
    """Breaks complex NL queries into atomic sub-problems, one per data source."""

    def __init__(self, llm: Any, default_source: str = ""):
        self._llm = llm
        self._default_source = default_source

    async def plan(
        self,
        query: str,
        sources: list[dict],
    ) -> DecompositionPlan:
        """Decompose a query into sub-problems."""
        source_desc = "\n".join(
            f"- {s.get('source_id')}: {s.get('dialect')} with tables {s.get('tables', [])}"
            for s in sources
        )

        prompt = (
            f"Decompose this question into sub-queries, one per data source.\n\n"
            f"Sources:\n{source_desc}\n\n"
            f"Question: {query}\n\n"
            f"Return JSON: {{\"sub_problems\": [{{\"id\": \"sq_a\", \"nl_description\": \"...\", "
            f"\"target_source\": \"...\", \"expected_columns\": [...]}}], "
            f"\"synthesis_strategy\": \"join|union\", \"join_key\": \"...\"}}"
        )

        resp = await self._llm.complete(
            [{"role": "user", "content": prompt}],
            json_mode=True,
        )

        try:
            data = json.loads(resp.content)
        except json.JSONDecodeError:
            return DecompositionPlan()

        return DecompositionPlan(
            sub_problems=[
                SubProblem(
                    id=sp.get("id", f"sq_{i}"),
                    nl_description=sp.get("nl_description", ""),
                    target_source=sp.get("target_source", ""),
                    expected_columns=sp.get("expected_columns", []),
                )
                for i, sp in enumerate(data.get("sub_problems", []))
            ],
            synthesis_strategy=data.get("synthesis_strategy", "join"),
            join_key=data.get("join_key", ""),
        )


# ═══════════════════════════════════════════════════════════════════════════════
# SYNTHESIS AGENT — merges sub-query results via DuckDB
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SynthesisResult:
    rows: list[dict] = field(default_factory=list)
    columns: list[str] = field(default_factory=list)
    row_count: int = 0
    synthesis_sql: str = ""
    succeeded: bool = False
    error: str = ""


class SynthesisAgent:
    """Merges N result DataFrames from parallel sub-queries via DuckDB in-memory JOIN."""

    def __init__(self, llm: Any = None):
        self._llm = llm

    async def synthesize(
        self,
        sub_results: list[dict],
        plan: DecompositionPlan,
    ) -> SynthesisResult:
        """Join sub-query results in DuckDB."""
        import duckdb
        import pandas as pd

        con = duckdb.connect(":memory:")

        # Load sub-results as tables
        for sr in sub_results:
            if sr.get("succeeded") and sr.get("rows"):
                df = pd.DataFrame(sr["rows"])
                con.register(sr["sub_query_id"], df)

        successful = [sr for sr in sub_results if sr.get("succeeded") and sr.get("rows")]
        if len(successful) < 2:
            if successful:
                return SynthesisResult(
                    rows=successful[0].get("rows", []),
                    columns=successful[0].get("columns", []),
                    row_count=successful[0].get("row_count", 0),
                    succeeded=True,
                )
            return SynthesisResult(error="No successful sub-queries to synthesize")

        # Build JOIN SQL
        join_key = plan.join_key
        if plan.synthesis_strategy == "join" and join_key:
            left = successful[0]["sub_query_id"]
            right = successful[1]["sub_query_id"]
            sql = f"SELECT * FROM {left} a JOIN {right} b ON a.{join_key} = b.{join_key} LIMIT 10000"
        elif plan.synthesis_strategy == "union":
            tables = [sr["sub_query_id"] for sr in successful]
            sql = " UNION ALL ".join(f"SELECT * FROM {t}" for t in tables)
        else:
            left = successful[0]["sub_query_id"]
            right = successful[1]["sub_query_id"]
            sql = f"SELECT * FROM {left} a, {right} b LIMIT 10000"

        try:
            result = con.execute(sql).fetchdf()
            rows = result.to_dict("records")
            columns = list(result.columns)
            con.close()
            return SynthesisResult(
                rows=rows, columns=columns, row_count=len(rows),
                synthesis_sql=sql, succeeded=True,
            )
        except Exception as e:
            con.close()
            return SynthesisResult(error=str(e), synthesis_sql=sql)


# ═══════════════════════════════════════════════════════════════════════════════
# RESPONSE GENERATOR — NL summary + follow-ups + chart config
# ═══════════════════════════════════════════════════════════════════════════════

class ResponseGenerator:
    """Generates human-readable response from query results."""

    def __init__(self, llm: Any):
        self._llm = llm

    async def generate(
        self,
        nl_query: str,
        sql: str,
        rows: list[dict],
        columns: list[str],
    ) -> dict:
        """Generate NL summary, follow-ups, and chart suggestion."""
        sample = rows[:5]
        prompt = (
            f"Question: {nl_query}\nSQL: {sql}\n"
            f"Results ({len(rows)} rows, first 5): {json.dumps(sample, default=str)}\n\n"
            f"Return JSON: {{\"summary\": \"2-3 sentence answer with **bold** key numbers\", "
            f"\"follow_ups\": [\"question1\", \"question2\", \"question3\"], "
            f"\"chart_type\": \"bar|line|pie|table|none\"}}"
        )

        resp = await self._llm.complete(
            [{"role": "user", "content": prompt}],
            json_mode=True,
        )

        try:
            return json.loads(resp.content)
        except json.JSONDecodeError:
            return {"summary": resp.content, "follow_ups": [], "chart_type": "table"}


# ═══════════════════════════════════════════════════════════════════════════════
# OUTPUT VALIDATOR — sanity checks on query results
# ═══════════════════════════════════════════════════════════════════════════════

class OutputValidator:
    """Validates query results before showing to user."""

    @staticmethod
    def validate(rows: list[dict], columns: list[str], nl_query: str) -> dict:
        """Run sanity checks. Returns {passed: bool, checks: [...]}."""
        checks = []

        # Non-empty
        checks.append({
            "check": "non_empty",
            "passed": len(rows) > 0,
            "detail": f"{len(rows)} rows returned",
        })

        # Not all nulls
        if rows:
            all_null = all(
                all(v is None for v in row.values())
                for row in rows[:10]
            )
            checks.append({
                "check": "not_all_nulls",
                "passed": not all_null,
                "detail": "All values are NULL" if all_null else "Contains non-null values",
            })

        # Reasonable row count
        checks.append({
            "check": "row_count_reasonable",
            "passed": len(rows) <= 100000,
            "detail": f"{len(rows)} rows",
        })

        passed = all(c["passed"] for c in checks)
        return {"passed": passed, "checks": checks}


# ═══════════════════════════════════════════════════════════════════════════════
# LEARNING LOOP — trace-aware feedback → training
# ═══════════════════════════════════════════════════════════════════════════════

class LearningLoop:
    """
    Trace-aware learning: converts user feedback into prioritized training data.

    Three tiers of feedback quality:
      auto_learned   (priority 1) — every successful query
      user_verified  (priority 2) — thumbs up
      user_corrected (priority 3) — thumbs down + correction; highest retrieval weight

    Corrections carry trace context: which pipeline stage failed, what schema
    elements were missing, and the corrected SQL — so future similar queries
    retrieve the right example AND know which columns to include.
    """

    def __init__(self, example_store: Any):
        self._store = example_store

    async def on_thumbs_up(self, nl_query: str, sql: str, source_id: str = "") -> str:
        """User confirmed the result — save as verified training pair."""
        if self._store:
            return await self._store.add(
                nl_query=nl_query,
                sql=sql,
                source_id=source_id,
                generator="user_verified",
                verified=True,
            )
        return ""

    async def on_correction(
        self,
        nl_query: str,
        corrected_sql: str,
        original_sql: str = "",
        trace_events: list = None,
        failure_type: str = "",
        failed_node: str = "",
        correction_note: str = "",
        source_id: str = "",
    ) -> dict:
        """
        Trace-aware correction — the core of the Learn Agent.

        Analyzes which pipeline stage produced the wrong result, annotates the
        NL query with the lesson learned, and stores with highest retrieval
        priority so future similar queries get the correction first.

        Returns analysis dict: failed_stage, schema_hint, message
        """
        analysis = _analyze_trace_for_failure(
            trace_events or [], failure_type, failed_node, original_sql, corrected_sql
        )

        # Annotate the NL query with the learned context so vector retrieval
        # surfaces this pair when similar questions are asked in future
        context_parts = []
        if correction_note:
            context_parts.append(correction_note)
        if analysis["failed_stage"]:
            stage_lessons = {
                "schema":     "schema pruning selected wrong tables/columns",
                "retrieval":  "wrong example retrieved from memory",
                "planning":   "incorrect query strategy chosen",
                "generation": "SQL logic was incorrect",
                "filtering":  "WHERE/HAVING conditions were wrong",
            }
            lesson = stage_lessons.get(analysis["failed_stage"], analysis["failed_stage"])
            if not correction_note or lesson not in correction_note:
                context_parts.append(f"fix: {lesson}")
        if analysis["schema_hint"]:
            context_parts.append(f"use: {analysis['schema_hint']}")

        nl_annotated = nl_query
        if context_parts:
            nl_annotated = f"{nl_query}\n[learn: {'; '.join(context_parts)}]"

        pair_id = ""
        if self._store:
            pair_id = await self._store.add(
                nl_query=nl_annotated,
                sql=corrected_sql,
                source_id=source_id,
                generator="user_corrected",
                verified=True,
            )

        return {
            "pair_id": pair_id,
            "failed_stage": analysis["failed_stage"],
            "schema_hint":  analysis["schema_hint"],
            "message":      _correction_message(analysis),
        }


# ── Trace analysis helpers ────────────────────────────────────────────────────

def _analyze_trace_for_failure(
    trace_events: list,
    failure_type: str,
    failed_node: str,
    original_sql: str,
    corrected_sql: str,
) -> dict:
    """
    Determine which pipeline stage caused the bad result and extract schema hints.

    Priority: explicit failed_node > user failure_type > trace event analysis > fallback
    """
    FAILURE_TYPE_TO_STAGE = {
        "wrong_tables":      "schema",
        "wrong_columns":     "schema",
        "bad_example":       "retrieval",
        "wrong_plan":        "planning",
        "wrong_logic":       "generation",
        "wrong_filter":      "filtering",
        "wrong_aggregation": "generation",
    }
    NODE_TO_STAGE = {
        "prune":    "schema",
        "retrieve": "retrieval",
        "plan":     "planning",
        "generate": "generation",
        "execute":  "execution",
        "correct":  "correction",
    }

    failed_stage = ""
    if failed_node:
        failed_stage = NODE_TO_STAGE.get(failed_node, failed_node)
    elif failure_type:
        failed_stage = FAILURE_TYPE_TO_STAGE.get(failure_type, failure_type)
    elif trace_events:
        for evt in trace_events:
            if evt.get("status") == "failed":
                failed_stage = NODE_TO_STAGE.get(evt.get("node", ""), "")
                break
        if not failed_stage:
            failed_stage = "generation"  # default: SQL was wrong

    schema_hint = _extract_schema_hint(original_sql, corrected_sql)
    return {"failed_stage": failed_stage, "schema_hint": schema_hint}


def _extract_schema_hint(original_sql: str, corrected_sql: str) -> str:
    """
    Diff the table references between original and corrected SQL.
    Returns a compact hint: "table: <new_tables>" for schema pruning guidance.
    """
    if not corrected_sql or not original_sql:
        return ""
    try:
        import re
        pattern = re.compile(r'\b(?:FROM|JOIN)\s+(["`\w]+(?:\.["`\w]+)?)', re.IGNORECASE)
        orig_tables = {t.strip('`"').lower() for t in pattern.findall(original_sql)}
        corr_tables = {t.strip('`"').lower() for t in pattern.findall(corrected_sql)}
        new_tables = corr_tables - orig_tables
        if new_tables:
            return "table: " + ", ".join(sorted(new_tables))
    except Exception as exc:
        logger.debug("agents.operation_failed", error=str(exc))
    return ""


def _correction_message(analysis: dict) -> str:
    MESSAGES = {
        "schema":     "Schema context updated — agent will include correct tables next time",
        "retrieval":  "Example store updated — correction surfaced first on similar queries",
        "planning":   "Query strategy saved — agent will plan differently for this pattern",
        "generation": "SQL pattern registered — agent will generate correct SQL next time",
        "filtering":  "Filter logic saved — agent will apply correct conditions",
        "":           "Correction registered — agent will improve on similar queries",
    }
    return MESSAGES.get(analysis.get("failed_stage", ""), MESSAGES[""])
