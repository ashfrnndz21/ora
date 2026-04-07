"""Semantic Analysis Agent — runs once at data-source connect time.

Analyzes schema + sample values to build semantic understanding:
  - What does each column mean (obs_value = employment rate %)
  - What abbreviation schemes are in use (MYS = Malaysia, sex = Total/Male/Female)
  - Which columns are dimensions vs measures
  - What domain is this data from
  - Query tips (e.g. "filter sex='Total' for aggregate queries")

Output is saved to disk and injected into every SQL generation prompt,
giving the LLM permanent data context at zero per-query cost.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import structlog

logger = structlog.get_logger()


@dataclass
class SourceSemanticContext:
    """Semantic understanding of one data source, built once at connect time."""

    source_id: str
    domain: str = ""                                    # "ILO labour market statistics"
    column_meanings: dict[str, str] = field(default_factory=dict)    # col → plain-English meaning
    abbreviation_maps: dict[str, dict[str, str]] = field(default_factory=dict)  # col → {abbrev: full}
    dimension_columns: list[str] = field(default_factory=list)       # grouping cols
    measure_columns: list[str] = field(default_factory=list)         # numeric value cols
    filter_tips: list[str] = field(default_factory=list)             # query heuristics

    def to_context_block(self) -> str:
        """Formatted text block injected into SQL generation prompts."""
        lines: list[str] = []

        if self.domain:
            lines.append(f"Data domain: {self.domain}")

        if self.column_meanings:
            lines.append("Column meanings:")
            for col, meaning in self.column_meanings.items():
                lines.append(f"  {col}: {meaning}")

        if self.abbreviation_maps:
            lines.append("Value abbreviations (use these in WHERE clauses):")
            for col, abbrevs in self.abbreviation_maps.items():
                # Show up to 8 entries; the LLM doesn't need the full list
                sample_pairs = list(abbrevs.items())[:8]
                pairs_str = ", ".join(f"{k}={v}" for k, v in sample_pairs)
                suffix = "…" if len(abbrevs) > 8 else ""
                lines.append(f"  {col}: {pairs_str}{suffix}")

        if self.filter_tips:
            lines.append("Query tips:")
            for tip in self.filter_tips:
                lines.append(f"  • {tip}")

        return "\n".join(lines) if lines else ""

    def to_dict(self) -> dict:
        return {
            "source_id": self.source_id,
            "domain": self.domain,
            "column_meanings": self.column_meanings,
            "abbreviation_maps": self.abbreviation_maps,
            "dimension_columns": self.dimension_columns,
            "measure_columns": self.measure_columns,
            "filter_tips": self.filter_tips,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SourceSemanticContext":
        return cls(
            source_id=data.get("source_id", ""),
            domain=data.get("domain", ""),
            column_meanings=data.get("column_meanings", {}),
            abbreviation_maps=data.get("abbreviation_maps", {}),
            dimension_columns=data.get("dimension_columns", []),
            measure_columns=data.get("measure_columns", []),
            filter_tips=data.get("filter_tips", []),
        )


async def analyze_source(
    source_id: str,
    connector: object,
    llm: object,
) -> SourceSemanticContext | None:
    """Run the semantic analysis agent against one data source.

    Makes a single LLM call with schema + sample data.
    Returns None on any failure — never blocks the connect flow.
    """
    try:
        snap = await connector.introspect()
        if not snap.tables:
            return None

        # Build schema + sample section for the LLM
        schema_lines: list[str] = []
        for table in snap.tables[:5]:  # cap at 5 tables
            col_parts = []
            for col in table.columns[:30]:  # cap at 30 cols
                ex = f"  [values: {', '.join(str(v) for v in col.examples[:6])}]" if col.examples else ""
                col_parts.append(f"    {col.name} {col.data_type}{ex}")
            schema_lines.append(
                f"Table: {table.name}  ({table.row_count_estimate:,} rows)\n" + "\n".join(col_parts)
            )

        schema_text = "\n\n".join(schema_lines)

        prompt = f"""\
You are a data analyst. Given the schema and sample values below, build a semantic understanding of this data source.

SOURCE ID: {source_id}

SCHEMA:
{schema_text}

Return a JSON object with these fields:
{{
  "domain": "brief domain description (e.g. 'ILO labour market statistics', 'retail sales transactions')",
  "column_meanings": {{
    "<col_name>": "plain-English meaning of the column (e.g. 'employment rate as % of working-age population')"
  }},
  "abbreviation_maps": {{
    "<col_name>": {{"<abbrev_or_code>": "<full_value>", ...}}
  }},
  "dimension_columns": ["col1", "col2"],
  "measure_columns": ["col3"],
  "filter_tips": [
    "tip about how to query correctly, e.g. 'filter sex=Total for population-level aggregates'"
  ]
}}

Rules:
- column_meanings: describe EVERY column in plain English. Be specific about units, scale, and what the values represent.
- abbreviation_maps: ONLY include columns where the stored values are codes or abbreviations (not already human-readable). For each such column, include 5-10 representative sample mappings from the actual values visible in the schema above — enough to show the code FORMAT and style (e.g. 3-letter ISO codes, numeric IDs, short category codes). You do NOT need to list every value — just enough to make the format unambiguous. Skip columns with already human-readable plain text values.
- dimension_columns: columns that define groupings (country, year, gender, category, etc.)
- measure_columns: columns that contain numeric measurements (rates, amounts, counts, etc.)
- filter_tips: practical SQL hints — default filter values, how to avoid double-counting, required GROUP BY columns, etc.
- Return ONLY valid JSON, no markdown, no explanation.
"""

        resp = await llm.complete([{"role": "user", "content": prompt}], json_mode=True)
        raw = resp.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip().rstrip("```").strip()

        parsed = json.loads(raw)

        ctx = SourceSemanticContext(
            source_id=source_id,
            domain=parsed.get("domain", ""),
            column_meanings=parsed.get("column_meanings", {}),
            abbreviation_maps=parsed.get("abbreviation_maps", {}),
            dimension_columns=parsed.get("dimension_columns", []),
            measure_columns=parsed.get("measure_columns", []),
            filter_tips=parsed.get("filter_tips", []),
        )
        logger.info(
            "semantic.analyzed",
            source_id=source_id,
            domain=ctx.domain,
            columns=len(ctx.column_meanings),
            abbrev_cols=len(ctx.abbreviation_maps),
            tips=len(ctx.filter_tips),
        )
        return ctx

    except Exception as exc:
        logger.warning("semantic.analyze_failed", source_id=source_id, error=str(exc))
        return None


def persist_context(ctx: SourceSemanticContext, workspace_id: str) -> None:
    """Save semantic context to disk so it survives server restarts."""
    try:
        base = os.path.join(
            os.path.expanduser("~"), ".sqlagent", "uploads", workspace_id
        )
        os.makedirs(base, exist_ok=True)
        path = os.path.join(base, f"semantic_{ctx.source_id}.json")
        with open(path, "w") as f:
            json.dump(ctx.to_dict(), f, indent=2)
        logger.info("semantic.persisted", source_id=ctx.source_id, path=path)
    except Exception as exc:
        logger.warning("semantic.persist_failed", source_id=ctx.source_id, error=str(exc))


# ═══════════════════════════════════════════════════════════════════════════════
# SEMANTIC MANIFEST — versioned snapshot of semantic layer state
# ═══════════════════════════════════════════════════════════════════════════════


def _load_manifest(workspace_id: str) -> dict:
    """Load the current semantic manifest (or create a fresh one)."""
    ws_dir = os.path.join(
        os.path.expanduser("~"), ".sqlagent", "uploads", workspace_id
    )
    path = os.path.join(ws_dir, "semantic_manifest.json")
    if os.path.exists(path):
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            pass
    return {"iteration_id": 0, "history": []}


def _save_manifest(workspace_id: str, manifest: dict) -> None:
    """Persist the semantic manifest to disk."""
    ws_dir = os.path.join(
        os.path.expanduser("~"), ".sqlagent", "uploads", workspace_id
    )
    os.makedirs(ws_dir, exist_ok=True)
    path = os.path.join(ws_dir, "semantic_manifest.json")
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)


# ═══════════════════════════════════════════════════════════════════════════════
# LIGHTWEIGHT TAXONOMY — replaces exhaustive bootstrap at connect time
# ═══════════════════════════════════════════════════════════════════════════════


async def build_initial_taxonomy(
    workspace_id: str,
    connectors: dict,
    llm: object,
    semantic_contexts: dict[str, SourceSemanticContext] | None = None,
) -> dict:
    """Build a lightweight semantic taxonomy at connect time.

    Instead of pre-computing aliases for every value (expensive, 90% confidence),
    this function:
    1. Detects cross-source join candidates (no LLM — column name/type matching)
    2. Classifies entity types and detects group memberships (single LLM call)
    3. Saves as iteration_id=1 in the semantic manifest

    The taxonomy evolves through actual queries via evolve_semantic_layer().
    """
    from datetime import datetime, timezone

    manifest = _load_manifest(workspace_id)
    result = {
        "cross_source_joins": [],
        "entity_types": {},
        "group_memberships": {},
        "source_summaries": {},
    }

    # ── Phase 1: Cross-source join detection (no LLM) ──────────────────
    all_tables: dict[str, list[dict]] = {}  # source_id → [{table, columns}]

    for sid, conn in connectors.items():
        try:
            snap = await conn.introspect()
            tables_info = []
            for table in snap.tables[:10]:
                cols = [
                    {"name": col.name, "type": (col.data_type or "").lower()}
                    for col in table.columns
                ]
                tables_info.append({"table": table.name, "columns": cols})
            all_tables[sid] = tables_info
        except Exception:
            continue

    # Compare columns across all table pairs (within and across sources)
    all_flat = []
    for sid, tables in all_tables.items():
        for t in tables:
            for col in t["columns"]:
                all_flat.append({
                    "source": sid, "table": t["table"],
                    "column": col["name"], "type": col["type"],
                })

    for i, a in enumerate(all_flat):
        for b in all_flat[i + 1:]:
            if a["source"] == b["source"] and a["table"] == b["table"]:
                continue
            # Same column name = candidate join
            a_name = a["column"].lower().rstrip("_id")
            b_name = b["column"].lower().rstrip("_id")
            if a_name == b_name and a_name:
                # Check type compatibility
                a_type = a["type"]
                b_type = b["type"]
                compatible = (
                    a_type == b_type
                    or (any(t in a_type for t in ("varchar", "text", "char"))
                        and any(t in b_type for t in ("varchar", "text", "char")))
                    or (any(t in a_type for t in ("int", "bigint", "numeric"))
                        and any(t in b_type for t in ("int", "bigint", "numeric")))
                )
                if compatible:
                    join = {
                        "from": f"{a['source']}.{a['table']}.{a['column']}",
                        "to": f"{b['source']}.{b['table']}.{b['column']}",
                        "confidence": 0.9 if a["column"] == b["column"] else 0.7,
                        "method": "name_match",
                    }
                    # Deduplicate
                    key = tuple(sorted([join["from"], join["to"]]))
                    if not any(
                        tuple(sorted([j["from"], j["to"]])) == key
                        for j in result["cross_source_joins"]
                    ):
                        result["cross_source_joins"].append(join)

    # ── Phase 2: Entity classification (single LLM call) ──────────────
    # Build a compact schema summary for the LLM
    schema_summary = []
    for sid, tables in all_tables.items():
        ctx = (semantic_contexts or {}).get(sid)
        for t in tables:
            dim_cols = []
            for col in t["columns"]:
                is_dim = ctx and col["name"] in (ctx.dimension_columns or [])
                if is_dim or any(
                    kw in col["name"].lower()
                    for kw in ("code", "iso", "name", "type", "country", "region", "category")
                ):
                    dim_cols.append(col["name"])
            if dim_cols:
                # Get sample values from semantic context if available
                samples = ""
                if ctx and ctx.abbreviation_maps:
                    for dc in dim_cols[:3]:
                        if dc in ctx.abbreviation_maps:
                            vals = list(ctx.abbreviation_maps[dc].keys())[:8]
                            samples += f"\n      {dc} samples: {vals}"
                schema_summary.append(
                    f"  {sid}.{t['table']}: dimensions=[{', '.join(dim_cols)}]{samples}"
                )

    if schema_summary:
        taxonomy_prompt = f"""\
Analyze these data source dimensions and classify the entity types.

SOURCES AND DIMENSIONS:
{chr(10).join(schema_summary)}

CROSS-SOURCE JOINS DETECTED:
{json.dumps(result['cross_source_joins'][:10], indent=2) if result['cross_source_joins'] else 'None detected'}

Return JSON:
{{
  "entity_types": {{
    "column_name": "entity_type (e.g. country_code, year, industry, gender, age_group)"
  }},
  "group_memberships": {{
    "column_name": {{
      "group_name": ["member1", "member2", "..."]
    }}
  }}
}}

Rules:
- entity_types: map EVERY dimension column to its semantic type
- group_memberships: ONLY include well-known groups you can identify from samples
  (e.g. ASEAN countries, G7, EU, age brackets, gender categories)
- For country/region columns: identify which geopolitical groups the sample values belong to
- Return ONLY valid JSON
"""
        try:
            resp = await llm.complete(
                [{"role": "user", "content": taxonomy_prompt}],
                temperature=0.0, max_tokens=1024, json_mode=True,
            )
            raw = resp.content.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip().rstrip("```").strip()
            parsed = json.loads(raw)
            result["entity_types"] = parsed.get("entity_types", {})
            result["group_memberships"] = parsed.get("group_memberships", {})
        except Exception as exc:
            logger.warning("semantic.taxonomy.llm_failed", error=str(exc))

    # ── Phase 3: Save manifest with iteration_id ──────────────────────
    manifest["iteration_id"] += 1
    manifest["history"].append({
        "iteration_id": manifest["iteration_id"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "trigger": "connect",
        "cross_source_joins": len(result["cross_source_joins"]),
        "entity_types": len(result["entity_types"]),
        "groups_detected": list(
            g for col_groups in result["group_memberships"].values()
            for g in col_groups.keys()
        ),
    })
    # Keep history bounded
    manifest["history"] = manifest["history"][-50:]

    # Save taxonomy data into manifest
    manifest["taxonomy"] = result
    _save_manifest(workspace_id, manifest)

    logger.info(
        "semantic.taxonomy.built",
        workspace_id=workspace_id,
        iteration_id=manifest["iteration_id"],
        joins=len(result["cross_source_joins"]),
        entity_types=len(result["entity_types"]),
        groups=len(result["group_memberships"]),
    )

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# SEMANTIC LAYER EVOLUTION — called by Learn Agent after every successful query
# ═══════════════════════════════════════════════════════════════════════════════


def evolve_semantic_layer(
    workspace_id: str,
    query_result: dict,
) -> dict:
    """Update the semantic layer with discoveries from a successful query.

    Called by the Learn Agent after every successful query. Persists:
    1. Schema Agent findings (entity values found via search)
    2. Confirmed table relationships (joins that worked)
    3. Query patterns (common filters, default values)
    4. Column meaning enrichment (what the column was used for)

    Returns a summary of what was learned.
    """
    learned = {"aliases": 0, "relationships": 0, "patterns": 0, "enrichments": 0}
    # Track actual items for history detail
    learned_details = {"aliases": [], "relationships": [], "patterns": [], "enrichments": []}

    ws_dir = os.path.join(
        os.path.expanduser("~"), ".sqlagent", "uploads", workspace_id
    )
    os.makedirs(ws_dir, exist_ok=True)

    # ── 1. Save new aliases from semantic reasoning ──────────────────────
    sem_reasoning = query_result.get("semantic_reasoning")
    if sem_reasoning and sem_reasoning.get("new_aliases"):
        target_sources = query_result.get("target_sources", [])
        for sid in target_sources:
            existing = _load_learned_aliases(workspace_id, sid)
            new_confs: dict[str, float] = {}
            for term, value in sem_reasoning["new_aliases"].items():
                term_lower = term.lower().strip()
                if term_lower and term_lower not in existing:
                    existing[term_lower] = value
                    new_confs[term_lower] = sem_reasoning.get("confidence", 0.85)
                    learned["aliases"] += 1
                    learned_details["aliases"].append(f"{term_lower} → {value}")
            if new_confs:
                _save_learned_aliases(workspace_id, sid, existing, new_confs)

    # ── 1b. Learn group memberships from resolved filters ─────────────
    # If a filter has a list of values (e.g., ASEAN → 10 countries),
    # save the group mapping so future queries resolve instantly
    if sem_reasoning and sem_reasoning.get("filters"):
        for flt in sem_reasoning["filters"]:
            val = flt.get("value", "")
            if isinstance(val, list) and len(val) >= 3:
                # This is a group resolution — save it
                # Check if any alias maps to this group
                reasoning = sem_reasoning.get("reasoning", "").lower()
                for alias_term, alias_val in sem_reasoning.get("new_aliases", {}).items():
                    if isinstance(alias_val, str) and alias_val.lower() in reasoning:
                        # Save the group: term → list of values
                        target_sources = query_result.get("target_sources", [])
                        for sid in target_sources:
                            existing = _load_learned_aliases(workspace_id, sid)
                            key = alias_term.lower().strip()
                            if key not in existing or not isinstance(existing.get(key), list):
                                existing[key] = val  # Save the list, not the description
                                _save_learned_aliases(workspace_id, sid, existing, {key: 0.9})
                                learned["aliases"] += 1
                                learned_details["aliases"].append(f"{key} → [{len(val)} values]")
                                break

    # ── 1c. Learn which tables were used together ─────────────────────
    # If the query successfully used multiple tables, save which tables
    # answer what kind of question (e.g., "both rates" = both tables)
    sql = query_result.get("sql", "")

    # ── 2. Save confirmed relationships ──────────────────────────────────
    if sem_reasoning and len(sem_reasoning.get("tables", [])) >= 2:
        rels_path = os.path.join(ws_dir, "relationships.json")
        try:
            existing_rels = []
            if os.path.exists(rels_path):
                with open(rels_path) as f:
                    existing_rels = json.load(f)

            tables = sem_reasoning["tables"]
            reasoning = sem_reasoning.get("reasoning", "")
            # Extract join info from reasoning
            for i, t1 in enumerate(tables):
                for t2 in tables[i + 1:]:
                    rel_key = f"{t1}:{t2}"
                    existing_keys = {f"{r['from_table']}:{r['to_table']}" for r in existing_rels}
                    if rel_key not in existing_keys:
                        existing_rels.append({
                            "from_table": t1,
                            "to_table": t2,
                            "source": "query_confirmed",
                            "confidence": 0.85,
                            "query_count": 1,
                        })
                        learned["relationships"] += 1
                        learned_details["relationships"].append(f"{t1} ↔ {t2}")
                    else:
                        # Strengthen existing relationship
                        for r in existing_rels:
                            if f"{r['from_table']}:{r['to_table']}" == rel_key:
                                r["query_count"] = r.get("query_count", 0) + 1
                                r["confidence"] = min(r.get("confidence", 0.85) + 0.03, 0.99)

            with open(rels_path, "w") as f:
                json.dump(existing_rels, f, indent=2)
        except Exception as exc:
            logger.debug("semantic.evolve.relationships_failed", error=str(exc))

    # ── 3. Save query patterns (common filters) ─────────────────────────
    if sem_reasoning and sem_reasoning.get("filters"):
        patterns_path = os.path.join(ws_dir, "query_patterns.json")
        try:
            existing_patterns = []
            if os.path.exists(patterns_path):
                with open(patterns_path) as f:
                    existing_patterns = json.load(f)

            for flt in sem_reasoning["filters"]:
                col = flt.get("column", "")
                val = flt.get("value", "")
                tbl = flt.get("table", "")
                if not col or not val:
                    continue
                # Check if this pattern already exists
                pattern_key = f"{tbl}.{col}={val}"
                found = False
                for p in existing_patterns:
                    if p.get("key") == pattern_key:
                        p["count"] = p.get("count", 0) + 1
                        found = True
                        break
                if not found:
                    existing_patterns.append({
                        "key": pattern_key,
                        "table": tbl,
                        "column": col,
                        "value": val,
                        "count": 1,
                    })
                    learned["patterns"] += 1
                    learned_details["patterns"].append(f"{tbl}.{col} = '{val}'")

            with open(patterns_path, "w") as f:
                json.dump(existing_patterns, f, indent=2)
        except Exception as exc:
            logger.debug("semantic.evolve.patterns_failed", error=str(exc))

    # ── 4. Enrich column meanings from query context ─────────────────────
    if query_result.get("succeeded") and sem_reasoning:
        sql = query_result.get("sql", "")
        nl_query = query_result.get("nl_query", "")
        tables_used = sem_reasoning.get("tables", [])
        metrics_used = sem_reasoning.get("metrics", [])

        if tables_used and metrics_used:
            enrichments_path = os.path.join(ws_dir, "column_enrichments.json")
            try:
                enrichments = {}
                if os.path.exists(enrichments_path):
                    with open(enrichments_path) as f:
                        enrichments = json.load(f)

                for metric in metrics_used:
                    key = f"{tables_used[0]}.{metric}"
                    if key not in enrichments:
                        enrichments[key] = {
                            "column": metric,
                            "table": tables_used[0],
                            "used_for": [nl_query[:100]],
                            "query_count": 1,
                        }
                        learned["enrichments"] += 1
                        learned_details["enrichments"].append(f"{tables_used[0]}.{metric} used for '{nl_query[:50]}'")
                    else:
                        enrichments[key]["query_count"] += 1
                        if nl_query[:100] not in enrichments[key].get("used_for", []):
                            enrichments[key]["used_for"].append(nl_query[:100])
                            enrichments[key]["used_for"] = enrichments[key]["used_for"][-5:]

                with open(enrichments_path, "w") as f:
                    json.dump(enrichments, f, indent=2)
            except Exception as exc:
                logger.debug("semantic.evolve.enrichments_failed", error=str(exc))

    if any(v > 0 for v in learned.values()):
        logger.info("semantic.evolved", **learned, workspace_id=workspace_id)

        # ── Increment semantic manifest iteration ────────────────────────
        try:
            from datetime import datetime as _dt, timezone as _tz
            manifest = _load_manifest(workspace_id)
            manifest["iteration_id"] += 1
            manifest["history"].append({
                "iteration_id": manifest["iteration_id"],
                "timestamp": _dt.now(_tz.utc).isoformat(),
                "trigger": "query_success",
                "learned": learned,
                "details": {k: v[:5] for k, v in learned_details.items() if v},
                "nl_query": query_result.get("nl_query", "")[:100],
            })
            manifest["history"] = manifest["history"][-50:]
            _save_manifest(workspace_id, manifest)
        except Exception:
            pass  # Never fail a query over manifest tracking

    return learned


async def bootstrap_inference_graph(
    source_id: str,
    workspace_id: str,
    connector: object,
    llm: object,
    semantic_context: SourceSemanticContext | None = None,
) -> dict[str, str]:
    """Build a complete inferential alias map for all entity values in a source.

    This is the intelligence layer. On first connect (or when triggered by admin),
    the Semantic Agent:

    1. Samples ALL distinct values from each coded/abbreviation column
    2. Asks the LLM to infer EVERY way a user might refer to each value
       (ISO2, ISO3, currency codes, nicknames, misspellings, alternate names)
    3. Builds a complete alias → canonical map with confidence levels
    4. Persists the map so all future queries resolve instantly (zero LLM cost)

    Example output for iso_code column with value 'MYS':
      my → MYS (ISO2 code, confidence 0.98)
      myr → MYS (currency code Malaysian Ringgit, confidence 0.95)
      malaysia → MYS (full country name, confidence 0.99)
      malay → MYS (common abbreviation, confidence 0.85)
      msia → MYS (informal abbreviation, confidence 0.80)

    This runs ONCE per source. After that, every user query resolves
    from the pre-built map — no LLM call needed.
    """
    all_aliases: dict[str, str] = {}

    # Load any existing learned aliases
    existing = _load_learned_aliases(workspace_id, source_id)
    all_aliases.update(existing)

    # Identify ALL dimension columns that could benefit from alias inference.
    # Not just coded columns — ANY column with categorical/entity values
    # should have inferential aliases built so the Semantic Agent can resolve
    # user terms like "MY" → "Malaysia", "genai" → "GenAI", etc.
    coded_columns: dict[str, list[str]] = {}  # col_path → all distinct values

    try:
        snap = await connector.introspect()
        for table in snap.tables[:8]:
            for col in table.columns:
                col_lower = col.name.lower()
                dt_lower = (col.data_type or "").lower()

                # Include this column if ANY of:
                # 1. It was identified as having abbreviations by analyze_source
                # 2. Column name suggests it's an entity/category column
                # 3. It's a text/string column classified as a dimension
                # 4. It has low cardinality (< 200 distinct values = likely categorical)
                is_dimension = False

                if semantic_context:
                    if col.name in (semantic_context.abbreviation_maps or {}):
                        is_dimension = True
                    if col.name in (semantic_context.dimension_columns or []):
                        is_dimension = True

                if any(kw in col_lower for kw in (
                    "code", "iso", "name", "type", "category", "status", "region",
                    "country", "city", "state", "segment", "tier", "channel",
                    "product", "service", "vendor", "provider", "department",
                )):
                    is_dimension = True

                # Text/string columns are likely dimensions
                if any(t in dt_lower for t in ("varchar", "text", "string", "nvarchar", "char")):
                    is_dimension = True

                # Skip numeric, date, and boolean columns
                if any(t in dt_lower for t in (
                    "int", "float", "double", "decimal", "numeric", "date",
                    "time", "bool", "bigint", "money", "real",
                )):
                    is_dimension = False

                if is_dimension:
                    try:
                        res = await connector.execute(
                            f'SELECT DISTINCT "{col.name}" FROM "{table.name}" '
                            f'WHERE "{col.name}" IS NOT NULL LIMIT 200'
                        )
                        import pandas as _pd
                        if isinstance(res, _pd.DataFrame) and not res.empty:
                            vals = [str(v) for v in res.iloc[:, 0].dropna().tolist()]
                            if vals:
                                coded_columns[f"{table.name}.{col.name}"] = vals
                    except Exception:
                        pass
    except Exception as exc:
        logger.warning("semantic.bootstrap.introspect_failed", error=str(exc))
        return all_aliases

    if not coded_columns:
        logger.info("semantic.bootstrap.no_dimension_columns", source_id=source_id)
        return all_aliases

    logger.info(
        "semantic.bootstrap.starting",
        source_id=source_id,
        dimension_columns=list(coded_columns.keys()),
        total_values=sum(len(v) for v in coded_columns.values()),
    )

    # Accumulate confidence scores across all columns
    all_confidences: dict[str, float] = {}

    # Build inference prompt for each coded column
    for col_path, values in coded_columns.items():
        # Skip if we already have extensive aliases for these values
        existing_canonicals = set(all_aliases.values())
        new_values = [v for v in values if v not in existing_canonicals
                      and v.lower() not in all_aliases]
        if len(new_values) < 2:
            continue

        # Include existing abbreviation map context if available
        abbrev_context = ""
        col_name = col_path.split(".")[-1]
        if semantic_context and col_name in semantic_context.abbreviation_maps:
            known = semantic_context.abbreviation_maps[col_name]
            abbrev_context = "\nKnown mappings for this column:\n" + "\n".join(
                f"  {code} = {name}" for code, name in known.items()
            )

        prompt = f"""\
You are building a semantic alias map for a database column so that a natural language
query system can resolve user terms to actual stored values.

Column: {col_path}
All distinct values in the database: {values[:100]}
{abbrev_context}

For EACH value, generate ALL possible ways a user might refer to it in a question.
Think about:
- Abbreviations and acronyms (e.g. "AWS" for "Amazon Web Services", "SG" for "Singapore")
- ISO codes if applicable (MY, MYS, MYR for Malaysia)
- Informal names, nicknames, short forms (e.g. "genai" for "Gen AI", "indo" for "Indonesia")
- Alternate spellings or casing (e.g. "Viet Nam" vs "Vietnam")
- Related concepts (e.g. "cloud" might map to a specific service category)
- Plural/singular forms
- Common user-facing labels vs internal database values

Return JSON:
{{
  "aliases": [
    {{"canonical": "Malaysia", "aliases": ["my", "mys", "myr", "malay", "msia", "malaysian"], "confidence": 0.95}},
    {{"canonical": "Gen AI", "aliases": ["genai", "generative ai", "gen-ai", "llm services"], "confidence": 0.90}}
  ]
}}

Rules:
- canonical MUST be the EXACT value as stored in the database (case-sensitive match)
- aliases should be LOWERCASE
- Only include aliases with confidence > 0.85
- Be conservative — only map aliases you are VERY sure about
- If a value is already a plain English word with no obvious aliases, you can skip it
- Return ONLY valid JSON, no markdown
"""

        try:
            resp = await llm.complete(
                [{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=4096,
                json_mode=True,
            )

            raw = resp.content.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip().rstrip("```").strip()

            parsed = json.loads(raw)
            entries = parsed.get("aliases", [])

            new_count = 0
            for entry in entries:
                canonical = entry.get("canonical", "")
                aliases = entry.get("aliases", [])
                conf = entry.get("confidence", 0.8)

                if not canonical or conf < CONFIDENCE_THRESHOLD:
                    logger.debug(
                        "semantic.bootstrap.below_threshold",
                        canonical=canonical, confidence=conf,
                        threshold=CONFIDENCE_THRESHOLD,
                    )
                    continue

                for alias in aliases:
                    alias_lower = alias.lower().strip()
                    if not alias_lower or alias_lower == canonical.lower():
                        continue
                    if alias_lower in all_aliases:
                        continue

                    # Structural confidence adjustment — LLMs are bad at
                    # self-reporting confidence. Apply rules:
                    # 1. Single common word (< 4 chars or in common words) → reject
                    # 2. Single word that's a generic English word → reduce confidence
                    # 3. Acronyms/codes (all caps, 2-4 chars) → boost confidence
                    adjusted_conf = conf
                    words = alias_lower.split()

                    if len(words) == 1 and len(alias_lower) <= 3:
                        # Short codes like "my", "sg", "ph" — likely valid codes
                        # Keep confidence as-is
                        pass
                    elif len(words) == 1 and len(alias_lower) <= 5 and alias_lower.isalpha():
                        # Single short word: could be ambiguous
                        # "power", "web", "food", "green" etc. are too generic
                        adjusted_conf = min(conf, 0.70)  # will be below 0.85 threshold
                    elif len(words) == 1 and alias_lower.isalpha() and len(alias_lower) >= 6:
                        # Longer single words: "banking", "telecom", "logistics"
                        # Slightly ambiguous but more specific
                        adjusted_conf = min(conf, 0.82)  # just below threshold

                    if adjusted_conf >= CONFIDENCE_THRESHOLD:
                        all_aliases[alias_lower] = canonical
                        all_confidences[alias_lower] = adjusted_conf
                        new_count += 1

            logger.info(
                "semantic.bootstrap.column_done",
                column=col_path,
                values_count=len(values),
                new_aliases=new_count,
                total_aliases=len(all_aliases),
            )

        except Exception as exc:
            logger.warning(
                "semantic.bootstrap.llm_failed",
                column=col_path, error=str(exc),
            )

    # Persist the complete alias map with confidence scores
    _save_learned_aliases(workspace_id, source_id, all_aliases, all_confidences)

    logger.info(
        "semantic.bootstrap.completed",
        source_id=source_id,
        total_aliases=len(all_aliases),
        coded_columns=len(coded_columns),
    )
    return all_aliases


def load_context(source_id: str, workspace_id: str) -> SourceSemanticContext | None:
    """Load persisted semantic context from disk."""
    try:
        path = os.path.join(
            os.path.expanduser("~"), ".sqlagent", "uploads", workspace_id,
            f"semantic_{source_id}.json"
        )
        if not os.path.exists(path):
            return None
        with open(path) as f:
            return SourceSemanticContext.from_dict(json.load(f))
    except Exception as exc:
        logger.warning("semantic.load_failed", source_id=source_id, error=str(exc))
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# PER-QUERY SEMANTIC RESOLUTION (v2.0)
# ═══════════════════════════════════════════════════════════════════════════════
#
# This is the Semantic Agent's per-query work. Called between the understand
# node and the prune node in the LangGraph pipeline.
#
# Strategy (fully agentic — no difflib, no regex, no keyword lists):
#   1. Check learned aliases (persisted from previous queries) — instant
#   2. Check abbreviation_maps from connect-time analysis — instant
#   3. LLM reasoning with actual column values as context — 1 call
#   4. Save resolved aliases for future queries — persistent learning
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class SemanticResolution:
    """Result of the Semantic Agent's per-query resolution."""

    entity_map: dict[str, str] = field(default_factory=dict)
    """User term → canonical DB value. E.g. {'MYR': 'MYS', 'PHP': 'PHL'}"""

    synonyms: dict[str, str] = field(default_factory=dict)
    """Natural language term → column.value. E.g. {'Malaysia': 'MYS'}"""

    filter_hints: list[str] = field(default_factory=list)
    """SQL WHERE fragments. E.g. ["iso_code IN ('MYS','PHL')"]"""

    reasoning: str = ""
    """LLM's reasoning trace for the resolution."""

    resolved_from: dict[str, str] = field(default_factory=dict)
    """Provenance per entity: term → source. E.g. {'MYR': 'llm', 'PHP': 'learned'}"""

    confidence: float = 0.0
    """Overall resolution confidence 0.0–1.0"""

    llm_called: bool = False
    """Whether an LLM call was needed (False = all resolved from cache/memory)"""


# ── Confidence threshold ──────────────────────────────────────────────────────
# Only aliases above this threshold are saved to the semantic layer.
# Below this → the Semantic Agent asks the LLM per-query instead of assuming.
CONFIDENCE_THRESHOLD = 0.85

# Persisted learned aliases — source_id → {user_term_lower → canonical_value}
# Each alias also has a confidence score stored in a parallel dict.
_learned_aliases: dict[str, dict[str, str]] = {}
_alias_confidence: dict[str, dict[str, float]] = {}


def _load_learned_aliases(workspace_id: str, source_id: str) -> dict[str, str]:
    """Load learned aliases from disk."""
    key = f"{workspace_id}:{source_id}"
    if key in _learned_aliases:
        return _learned_aliases[key]
    try:
        path = os.path.join(
            os.path.expanduser("~"), ".sqlagent", "uploads", workspace_id,
            f"aliases_{source_id}.json"
        )
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            # Support both old format (flat dict) and new format (with confidence)
            if isinstance(data, dict) and "_confidence" in data:
                aliases = {k: v for k, v in data.items() if k != "_confidence"}
                _learned_aliases[key] = aliases
                _alias_confidence[key] = data["_confidence"]
            else:
                _learned_aliases[key] = data
                _alias_confidence[key] = {k: 0.9 for k in data}
            return _learned_aliases[key]
    except Exception:
        pass
    _learned_aliases[key] = {}
    _alias_confidence[key] = {}
    return _learned_aliases[key]


def _get_alias_confidence(workspace_id: str, source_id: str, alias: str) -> float:
    """Get confidence for a specific alias."""
    key = f"{workspace_id}:{source_id}"
    return _alias_confidence.get(key, {}).get(alias.lower(), 0.5)


def _save_learned_aliases(
    workspace_id: str,
    source_id: str,
    aliases: dict[str, str],
    confidence: dict[str, float] | None = None,
) -> None:
    """Persist learned aliases to disk with confidence scores."""
    key = f"{workspace_id}:{source_id}"
    _learned_aliases[key] = aliases
    if confidence:
        if key not in _alias_confidence:
            _alias_confidence[key] = {}
        _alias_confidence[key].update(confidence)

    try:
        base = os.path.join(
            os.path.expanduser("~"), ".sqlagent", "uploads", workspace_id
        )
        os.makedirs(base, exist_ok=True)
        path = os.path.join(base, f"aliases_{source_id}.json")
        # Save with confidence metadata
        save_data = dict(aliases)
        save_data["_confidence"] = _alias_confidence.get(key, {})
        with open(path, "w") as f:
            json.dump(save_data, f, indent=2)
    except Exception as exc:
        logger.warning("semantic.save_aliases_failed", error=str(exc))


def strengthen_alias(workspace_id: str, source_id: str, alias: str, canonical: str) -> None:
    """Strengthen confidence for an alias that was confirmed by a successful query.

    Called by the learn node after a query succeeds — progressively builds
    confidence in the semantic layer.
    """
    key = f"{workspace_id}:{source_id}"
    aliases = _load_learned_aliases(workspace_id, source_id)
    alias_lower = alias.lower().strip()

    # Add or update the alias
    aliases[alias_lower] = canonical

    # Increase confidence (cap at 0.99)
    if key not in _alias_confidence:
        _alias_confidence[key] = {}
    current = _alias_confidence[key].get(alias_lower, 0.85)
    new_conf = min(current + 0.03, 0.99)  # Each confirmation adds 3%
    _alias_confidence[key][alias_lower] = new_conf

    _save_learned_aliases(workspace_id, source_id, aliases, _alias_confidence.get(key))
    logger.info(
        "semantic.alias_strengthened",
        alias=alias, canonical=canonical,
        confidence=round(new_conf, 3),
    )


def _save_resolution_log(
    workspace_id: str,
    question: str,
    filters: list[dict],
    confidence: float,
    new_aliases: dict[str, str],
) -> None:
    """Append a resolution entry to the resolution log.

    Tracks every resolution attempt so the agent can reflect on failures
    and avoid repeating mistakes.
    """
    ws_dir = os.path.join(
        os.path.expanduser("~"), ".sqlagent", "uploads", workspace_id
    )
    os.makedirs(ws_dir, exist_ok=True)
    log_path = os.path.join(ws_dir, "resolution_log.json")

    try:
        existing: list[dict] = []
        if os.path.exists(log_path):
            with open(log_path) as f:
                existing = json.load(f)

        # Add entry for each resolved filter
        from datetime import datetime, timezone
        ts = datetime.now(timezone.utc).isoformat()
        for flt in filters:
            existing.append({
                "question": question[:200],
                "term": flt.get("reasoning", "")[:100],
                "resolved_as": flt.get("value", ""),
                "column": flt.get("column", ""),
                "table": flt.get("table", ""),
                "confidence": confidence,
                "succeeded": True,  # Updated to False by record_resolution_failure
                "timestamp": ts,
            })

        # Keep last 200 entries
        existing = existing[-200:]
        with open(log_path, "w") as f:
            json.dump(existing, f, indent=2)
    except Exception as exc:
        logger.debug("semantic.resolution_log.save_failed", error=str(exc))


def record_resolution_failure(
    workspace_id: str,
    question: str,
    failed_filters: list[dict],
    error: str,
    correct_value: str = "",
) -> None:
    """Record a negative signal — a resolution that led to a failed query.

    Called by the orchestrator when a query fails after semantic resolution.
    The failure context is loaded by future reason_about_query() calls to
    avoid repeating the same mistake.
    """
    ws_dir = os.path.join(
        os.path.expanduser("~"), ".sqlagent", "uploads", workspace_id
    )
    os.makedirs(ws_dir, exist_ok=True)
    log_path = os.path.join(ws_dir, "resolution_log.json")

    try:
        existing: list[dict] = []
        if os.path.exists(log_path):
            with open(log_path) as f:
                existing = json.load(f)

        from datetime import datetime, timezone
        ts = datetime.now(timezone.utc).isoformat()
        for flt in failed_filters:
            existing.append({
                "question": question[:200],
                "term": flt.get("reasoning", "")[:100],
                "resolved_as": flt.get("value", ""),
                "column": flt.get("column", ""),
                "table": flt.get("table", ""),
                "confidence": 0.0,
                "succeeded": False,
                "error": error[:200],
                "correct_value": correct_value,
                "timestamp": ts,
            })

        existing = existing[-200:]
        with open(log_path, "w") as f:
            json.dump(existing, f, indent=2)

        logger.info("semantic.resolution_failure_recorded",
                     filters=len(failed_filters), question=question[:60])
    except Exception as exc:
        logger.debug("semantic.resolution_log.failure_save_failed", error=str(exc))


async def resolve_entities(
    entities: list[str],
    source_id: str,
    workspace_id: str,
    connector: object,
    llm: object,
    semantic_context: SourceSemanticContext | None = None,
) -> SemanticResolution:
    """Resolve user entity terms to canonical DB values using LLM reasoning.

    This is the Semantic Agent's core per-query function. It replaces the old
    difflib fuzzy matching with agentic LLM-powered resolution.

    Args:
        entities: User's entity terms, e.g. ['MYR', 'PHP', 'Malaysia']
        source_id: Which data source to resolve against
        workspace_id: For loading/saving persistent aliases
        connector: DB connector for sampling column values
        llm: LLM provider for reasoning
        semantic_context: Connect-time analysis (abbreviation_maps, etc.)

    Returns:
        SemanticResolution with entity_map, synonyms, filter_hints
    """
    if not entities:
        return SemanticResolution(confidence=1.0)

    result = SemanticResolution()
    unresolved: list[str] = []
    learned = _load_learned_aliases(workspace_id, source_id)

    # ── Step 1: Check learned aliases (instant, no cost) ─────────────────
    for entity in entities:
        key = entity.lower().strip()
        if key in learned:
            canonical = learned[key]
            result.entity_map[entity] = canonical
            result.resolved_from[entity] = "learned"
            logger.info("semantic.resolve.learned", entity=entity, canonical=canonical)

    # ── Step 2: Check abbreviation_maps from connect-time analysis ───────
    if semantic_context and semantic_context.abbreviation_maps:
        for entity in entities:
            if entity in result.entity_map:
                continue  # already resolved
            entity_lower = entity.lower().strip()
            for col, abbrev_map in semantic_context.abbreviation_maps.items():
                # Check both keys and values in the abbreviation map
                for code, full_name in abbrev_map.items():
                    if (code.lower() == entity_lower
                            or full_name.lower() == entity_lower):
                        result.entity_map[entity] = code
                        result.resolved_from[entity] = "abbreviation_map"
                        logger.info(
                            "semantic.resolve.abbrev",
                            entity=entity, canonical=code, column=col,
                        )
                        break
                if entity in result.entity_map:
                    break

    # ── Step 3: Collect unresolved entities for LLM ──────────────────────
    for entity in entities:
        if entity not in result.entity_map:
            unresolved.append(entity)

    if not unresolved:
        result.confidence = 1.0
        result.llm_called = False
        result.reasoning = "All entities resolved from learned aliases or abbreviation maps."
        _build_filter_hints(result, semantic_context)
        return result

    # ── Step 4: Sample actual column values for LLM context ──────────────
    actual_values: dict[str, list[str]] = {}
    try:
        snap = await connector.introspect()
        for table in snap.tables[:5]:
            for col in table.columns:
                if col.examples:
                    actual_values[f"{table.name}.{col.name}"] = [
                        str(v) for v in col.examples[:20]
                    ]
                # Also try to get more values for likely entity columns
                col_lower = col.name.lower()
                if any(kw in col_lower for kw in ("code", "iso", "country", "name", "id")):
                    try:
                        res = await connector.execute(
                            f'SELECT DISTINCT "{col.name}" FROM "{table.name}" '
                            f'WHERE "{col.name}" IS NOT NULL LIMIT 50'
                        )
                        import pandas as _pd
                        if isinstance(res, _pd.DataFrame) and not res.empty:
                            vals = [str(v) for v in res.iloc[:, 0].dropna().tolist()]
                            actual_values[f"{table.name}.{col.name}"] = vals
                    except Exception:
                        pass
    except Exception:
        pass

    # ── Step 5: LLM-powered resolution ───────────────────────────────────
    values_context = ""
    for col_path, vals in actual_values.items():
        values_context += f"  {col_path}: {vals[:30]}\n"

    # Include already-resolved entities as context for the LLM
    already_resolved = ""
    if result.entity_map:
        already_resolved = "\nAlready resolved:\n" + "\n".join(
            f"  {k} → {v}" for k, v in result.entity_map.items()
        )

    prompt = f"""\
You are an entity resolution agent for a database query system.

The user mentioned these entities that need to be mapped to actual values in the database:
  Unresolved: {unresolved}
{already_resolved}

Here are the actual values stored in the database columns:
{values_context}

For each unresolved entity, determine which actual database value it refers to.
Consider:
- ISO country codes (MYS=Malaysia, PHL=Philippines, VNM=Vietnam)
- Currency codes (MYR=Malaysian Ringgit→Malaysia, PHP=Philippine Peso→Philippines)
- Common abbreviations, nicknames, partial matches
- The entity might be a different representation of the same thing

Return JSON:
{{
  "resolutions": [
    {{"entity": "MYR", "canonical": "MYS", "reasoning": "MYR is Malaysian Ringgit currency code, maps to Malaysia (MYS)", "confidence": 0.95}},
  ],
  "overall_reasoning": "brief summary"
}}

Rules:
- canonical MUST be an exact value that exists in the database columns shown above
- If you cannot confidently resolve an entity, set confidence < 0.5
- Return ONLY valid JSON, no markdown
"""

    try:
        resp = await llm.complete(
            [{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=1024,
            json_mode=True,
        )
        result.llm_called = True

        raw = resp.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip().rstrip("```").strip()

        parsed = json.loads(raw)
        resolutions = parsed.get("resolutions", [])
        result.reasoning = parsed.get("overall_reasoning", "")

        total_conf = 0.0
        for res in resolutions:
            entity = res.get("entity", "")
            canonical = res.get("canonical", "")
            conf = res.get("confidence", 0.0)
            reasoning = res.get("reasoning", "")

            if entity and canonical and conf >= 0.5:
                result.entity_map[entity] = canonical
                result.resolved_from[entity] = "llm"
                total_conf += conf

                # Save to learned aliases only if confidence ≥ threshold
                if conf >= CONFIDENCE_THRESHOLD:
                    learned[entity.lower().strip()] = canonical
                logger.info(
                    "semantic.resolve.llm",
                    entity=entity, canonical=canonical,
                    confidence=conf, reasoning=reasoning,
                )
            else:
                logger.warning(
                    "semantic.resolve.low_confidence",
                    entity=entity, canonical=canonical,
                    confidence=conf, reasoning=reasoning,
                )

        # Persist learned aliases with confidence scores
        if any(v == "llm" for v in result.resolved_from.values()):
            new_confs = {}
            for res in resolutions:
                e_key = res.get("entity", "").lower().strip()
                if e_key and res.get("confidence", 0) >= CONFIDENCE_THRESHOLD:
                    new_confs[e_key] = res.get("confidence", 0.85)
            _save_learned_aliases(workspace_id, source_id, learned, new_confs)

        # Calculate overall confidence
        all_confs = [
            1.0 if result.resolved_from.get(e) in ("learned", "abbreviation_map") else
            next((r.get("confidence", 0) for r in resolutions if r.get("entity") == e), 0)
            for e in entities
        ]
        result.confidence = sum(all_confs) / len(all_confs) if all_confs else 0.0

    except Exception as exc:
        logger.warning("semantic.resolve.llm_failed", error=str(exc))
        result.reasoning = f"LLM resolution failed: {exc}"
        result.confidence = 0.3

    _build_filter_hints(result, semantic_context)
    return result


def _build_filter_hints(result: SemanticResolution, ctx: SourceSemanticContext | None) -> None:
    """Generate SQL WHERE fragments from resolved entities."""
    if not result.entity_map:
        return

    # Group resolved entities by which column they likely belong to
    if ctx and ctx.abbreviation_maps:
        for col, abbrev_map in ctx.abbreviation_maps.items():
            matched_values = []
            for entity, canonical in result.entity_map.items():
                if canonical in abbrev_map:
                    matched_values.append(canonical)
            if matched_values:
                vals_str = ", ".join(f"'{v}'" for v in matched_values)
                result.filter_hints.append(f"{col} IN ({vals_str})")

    # Also build synonyms map
    for entity, canonical in result.entity_map.items():
        if entity.lower() != canonical.lower():
            result.synonyms[entity] = canonical


# ═══════════════════════════════════════════════════════════════════════════════
# SEMANTIC REASONING AGENT (v2.0 — replaces lookup-based resolution)
# ═══════════════════════════════════════════════════════════════════════════════
#
# Single LLM call with full schema context. The LLM REASONS about what the
# user means in the context of the actual data — no dictionaries, no rules,
# no confidence thresholds, no alias maps as primary resolution.
#
# Previously learned aliases are included as FEW-SHOT CONTEXT, not lookups.
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class SemanticReasoning:
    """Output of the Semantic Reasoning Agent — structured query interpretation."""

    resolved_query: str = ""
    """The user's question rewritten in precise data terms."""

    filters: list[dict] = field(default_factory=list)
    """[{column, operator, value, table, reasoning}] — exact SQL WHERE conditions."""

    metrics: list[str] = field(default_factory=list)
    """Column names to SELECT/aggregate."""

    tables: list[str] = field(default_factory=list)
    """Which tables to query."""

    group_by: list[str] = field(default_factory=list)
    """Columns to GROUP BY."""

    reasoning: str = ""
    """How each user term was mapped to data attributes."""

    confidence: float = 0.0
    """Overall confidence in the interpretation (0.0–1.0)."""

    new_aliases: dict[str, str] = field(default_factory=dict)
    """New term→value mappings discovered, to save for future queries."""

    def to_sql_context(self) -> str:
        """Format as structured context for the SQL Agent prompt."""
        lines = []
        if self.resolved_query:
            lines.append(f"Interpreted question: {self.resolved_query}")
        if self.filters:
            lines.append("Required SQL filters (use these EXACT values):")
            for f in self.filters:
                lines.append(f"  WHERE {f['table']}.{f['column']} {f.get('operator','=')} '{f['value']}'")
        if self.metrics:
            lines.append(f"Key columns to include: {', '.join(self.metrics)}")
        if self.tables:
            lines.append(f"Primary tables: {', '.join(self.tables)}")
        if self.group_by:
            lines.append(f"Group by: {', '.join(self.group_by)}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "resolved_query": self.resolved_query,
            "filters": self.filters,
            "metrics": self.metrics,
            "tables": self.tables,
            "group_by": self.group_by,
            "reasoning": self.reasoning,
            "confidence": self.confidence,
            "new_aliases": self.new_aliases,
        }


async def _schema_search(connectors: dict, column: str, search_term: str, table_hint: str = "") -> list[dict]:
    """Ask the Schema Agent: search a column for a value across all tables.

    This is how the Semantic Agent interacts with the Schema Agent —
    targeted searches for unresolved entities.
    """
    results = []
    import pandas as _pd
    for sid, conn in connectors.items():
        try:
            snap = await conn.introspect()
            for tbl in snap.tables:
                if table_hint and table_hint.lower() not in tbl.name.lower():
                    continue
                for col in tbl.columns:
                    if col.name.lower() == column.lower():
                        try:
                            res = await conn.execute(
                                f"SELECT DISTINCT \"{col.name}\" FROM \"{tbl.name}\" "
                                f"WHERE LOWER(\"{col.name}\") LIKE LOWER('%{search_term}%') LIMIT 10"
                            )
                            if isinstance(res, _pd.DataFrame) and not res.empty:
                                for v in res.iloc[:, 0].dropna().tolist():
                                    results.append({
                                        "value": str(v),
                                        "table": tbl.name,
                                        "column": col.name,
                                        "source_id": sid,
                                    })
                        except Exception:
                            pass
        except Exception:
            pass
    return results


async def _schema_check_join(connectors: dict, table_a: str, table_b: str) -> dict:
    """Ask the Schema Agent: can these two tables be joined? On what key?"""
    result = {"joinable": False, "join_column": "", "reasoning": ""}
    try:
        cols_a = set()
        cols_b = set()
        for sid, conn in connectors.items():
            snap = await conn.introspect()
            for tbl in snap.tables:
                if tbl.name.lower() == table_a.lower():
                    cols_a = {c.name.lower() for c in tbl.columns}
                if tbl.name.lower() == table_b.lower():
                    cols_b = {c.name.lower() for c in tbl.columns}
        shared = cols_a & cols_b
        # Prefer dimension-like join keys
        for candidate in ["country", "iso_code", "region", "id", "customer_id", "industry"]:
            if candidate in shared:
                result = {"joinable": True, "join_column": candidate,
                          "reasoning": f"Shared column '{candidate}' in both tables"}
                return result
        if shared:
            col = next(iter(shared))
            result = {"joinable": True, "join_column": col,
                      "reasoning": f"Shared column '{col}' in both tables"}
    except Exception:
        pass
    return result


async def reason_about_query(
    question: str,
    source_ids: list[str],
    workspace_id: str,
    connectors: dict,
    llm: object,
    semantic_contexts: dict[str, SourceSemanticContext] | None = None,
) -> SemanticReasoning:
    """Semantic Reasoning Agent — iterative reasoning loop with Schema Agent interaction.

    NOT a single LLM call. The agent:
    1. Starts with pre-loaded knowledge (domain, aliases, column meanings)
    2. Makes an initial reasoning pass to identify what it knows vs doesn't
    3. For unresolved entities, ASKS the Schema Agent (targeted DB search)
    4. Checks if multi-table join is feasible via Schema Agent
    5. Returns ONLY when confident it has the full picture

    The Semantic Agent interacts with the Schema Agent through:
    - _schema_search(): find entity values across tables
    - _schema_check_join(): verify if tables can be joined
    """
    result = SemanticReasoning()

    # ── Build schema context with sample values ──────────────────────────
    schema_sections: list[str] = []

    effective_sources = source_ids if source_ids else list(connectors.keys())
    if not any(connectors.get(sid) for sid in effective_sources):
        effective_sources = list(connectors.keys())

    for sid in effective_sources:
        conn = connectors.get(sid)
        if not conn:
            continue

        try:
            snap = await conn.introspect()
        except Exception:
            continue

        for table in snap.tables[:10]:
            col_lines = []
            for col in table.columns[:25]:
                dt = col.data_type or "unknown"
                examples = getattr(col, "examples", None) or []

                # For dimension columns, sample more values from the DB
                dt_lower = dt.lower()
                is_text = any(t in dt_lower for t in ("varchar", "text", "string", "char"))

                sample_vals = [str(v) for v in examples[:8]]
                if is_text and len(sample_vals) < 5:
                    try:
                        res = await conn.execute(
                            f'SELECT DISTINCT "{col.name}" FROM "{table.name}" '
                            f'WHERE "{col.name}" IS NOT NULL LIMIT 30'
                        )
                        import pandas as _pd
                        if isinstance(res, _pd.DataFrame) and not res.empty:
                            sample_vals = [str(v) for v in res.iloc[:, 0].dropna().tolist()[:30]]
                    except Exception:
                        pass

                samples_str = f"  values: {sample_vals}" if sample_vals else ""
                col_lines.append(f"    {col.name} ({dt}){samples_str}")

            schema_sections.append(
                f"Table: {table.name}\n" + "\n".join(col_lines)
            )

    if not schema_sections:
        logger.warning(
            "semantic.reasoning.no_schema",
            question=question[:50],
            source_ids=source_ids,
            effective_sources=effective_sources,
            connectors_available=list(connectors.keys()),
        )
        result.reasoning = "No schema available for reasoning."
        return result

    logger.info(
        "semantic.reasoning.schema_built",
        tables=len(schema_sections),
        question=question[:50],
    )

    schema_text = "\n\n".join(schema_sections)

    # ── Build learned context (few-shot, not lookup) ─────────────────────
    learned_context = ""
    for sid in source_ids:
        aliases = _load_learned_aliases(workspace_id, sid)
        if aliases:
            sample_aliases = list(aliases.items())[:20]
            learned_context += "\nPreviously learned mappings:\n"
            learned_context += "\n".join(
                f"  '{k}' → '{v}'" for k, v in sample_aliases
            )

    # ── Pre-check: resolve high-confidence aliases deterministically ─────
    # Aliases with confidence >= 0.93 are treated as ground truth.
    # These are injected as pre-resolved filters so the LLM doesn't
    # have to re-discover them. Only applies to entities found in the
    # question text (case-insensitive substring match).
    pre_resolved: list[dict] = []
    question_lower = question.lower()
    for sid in (source_ids or list(connectors.keys())):
        aliases = _load_learned_aliases(workspace_id, sid)
        key = f"{workspace_id}:{sid}"
        confs = _alias_confidence.get(key, {})
        for term, canonical in aliases.items():
            if term in question_lower and confs.get(term, 0) >= 0.93:
                pre_resolved.append({
                    "term": term,
                    "canonical": canonical,
                    "confidence": confs[term],
                    "source_id": sid,
                })
    if pre_resolved:
        learned_context += "\n\nHIGH-CONFIDENCE pre-resolved entities (use these directly):\n"
        for pr in pre_resolved:
            learned_context += f"  '{pr['term']}' = '{pr['canonical']}' (confidence: {pr['confidence']:.2f})\n"
        logger.info("semantic.reasoning.pre_resolved",
                     count=len(pre_resolved),
                     terms=[p["term"] for p in pre_resolved])

    # ── Load known query patterns (auto-inject common filters) ───────────
    patterns_context = ""
    if workspace_id:
        patterns_path = os.path.join(
            os.path.expanduser("~"), ".sqlagent", "uploads", workspace_id,
            "query_patterns.json"
        )
        try:
            if os.path.exists(patterns_path):
                with open(patterns_path) as f:
                    patterns = json.load(f)
                # Only inject patterns seen 3+ times (confirmed by repeated success)
                frequent = [p for p in patterns if p.get("count", 0) >= 3]
                if frequent:
                    patterns_context = "\nKnown default filters (apply these unless the user explicitly asks otherwise):\n"
                    for p in frequent[:10]:
                        patterns_context += (
                            f"  {p['table']}.{p['column']} = '{p['value']}' "
                            f"(confirmed {p['count']} times)\n"
                        )
        except Exception:
            pass

    # ── Load column enrichments (usage context from past queries) ────────
    enrichments_context = ""
    if workspace_id:
        enrichments_path = os.path.join(
            os.path.expanduser("~"), ".sqlagent", "uploads", workspace_id,
            "column_enrichments.json"
        )
        try:
            if os.path.exists(enrichments_path):
                with open(enrichments_path) as f:
                    enrichments = json.load(f)
                if enrichments:
                    enrichments_context = "\nColumn usage insights (from past successful queries):\n"
                    for key_col, info in list(enrichments.items())[:10]:
                        used_for = info.get("used_for", [])
                        if used_for:
                            enrichments_context += (
                                f"  {key_col}: used for queries like '{used_for[0]}' "
                                f"({info.get('query_count', 0)} times)\n"
                            )
        except Exception:
            pass

    # ── Load failure log (avoid repeating past mistakes) ─────────────────
    failure_context = ""
    if workspace_id:
        resolution_log_path = os.path.join(
            os.path.expanduser("~"), ".sqlagent", "uploads", workspace_id,
            "resolution_log.json"
        )
        try:
            if os.path.exists(resolution_log_path):
                with open(resolution_log_path) as f:
                    res_log = json.load(f)
                # Show recent failures so the agent avoids them
                failures = [r for r in res_log if not r.get("succeeded", True)][-5:]
                if failures:
                    failure_context = "\nPAST FAILURES (avoid these exact resolutions — they were wrong):\n"
                    for f_entry in failures:
                        failure_context += (
                            f"  '{f_entry.get('term','')}' resolved as "
                            f"'{f_entry.get('resolved_as','')}' → FAILED "
                            f"(correct: {f_entry.get('correct_value','unknown')})\n"
                        )
        except Exception:
            pass

    # ── Build semantic analysis context ──────────────────────────────────
    domain_context = ""
    if semantic_contexts:
        for sid, ctx in semantic_contexts.items():
            if ctx.domain:
                domain_context += f"\nData domain: {ctx.domain}"
            if ctx.column_meanings:
                domain_context += "\nColumn meanings:"
                for col, meaning in list(ctx.column_meanings.items())[:15]:
                    domain_context += f"\n  {col}: {meaning}"
            if ctx.filter_tips:
                domain_context += "\nQuery tips:"
                for tip in ctx.filter_tips[:5]:
                    domain_context += f"\n  • {tip}"

    # ══════════════════════════════════════════════════════════════════════
    # ITERATIVE REASONING LOOP
    # The Semantic Agent reasons in multiple passes, interacting with
    # the Schema Agent to fill in gaps.
    # ══════════════════════════════════════════════════════════════════════

    # ── Pass 1: Initial reasoning with pre-loaded knowledge ──────────────
    prompt_pass1 = f"""\
You are a Semantic Reasoning Agent. Analyze this question and map EVERY term
to the actual database schema.

Question: "{question}"

DATABASE SCHEMA:
{schema_text}
{domain_context}
{learned_context}
{patterns_context}
{enrichments_context}
{failure_context}

INSTRUCTIONS:
1. Identify EVERY entity the user mentions. For each, determine:
   - Which column it maps to
   - The EXACT value in the database (from sample values above)
   - If you CANNOT find an exact match in the samples, mark it as "UNRESOLVED"

2. For correlation/comparison queries: identify ALL datasets needed (not just one side)

3. Identify all metrics/measurements the user wants

4. Identify which tables are needed

Return JSON:
{{
  "resolved_query": "Question rewritten with exact DB terminology",
  "filters": [
    {{"column": "col", "operator": "=", "value": "exact_db_value", "table": "tbl", "reasoning": "why"}}
  ],
  "unresolved": [
    {{"term": "True's AI", "likely_column": "Customer", "search_hint": "True",
      "reasoning": "Not in sample values — needs targeted search"}}
  ],
  "metrics": ["col1", "col2"],
  "tables": ["table1", "table2"],
  "group_by": ["col"],
  "reasoning": "Full reasoning trace",
  "confidence": 0.7,
  "new_aliases": {{"th": "Thailand"}},
  "is_multi_dataset": false
}}

Rules:
- filter values MUST be EXACT character-for-character matches from the sample values shown above
  Example: if samples show 'Viet Nam', use 'Viet Nam' — NOT 'Vietnam'
  Example: if samples show '15+', use '15+' — NOT '15 years and over'
- Look at the ACTUAL sample values in the schema. Use ONLY values you can see.
- For time ranges: convert "1990s" to 1990-1999, "2000s" to 2000-2009
- For ratio/calculation queries: identify ALL source tables needed
- For correlation queries, identify BOTH datasets (set is_multi_dataset: true)
- Mark anything you're not sure about as "unresolved" — don't guess
- Return ONLY valid JSON
"""

    try:
        resp1 = await llm.complete(
            [{"role": "user", "content": prompt_pass1}],
            temperature=0.0, max_tokens=2048, json_mode=True,
        )
        raw = resp1.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip().rstrip("```").strip()
        parsed = json.loads(raw)

        result.resolved_query = parsed.get("resolved_query", "")
        result.filters = parsed.get("filters", [])
        result.metrics = parsed.get("metrics", [])
        result.tables = parsed.get("tables", [])
        result.group_by = parsed.get("group_by", [])
        result.reasoning = parsed.get("reasoning", "")
        result.confidence = parsed.get("confidence", 0.5)
        result.new_aliases = parsed.get("new_aliases", {})
        unresolved = parsed.get("unresolved", [])

        logger.info(
            "semantic.reasoning.pass1",
            question=question[:60],
            filters=len(result.filters),
            unresolved=len(unresolved),
            tables=result.tables,
            confidence=result.confidence,
        )

        # ── Inject pre-resolved aliases as filters ────────────────────
        # High-confidence aliases resolved deterministically before the
        # LLM call are now merged into the filter set. If the LLM already
        # produced a matching filter, skip the duplicate.
        if pre_resolved:
            existing_values = {
                (f.get("value", "").lower(), f.get("column", "").lower())
                for f in result.filters
            }
            for pr in pre_resolved:
                val_lower = pr["canonical"].lower()
                # Don't duplicate if LLM already got it right
                if not any(val_lower == ev[0] for ev in existing_values):
                    result.new_aliases[pr["term"]] = pr["canonical"]
                    result.reasoning += (
                        f"\n  Pre-resolved: '{pr['term']}' → '{pr['canonical']}' "
                        f"(confidence {pr['confidence']:.2f})"
                    )

            # Remove pre-resolved terms from unresolved list
            pre_terms = {pr["term"].lower() for pr in pre_resolved}
            unresolved = [
                u for u in unresolved
                if u.get("term", "").lower() not in pre_terms
            ]

        # ── Pass 2: Resolve unresolved entities via Schema Agent ─────────
        if unresolved:
            schema_findings = []
            for item in unresolved:
                term = item.get("term", "")
                likely_col = item.get("likely_column", "")
                search_hint = item.get("search_hint", term)

                if not search_hint:
                    continue

                # Ask Schema Agent: search for this entity across all tables
                found = await _schema_search(
                    connectors, likely_col, search_hint,
                )
                if not found and likely_col:
                    # Try without column hint — search all text columns
                    for sid, conn in connectors.items():
                        try:
                            snap = await conn.introspect()
                            for tbl in snap.tables:
                                for col in tbl.columns:
                                    dt = (col.data_type or "").lower()
                                    if any(t in dt for t in ("varchar", "text", "string")):
                                        search_results = await _schema_search(
                                            connectors, col.name, search_hint, tbl.name,
                                        )
                                        found.extend(search_results)
                                        if found:
                                            break
                                if found:
                                    break
                        except Exception:
                            pass
                        if found:
                            break

                if found:
                    best = found[0]
                    schema_findings.append({
                        "term": term,
                        "resolved_value": best["value"],
                        "column": best["column"],
                        "table": best["table"],
                    })
                    # Add as a filter
                    result.filters.append({
                        "column": best["column"],
                        "operator": "=",
                        "value": best["value"],
                        "table": best["table"],
                        "reasoning": f"Schema Agent found '{best['value']}' via search for '{search_hint}'",
                    })
                    # Add to aliases
                    result.new_aliases[term.lower()] = best["value"]
                    # Add table if not already listed
                    if best["table"] not in result.tables:
                        result.tables.append(best["table"])

                    logger.info(
                        "semantic.reasoning.schema_resolved",
                        term=term, value=best["value"],
                        table=best["table"], column=best["column"],
                    )

            # ── Pass 2b: Check join feasibility for multi-table queries ──
            if len(result.tables) >= 2:
                for i, t1 in enumerate(result.tables):
                    for t2 in result.tables[i + 1:]:
                        join_info = await _schema_check_join(connectors, t1, t2)
                        if join_info["joinable"]:
                            result.reasoning += (
                                f"\n  Join: {t1} ↔ {t2} on {join_info['join_column']}"
                            )

            # ── Pass 3: Final reasoning with Schema Agent findings ───────
            if schema_findings:
                findings_text = "\n".join(
                    f"  '{f['term']}' → {f['table']}.{f['column']} = '{f['resolved_value']}'"
                    for f in schema_findings
                )
                prompt_pass3 = f"""\
The Schema Agent found these additional entities:
{findings_text}

Original question: "{question}"
Previously resolved: {json.dumps(result.filters[:5])}
Tables needed: {result.tables}

Rewrite the resolved_query incorporating ALL findings. Update confidence.

Return JSON:
{{
  "resolved_query": "Complete rewritten query with ALL entities resolved",
  "confidence": 0.9,
  "reasoning": "Updated reasoning with schema findings"
}}
Return ONLY valid JSON.
"""
                try:
                    resp3 = await llm.complete(
                        [{"role": "user", "content": prompt_pass3}],
                        temperature=0.0, max_tokens=512, json_mode=True,
                    )
                    raw3 = resp3.content.strip()
                    if raw3.startswith("```"):
                        raw3 = raw3.split("```")[1]
                        if raw3.startswith("json"):
                            raw3 = raw3[4:]
                        raw3 = raw3.strip().rstrip("```").strip()
                    parsed3 = json.loads(raw3)
                    if parsed3.get("resolved_query"):
                        result.resolved_query = parsed3["resolved_query"]
                    if parsed3.get("confidence"):
                        result.confidence = parsed3["confidence"]
                    if parsed3.get("reasoning"):
                        result.reasoning += f"\n  Final: {parsed3['reasoning']}"
                except Exception:
                    pass  # Pass 3 is enhancement — don't fail on it

        # ── Save learned aliases ─────────────────────────────────────────
        if result.new_aliases:
            for sid in (source_ids or list(connectors.keys())):
                aliases = _load_learned_aliases(workspace_id, sid)
                new_confs: dict[str, float] = {}
                for term, value in result.new_aliases.items():
                    term_lower = term.lower().strip()
                    if term_lower and term_lower not in aliases:
                        aliases[term_lower] = value
                        new_confs[term_lower] = result.confidence
                if new_confs:
                    _save_learned_aliases(workspace_id, sid, aliases, new_confs)

        logger.info(
            "semantic.reasoning.completed",
            question=question[:80],
            filters=len(result.filters),
            metrics=len(result.metrics),
            tables=result.tables,
            confidence=result.confidence,
            new_aliases=len(result.new_aliases),
            passes="1" if not unresolved else f"1+2{'+3' if unresolved else ''}",
        )

    except Exception as exc:
        logger.warning("semantic.reasoning.failed", error=str(exc))
        result.reasoning = f"Reasoning failed: {exc}"
        result.confidence = 0.3

    # ── Save resolution log entry ────────────────────────────────────────
    # Track every resolution attempt for reflection and debugging.
    if workspace_id and result.filters:
        try:
            _save_resolution_log(
                workspace_id, question, result.filters,
                result.confidence, result.new_aliases,
            )
        except Exception:
            pass  # Never fail a query over logging

    return result
