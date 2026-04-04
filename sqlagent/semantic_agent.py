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


# Persisted learned aliases — source_id → {user_term_lower → canonical_value}
_learned_aliases: dict[str, dict[str, str]] = {}


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
                aliases = json.load(f)
            _learned_aliases[key] = aliases
            return aliases
    except Exception:
        pass
    _learned_aliases[key] = {}
    return _learned_aliases[key]


def _save_learned_aliases(workspace_id: str, source_id: str, aliases: dict[str, str]) -> None:
    """Persist learned aliases to disk."""
    key = f"{workspace_id}:{source_id}"
    _learned_aliases[key] = aliases
    try:
        base = os.path.join(
            os.path.expanduser("~"), ".sqlagent", "uploads", workspace_id
        )
        os.makedirs(base, exist_ok=True)
        path = os.path.join(base, f"aliases_{source_id}.json")
        with open(path, "w") as f:
            json.dump(aliases, f, indent=2)
    except Exception as exc:
        logger.warning("semantic.save_aliases_failed", error=str(exc))


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

                # Save to learned aliases for next time
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

        # Persist learned aliases
        if any(v == "llm" for v in result.resolved_from.values()):
            _save_learned_aliases(workspace_id, source_id, learned)

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
