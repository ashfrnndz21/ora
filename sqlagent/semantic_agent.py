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
