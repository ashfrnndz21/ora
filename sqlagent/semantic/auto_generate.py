"""Auto-generate OraSpec semantic model from database schema introspection.

Fully agentic: uses LLM to classify columns (dimension/measure/time),
generate descriptions, and detect relationships — no keyword lists or heuristics.
"""

from __future__ import annotations

from typing import Any

import structlog

from sqlagent.semantic.model import (
    AggregationType,
    DataType,
    Dimension,
    LogicalTable,
    Measure,
    Metric,
    OraSpec,
    Relationship,
    TimeDimension,
    TimeGrain,
)

logger = structlog.get_logger()

# Map SQL type strings to OraSpec DataType (structural, not heuristic)
SQL_TYPE_MAP: dict[str, DataType] = {
    "varchar": DataType.STRING,
    "text": DataType.STRING,
    "char": DataType.STRING,
    "nvarchar": DataType.STRING,
    "string": DataType.STRING,
    "int": DataType.INTEGER,
    "integer": DataType.INTEGER,
    "bigint": DataType.INTEGER,
    "smallint": DataType.INTEGER,
    "tinyint": DataType.INTEGER,
    "float": DataType.FLOAT,
    "double": DataType.FLOAT,
    "real": DataType.FLOAT,
    "decimal": DataType.DECIMAL,
    "numeric": DataType.DECIMAL,
    "number": DataType.DECIMAL,
    "money": DataType.DECIMAL,
    "boolean": DataType.BOOLEAN,
    "bool": DataType.BOOLEAN,
    "date": DataType.DATE,
    "datetime": DataType.TIMESTAMP,
    "timestamp": DataType.TIMESTAMP,
    "timestamptz": DataType.TIMESTAMP,
    "timestamp_ntz": DataType.TIMESTAMP,
    "json": DataType.JSON,
    "jsonb": DataType.JSON,
    "variant": DataType.JSON,
}


def _map_type(sql_type: str) -> DataType:
    """Map a SQL type string to OraSpec DataType."""
    type_lower = sql_type.lower().split("(")[0].strip()
    return SQL_TYPE_MAP.get(type_lower, DataType.STRING)


def _humanize(name: str) -> str:
    """Convert snake_case/camelCase to Title Case."""
    import re

    name = re.sub(r"([a-z])([A-Z])", r"\1 \2", name)
    name = name.replace("_", " ").replace("-", " ")
    return name.title().strip()


async def auto_generate_oraspec(
    connector: Any,
    llm: Any,
    spec_name: str = "auto_generated",
    description: str = "Auto-generated semantic model",
    sample_values_limit: int = 10,
    include_tables: list[str] | None = None,
    exclude_tables: list[str] | None = None,
    backend: str | None = None,
) -> OraSpec:
    """Introspect a database and generate a starter OraSpec using LLM classification.

    The LLM classifies each column as dimension/measure/time_dimension,
    generates business descriptions, suggests synonyms, and detects relationships.
    No keyword lists or heuristic rules.

    Args:
        connector: A sqlagent database connector with introspect() method
        llm: LLM provider for agentic classification
        spec_name: Name for the generated spec
        description: Description for the spec
        sample_values_limit: Max sample values per string dimension
        include_tables: Whitelist of tables (None = all)
        exclude_tables: Blacklist of tables (None = none)
        backend: Backend identifier for federation ("postgres", "snowflake", etc.)

    Returns:
        OraSpec with LLM-classified dimensions, time_dimensions, measures,
        descriptions, synonyms, and detected relationships.
    """
    exclude = {t.lower() for t in (exclude_tables or [])}

    # Get schema snapshot from connector
    snap = await connector.introspect()

    tables: list[LogicalTable] = []
    relationships: list[Relationship] = []

    for table_info in snap.tables:
        table_name = table_info.name
        if table_name.lower() in exclude:
            continue
        if include_tables and table_name.lower() not in {t.lower() for t in include_tables}:
            continue

        # Build column summary for LLM
        columns_desc = []
        for col in table_info.columns:
            sample_vals = getattr(col, "sample_values", []) or []
            sample_str = f", samples: {sample_vals[:5]}" if sample_vals else ""
            columns_desc.append(f"  - {col.name} ({col.data_type}{sample_str})")

        columns_text = "\n".join(columns_desc)

        # LLM classifies all columns + generates descriptions in one call
        classification_prompt = (
            f"You are analyzing a database table for a semantic data model.\n\n"
            f"Table: {table_name}\nColumns:\n{columns_text}\n\n"
            f"For each column, classify it as one of:\n"
            f'  - "dimension" (categorical/text used for grouping/filtering)\n'
            f'  - "measure" (numeric values that should be aggregated: SUM, AVG, COUNT)\n'
            f'  - "time_dimension" (dates, timestamps, temporal columns)\n\n'
            f"Also provide:\n"
            f'  - A short business description (1 sentence)\n'
            f'  - 2-3 synonyms a business user might use\n'
            f'  - For measures: the default aggregation (sum/count/avg/count_distinct/min/max)\n\n'
            f"Respond as JSON array:\n"
            f'[{{"name":"col_name","classification":"dimension|measure|time_dimension",'
            f'"description":"...","synonyms":["..."],"aggregation":"sum"}}]\n\n'
            f"JSON only, no markdown."
        )

        try:
            resp = await llm.complete(
                [{"role": "user", "content": classification_prompt}],
                temperature=0.0,
                max_tokens=2048,
                json_mode=True,
            )

            import json

            # Parse LLM response
            content = resp.content.strip()
            # Handle markdown code blocks
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            classifications = json.loads(content)
        except Exception as e:
            logger.warning(
                "auto_generate.llm_classify_failed",
                table=table_name,
                error=str(e),
            )
            # Fallback: classify by SQL type only (structural, not heuristic)
            classifications = []
            for col in table_info.columns:
                dt = _map_type(col.data_type)
                if dt in (DataType.DATE, DataType.TIMESTAMP):
                    cls = "time_dimension"
                elif dt in (DataType.INTEGER, DataType.FLOAT, DataType.DECIMAL):
                    cls = "measure"
                else:
                    cls = "dimension"
                classifications.append(
                    {
                        "name": col.name,
                        "classification": cls,
                        "description": f"{_humanize(col.name)}",
                        "synonyms": [],
                        "aggregation": "sum",
                    }
                )

        # Build column models from LLM classifications
        dimensions: list[Dimension] = []
        time_dimensions: list[TimeDimension] = []
        measures: list[Measure] = []

        col_type_map = {col.name.lower(): col.data_type for col in table_info.columns}
        col_samples_map = {
            col.name.lower(): (getattr(col, "sample_values", None) or [])
            for col in table_info.columns
        }

        for item in classifications:
            col_name = item.get("name", "")
            cls = item.get("classification", "dimension")
            desc = item.get("description", _humanize(col_name))
            syns = item.get("synonyms", [])
            agg = item.get("aggregation", "sum")

            sql_type = col_type_map.get(col_name.lower(), "string")

            if cls == "time_dimension":
                time_dimensions.append(
                    TimeDimension(
                        name=col_name,
                        display_name=_humanize(col_name),
                        description=desc,
                        expr=col_name,
                        time_grain=TimeGrain.DAY,
                        synonyms=syns,
                        source="auto",
                        confidence=0.7,
                    )
                )
            elif cls == "measure":
                agg_map = {
                    "sum": AggregationType.SUM,
                    "count": AggregationType.COUNT,
                    "count_distinct": AggregationType.COUNT_DISTINCT,
                    "avg": AggregationType.AVG,
                    "min": AggregationType.MIN,
                    "max": AggregationType.MAX,
                }
                measures.append(
                    Measure(
                        name=col_name,
                        display_name=_humanize(col_name),
                        description=desc,
                        expr=col_name,
                        data_type=_map_type(sql_type),
                        aggregation=agg_map.get(agg, AggregationType.SUM),
                        synonyms=syns,
                        source="auto",
                        confidence=0.7,
                    )
                )
            else:
                samples = col_samples_map.get(col_name.lower(), [])[:sample_values_limit]
                dimensions.append(
                    Dimension(
                        name=col_name,
                        display_name=_humanize(col_name),
                        description=desc,
                        expr=col_name,
                        data_type=_map_type(sql_type),
                        synonyms=syns,
                        sample_values=[str(v) for v in samples],
                        source="auto",
                        confidence=0.7,
                    )
                )

        # Generate table-level description via LLM
        table_desc = f"Table: {_humanize(table_name)}"
        try:
            desc_resp = await llm.complete(
                [
                    {
                        "role": "user",
                        "content": (
                            f"Write a one-sentence business description for this database table.\n\n"
                            f"Table: {table_name}\n"
                            f"Columns: {', '.join(c.name for c in table_info.columns)}\n\n"
                            f"One sentence only, no quotes."
                        ),
                    }
                ],
                temperature=0.0,
                max_tokens=100,
            )
            table_desc = desc_resp.content.strip().strip('"').strip("'")
        except Exception:
            pass

        tables.append(
            LogicalTable(
                name=table_name,
                display_name=_humanize(table_name),
                description=table_desc,
                table=table_name,
                backend=backend,
                dimensions=dimensions,
                time_dimensions=time_dimensions,
                measures=measures,
                source="auto",
                confidence=0.7,
            )
        )

        # Auto-detect FK relationships from schema metadata
        for col in table_info.columns:
            fk = getattr(col, "foreign_key", None)
            if fk and isinstance(fk, dict):
                relationships.append(
                    Relationship(
                        from_table=table_name,
                        from_column=col.name,
                        to_table=fk.get("table", ""),
                        to_column=fk.get("column", ""),
                        join_type="inner",
                        relationship_type="many_to_one",
                        source="fk",
                        confidence=0.95,
                        confirmed=True,
                    )
                )

    # LLM-assisted relationship inference for tables without FK
    if len(tables) > 1 and llm:
        try:
            table_cols_summary = "\n".join(
                f"  {t.name}: [{', '.join(t.all_column_names())}]" for t in tables
            )
            rel_prompt = (
                f"Given these database tables and their columns:\n{table_cols_summary}\n\n"
                f"Known relationships (from foreign keys):\n"
                + (
                    "\n".join(
                        f"  {r.from_table}.{r.from_column} → {r.to_table}.{r.to_column}"
                        for r in relationships
                    )
                    or "  (none detected)"
                )
                + "\n\n"
                f"Suggest additional JOIN relationships that are likely based on "
                f"column names and types. Only suggest high-confidence matches.\n\n"
                f"Respond as JSON array:\n"
                f'[{{"from_table":"...","from_column":"...","to_table":"...","to_column":"...",'
                f'"join_type":"inner","confidence":0.8}}]\n\n'
                f"JSON only. Empty array [] if no additional relationships detected."
            )
            rel_resp = await llm.complete(
                [{"role": "user", "content": rel_prompt}],
                temperature=0.0,
                max_tokens=1024,
                json_mode=True,
            )
            import json

            content = rel_resp.content.strip()
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            inferred_rels = json.loads(content)
            for ir in inferred_rels:
                # Don't duplicate existing FK relationships
                existing = any(
                    r.from_table.lower() == ir.get("from_table", "").lower()
                    and r.from_column.lower() == ir.get("from_column", "").lower()
                    and r.to_table.lower() == ir.get("to_table", "").lower()
                    for r in relationships
                )
                if not existing:
                    relationships.append(
                        Relationship(
                            from_table=ir.get("from_table", ""),
                            from_column=ir.get("from_column", ""),
                            to_table=ir.get("to_table", ""),
                            to_column=ir.get("to_column", ""),
                            join_type=ir.get("join_type", "inner"),
                            source="inferred",
                            confidence=ir.get("confidence", 0.6),
                            confirmed=False,
                        )
                    )
        except Exception as e:
            logger.warning("auto_generate.relationship_inference_failed", error=str(e))

    spec = OraSpec(
        name=spec_name,
        description=description,
        tables=tables,
        relationships=relationships,
    )

    stats = spec.stats()
    logger.info(
        "auto_generate.completed",
        name=spec_name,
        tables=stats["tables"],
        dimensions=stats["dimensions"],
        measures=stats["measures"],
        relationships=stats["relationships"],
        avg_confidence=stats["avg_confidence"],
    )

    return spec
