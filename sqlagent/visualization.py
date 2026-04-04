"""Auto-Visualization Engine — LLM-selected chart for every query result.

The LLM analyzes the DataFrame shape, column types, cardinality, and the
original question to select the most appropriate chart type and generate
a Vega-Lite specification.

No heuristic rules — the LLM reasons about what visualization best
communicates the answer.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import structlog

logger = structlog.get_logger()


@dataclass
class ChartSpec:
    """Auto-generated chart specification."""

    chart_type: str = "table"  # "bar", "line", "scatter", "pie", "kpi", "table"
    vega_lite: dict = field(default_factory=dict)  # Full Vega-Lite JSON spec
    title: str = ""
    description: str = ""  # "Revenue by region, ordered descending"
    reasoning: str = ""  # Why this chart type was selected

    def to_dict(self) -> dict:
        return {
            "chart_type": self.chart_type,
            "vega_lite": self.vega_lite,
            "title": self.title,
            "description": self.description,
            "reasoning": self.reasoning,
        }


async def generate_chart(
    question: str,
    sql: str,
    columns: list[str],
    rows: list[dict],
    row_count: int,
    llm: object,
) -> ChartSpec:
    """Generate a Vega-Lite chart spec using LLM reasoning.

    The LLM sees the data shape and picks the best visualization.

    Args:
        question: Original user question
        sql: SQL that was executed
        columns: Column names in the result
        rows: First few rows of data (for type inference)
        row_count: Total rows
        llm: LLM provider

    Returns:
        ChartSpec with Vega-Lite JSON
    """
    if row_count == 0 or not columns:
        return ChartSpec(chart_type="table", reasoning="No data to visualize")

    # Build data sample for the LLM
    sample_rows = rows[:5] if rows else []
    sample_str = ""
    for r in sample_rows:
        sample_str += "  " + str({k: r.get(k) for k in columns[:6]}) + "\n"

    prompt = f"""\
You are a data visualization expert. Given a query result, generate the best chart.

QUESTION: {question}
COLUMNS: {columns}
ROW COUNT: {row_count}
SAMPLE DATA:
{sample_str}

Pick the best chart type and generate a Vega-Lite spec.

Chart selection reasoning:
- 1 categorical + 1 numeric → bar chart
- 1 time/date + 1 numeric → line chart
- 2 numeric columns → scatter plot
- 1 numeric only (1 row) → KPI / single value display
- 1 categorical + multiple numerics → grouped bar
- Parts of a whole → pie/donut (only if ≤ 8 slices)
- Too many rows (>50) or complex structure → table (no chart)

Return JSON:
{{
  "chart_type": "bar",
  "title": "Revenue by Region",
  "description": "Horizontal bar chart showing total revenue per region",
  "reasoning": "1 categorical column (region) + 1 numeric (revenue) = bar chart",
  "vega_lite": {{
    "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
    "mark": "bar",
    "encoding": {{
      "x": {{"field": "revenue", "type": "quantitative"}},
      "y": {{"field": "region", "type": "nominal", "sort": "-x"}}
    }}
  }}
}}

Rules:
- vega_lite must be valid Vega-Lite v5 spec (no "data" field — we inject that client-side)
- Use the EXACT column names from the COLUMNS list above
- For "table" type, return empty vega_lite {{}}
- Return ONLY valid JSON, no markdown
"""

    try:
        resp = await llm.complete(
            [{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=1024,
            json_mode=True,
        )

        import json
        raw = resp.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip().rstrip("```").strip()

        parsed = json.loads(raw)
        result = ChartSpec(
            chart_type=parsed.get("chart_type", "table"),
            vega_lite=parsed.get("vega_lite", {}),
            title=parsed.get("title", ""),
            description=parsed.get("description", ""),
            reasoning=parsed.get("reasoning", ""),
        )

        logger.info(
            "visualization.generated",
            chart_type=result.chart_type,
            columns=len(columns),
            rows=row_count,
        )
        return result

    except Exception as exc:
        logger.warning("visualization.failed", error=str(exc))
        return ChartSpec(chart_type="table", reasoning=f"Chart generation failed: {exc}")
