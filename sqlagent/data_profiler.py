"""DuckDB + LLM data profiler.

One LLM call at upload time generates the optimal read_csv_auto() / read_json_auto()
argument set for this specific file. The result is stored and reused on every query
— zero LLM cost at query time, zero custom parsing code.

Flow:
  upload → _read_raw_sample() → _compute_stats() → llm.complete() → _parse_config()
         → FileConnector stores ReadConfig → _ingest_file() uses it

For Excel / Parquet files the profiler is skipped (those formats carry type metadata
natively and DuckDB / pandas handle them well already).
"""

from __future__ import annotations

import csv
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any

import structlog

logger = structlog.get_logger()

# ── Data classes ──────────────────────────────────────────────────────────────


@dataclass
class ColumnProfile:
    name: str
    raw_dtype: str  # DuckDB's initial guess before profiling
    null_pct: float  # fraction of rows that are null / empty
    sample_values: list  # up to 10 non-null raw string values
    found_null_strings: list = field(
        default_factory=list
    )  # actual null-encoded strings seen in this column
    looks_numeric: bool = False
    looks_date: bool = False
    looks_boolean: bool = False
    currency_prefix: str = ""  # "$", "€", "£", "" etc.


@dataclass
class ReadConfig:
    """Stored per-file. Contains the DuckDB arguments that load this file cleanly."""

    source_id: str
    file_path: str

    # Arguments to pass to read_csv_auto() or read_json_auto()
    duckdb_args: dict[str, Any] = field(default_factory=dict)

    # Post-load CAST expressions: {column_name: "CAST(col AS DOUBLE)"} etc.
    # Applied as a SELECT projection on top of the raw read.
    cast_exprs: dict[str, str] = field(default_factory=dict)

    # Null-encoding strings found in this file beyond DuckDB's defaults.
    extra_null_strings: list[str] = field(default_factory=list)

    llm_model: str = ""
    profiled_rows: int = 0

    def build_read_sql(self, table_name: str) -> str:
        """Return the CREATE TABLE … AS SELECT … FROM read_csv_auto(…) statement."""
        ext = os.path.splitext(self.file_path)[1].lower()
        escaped = self.file_path.replace("'", "''")

        # Build keyword arg string
        args: list[str] = [f"'{escaped}'"]

        if self.extra_null_strings:
            null_list = ", ".join(f"'{s}'" for s in self.extra_null_strings)
            # Merge with DuckDB's built-in null strings
            args.append(f"nullstr=[{null_list}]")

        args.append("normalize_names=true")
        args.append("null_padding=true")
        args.append("ignore_errors=true")

        for k, v in self.duckdb_args.items():
            if isinstance(v, bool):
                args.append(f"{k}={'true' if v else 'false'}")
            elif isinstance(v, str):
                args.append(f"{k}='{v}'")
            elif isinstance(v, dict):
                # e.g. types={'salary': 'DOUBLE'}
                inner = ", ".join(f"'{ck}': '{cv}'" for ck, cv in v.items())
                args.append(f"{k}={{{inner}}}")
            else:
                args.append(f"{k}={v}")

        arg_str = ", ".join(args)

        if ext in (".csv", ".tsv", ".txt"):
            reader = f"read_csv_auto({arg_str})"
        elif ext == ".json":
            reader = f"read_json_auto('{escaped}', ignore_errors=true)"
        else:
            reader = f"read_csv_auto({arg_str})"

        # Build SELECT with any CAST expressions
        if self.cast_exprs:
            # We can't know all columns at SQL-build time (before the table exists).
            # Emit a raw SELECT * wrapped in a second pass instead.
            # The CAST exprs are applied after initial load via ALTER / SELECT.
            select = "SELECT * FROM raw_load"
        else:
            select = f"SELECT * FROM {reader}"

        if not self.cast_exprs:
            return f'CREATE TABLE "{table_name}" AS {select}'

        # Two-pass: load raw, then select with casts
        cast_cols = ", ".join(
            f'{expr} AS "{col}"' if expr != col else f'"{col}"'
            for col, expr in self.cast_exprs.items()
        )
        return (
            f"CREATE TEMP VIEW raw_load AS SELECT * FROM {reader};\n"
            f'CREATE TABLE "{table_name}" AS SELECT *, {cast_cols} FROM raw_load'
        )


# ── Sampling helpers ──────────────────────────────────────────────────────────

_MAX_SAMPLE_ROWS = 80
_MAX_SAMPLE_BYTES = 16_000  # stay well inside a typical 8k-token context window


def _read_raw_sample(file_path: str) -> tuple[str, list[str], int]:
    """Return (raw_text, header_columns, row_count_sampled) from a CSV/TSV file."""
    ext = os.path.splitext(file_path)[1].lower()
    delim = "\t" if ext == ".tsv" else ","

    lines: list[str] = []
    headers: list[str] = []

    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as fh:
            reader = csv.reader(fh, delimiter=delim)
            for i, row in enumerate(reader):
                if i == 0:
                    headers = row
                line = delim.join(row)
                lines.append(line)
                if i >= _MAX_SAMPLE_ROWS:
                    break
                if sum(len(ln) for ln in lines) > _MAX_SAMPLE_BYTES:
                    break
    except Exception as exc:
        logger.warning("profiler.sample_failed", error=str(exc))
        return "", [], 0

    raw = "\n".join(lines)
    return raw, headers, len(lines) - 1  # -1 for header


def _compute_stats(file_path: str, headers: list[str], raw: str) -> list[ColumnProfile]:
    """Quick in-Python column stats — no LLM, no DuckDB needed."""
    if not headers or not raw:
        return []

    ext = os.path.splitext(file_path)[1].lower()
    delim = "\t" if ext == ".tsv" else ","

    # Parse rows
    rows: list[list[str]] = []
    for line in raw.split("\n")[1:]:  # skip header
        if line.strip():
            rows.append(next(csv.reader([line], delimiter=delim), []))

    if not rows:
        return []

    profiles: list[ColumnProfile] = []
    for col_idx, col_name in enumerate(headers):
        values = [r[col_idx].strip() if col_idx < len(r) else "" for r in rows]

        null_indicators = {"", "n/a", "na", "null", "none", "-", "--", "nan", "#n/a", "nil"}
        non_null = [v for v in values if v.lower() not in null_indicators]
        null_pct = 1.0 - len(non_null) / max(len(values), 1)

        sample_values = non_null[:10]

        # Track which non-default null-encoding strings were actually seen in this column.
        # DuckDB already handles "", "NULL", "NA", "\N" — only flag extras.
        duckdb_defaults = {"", "null", "na", "\\n"}
        found_null_strings = list(
            {
                v
                for v in values
                if v.lower() in null_indicators and v.lower() not in duckdb_defaults and v != ""
            }
        )

        # Heuristics
        looks_numeric = False
        currency_prefix = ""
        if non_null:
            cleaned = []
            for v in non_null[:20]:
                m = re.match(r"^([€$£¥₹])\s*", v)
                if m:
                    currency_prefix = m.group(1)
                    v = v[m.end() :]
                cleaned.append(re.sub(r"[,_\s]", "", v))
            numeric_count = sum(
                1 for v in cleaned if re.match(r"^-?\d+(\.\d+)?([eE][+-]?\d+)?$", v)
            )
            looks_numeric = numeric_count / max(len(cleaned), 1) >= 0.80

        looks_date = False
        if non_null and not looks_numeric:
            date_patterns = [
                r"^\d{4}-\d{2}-\d{2}",  # ISO
                r"^\d{1,2}/\d{1,2}/\d{2,4}",  # US/EU
                r"^\d{1,2}-\d{1,2}-\d{2,4}",
                r"^\d{4}/\d{2}/\d{2}",
                r"^\w{3}\s+\d{1,2},?\s+\d{4}",  # Jan 1, 2024
            ]
            date_count = sum(1 for v in non_null[:20] if any(re.match(p, v) for p in date_patterns))
            looks_date = date_count / max(len(non_null[:20]), 1) >= 0.70

        looks_boolean = False
        if non_null:
            bool_vals = {"yes", "no", "true", "false", "y", "n", "1", "0", "on", "off"}
            bool_count = sum(1 for v in non_null[:20] if v.lower() in bool_vals)
            looks_boolean = bool_count / max(len(non_null[:20]), 1) >= 0.90

        profiles.append(
            ColumnProfile(
                name=col_name,
                raw_dtype="VARCHAR",
                null_pct=round(null_pct, 3),
                sample_values=sample_values,
                found_null_strings=found_null_strings,
                looks_numeric=looks_numeric,
                looks_date=looks_date,
                looks_boolean=looks_boolean,
                currency_prefix=currency_prefix,
            )
        )

    return profiles


# ── LLM prompt ────────────────────────────────────────────────────────────────

_SYSTEM = """\
You are a DuckDB data engineer. You will receive a CSV file sample and column statistics.
Your job is to return a JSON object that describes how to load this file cleanly using
DuckDB's read_csv_auto() function.

Rules:
- Output ONLY valid JSON, no markdown fences, no explanation.
- Detect null-encoding strings (common ones: "N/A", "NULL", "-", "--", "nan", "none", "#N/A").
  Return them in extra_null_strings (only strings NOT already handled by DuckDB by default).
  DuckDB default null strings: "", "NULL", "NA", "\\N" — do NOT include these.
- For date columns, detect the format string (strftime style, e.g. "%d/%m/%Y").
  Return in duckdb_args.dateformat if a non-ISO format is detected.
- For currency columns (values like "$1,234.56"), add a cast expression that strips the
  symbol and comma then casts to DOUBLE.
- For columns that look like they store delimited values (e.g. ";" or "|" as secondary
  delimiter in a cell value), note them in notes only — do not break the load.
- If the file delimiter is NOT a comma, set duckdb_args.delim.
- cast_exprs maps column_name → SQL expression (use the normalized column name with
  spaces replaced by underscores and lowercased, matching normalize_names=true behavior).

Output schema (all fields optional except source_id):
{
  "extra_null_strings": ["N/A", "na"],
  "duckdb_args": {
    "delim": ";",
    "dateformat": "%d/%m/%Y",
    "types": {"employment_rate": "DOUBLE"}
  },
  "cast_exprs": {
    "salary": "CAST(REPLACE(REPLACE(salary, '$', ''), ',', '') AS DOUBLE)"
  },
  "notes": "brief free text — not used programmatically"
}
"""

_USER_TMPL = """\
FILE: {filename}
SAMPLE ({n_rows} rows):
{sample}

COLUMN STATISTICS:
{stats}
"""


def _build_stats_text(profiles: list[ColumnProfile]) -> str:
    lines = []
    for p in profiles:
        flags = []
        if p.looks_numeric:
            flags.append("numeric")
        if p.looks_date:
            flags.append("date")
        if p.looks_boolean:
            flags.append("boolean")
        if p.currency_prefix:
            flags.append(f"currency:{p.currency_prefix}")
        if p.null_pct > 0.05:
            flags.append(f"null:{p.null_pct:.0%}")
        flag_str = ", ".join(flags) if flags else "text"
        sample_str = " | ".join(str(v) for v in p.sample_values[:5])
        lines.append(f"  {p.name!r:30s} [{flag_str}]  samples: {sample_str}")
    return "\n".join(lines)


def _parse_llm_response(raw: str, source_id: str, file_path: str, n_rows: int) -> ReadConfig:
    """Parse LLM JSON response into a ReadConfig. Tolerant of junk output."""
    # Strip markdown fences if the LLM added them
    text = re.sub(r"```(?:json)?|```", "", raw).strip()

    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        # Try extracting the first {...} block
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                obj = json.loads(m.group())
            except json.JSONDecodeError:
                logger.warning("profiler.parse_failed", raw=text[:200])
                obj = {}
        else:
            obj = {}

    return ReadConfig(
        source_id=source_id,
        file_path=file_path,
        duckdb_args=obj.get("duckdb_args", {}),
        cast_exprs=obj.get("cast_exprs", {}),
        extra_null_strings=obj.get("extra_null_strings", []),
        profiled_rows=n_rows,
    )


# ── Public API ────────────────────────────────────────────────────────────────


async def profile_file(
    file_path: str,
    source_id: str,
    llm=None,  # LLMProvider — optional; if None, returns a sensible default config
    model: str = "",
) -> ReadConfig:
    """Profile a CSV/TSV file and return a ReadConfig with optimal DuckDB load args.

    If `llm` is None (no API key configured), returns a ReadConfig with safe defaults
    (normalize_names, null_padding, ignore_errors) — better than nothing.

    For Excel / Parquet, returns an empty ReadConfig immediately (those formats carry
    type metadata natively).
    """
    ext = os.path.splitext(file_path)[1].lower()

    # Non-CSV formats: skip profiling, return empty config
    if ext not in (".csv", ".tsv", ".txt", ".json"):
        logger.debug("profiler.skipped", file=file_path, reason="non-csv format")
        return ReadConfig(source_id=source_id, file_path=file_path)

    # Read raw sample
    raw, headers, n_rows = _read_raw_sample(file_path)
    if not raw or not headers:
        logger.warning("profiler.empty_sample", file=file_path)
        return ReadConfig(source_id=source_id, file_path=file_path)

    # Compute stats (free — no LLM)
    profiles = _compute_stats(file_path, headers, raw)

    # If no LLM available, use heuristics only
    if llm is None:
        return _heuristic_config(source_id, file_path, profiles, n_rows)

    # Build prompt
    stats_text = _build_stats_text(profiles)
    user_msg = _USER_TMPL.format(
        filename=os.path.basename(file_path),
        n_rows=n_rows,
        sample=raw[:_MAX_SAMPLE_BYTES],
        stats=stats_text,
    )

    try:
        resp = await llm.complete(
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.0,
            max_tokens=512,
        )
        config = _parse_llm_response(resp.content, source_id, file_path, n_rows)
        config.llm_model = model or getattr(resp, "model", "")
        logger.info(
            "profiler.done",
            file=os.path.basename(file_path),
            n_rows=n_rows,
            null_strings=config.extra_null_strings,
            duckdb_args=config.duckdb_args,
            cast_exprs=list(config.cast_exprs.keys()),
        )
        return config
    except Exception as exc:
        logger.warning("profiler.llm_failed", error=str(exc), file=file_path)
        # Fall back to heuristics — never block the upload
        return _heuristic_config(source_id, file_path, profiles, n_rows)


def _heuristic_config(
    source_id: str,
    file_path: str,
    profiles: list[ColumnProfile],
    n_rows: int,
) -> ReadConfig:
    """Build a ReadConfig from pure column-stat heuristics, no LLM."""
    duckdb_args: dict[str, Any] = {}
    cast_exprs: dict[str, str] = {}

    # Detect non-comma delimiter from extension
    if file_path.endswith(".tsv"):
        duckdb_args["delim"] = "\t"

    # Build CAST expressions for currency columns
    for p in profiles:
        if p.currency_prefix and p.looks_numeric:
            safe_col = re.sub(r"[^a-z0-9_]", "_", p.name.lower()).strip("_")
            cast_exprs[safe_col] = (
                f"CAST(REPLACE(REPLACE({safe_col}, '{p.currency_prefix}', ''), ',', '') AS DOUBLE)"
            )

    # Collect extra null-encoding strings seen across all columns (deduped, preserving case)
    seen: dict[str, str] = {}  # lower → original (first occurrence wins)
    for p in profiles:
        for s in p.found_null_strings:
            seen.setdefault(s.lower(), s)
    extra_nulls = list(seen.values())

    return ReadConfig(
        source_id=source_id,
        file_path=file_path,
        duckdb_args=duckdb_args,
        cast_exprs=cast_exprs,
        extra_null_strings=extra_nulls,
        profiled_rows=n_rows,
    )
