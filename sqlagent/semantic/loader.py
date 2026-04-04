"""Load, validate, and write OraSpec YAML files.

Structural I/O — no LLM involved. The Semantic Agent calls these to persist
and restore its state.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from sqlagent.semantic.model import OraSpec

import structlog

logger = structlog.get_logger()


def load_oraspec(path: str | Path) -> OraSpec:
    """Load an OraSpec from a YAML file.

    Args:
        path: Path to the .yaml or .yml file

    Returns:
        Parsed and validated OraSpec

    Raises:
        FileNotFoundError: If path doesn't exist
        ValueError: If YAML is invalid or doesn't match OraSpec schema
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"OraSpec file not found: {path}")

    with open(path, "r") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"OraSpec file must contain a YAML mapping, got {type(data).__name__}")

    # Normalize YAML keys: allow both snake_case and camelCase
    data = _normalize_keys(data)

    try:
        spec = OraSpec(**data)
    except Exception as e:
        raise ValueError(f"Invalid OraSpec: {e}") from e

    logger.info(
        "oraspec.loaded",
        path=str(path),
        name=spec.name,
        tables=len(spec.tables),
    )
    return spec


def save_oraspec(spec: OraSpec, path: str | Path) -> None:
    """Write an OraSpec to a YAML file.

    Creates parent directories if needed. Uses human-friendly YAML output
    (no flow style, sorted keys disabled for readability).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = spec.model_dump(exclude_none=True, exclude_defaults=False, mode="json")
    # Clean up empty lists for readability
    data = _clean_empty(data)

    with open(path, "w") as f:
        yaml.dump(
            data,
            f,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
            width=120,
        )

    logger.info("oraspec.saved", path=str(path), name=spec.name)


def validate_oraspec(spec: OraSpec) -> list[str]:
    """Return a list of validation warnings (empty = valid).

    These are structural checks — no LLM. Checks referential integrity
    between tables, relationships, metrics, and verified queries.
    """
    warnings: list[str] = []
    table_names = {t.name.lower() for t in spec.tables}

    # Check relationships reference valid tables
    for rel in spec.relationships:
        if rel.from_table.lower() not in table_names:
            warnings.append(f"Relationship references unknown table: {rel.from_table}")
        if rel.to_table.lower() not in table_names:
            warnings.append(f"Relationship references unknown table: {rel.to_table}")

    # Check metrics reference valid measures
    measure_names: set[str] = set()
    for t in spec.tables:
        for m in t.measures:
            measure_names.add(m.name.lower())
    for metric in spec.metrics:
        for dep in metric.depends_on:
            if dep.lower() not in measure_names:
                warnings.append(f"Metric '{metric.name}' depends on unknown measure: {dep}")

    # Check for tables without descriptions (accuracy will suffer)
    for t in spec.tables:
        if not t.description:
            warnings.append(f"Table '{t.name}' has no description — accuracy will suffer")

    # Check for string dimensions without sample values (value matching impaired)
    for t in spec.tables:
        for dim in t.dimensions:
            if not dim.sample_values and dim.data_type.value == "string":
                warnings.append(
                    f"Dimension '{t.name}.{dim.name}' has no sample values — "
                    "value matching will be impaired"
                )

    # Check verified queries reference valid tables
    for vq in spec.verified_queries:
        for tbl in vq.tables_used:
            if tbl.lower() not in table_names:
                warnings.append(
                    f"Verified query references unknown table: {tbl} "
                    f"(in: '{vq.question[:50]}...')"
                )

    # Check for orphan tables (no relationships)
    if len(spec.tables) > 1:
        connected = set()
        for r in spec.relationships:
            connected.add(r.from_table.lower())
            connected.add(r.to_table.lower())
        for t in spec.tables:
            if t.name.lower() not in connected and len(spec.tables) > 1:
                warnings.append(
                    f"Table '{t.name}' has no relationships — it may be unreachable in JOINs"
                )

    return warnings


def oraspec_from_dict(data: dict) -> OraSpec:
    """Create an OraSpec from a dictionary (e.g., from JSON API body)."""
    data = _normalize_keys(data)
    return OraSpec(**data)


# ── Private helpers ──────────────────────────────────────────────────────────


def _normalize_keys(data: Any) -> Any:
    """Recursively normalize YAML keys (handle common variants)."""
    if isinstance(data, dict):
        normalized = {}
        for k, v in data.items():
            # Handle common YAML key variations
            key = k.replace("-", "_").replace(" ", "_")
            # Map common aliases
            if key == "physical_table":
                key = "table"
            elif key == "agg":
                key = "aggregation"
            elif key == "time_granularities":
                # Ignore — we use time_grain singular
                continue
            # Handle custom_instructions as multiline string → list of CustomInstruction dicts
            if key == "custom_instructions" and isinstance(v, str):
                lines = [line.strip().lstrip("- ").strip('"').strip("'")
                         for line in v.strip().splitlines() if line.strip()]
                v = [{"instruction": line, "scope": "global", "priority": 0} for line in lines]

            normalized[key] = _normalize_keys(v)
        return normalized
    elif isinstance(data, list):
        return [_normalize_keys(item) for item in data]
    return data


def _clean_empty(data: Any) -> Any:
    """Remove empty lists and None values for cleaner YAML output."""
    if isinstance(data, dict):
        cleaned = {}
        for k, v in data.items():
            v = _clean_empty(v)
            if v is None:
                continue
            if isinstance(v, list) and len(v) == 0:
                continue
            if isinstance(v, dict) and len(v) == 0:
                continue
            cleaned[k] = v
        return cleaned
    elif isinstance(data, list):
        return [_clean_empty(item) for item in data]
    return data
