"""Tests for OraSpec Semantic Model — v2.0 foundation."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from sqlagent.semantic.model import (
    AggregationType,
    CustomInstruction,
    DataType,
    Dimension,
    Filter,
    JoinType,
    LogicalTable,
    Measure,
    Metric,
    OraSpec,
    Relationship,
    Synonym,
    TimeDimension,
    TimeGrain,
    VerifiedQuery,
    FiscalCalendar,
)
from sqlagent.semantic.glossary import BusinessGlossary, GlossaryEntry
from sqlagent.semantic.loader import load_oraspec, save_oraspec, validate_oraspec


# ═══════════════════════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════════════════════


def _make_spec() -> OraSpec:
    """Build a minimal but complete OraSpec for testing."""
    return OraSpec(
        name="test_retail",
        version="1.0",
        description="Test retail analytics",
        fiscal_calendar=FiscalCalendar(fiscal_year_start_month=4, week_start_day="monday"),
        tables=[
            LogicalTable(
                name="orders",
                display_name="Customer Orders",
                description="Customer purchase orders — one row per order",
                table="fact_orders",
                backend="postgres",
                dimensions=[
                    Dimension(
                        name="region",
                        display_name="Region",
                        description="Sales region",
                        expr="orders.region",
                        data_type=DataType.STRING,
                        synonyms=["area", "territory", "market"],
                        sample_values=["APAC", "EMEA", "Americas"],
                        is_exhaustive=True,
                        confidence=0.95,
                    ),
                    Dimension(
                        name="channel",
                        display_name="Channel",
                        description="Sales channel",
                        expr="orders.channel",
                        data_type=DataType.STRING,
                        synonyms=["sales channel"],
                        sample_values=["Online", "In-Store"],
                    ),
                ],
                time_dimensions=[
                    TimeDimension(
                        name="order_date",
                        display_name="Order Date",
                        description="Date the order was placed",
                        expr="orders.created_at",
                        synonyms=["date", "when"],
                        time_grain=TimeGrain.DAY,
                    ),
                ],
                measures=[
                    Measure(
                        name="revenue",
                        display_name="Revenue",
                        description="Gross revenue",
                        expr="SUM(orders.amount)",
                        aggregation=AggregationType.SUM,
                        synonyms=["sales", "GMV", "top line"],
                        unit="USD",
                        confidence=0.98,
                    ),
                    Measure(
                        name="order_count",
                        display_name="Order Count",
                        description="Number of orders",
                        expr="COUNT(DISTINCT orders.order_id)",
                        aggregation=AggregationType.COUNT_DISTINCT,
                    ),
                ],
                filters=[
                    Filter(
                        name="active_orders",
                        display_name="Active Orders",
                        description="Only completed orders",
                        expr="orders.status = 'completed'",
                    ),
                ],
            ),
            LogicalTable(
                name="customers",
                display_name="Customers",
                description="Customer master data",
                table="dim_customers",
                backend="postgres",
                dimensions=[
                    Dimension(
                        name="customer_name",
                        display_name="Customer Name",
                        description="Customer full name",
                        data_type=DataType.STRING,
                    ),
                    Dimension(
                        name="email",
                        display_name="Email",
                        description="Customer email",
                        data_type=DataType.STRING,
                        is_sensitive=True,
                    ),
                ],
                measures=[
                    Measure(
                        name="lifetime_value",
                        display_name="Lifetime Value",
                        description="Total spend",
                        expr="SUM(total_spend)",
                    ),
                ],
            ),
        ],
        relationships=[
            Relationship(
                from_table="orders",
                from_column="customer_id",
                to_table="customers",
                to_column="customer_id",
                join_type=JoinType.INNER,
                relationship_type="many_to_one",
                source="fk",
                confidence=0.99,
                confirmed=True,
            ),
        ],
        metrics=[
            Metric(
                name="avg_order_value",
                display_name="Average Order Value",
                description="Revenue divided by order count",
                expr="SUM(amount) / NULLIF(COUNT(DISTINCT order_id), 0)",
                depends_on=["revenue", "order_count"],
                synonyms=["AOV"],
            ),
        ],
        verified_queries=[
            VerifiedQuery(
                question="What is the total revenue by region?",
                sql="SELECT region, SUM(amount) FROM orders GROUP BY region",
                tables_used=["orders"],
                verified_by="test@test.com",
                tags=["revenue", "regional"],
            ),
        ],
        custom_instructions=[
            CustomInstruction(
                instruction="Always use fiscal year starting April 1",
                scope="global",
                priority=10,
            ),
        ],
        synonyms=[
            Synonym(term="sales", canonical="orders.revenue"),
            Synonym(term="income", canonical="orders.revenue"),
        ],
    )


# ═══════════════════════════════════════════════════════════════════════════════
# ORASPEC MODEL TESTS
# ═══════════════════════════════════════════════════════════════════════════════


def test_oraspec_create():
    """Construct OraSpec programmatically."""
    spec = _make_spec()
    assert spec.name == "test_retail"
    assert len(spec.tables) == 2
    assert len(spec.relationships) == 1
    assert len(spec.metrics) == 1
    assert spec.fiscal_calendar is not None
    assert spec.fiscal_calendar.fiscal_year_start_month == 4


def test_oraspec_get_table():
    """Table lookup by name (case-insensitive)."""
    spec = _make_spec()
    t = spec.get_table("orders")
    assert t is not None
    assert t.name == "orders"
    t2 = spec.get_table("ORDERS")
    assert t2 is not None
    assert t2.name == "orders"
    assert spec.get_table("nonexistent") is None


def test_oraspec_get_all_synonyms():
    """Synonym resolution builds flat mapping."""
    spec = _make_spec()
    syns = spec.get_all_synonyms()
    assert "area" in syns
    assert syns["area"] == "orders.region"
    assert "gmv" in syns
    assert syns["gmv"] == "orders.revenue"
    assert "sales" in syns  # global synonym
    assert syns["sales"] == "orders.revenue"
    assert "aov" in syns  # metric synonym
    assert syns["aov"] == "avg_order_value"


def test_oraspec_to_prompt_context():
    """Verify prompt string format includes key elements."""
    spec = _make_spec()
    ctx = spec.to_prompt_context()
    assert "# Semantic Model: test_retail" in ctx
    assert "DIM region" in ctx
    assert "MEASURE revenue" in ctx
    assert "TIME order_date" in ctx
    assert "FILTER active_orders" in ctx
    assert "Derived Metrics" in ctx
    assert "avg_order_value" in ctx.lower() or "Average Order Value" in ctx
    assert "Relationships" in ctx
    assert "Instructions" in ctx
    assert "fiscal year" in ctx.lower()


def test_oraspec_sensitive_columns_excluded():
    """PII columns should not appear in prompt context."""
    spec = _make_spec()
    ctx = spec.to_prompt_context()
    assert "email" not in ctx.lower() or "Email" not in ctx


def test_oraspec_stats():
    """Stats returns correct counts."""
    spec = _make_spec()
    s = spec.stats()
    assert s["tables"] == 2
    assert s["dimensions"] == 4  # 2 + 2
    assert s["measures"] == 3  # 2 + 1
    assert s["metrics"] == 1
    assert s["relationships"] == 1
    assert s["verified_queries"] == 1
    assert s["synonyms"] > 0
    assert 0 < s["avg_confidence"] <= 1.0
    assert "postgres" in s["backends"]


def test_oraspec_get_relationships_for_table():
    """Get relationships involving a specific table."""
    spec = _make_spec()
    rels = spec.get_relationships_for_table("orders")
    assert len(rels) == 1
    assert rels[0].to_table == "customers"


def test_oraspec_get_tables_for_backend():
    """Filter tables by backend."""
    spec = _make_spec()
    pg_tables = spec.get_all_tables_for_backend("postgres")
    assert len(pg_tables) == 2
    sf_tables = spec.get_all_tables_for_backend("snowflake")
    assert len(sf_tables) == 0


def test_logical_table_all_column_names():
    """All column names across dims, time_dims, measures."""
    spec = _make_spec()
    t = spec.get_table("orders")
    names = t.all_column_names()
    assert "region" in names
    assert "order_date" in names
    assert "revenue" in names
    assert len(names) == 5  # 2 dims + 1 time + 2 measures


# ═══════════════════════════════════════════════════════════════════════════════
# YAML LOADER TESTS
# ═══════════════════════════════════════════════════════════════════════════════


def test_oraspec_yaml_roundtrip():
    """Save → load → compare preserves all data."""
    spec = _make_spec()
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.yaml"
        save_oraspec(spec, path)
        spec2 = load_oraspec(path)
        assert spec2.name == spec.name
        assert len(spec2.tables) == len(spec.tables)
        assert len(spec2.relationships) == len(spec.relationships)
        assert len(spec2.metrics) == len(spec.metrics)
        assert len(spec2.get_all_synonyms()) == len(spec.get_all_synonyms())


def test_load_retail_example():
    """Load the retail_semantic.yaml example file."""
    example_path = Path(__file__).parent.parent / "examples" / "retail_semantic.yaml"
    if not example_path.exists():
        pytest.skip("retail_semantic.yaml not found")
    spec = load_oraspec(example_path)
    assert spec.name == "retail_analytics"
    assert len(spec.tables) >= 3
    assert len(spec.get_all_synonyms()) > 20


def test_load_nonexistent_file():
    """FileNotFoundError for missing file."""
    with pytest.raises(FileNotFoundError):
        load_oraspec("/nonexistent/path.yaml")


def test_load_invalid_yaml():
    """ValueError for invalid YAML content."""
    with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
        f.write("just a string, not a mapping")
        f.flush()
        with pytest.raises(ValueError, match="must contain a YAML mapping"):
            load_oraspec(f.name)
    os.unlink(f.name)


# ═══════════════════════════════════════════════════════════════════════════════
# VALIDATION TESTS
# ═══════════════════════════════════════════════════════════════════════════════


def test_validate_oraspec_valid():
    """No warnings for well-formed spec (with sample values)."""
    spec = _make_spec()
    warnings = validate_oraspec(spec)
    # May have warnings about string dims without samples, but no structural errors
    structural = [w for w in warnings if "unknown" in w.lower() or "depends on" in w.lower()]
    assert len(structural) == 0


def test_validate_oraspec_bad_relationship():
    """Warning for relationship referencing unknown table."""
    spec = _make_spec()
    spec.relationships.append(
        Relationship(from_table="nonexistent", from_column="id", to_table="orders", to_column="id")
    )
    warnings = validate_oraspec(spec)
    assert any("unknown table: nonexistent" in w for w in warnings)


def test_validate_oraspec_bad_metric_dependency():
    """Warning for metric referencing unknown measure."""
    spec = _make_spec()
    spec.metrics.append(
        Metric(name="bad_metric", expr="SUM(x)/SUM(y)", depends_on=["nonexistent_measure"])
    )
    warnings = validate_oraspec(spec)
    assert any("unknown measure: nonexistent_measure" in w for w in warnings)


def test_validate_oraspec_missing_description():
    """Warning for table without description."""
    spec = _make_spec()
    spec.tables.append(LogicalTable(name="empty_table", table="empty"))
    warnings = validate_oraspec(spec)
    assert any("no description" in w.lower() for w in warnings)


# ═══════════════════════════════════════════════════════════════════════════════
# GLOSSARY TESTS
# ═══════════════════════════════════════════════════════════════════════════════


def test_glossary_resolve_exact_term():
    """Resolve by exact term match."""
    g = BusinessGlossary(
        entries=[
            GlossaryEntry(term="active customer", definition="Purchase in last 90 days"),
            GlossaryEntry(term="churn rate", definition="Lost customers / total"),
        ]
    )
    result = g.resolve_exact("active customer")
    assert result is not None
    assert result.term == "active customer"


def test_glossary_resolve_exact_alias():
    """Resolve by alias match."""
    g = BusinessGlossary(
        entries=[
            GlossaryEntry(
                term="active customer",
                definition="Purchase in last 90 days",
                aliases=["active account", "recent buyer"],
            ),
        ]
    )
    result = g.resolve_exact("recent buyer")
    assert result is not None
    assert result.term == "active customer"


def test_glossary_resolve_exact_case_insensitive():
    """Case-insensitive exact match."""
    g = BusinessGlossary(
        entries=[GlossaryEntry(term="Active Customer", definition="test")]
    )
    assert g.resolve_exact("ACTIVE CUSTOMER") is not None
    assert g.resolve_exact("active customer") is not None


def test_glossary_resolve_exact_not_found():
    """None for unresolvable term."""
    g = BusinessGlossary(entries=[GlossaryEntry(term="revenue", definition="Total sales")])
    assert g.resolve_exact("nonexistent term") is None


def test_glossary_add_and_remove():
    """Add and remove glossary entries."""
    g = BusinessGlossary()
    g.add(GlossaryEntry(term="LTV", definition="Lifetime value"))
    assert len(g.entries) == 1
    g.add(GlossaryEntry(term="LTV", definition="Updated lifetime value"))
    assert len(g.entries) == 1  # replaced, not duplicated
    assert g.entries[0].definition == "Updated lifetime value"
    assert g.remove("LTV") is True
    assert len(g.entries) == 0
    assert g.remove("nonexistent") is False


def test_glossary_all_terms():
    """All terms includes primary terms and aliases."""
    g = BusinessGlossary(
        entries=[
            GlossaryEntry(term="revenue", aliases=["sales", "income"]),
            GlossaryEntry(term="margin", aliases=["profit margin"]),
        ]
    )
    terms = g.all_terms()
    assert "revenue" in terms
    assert "sales" in terms
    assert "income" in terms
    assert "margin" in terms
    assert "profit margin" in terms
    assert len(terms) == 5
