"""Test all data models serialize/deserialize correctly."""

from sqlagent.models import (
    SchemaColumn, SchemaTable, SchemaSnapshot, ForeignKey,
    KnowledgeGraph, KGNode, KGEdge, KGLayer, EdgeType, SemanticEntry,
    Trace, TraceNode, TraceStatus,
    Workspace, WorkspaceStatus, User,
    Candidate, TrainingExample, PipelineResult,
    QueryStarted, SchemaPruned, CandidateGenerated, FinalResult,
    SchemaAnalysis, InferredRelationship, EntityGrouping, ColumnSemantic,
)
from sqlagent.config import AgentConfig, DataSourceConfig, AgentCoreConfig
from sqlagent.exceptions import SQLAgentError, PolicyViolation, SQLExecutionFailed


# ── Schema Models ─────────────────────────────────────────────────────────────

def test_schema_column_defaults():
    col = SchemaColumn(name="id", data_type="INTEGER")
    assert col.nullable is True
    assert col.is_primary_key is False
    assert col.aliases == []


def test_schema_snapshot_properties(northwind_snapshot):
    assert northwind_snapshot.table_count == 4
    assert northwind_snapshot.column_count == 18
    assert northwind_snapshot.get_table("orders") is not None
    assert northwind_snapshot.get_table("nonexistent") is None


def test_schema_snapshot_foreign_keys(northwind_snapshot):
    assert len(northwind_snapshot.foreign_keys) == 3
    fk = northwind_snapshot.foreign_keys[0]
    assert fk.from_table == "orders"
    assert fk.to_table == "customers"


# ── Knowledge Graph ───────────────────────────────────────────────────────────

def test_knowledge_graph_to_dict():
    kg = KnowledgeGraph(
        graph_id="kg_1",
        workspace_id="ws_1",
        nodes=[
            KGNode(id="tbl:orders", type="table", name="orders", properties={"row_count": 100}),
            KGNode(id="col:orders.id", type="column", name="id"),
        ],
        edges=[
            KGEdge(id="e1", source="tbl:orders", target="tbl:customers", type=EdgeType.DECLARED_FK),
        ],
        layers=[
            KGLayer(id="l1", name="Sales", tables=["tbl:orders"]),
        ],
        glossary=[
            SemanticEntry(term="revenue", maps_to="orders.total_amount"),
        ],
    )
    d = kg.to_dict()
    assert d["graph_id"] == "kg_1"
    assert d["stats"]["tables"] == 1
    assert d["stats"]["columns"] == 1
    assert d["stats"]["edges"] == 1
    assert len(d["glossary"]) == 1
    assert d["edges"][0]["type"] == "declared_fk"


def test_knowledge_graph_empty():
    kg = KnowledgeGraph()
    assert kg.table_count == 0
    assert kg.column_count == 0
    d = kg.to_dict()
    assert d["stats"]["tables"] == 0


# ── Trace ─────────────────────────────────────────────────────────────────────

def test_trace_node_to_dict():
    child = TraceNode(node_id="n2", name="Generate", status=TraceStatus.COMPLETED, latency_ms=890)
    parent = TraceNode(
        node_id="n1", name="Query", status=TraceStatus.COMPLETED,
        children=[child], summary="3 steps",
    )
    d = parent.to_dict()
    assert d["node_id"] == "n1"
    assert len(d["children"]) == 1
    assert d["children"][0]["latency_ms"] == 890


def test_trace_to_dict():
    trace = Trace(
        trace_id="t1",
        nl_query="top customers",
        succeeded=True,
        total_latency_ms=1200,
        root=TraceNode(node_id="root", name="Query"),
    )
    d = trace.to_dict()
    assert d["trace_id"] == "t1"
    assert d["succeeded"] is True
    assert d["root"]["node_id"] == "root"


# ── Workspace + Auth ──────────────────────────────────────────────────────────

def test_workspace_defaults():
    ws = Workspace(workspace_id="ws_1", name="Test")
    assert ws.status == WorkspaceStatus.SETUP
    assert ws.query_count == 0
    assert ws.sources == []


def test_user_defaults():
    user = User(user_id="u1", email="a@b.com")
    assert user.display_name == ""
    assert user.provider == "email"


# ── Candidate + Pipeline Result ───────────────────────────────────────────────

def test_candidate_succeeded():
    c = Candidate(generator_id="fewshot", sql="SELECT 1")
    assert c.succeeded is True

    c2 = Candidate(generator_id="fewshot", sql="", error="parse error")
    assert c2.succeeded is False


def test_pipeline_result_dataframe():
    r = PipelineResult(
        rows=[{"a": 1, "b": 2}, {"a": 3, "b": 4}],
        columns=["a", "b"],
        row_count=2,
    )
    df = r.dataframe
    assert len(df) == 2
    assert list(df.columns) == ["a", "b"]


def test_pipeline_result_empty_dataframe():
    r = PipelineResult()
    df = r.dataframe
    assert len(df) == 0


# ── Events ────────────────────────────────────────────────────────────────────

def test_event_types():
    e = QueryStarted(query_id="q1", nl_query="test")
    assert e.event_type == "query.started"

    e2 = SchemaPruned(columns_before=100, columns_after=8)
    assert e2.columns_before == 100

    e3 = FinalResult(succeeded=True, row_count=10, total_cost_usd=0.004)
    assert e3.event_type == "query.final"


# ── Config ────────────────────────────────────────────────────────────────────

def test_agent_config_defaults():
    c = AgentConfig()
    assert c.llm_model == "gpt-4o"
    assert c.select_only is True
    assert c.max_corrections == 3
    assert c.agentcore.enabled is False


def test_data_source_config():
    ds = DataSourceConfig(source_id="pg1", type="postgresql", connection_string="postgresql://localhost/test")
    assert ds.read_only is True


# ── Exceptions ────────────────────────────────────────────────────────────────

def test_policy_violation():
    e = PolicyViolation(rule_id="no_ddl", reason="DROP not allowed", sql="DROP TABLE x")
    assert "no_ddl" in str(e)
    assert e.rule_id == "no_ddl"


def test_sql_execution_failed():
    e = SQLExecutionFailed(sql="SELECT bad", error="column not found", sql_state="42703")
    assert e.sql == "SELECT bad"
    assert e.sql_state == "42703"


# ── Schema Analysis ───────────────────────────────────────────────────────────

def test_schema_analysis():
    sa = SchemaAnalysis(
        analysis_id="a1",
        inferred_relationships=[
            InferredRelationship(from_table="orders", from_column="store_id",
                                 to_table="stores", to_column="store_id", confidence=0.97),
        ],
        entity_groups=[
            EntityGrouping(name="Sales", tables=["orders", "order_items"]),
        ],
        column_semantics=[
            ColumnSemantic(table="orders", column="total_amount", semantic_type="currency"),
        ],
    )
    assert len(sa.inferred_relationships) == 1
    assert sa.inferred_relationships[0].confidence == 0.97
    assert sa.entity_groups[0].name == "Sales"
