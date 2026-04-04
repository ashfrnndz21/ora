"""Ora Semantic Layer — OraSpec model, glossary, loader, and auto-generation.

The Semantic Agent owns this layer. All reads and writes to the semantic model
flow through the agent — no direct mutation by other components.
"""

from sqlagent.semantic.model import (
    OraSpec,
    LogicalTable,
    Dimension,
    TimeDimension,
    Measure,
    Metric,
    Relationship,
    VerifiedQuery,
    Filter,
    Synonym,
    CustomInstruction,
    DataType,
    AggregationType,
    TimeGrain,
    JoinType,
)
from sqlagent.semantic.glossary import BusinessGlossary, GlossaryEntry
from sqlagent.semantic.loader import load_oraspec, save_oraspec, validate_oraspec

__all__ = [
    "OraSpec",
    "LogicalTable",
    "Dimension",
    "TimeDimension",
    "Measure",
    "Metric",
    "Relationship",
    "VerifiedQuery",
    "Filter",
    "Synonym",
    "CustomInstruction",
    "DataType",
    "AggregationType",
    "TimeGrain",
    "JoinType",
    "BusinessGlossary",
    "GlossaryEntry",
    "load_oraspec",
    "save_oraspec",
    "validate_oraspec",
]
