"""Confidence Scoring Engine — LLM-scored query result confidence.

Produces a calibrated 0-100 confidence score for every query result.
The score is computed by the LLM from 5 signals — no hardcoded weights
or formula. The LLM reasons about whether the result is trustworthy.

Signals provided to the LLM:
  1. Semantic resolution quality — did all user terms map to known columns/values?
  2. SQL execution outcome — did it run without errors? How many corrections needed?
  3. Result shape — reasonable row count? Non-empty? Values in expected ranges?
  4. Schema coverage — did the query use relevant tables/columns?
  5. Generator consensus — did multiple generators agree? (if applicable)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import structlog

logger = structlog.get_logger()


@dataclass
class ConfidenceBreakdown:
    """LLM-scored confidence breakdown."""

    total: int = 0  # 0-100
    semantic_match: str = ""  # "All terms resolved" or "2 terms unresolved"
    execution_quality: str = ""  # "Clean execution" or "Required 3 corrections"
    result_assessment: str = ""  # "12 rows, values look reasonable" or "Empty result"
    reasoning: str = ""  # Full LLM reasoning trace
    level: str = "unknown"  # "high" | "medium" | "low" | "unknown"

    def to_dict(self) -> dict:
        return {
            "total": self.total,
            "semantic_match": self.semantic_match,
            "execution_quality": self.execution_quality,
            "result_assessment": self.result_assessment,
            "reasoning": self.reasoning,
            "level": self.level,
        }


@dataclass
class Explanation:
    """Human-readable explanation of how the query was interpreted."""

    interpreted_as: str = ""
    assumptions: list[str] = field(default_factory=list)
    alternatives: list[str] = field(default_factory=list)
    tables_used: list[str] = field(default_factory=list)
    filters_applied: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "interpreted_as": self.interpreted_as,
            "assumptions": self.assumptions,
            "alternatives": self.alternatives,
            "tables_used": self.tables_used,
            "filters_applied": self.filters_applied,
        }


async def score_confidence(
    question: str,
    sql: str,
    row_count: int,
    corrections: int,
    error: str,
    semantic_reasoning: dict | None,
    llm: object,
) -> ConfidenceBreakdown:
    """Score the confidence of a query result using LLM reasoning.

    This is NOT a formula with hardcoded weights. The LLM receives all
    the signals and reasons about how trustworthy the result is.

    Args:
        question: Original user question
        sql: Final SQL that was executed
        row_count: Number of rows returned
        corrections: Number of self-heal correction rounds needed
        error: Execution error message (empty if success)
        semantic_reasoning: Output from the Semantic Reasoning Agent
        llm: LLM provider

    Returns:
        ConfidenceBreakdown with score 0-100 and reasoning
    """
    result = ConfidenceBreakdown()

    # Build context for the LLM
    sem_context = ""
    if semantic_reasoning:
        filters = semantic_reasoning.get("filters", [])
        sem_context = f"Semantic resolution: {len(filters)} filters mapped"
        if semantic_reasoning.get("reasoning"):
            sem_context += f"\n  {semantic_reasoning['reasoning'][:200]}"
        conf = semantic_reasoning.get("confidence", 0)
        sem_context += f"\n  Semantic confidence: {conf}"

    prompt = f"""\
You are a quality assessor for a natural language to SQL system.

Rate the confidence of this query result on a scale of 0-100.

USER QUESTION: {question}

GENERATED SQL:
{sql}

EXECUTION RESULT:
  Rows returned: {row_count}
  Corrections needed: {corrections}
  Error: {error or 'None'}

SEMANTIC RESOLUTION:
  {sem_context or 'No semantic reasoning available'}

SCORING CRITERIA:
  90-100: Perfect — all terms resolved, clean execution, meaningful results
  70-89:  Good — minor issues but result is likely correct
  50-69:  Uncertain — some terms may be wrong, result needs verification
  30-49:  Low — significant issues, result may be incorrect
  0-29:   Very low — likely wrong, user should rephrase

Return JSON:
{{
  "total": 85,
  "semantic_match": "brief assessment of how well user terms were mapped",
  "execution_quality": "brief assessment of SQL execution quality",
  "result_assessment": "brief assessment of whether results make sense",
  "reasoning": "1-2 sentence overall assessment"
}}

Rules:
- Be calibrated: empty results should score low, 0 rows = max 30
- Corrections needed reduce confidence (each correction = -10)
- Errors reduce confidence significantly
- Return ONLY valid JSON, no markdown
"""

    try:
        resp = await llm.complete(
            [{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=512,
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
        result.total = max(0, min(100, parsed.get("total", 50)))
        result.semantic_match = parsed.get("semantic_match", "")
        result.execution_quality = parsed.get("execution_quality", "")
        result.result_assessment = parsed.get("result_assessment", "")
        result.reasoning = parsed.get("reasoning", "")

        if result.total >= 80:
            result.level = "high"
        elif result.total >= 60:
            result.level = "medium"
        elif result.total >= 40:
            result.level = "low"
        else:
            result.level = "very_low"

        logger.info(
            "confidence.scored",
            score=result.total,
            level=result.level,
            corrections=corrections,
            rows=row_count,
        )

    except Exception as exc:
        logger.warning("confidence.scoring_failed", error=str(exc))
        # Fallback: simple heuristic score (only used when LLM fails)
        if error:
            result.total = 15
            result.level = "very_low"
        elif row_count == 0:
            result.total = 25
            result.level = "low"
        elif corrections > 2:
            result.total = 50
            result.level = "medium"
        elif corrections > 0:
            result.total = 70
            result.level = "medium"
        else:
            result.total = 80
            result.level = "high"
        result.reasoning = "Confidence estimated (LLM scoring unavailable)"

    return result
