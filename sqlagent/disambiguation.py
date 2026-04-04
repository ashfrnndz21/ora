"""Disambiguation Agent — asks instead of guessing wrong.

When the Semantic Reasoning Agent detects ambiguity in a user's question
(multiple possible interpretations, unknown terms, vague scope), this agent
generates a focused clarification question with 2-3 specific options.

Fully agentic — the LLM detects ambiguity and generates options from
the actual data context. No hardcoded patterns or rules.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import structlog

logger = structlog.get_logger()


@dataclass
class ClarificationQuestion:
    """A clarification question to present to the user."""

    question: str = ""  # "Did you mean gross revenue or net revenue?"
    options: list[dict] = field(default_factory=list)
    # [{"label": "Gross revenue", "description": "SUM(amount) before discounts", "value": "gross"}]
    default_option: str = ""  # Most likely option
    ambiguous_term: str = ""  # The term that's ambiguous
    reasoning: str = ""  # Why this is ambiguous
    confidence_without_clarification: float = 0.0  # How confident if we just guess

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "options": self.options,
            "default_option": self.default_option,
            "ambiguous_term": self.ambiguous_term,
            "reasoning": self.reasoning,
            "confidence_without_clarification": self.confidence_without_clarification,
        }


async def detect_disambiguation(
    question: str,
    semantic_reasoning: dict | None,
    schema_context: str,
    llm: object,
    threshold: float = 0.6,
) -> ClarificationQuestion | None:
    """Detect if a query needs clarification before execution.

    Called after the Semantic Reasoning Agent — if the reasoning confidence
    is below the threshold, or if the reasoning explicitly flagged ambiguity,
    this generates a clarification question.

    Args:
        question: User's original question
        semantic_reasoning: Output from reason_about_query()
        schema_context: Schema summary for context
        llm: LLM provider
        threshold: Confidence below which to ask for clarification

    Returns:
        ClarificationQuestion if ambiguity detected, None if clear
    """
    # Check if semantic reasoning already has high confidence
    if semantic_reasoning:
        conf = semantic_reasoning.get("confidence", 1.0)
        if conf >= threshold:
            return None  # Confident enough — no clarification needed

    prompt = f"""\
You are an ambiguity detection agent for a natural language to SQL system.

The user asked: "{question}"

Semantic reasoning result:
{semantic_reasoning or 'No semantic reasoning available'}

Schema context (abbreviated):
{schema_context[:1500]}

TASK: Determine if the question is ambiguous and needs clarification.

Ambiguity signals:
- A term maps to multiple possible columns or values
- A term is completely unknown (not in schema)
- Temporal reference is vague ("recently", "this year" without context)
- A metric could mean different things ("revenue" = gross? net? ARR?)
- Scope is unclear ("all customers" = active only? all-time?)

If ambiguous, generate ONE focused clarification question with 2-3 specific options.

Return JSON:
{{
  "is_ambiguous": true,
  "question": "Did you mean X or Y?",
  "options": [
    {{"label": "Option A", "description": "Detailed explanation", "value": "option_a"}},
    {{"label": "Option B", "description": "Detailed explanation", "value": "option_b"}}
  ],
  "default_option": "option_a",
  "ambiguous_term": "the ambiguous term",
  "reasoning": "Why this is ambiguous",
  "confidence_without_clarification": 0.45
}}

If NOT ambiguous (question is clear enough to proceed):
{{
  "is_ambiguous": false
}}

Rules:
- Only ask ONE question (the most impactful ambiguity)
- Provide 2-3 SPECIFIC options, not open-ended
- Include a default (most likely interpretation)
- Only flag as ambiguous if confidence_without_clarification < {threshold}
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

        if not parsed.get("is_ambiguous"):
            return None

        result = ClarificationQuestion(
            question=parsed.get("question", ""),
            options=parsed.get("options", []),
            default_option=parsed.get("default_option", ""),
            ambiguous_term=parsed.get("ambiguous_term", ""),
            reasoning=parsed.get("reasoning", ""),
            confidence_without_clarification=parsed.get("confidence_without_clarification", 0.5),
        )

        logger.info(
            "disambiguation.detected",
            term=result.ambiguous_term,
            options=len(result.options),
            confidence=result.confidence_without_clarification,
        )
        return result

    except Exception as exc:
        logger.warning("disambiguation.failed", error=str(exc))
        return None
