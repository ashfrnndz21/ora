"""SOUL — User mental model inference and evolution.

Learns how a user thinks about their data:
- Accountability unit (store, region, sku, team)
- Time reference (vs last week, vs budget, vs 4-week avg)
- Vocabulary map (domain terms → SQL columns)
- Query cadence, entity preferences, decision patterns

Every 20 queries, the LLM evolves the SOUL profile from accumulated observations.
SOUL is always wrapped in try/except — it NEVER breaks a query.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import structlog

logger = structlog.get_logger()

_EVOLVE_SYSTEM = """You are analyzing a user's query patterns to infer their mental model.

Given their recent queries and the tables/columns they use, infer:
1. accountability_unit: What entity do they think in terms of? (store, region, sku, customer, team, product)
2. time_reference: How do they frame time comparisons? (vs_last_week, vs_last_month, vs_budget, vs_4w_avg, ytd)
3. decision_horizon: What are they using data for? (daily_ops, weekly_review, monthly_board, ad_hoc_analysis)
4. vocabulary_map: Domain terms they use → likely SQL column mappings
5. preferred_dimensions: Which GROUP BY columns they prefer
6. frequent_entities: Specific values they ask about (store names, regions, categories)
7. query_cadence: When they typically query (morning, end_of_day, monday_standup)

Return JSON with these fields. Be specific and evidence-based."""


@dataclass
class SOULInstinct:
    """An inferred behavioral pattern with confidence."""

    name: str
    value: str
    confidence: float = 0.5  # 0-1, decays over 90 days if not reinforced
    evidence: str = ""
    last_reinforced: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class SOULProfile:
    """A user's inferred mental model."""

    user_id: str = ""
    accountability_unit: str = ""  # store, region, sku, customer, team
    time_reference: str = ""  # vs_last_week, vs_budget, ytd
    decision_horizon: str = ""  # daily_ops, weekly_review, monthly_board
    vocabulary_map: dict[str, str] = field(
        default_factory=dict
    )  # "revenue" → "orders.total_amount"
    preferred_dimensions: list[str] = field(default_factory=list)
    frequent_entities: list[str] = field(default_factory=list)
    instincts: list[SOULInstinct] = field(default_factory=list)
    query_count: int = 0
    last_evolved: datetime | None = None
    version: int = 0

    def get_context_block(self) -> str:
        """Generate context block injected into schema pruning prompts."""
        parts = []
        if self.accountability_unit:
            parts.append(f"User thinks in terms of: {self.accountability_unit}")
        if self.time_reference:
            parts.append(f"Time framing: {self.time_reference}")
        if self.vocabulary_map:
            mappings = ", ".join(f"{k}→{v}" for k, v in self.vocabulary_map.items())
            parts.append(f"Vocabulary: {mappings}")
        if self.preferred_dimensions:
            parts.append(f"Preferred GROUP BY: {', '.join(self.preferred_dimensions)}")
        if self.frequent_entities:
            parts.append(f"Frequent entities: {', '.join(self.frequent_entities[:5])}")
        return "\n".join(parts)

    def to_markdown(self) -> str:
        """Export as human-readable markdown (user can edit to override)."""
        lines = [
            f"# SOUL Profile: {self.user_id}",
            "",
            f"**Accountability unit:** {self.accountability_unit or '(not yet inferred)'}",
            f"**Time reference:** {self.time_reference or '(not yet inferred)'}",
            f"**Decision horizon:** {self.decision_horizon or '(not yet inferred)'}",
            f"**Query count:** {self.query_count}",
            f"**Version:** {self.version}",
            "",
        ]
        if self.vocabulary_map:
            lines.append("## Vocabulary Map")
            for term, col in self.vocabulary_map.items():
                lines.append(f"- {term} → `{col}`")
            lines.append("")
        if self.instincts:
            lines.append("## Instincts")
            for inst in self.instincts:
                lines.append(f"- **{inst.name}:** {inst.value} (confidence: {inst.confidence:.0%})")
            lines.append("")
        return "\n".join(lines)


class UserSOUL:
    """Interface to a user's SOUL profile. Persists to ~/.sqlagent/soul/{user_id}.json"""

    def __init__(self, llm: Any = None, soul_dir: str = ""):
        self._llm = llm
        self._dir = soul_dir or os.path.join(os.path.expanduser("~"), ".sqlagent", "soul")
        self._profiles: dict[str, SOULProfile] = {}
        self._observations: dict[str, list[dict]] = {}  # user_id → recent observations

    def _profile_path(self, user_id: str) -> str:
        return os.path.join(self._dir, f"{user_id}.json")

    def get_profile(self, user_id: str) -> SOULProfile:
        """Get or create a SOUL profile for a user."""
        if user_id in self._profiles:
            return self._profiles[user_id]

        # Try loading from disk
        path = self._profile_path(user_id)
        if os.path.exists(path):
            try:
                with open(path) as f:
                    data = json.load(f)
                profile = SOULProfile(
                    user_id=user_id,
                    accountability_unit=data.get("accountability_unit", ""),
                    time_reference=data.get("time_reference", ""),
                    decision_horizon=data.get("decision_horizon", ""),
                    vocabulary_map=data.get("vocabulary_map", {}),
                    preferred_dimensions=data.get("preferred_dimensions", []),
                    frequent_entities=data.get("frequent_entities", []),
                    query_count=data.get("query_count", 0),
                    version=data.get("version", 0),
                )
                self._profiles[user_id] = profile
                return profile
            except Exception as exc:
                logger.debug("soul.operation_failed", error=str(exc))

        profile = SOULProfile(user_id=user_id)
        self._profiles[user_id] = profile
        return profile

    async def observe(
        self,
        user_id: str,
        query: str,
        tables: list[str],
        generator: str = "",
    ) -> None:
        """Record a behavioral observation. Non-blocking."""
        try:
            if user_id not in self._observations:
                self._observations[user_id] = []
            self._observations[user_id].append(
                {
                    "query": query,
                    "tables": tables,
                    "generator": generator,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            )

            profile = self.get_profile(user_id)
            profile.query_count += 1

            # Trigger evolution every N queries
            if profile.query_count % 20 == 0 and self._llm:
                await self.evolve(user_id)
        except Exception as e:
            logger.debug("soul.observe_failed", error=str(e))

    async def evolve(self, user_id: str) -> SOULProfile:
        """Re-infer the user's mental model from accumulated observations."""
        profile = self.get_profile(user_id)
        observations = self._observations.get(user_id, [])

        if not observations or not self._llm:
            return profile

        # Build observation summary for the LLM
        recent = observations[-50:]  # Last 50 observations
        obs_text = "\n".join(f'- Query: "{o["query"]}" → Tables: {o["tables"]}' for o in recent)

        prompt = (
            f"User has made {profile.query_count} queries.\n\n"
            f"Recent observations:\n{obs_text}\n\n"
            f"Current profile:\n{json.dumps({'accountability_unit': profile.accountability_unit, 'vocabulary_map': profile.vocabulary_map}, indent=2)}\n\n"
            f"Infer the updated mental model. Return JSON."
        )

        try:
            resp = await self._llm.complete(
                [
                    {"role": "system", "content": _EVOLVE_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                json_mode=True,
            )

            data = json.loads(resp.content)
            profile.accountability_unit = data.get(
                "accountability_unit", profile.accountability_unit
            )
            profile.time_reference = data.get("time_reference", profile.time_reference)
            profile.decision_horizon = data.get("decision_horizon", profile.decision_horizon)
            profile.vocabulary_map.update(data.get("vocabulary_map", {}))
            profile.preferred_dimensions = data.get(
                "preferred_dimensions", profile.preferred_dimensions
            )
            profile.frequent_entities = data.get("frequent_entities", profile.frequent_entities)
            profile.last_evolved = datetime.now(timezone.utc)
            profile.version += 1

            # Persist
            self._save_profile(profile)
            logger.info("soul.evolved", user=user_id, version=profile.version)

        except Exception as e:
            logger.warn("soul.evolve_failed", error=str(e))

        return profile

    def _save_profile(self, profile: SOULProfile) -> None:
        """Persist profile to disk."""
        try:
            os.makedirs(self._dir, exist_ok=True)
            path = self._profile_path(profile.user_id)
            data = {
                "user_id": profile.user_id,
                "accountability_unit": profile.accountability_unit,
                "time_reference": profile.time_reference,
                "decision_horizon": profile.decision_horizon,
                "vocabulary_map": profile.vocabulary_map,
                "preferred_dimensions": profile.preferred_dimensions,
                "frequent_entities": profile.frequent_entities,
                "query_count": profile.query_count,
                "version": profile.version,
            }
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.debug("soul.save_failed", error=str(e))

    async def interpret(self, user_id: str, nl_query: str) -> str:
        """Enrich a query with SOUL context (injected into schema pruning)."""
        try:
            profile = self.get_profile(user_id)
            return profile.get_context_block()
        except Exception as exc:
            logger.debug("soul.operation_failed", error=str(exc))
            return ""
