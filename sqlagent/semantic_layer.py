"""Semantic Reasoning Layer — dynamic, cache-backed semantic understanding.

Architecture:
  Per query → semantic_resolve_node:
    1. embed(nl_query + source_id) → search SemanticCache (Qdrant)
    2. cosine >= 0.85 → cache HIT: inject SemanticResolution, skip agent
    3. cache MISS → SemanticReasoningAgent (one LLM call):
         reads schema + actual sample values → reasons about:
           - what abbreviations/codes map to (MY → MYS, sex=Total, etc.)
           - which column name matches the concept (unemployment → obs_value)
           - what filters prevent double-counting
         returns SemanticResolution
    4. resolution injected into QueryState as context_notes for generate_node

  After successful query (no thumbs-down) → learn_node saves resolution to cache.
  Wrong resolutions are never cached — they die with the query.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any

import structlog

logger = structlog.get_logger()

CACHE_COLLECTION = "semantic_cache"
CACHE_HIT_THRESHOLD = 0.85


@dataclass
class SemanticResolution:
    """Output of the SemanticReasoningAgent for one query + source combination."""

    resolution_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    nl_query: str = ""
    source_id: str = ""

    # What the agent reasoned:
    entity_map: dict[str, str] = field(default_factory=dict)
    # e.g. {"Malaysia": "MYS", "Vietnam": "VNM"} — user term → actual stored value

    synonyms: dict[str, str] = field(default_factory=dict)
    # e.g. {"unemployment": "obs_value", "year": "_year"} — concept → column name

    filter_hints: list[str] = field(default_factory=list)
    # e.g. ["sex='Total' for aggregate queries to avoid double-counting"]

    sql_fragments: list[str] = field(default_factory=list)
    # e.g. ["iso_code IN ('MYS', 'VNM')"] — ready-to-use WHERE fragments

    reasoning: str = ""  # full chain-of-thought from the agent
    confidence: float = 0.0
    latency_ms: int = 0

    def to_context_block(self) -> str:
        """Formatted text injected into the SQL generation prompt."""
        lines: list[str] = ["SEMANTIC CONTEXT (reasoned from actual data — apply to SQL):"]
        if self.entity_map:
            lines.append("Entity mappings (use these exact values in WHERE clauses):")
            for term, value in self.entity_map.items():
                lines.append(f"  '{term}' → '{value}'")
        if self.synonyms:
            lines.append("Column synonyms:")
            for concept, col in self.synonyms.items():
                lines.append(f"  '{concept}' → column '{col}'")
        if self.filter_hints:
            lines.append("Filter hints:")
            for hint in self.filter_hints:
                lines.append(f"  • {hint}")
        if self.sql_fragments:
            lines.append("SQL fragments (use directly):")
            for frag in self.sql_fragments:
                lines.append(f"  {frag}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "SemanticResolution":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ═══════════════════════════════════════════════════════════════════════════════
# SEMANTIC CACHE — Qdrant-backed vector cache per workspace
# ═══════════════════════════════════════════════════════════════════════════════


class SemanticCache:
    """Vector cache for SemanticResolution objects.

    Key: embed(nl_query + source_id)
    Value: SemanticResolution JSON stored as Qdrant payload
    Hit threshold: cosine >= CACHE_HIT_THRESHOLD
    """

    def __init__(self, vector_store: Any, embedder: Any):
        self._store = vector_store
        self._embedder = embedder
        self._collection = CACHE_COLLECTION
        self._ready = False

    async def _ensure_ready(self) -> None:
        if self._ready:
            return
        try:
            await self._store.ensure_collection(
                name=self._collection,
                dimensions=384,
            )
            self._ready = True
        except Exception as exc:
            logger.warning("semantic_cache.init_failed", error=str(exc))

    def _cache_key(self, nl_query: str, source_id: str) -> str:
        return f"{nl_query.strip().lower()} [src:{source_id}]"

    async def get(self, nl_query: str, source_id: str) -> SemanticResolution | None:
        """Return cached resolution if similarity >= threshold, else None."""
        await self._ensure_ready()
        try:
            key = self._cache_key(nl_query, source_id)
            embedding = await self._embedder.embed(key)
            results = await self._store.search(
                vector=embedding,
                top_k=1,
            )
            if not results:
                return None
            top = results[0]
            if top.get("score", 0) < CACHE_HIT_THRESHOLD:
                return None
            payload = top.get("payload", {})
            resolution_data = payload.get("resolution")
            if not resolution_data:
                return None
            return SemanticResolution.from_dict(resolution_data)
        except Exception as exc:
            logger.debug("semantic_cache.get_failed", error=str(exc))
            return None

    async def save(self, resolution: SemanticResolution) -> None:
        """Embed and store a resolution in the cache."""
        await self._ensure_ready()
        try:
            key = self._cache_key(resolution.nl_query, resolution.source_id)
            embedding = await self._embedder.embed(key)
            await self._store.upsert(
                id=resolution.resolution_id,
                vector=embedding,
                payload={"resolution": resolution.to_dict()},
            )
            logger.info(
                "semantic_cache.saved",
                resolution_id=resolution.resolution_id,
                source_id=resolution.source_id,
                entities=len(resolution.entity_map),
                synonyms=len(resolution.synonyms),
            )
        except Exception as exc:
            logger.warning("semantic_cache.save_failed", error=str(exc))


# ═══════════════════════════════════════════════════════════════════════════════
# SEMANTIC REASONING AGENT — one LLM call, pure reasoning
# ═══════════════════════════════════════════════════════════════════════════════


async def run_semantic_agent(
    nl_query: str,
    source_id: str,
    schema: dict,
    llm: Any,
) -> SemanticResolution:
    """One LLM call that reasons about query vs data to produce a SemanticResolution.

    The agent sees:
      - the raw user query
      - the actual schema with real sample values from the data
      - no hardcoded rules — it reasons from evidence

    Returns a SemanticResolution with entity_map, synonyms, filter_hints, sql_fragments.
    """
    started = time.monotonic()

    # Build compact schema text with sample values
    schema_lines: list[str] = []
    for t in schema.get("tables", []):
        cols = []
        for c in t.get("columns", []):
            examples = c.get("examples", [])
            ex_str = f" [samples: {', '.join(str(v) for v in examples[:8])}]" if examples else ""
            cols.append(f"    {c['name']} {c.get('data_type', '')}{ex_str}")
        schema_lines.append(f"Table: {t['name']} ({t.get('row_count', 0)} rows)\n" + "\n".join(cols))

    schema_text = "\n\n".join(schema_lines) if schema_lines else "(no schema available)"

    prompt = f"""\
You are a semantic reasoning agent. Your job is to bridge the gap between what a user asks
and what is actually stored in the database.

USER QUERY: {nl_query}

DATA SCHEMA WITH ACTUAL SAMPLE VALUES:
{schema_text}

Reason step by step:
1. What entities does the user refer to? (countries, categories, time periods, etc.)
   Look at the sample values in relevant columns. What are the ACTUAL stored codes/values
   for those entities? (e.g. user says "Malaysia" but samples show "MYS")

2. What concepts does the user mention? (unemployment, revenue, trend, etc.)
   Which column names in this schema represent those concepts?

3. What filter is needed to avoid double-counting or aggregation artifacts?
   (e.g. a "sex" column with Total/Male/Female — need sex='Total' for population-level)

4. Can you write a ready-to-use SQL WHERE fragment for the entity filter?

Return a JSON object:
{{
  "entity_map": {{"<exact user term>": "<actual stored value>", ...}},
  "synonyms": {{"<user concept>": "<column name>", ...}},
  "filter_hints": ["<hint>", ...],
  "sql_fragments": ["<WHERE fragment>", ...],
  "reasoning": "<your step-by-step reasoning>",
  "confidence": <0.0-1.0>
}}

Rules:
- entity_map keys MUST be the EXACT strings the user wrote in their query — not inferred full names.
  If the user wrote "PHN", key = "PHN" (not "Philippines").
  If the user wrote "MYY", key = "MYY" (not "Malaysia").
  If the user wrote "VNT", key = "VNT" (not "Vietnam").
  If the user wrote "SG", key = "SG" (not "Singapore").
- entity_map values MUST be the ACTUAL stored values from the sample data above.
  e.g. user wrote "SG" → look at iso_code samples → see "SGP" → map "SG": "SGP"
- Include ALL entity terms from the user query in entity_map, even if you're not 100% certain.
  It is better to over-map than to leave a term unresolved.
- CRITICAL: If the user wrote a code/abbreviation that does NOT appear in the sample values
  (e.g. user wrote "VNT" but samples only show "VNM"), find the CLOSEST match from the actual
  sample values and map the user's wrong term to the correct stored value.
  Example: user wrote "VNT" (typo for Vietnam), samples show "VNM" → map "VNT": "VNM"
  Example: user wrote "PHN" (abbreviation for Philippines), samples show "PHL" → map "PHN": "PHL"
  Example: user wrote "MYY" (wrong code for Malaysia), samples show "MYS" → map "MYY": "MYS"
- VALIDATE: Every value in entity_map must actually appear in the sample values above.
  Never invent a mapping target — only map to values you can see in the samples.
- sql_fragments: Always include a ready-to-use WHERE fragment listing all entity values.
  e.g. "iso_code IN ('PHL', 'MYS')" — this is the most important output.
- synonyms: ONLY include entries where the column name is not obvious from the concept.
- Return ONLY valid JSON, no markdown.
"""

    try:
        resp = await llm.complete([{"role": "user", "content": prompt}], json_mode=True)
        raw = resp.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip().rstrip("```").strip()

        parsed = json.loads(raw)
        latency = int((time.monotonic() - started) * 1000)

        resolution = SemanticResolution(
            nl_query=nl_query,
            source_id=source_id,
            entity_map=parsed.get("entity_map", {}),
            synonyms=parsed.get("synonyms", {}),
            filter_hints=parsed.get("filter_hints", []),
            sql_fragments=parsed.get("sql_fragments", []),
            reasoning=parsed.get("reasoning", ""),
            confidence=float(parsed.get("confidence", 0.8)),
            latency_ms=latency,
        )
        logger.info(
            "semantic_agent.resolved",
            nl_query=nl_query[:60],
            source_id=source_id,
            entities=len(resolution.entity_map),
            synonyms=len(resolution.synonyms),
            latency_ms=latency,
        )
        return resolution

    except Exception as exc:
        logger.warning("semantic_agent.failed", error=str(exc))
        return SemanticResolution(
            nl_query=nl_query,
            source_id=source_id,
            latency_ms=int((time.monotonic() - started) * 1000),
        )


# ═══════════════════════════════════════════════════════════════════════════════
# SEMANTIC MEMORY — SQLite-backed persistent source facts
# ═══════════════════════════════════════════════════════════════════════════════


class SemanticMemory:
    """Persistent store of source-level semantic facts across sessions.

    Unlike SemanticCache (per-query Qdrant), SemanticMemory stores durable facts:
    - Which table/column holds entity names in each source
    - Which entity values are known to exist in each source
    - Concept → column mappings (e.g. 'unemployment' → 'obs_value')

    Used by Ora orchestrator to skip live SQL scans on cache hit.
    Written by learn_node after each successful query.
    """

    def __init__(self, db_path: str = "~/.sqlagent/semantic_memory.db"):
        self._db_path = os.path.expanduser(db_path)
        self._ready = False

    async def _ensure_ready(self) -> None:
        if self._ready:
            return
        os.makedirs(os.path.dirname(self._db_path), exist_ok=True)
        try:
            import aiosqlite
            async with aiosqlite.connect(self._db_path) as db:
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS source_facts (
                        source_id TEXT NOT NULL,
                        fact_type TEXT NOT NULL,
                        key       TEXT NOT NULL,
                        value     TEXT NOT NULL,
                        confidence REAL DEFAULT 1.0,
                        updated_at TEXT NOT NULL,
                        PRIMARY KEY (source_id, fact_type, key)
                    )
                """)
                await db.commit()
            self._ready = True
        except Exception as exc:
            logger.warning("semantic_memory.init_failed", error=str(exc))

    # ── Entity column location ────────────────────────────────────────────────

    async def get_entity_column(self, source_id: str) -> tuple[str, str] | None:
        """Return (table_name, column_name) where entity names are stored, or None."""
        await self._ensure_ready()
        try:
            import aiosqlite
            async with aiosqlite.connect(self._db_path) as db:
                async with db.execute(
                    "SELECT key, value FROM source_facts "
                    "WHERE source_id=? AND fact_type='entity_column' LIMIT 1",
                    (source_id,),
                ) as cur:
                    row = await cur.fetchone()
                    if row:
                        data = json.loads(row[1])
                        return (row[0], data["column"])
        except Exception as exc:
            logger.debug("semantic_memory.get_entity_column_failed", error=str(exc))
        return None

    async def save_entity_column(self, source_id: str, table: str, column: str) -> None:
        """Persist the entity column location for this source."""
        await self._ensure_ready()
        try:
            import aiosqlite
            now = datetime.now(timezone.utc).isoformat()
            async with aiosqlite.connect(self._db_path) as db:
                await db.execute(
                    "INSERT OR REPLACE INTO source_facts VALUES (?,?,?,?,?,?)",
                    (source_id, "entity_column", table, json.dumps({"column": column}), 1.0, now),
                )
                await db.commit()
        except Exception as exc:
            logger.debug("semantic_memory.save_entity_column_failed", error=str(exc))

    # ── Known entity values ───────────────────────────────────────────────────

    async def get_known_entities(self, source_id: str) -> list[str]:
        """Return all entity values known to exist in this source."""
        await self._ensure_ready()
        try:
            import aiosqlite
            async with aiosqlite.connect(self._db_path) as db:
                async with db.execute(
                    "SELECT value FROM source_facts "
                    "WHERE source_id=? AND fact_type='coverage' AND key='entities_found'",
                    (source_id,),
                ) as cur:
                    row = await cur.fetchone()
                    if row:
                        return json.loads(row[0])
        except Exception as exc:
            logger.debug("semantic_memory.get_known_entities_failed", error=str(exc))
        return []

    async def save_discovered_entities(
        self, source_id: str, table: str, column: str, entity_values: list[str]
    ) -> None:
        """Persist entity column location + merge newly discovered entity values."""
        await self._ensure_ready()
        try:
            import aiosqlite
            now = datetime.now(timezone.utc).isoformat()
            async with aiosqlite.connect(self._db_path) as db:
                # Save entity column location
                await db.execute(
                    "INSERT OR REPLACE INTO source_facts VALUES (?,?,?,?,?,?)",
                    (source_id, "entity_column", table, json.dumps({"column": column}), 1.0, now),
                )
                # Merge with existing known entities
                async with db.execute(
                    "SELECT value FROM source_facts "
                    "WHERE source_id=? AND fact_type='coverage' AND key='entities_found'",
                    (source_id,),
                ) as cur:
                    row = await cur.fetchone()
                existing: list[str] = json.loads(row[0]) if row else []
                merged = list({v.lower(): v for v in existing + entity_values}.values())
                await db.execute(
                    "INSERT OR REPLACE INTO source_facts VALUES (?,?,?,?,?,?)",
                    (source_id, "coverage", "entities_found", json.dumps(merged), 1.0, now),
                )
                await db.commit()
        except Exception as exc:
            logger.debug("semantic_memory.save_entities_failed", error=str(exc))

    # ── Concept → column mappings ─────────────────────────────────────────────

    async def get_concept_column(self, source_id: str, concept: str) -> str | None:
        """Return the column name that stores this concept (e.g. 'unemployment' → 'obs_value')."""
        await self._ensure_ready()
        try:
            import aiosqlite
            async with aiosqlite.connect(self._db_path) as db:
                async with db.execute(
                    "SELECT value FROM source_facts "
                    "WHERE source_id=? AND fact_type='concept_column' AND key=?",
                    (source_id, concept.lower()),
                ) as cur:
                    row = await cur.fetchone()
                    if row:
                        return row[0]
        except Exception as exc:
            logger.debug("semantic_memory.get_concept_column_failed", error=str(exc))
        return None

    async def save_concept_column(self, source_id: str, concept: str, column_name: str) -> None:
        """Persist concept → column_name mapping."""
        await self._ensure_ready()
        try:
            import aiosqlite
            now = datetime.now(timezone.utc).isoformat()
            async with aiosqlite.connect(self._db_path) as db:
                await db.execute(
                    "INSERT OR REPLACE INTO source_facts VALUES (?,?,?,?,?,?)",
                    (source_id, "concept_column", concept.lower(), column_name, 1.0, now),
                )
                await db.commit()
        except Exception as exc:
            logger.debug("semantic_memory.save_concept_column_failed", error=str(exc))

    async def entities_covered(self, source_id: str, requested: list[str]) -> dict[str, bool]:
        """Check which requested entities are known to exist in this source.

        Returns {entity: True/False}. Unknown entities get False (not cached = not known).
        Also checks saved entity aliases so 'VNT' resolves via 'Vietnam'.
        """
        known = {e.lower() for e in await self.get_known_entities(source_id)}
        result = {}
        for e in requested:
            if e.lower() in known:
                result[e] = True
            else:
                # check if there's a saved alias mapping for this entity
                canonical = await self.get_entity_alias(source_id, e)
                result[e] = (canonical is not None and canonical.lower() in known)
        return result

    async def save_entity_alias(self, source_id: str, alias: str, canonical: str) -> None:
        """Persist a user/alias → canonical entity name mapping.

        e.g. alias='VNT', canonical='Vietnam'
        Stored as fact_type='entity_alias', key=alias.lower(), value=canonical.
        """
        await self._ensure_ready()
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                """INSERT INTO source_facts(source_id, fact_type, key, value, confidence, updated_at)
                   VALUES(?,?,?,?,?,?)
                   ON CONFLICT(source_id, fact_type, key) DO UPDATE SET
                     value=excluded.value, confidence=excluded.confidence, updated_at=excluded.updated_at""",
                (source_id, "entity_alias", alias.lower(), canonical, 1.0,
                 datetime.now(timezone.utc).isoformat()),
            )
            await db.commit()

    async def get_entity_alias(self, source_id: str, alias: str) -> str | None:
        """Look up a saved alias → canonical name mapping, or None if not found."""
        await self._ensure_ready()
        async with aiosqlite.connect(self._db_path) as db:
            async with db.execute(
                "SELECT value FROM source_facts WHERE source_id=? AND fact_type='entity_alias' AND key=?",
                (source_id, alias.lower()),
            ) as cur:
                row = await cur.fetchone()
                return row[0] if row else None
