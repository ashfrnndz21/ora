"""Business Glossary — LLM-powered term resolution.

Resolves business terms to canonical column/metric definitions using
embedding-based semantic search, NOT substring matching or hardcoded rules.

The Semantic Agent calls these methods — external components do not write
directly to the glossary.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

import structlog

logger = structlog.get_logger()


class GlossaryEntry(BaseModel):
    """A business term with its definition and column binding."""

    term: str  # "active customer"
    definition: str = ""  # "Customer with purchase in last 90 days"
    sql_expr: Optional[str] = None  # "last_purchase_date >= CURRENT_DATE - INTERVAL '90 days'"
    table: Optional[str] = None  # "customers"
    column: Optional[str] = None  # "last_purchase_date"
    owner: Optional[str] = None  # "data-team@company.com"
    tags: list[str] = Field(default_factory=list)
    aliases: list[str] = Field(default_factory=list)  # ["active account", "recent buyer"]
    # Provenance
    source: str = "manual"  # "manual" | "imported" | "learned"
    confidence: float = 1.0


class BusinessGlossary(BaseModel):
    """Collection of business term definitions with LLM-powered resolution.

    Resolution strategy (fully agentic — no substring/regex):
      1. Exact match on term or alias (O(1) dict lookup — structural, not rules-based)
      2. Embedding search across all terms/aliases (via vector store if provided)
      3. LLM reasoning as fallback when embeddings are ambiguous
    """

    entries: list[GlossaryEntry] = Field(default_factory=list)

    def resolve_exact(self, term: str) -> Optional[GlossaryEntry]:
        """Fast exact lookup by term or alias. O(n) scan — no LLM needed."""
        term_lower = term.lower().strip()
        for entry in self.entries:
            if entry.term.lower() == term_lower:
                return entry
            for alias in entry.aliases:
                if alias.lower() == term_lower:
                    return entry
        return None

    async def resolve(
        self,
        term: str,
        embedder=None,
        llm=None,
        threshold: float = 0.82,
    ) -> Optional[GlossaryEntry]:
        """Resolve a business term to a glossary entry.

        Strategy:
          1. Exact match (term or alias) — instant, no cost
          2. Embedding similarity (if embedder provided) — fast, no LLM cost
          3. LLM reasoning (if llm provided) — slowest but handles novel terms

        Args:
            term: The business term to resolve
            embedder: Embedding model for semantic similarity
            llm: LLM provider for reasoning fallback
            threshold: Minimum cosine similarity for embedding match

        Returns:
            Matching GlossaryEntry or None
        """
        # 1. Exact match
        exact = self.resolve_exact(term)
        if exact:
            return exact

        if not self.entries:
            return None

        # 2. Embedding-based semantic search
        if embedder is not None:
            try:
                # Embed the query term
                query_embedding = await embedder.embed(term)

                # Embed all glossary terms + aliases and find best match
                best_score = 0.0
                best_entry: Optional[GlossaryEntry] = None

                for entry in self.entries:
                    # Build candidate texts: term + all aliases
                    candidates = [entry.term] + entry.aliases
                    if entry.definition:
                        candidates.append(entry.definition)

                    for candidate in candidates:
                        candidate_embedding = await embedder.embed(candidate)
                        score = _cosine_similarity(query_embedding, candidate_embedding)
                        if score > best_score:
                            best_score = score
                            best_entry = entry

                if best_entry and best_score >= threshold:
                    logger.info(
                        "glossary.embedding_match",
                        term=term,
                        matched=best_entry.term,
                        score=round(best_score, 3),
                    )
                    return best_entry
            except Exception as e:
                logger.warning("glossary.embedding_failed", error=str(e))

        # 3. LLM reasoning fallback
        if llm is not None:
            try:
                entry_descriptions = "\n".join(
                    f"- {e.term}: {e.definition} (aliases: {', '.join(e.aliases)})"
                    for e in self.entries
                )
                prompt = (
                    f"Given these business term definitions:\n{entry_descriptions}\n\n"
                    f"Which term best matches '{term}'? "
                    f"Respond with ONLY the exact term name, or 'NONE' if no match."
                )
                resp = await llm.complete(
                    [{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=50,
                )
                matched_term = resp.content.strip().strip('"').strip("'")
                if matched_term.upper() != "NONE":
                    for entry in self.entries:
                        if entry.term.lower() == matched_term.lower():
                            logger.info(
                                "glossary.llm_match",
                                term=term,
                                matched=entry.term,
                            )
                            return entry
            except Exception as e:
                logger.warning("glossary.llm_failed", error=str(e))

        return None

    def add(self, entry: GlossaryEntry) -> None:
        """Add or update a glossary entry."""
        # Replace if term already exists
        for i, existing in enumerate(self.entries):
            if existing.term.lower() == entry.term.lower():
                self.entries[i] = entry
                return
        self.entries.append(entry)

    def remove(self, term: str) -> bool:
        """Remove a glossary entry by term name. Returns True if found."""
        for i, entry in enumerate(self.entries):
            if entry.term.lower() == term.lower():
                self.entries.pop(i)
                return True
        return False

    def all_terms(self) -> list[str]:
        """Return all terms and aliases as a flat list."""
        terms = []
        for entry in self.entries:
            terms.append(entry.term)
            terms.extend(entry.aliases)
        return terms


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
