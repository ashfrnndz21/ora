"""Vector store + example store — training pair retrieval.

Stores NL→SQL pairs in Qdrant (in-process, no server needed).
Retrieves similar examples for few-shot prompting.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Protocol, runtime_checkable, Any

import structlog

from sqlagent.models import TrainingExample, SearchResult

logger = structlog.get_logger()


# ═══════════════════════════════════════════════════════════════════════════════
# VECTOR STORE PROTOCOL
# ═══════════════════════════════════════════════════════════════════════════════

@runtime_checkable
class VectorStore(Protocol):
    async def upsert(self, id: str, vector: list[float], payload: dict) -> None: ...
    async def search(self, vector: list[float], top_k: int = 3) -> list[dict]: ...
    async def delete(self, id: str) -> None: ...
    async def count(self) -> int: ...
    async def ensure_collection(self, name: str, dimensions: int) -> None: ...


# ═══════════════════════════════════════════════════════════════════════════════
# QDRANT VECTOR STORE (in-process)
# ═══════════════════════════════════════════════════════════════════════════════

class QdrantVectorStore:
    """In-process Qdrant vector store — no external server needed."""

    def __init__(self, path: str = "", collection_name: str = "training_examples"):
        self._path = path
        self._collection = collection_name
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            from qdrant_client import QdrantClient
            if self._path:
                self._client = QdrantClient(path=self._path)
            else:
                self._client = QdrantClient(":memory:")

    async def ensure_collection(self, name: str = "", dimensions: int = 384) -> None:
        self._ensure_client()
        collection = name or self._collection
        from qdrant_client.models import Distance, VectorParams
        try:
            self._client.get_collection(collection)
        except Exception as exc:
            logger.debug("retrieval.operation_failed", error=str(exc))
            self._client.create_collection(
                collection_name=collection,
                vectors_config=VectorParams(size=dimensions, distance=Distance.COSINE),
            )

    async def upsert(self, id: str, vector: list[float], payload: dict) -> None:
        self._ensure_client()
        from qdrant_client.models import PointStruct
        self._client.upsert(
            collection_name=self._collection,
            points=[PointStruct(id=id, vector=vector, payload=payload)],
        )

    async def search(self, vector: list[float], top_k: int = 3) -> list[dict]:
        self._ensure_client()
        try:
            results = self._client.search(
                collection_name=self._collection,
                query_vector=vector,
                limit=top_k,
            )
            return [
                {"payload": r.payload, "score": r.score, "id": r.id}
                for r in results
            ]
        except Exception as exc:
            logger.debug("retrieval.operation_failed", error=str(exc))
            return []

    async def delete(self, id: str) -> None:
        self._ensure_client()
        from qdrant_client.models import PointIdsList
        self._client.delete(
            collection_name=self._collection,
            points_selector=PointIdsList(points=[id]),
        )

    async def count(self) -> int:
        self._ensure_client()
        try:
            info = self._client.get_collection(self._collection)
            return info.points_count
        except Exception as exc:
            logger.debug("retrieval.operation_failed", error=str(exc))
            return 0


# ═══════════════════════════════════════════════════════════════════════════════
# EXAMPLE STORE (high-level: embed + store + search)
# ═══════════════════════════════════════════════════════════════════════════════

class ExampleStore:
    """High-level interface for NL→SQL training pair storage and retrieval.

    Automatically embeds text before storing/searching.
    """

    def __init__(self, vector_store: QdrantVectorStore, embedder: Any):
        self._store = vector_store
        self._embedder = embedder

    async def init(self, dimensions: int = 384) -> None:
        """Initialize the vector store collection."""
        await self._store.ensure_collection(dimensions=dimensions)

    async def add(
        self,
        nl_query: str,
        sql: str,
        ddl: str = "",
        source_id: str = "",
        generator: str = "",
        verified: bool = False,
    ) -> str:
        """Add a training example. Returns the example ID."""
        example_id = str(uuid.uuid4())

        # Embed the natural language query
        embeddings = await self._embedder.embed([nl_query])
        vector = embeddings[0]

        payload = {
            "nl_query": nl_query,
            "sql": sql,
            "ddl": ddl,
            "source_id": source_id,
            "generator": generator,
            "verified": verified,
        }

        await self._store.upsert(id=example_id, vector=vector, payload=payload)
        logger.info("example.added", nl=nl_query[:50], source=source_id)
        return example_id

    async def add_batch(self, examples: list[TrainingExample]) -> int:
        """Add multiple training examples. Returns count added."""
        count = 0
        for ex in examples:
            await self.add(
                nl_query=ex.nl_query,
                sql=ex.sql,
                ddl=ex.ddl,
                source_id=ex.source_id,
                generator=ex.generator,
                verified=ex.verified,
            )
            count += 1
        return count

    async def search(self, nl_query: str, top_k: int = 3) -> list[SearchResult]:
        """Find similar training examples for few-shot prompting."""
        embeddings = await self._embedder.embed([nl_query])
        vector = embeddings[0]

        results = await self._store.search(vector=vector, top_k=top_k)

        return [
            SearchResult(
                example=TrainingExample(
                    nl_query=r["payload"].get("nl_query", ""),
                    sql=r["payload"].get("sql", ""),
                    ddl=r["payload"].get("ddl", ""),
                    source_id=r["payload"].get("source_id", ""),
                    generator=r["payload"].get("generator", ""),
                    verified=r["payload"].get("verified", False),
                ),
                similarity=r.get("score", 0.0),
            )
            for r in results
        ]

    async def count(self) -> int:
        return await self._store.count()
