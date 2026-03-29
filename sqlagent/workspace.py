"""Workspace store — CRUD for multi-workspace persistence."""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone

import structlog

from sqlagent.models import Workspace, WorkspaceStatus
from sqlagent.exceptions import WorkspaceNotFound

logger = structlog.get_logger()


class WorkspaceStore:
    """SQLite-backed workspace CRUD."""

    def __init__(self, db_path: str = ""):
        if not db_path:
            db_path = os.path.join(os.path.expanduser("~"), ".sqlagent", "workspaces.db")
        self._db_path = db_path
        self._initialized = False

    async def init(self) -> None:
        if self._initialized:
            return
        import aiosqlite
        os.makedirs(os.path.dirname(self._db_path), exist_ok=True)
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS workspaces (
                    workspace_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    owner_id TEXT NOT NULL,
                    description TEXT DEFAULT '',
                    status TEXT DEFAULT 'setup',
                    sources_json TEXT DEFAULT '[]',
                    knowledge_graph_version INTEGER DEFAULT 0,
                    query_count INTEGER DEFAULT 0,
                    created_at TEXT,
                    updated_at TEXT
                )
            """)
            await db.commit()
        self._initialized = True

    async def create(
        self, name: str, owner_id: str, description: str = "",
    ) -> Workspace:
        await self.init()
        import aiosqlite
        ws_id = f"ws_{uuid.uuid4().hex[:12]}"
        now = datetime.now(timezone.utc).isoformat()
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                "INSERT INTO workspaces VALUES (?,?,?,?,?,?,?,?,?,?)",
                (ws_id, name, owner_id, description, "setup", "[]", 0, 0, now, now),
            )
            await db.commit()
        return Workspace(
            workspace_id=ws_id, name=name, owner_id=owner_id,
            description=description, status=WorkspaceStatus.SETUP,
        )

    async def get(self, workspace_id: str) -> Workspace:
        await self.init()
        import aiosqlite
        async with aiosqlite.connect(self._db_path) as db:
            cursor = await db.execute(
                "SELECT * FROM workspaces WHERE workspace_id = ?", (workspace_id,)
            )
            row = await cursor.fetchone()
            if not row:
                raise WorkspaceNotFound(f"Workspace {workspace_id} not found")
            return self._row_to_workspace(row)

    async def list_for_user(self, owner_id: str) -> list[Workspace]:
        await self.init()
        import aiosqlite
        async with aiosqlite.connect(self._db_path) as db:
            cursor = await db.execute(
                "SELECT * FROM workspaces WHERE owner_id = ? ORDER BY updated_at DESC",
                (owner_id,),
            )
            rows = await cursor.fetchall()
            return [self._row_to_workspace(r) for r in rows]

    async def update(self, workspace_id: str, **fields) -> Workspace:
        await self.init()
        import aiosqlite
        allowed = {"name", "description", "status", "sources_json", "knowledge_graph_version", "query_count"}
        updates = {k: v for k, v in fields.items() if k in allowed}
        if not updates:
            return await self.get(workspace_id)

        updates["updated_at"] = datetime.now(timezone.utc).isoformat()
        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = list(updates.values()) + [workspace_id]

        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                f"UPDATE workspaces SET {set_clause} WHERE workspace_id = ?",
                values,
            )
            await db.commit()
        return await self.get(workspace_id)

    async def add_source(self, workspace_id: str, source_config: dict) -> None:
        """Add a data source to a workspace."""
        ws = await self.get(workspace_id)
        sources = ws.sources + [source_config]
        await self.update(workspace_id, sources_json=json.dumps(sources))

    async def delete(self, workspace_id: str) -> None:
        await self.init()
        import aiosqlite
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute("DELETE FROM workspaces WHERE workspace_id = ?", (workspace_id,))
            await db.commit()

    @staticmethod
    def _row_to_workspace(row) -> Workspace:
        return Workspace(
            workspace_id=row[0], name=row[1], owner_id=row[2],
            description=row[3],
            status=WorkspaceStatus(row[4]) if row[4] in WorkspaceStatus.__members__.values() else WorkspaceStatus.SETUP,
            sources=json.loads(row[5]) if row[5] else [],
            knowledge_graph_version=row[6] or 0,
            query_count=row[7] or 0,
        )
