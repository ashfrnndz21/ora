"""Setup Agent — conversational workspace creation.

Re-exported from the legacy sqlagent/agents.py standalone file.
This wrapper ensures SetupAgent is importable from sqlagent.agents package.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, AsyncIterator


@dataclass
class SetupEvent:
    event_type: str = ""  # message, action, source_added, workspace_ready, error
    data: dict = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


_SETUP_SYSTEM = """You know data, SQL, and how analysts think. Talk like a sharp data colleague — direct, insightful, no filler.

No emojis. No markdown headers. No bullet points. Plain sentences only.
Never say "I'm here to help", "Great!", "Sure!", or any filler opening.

When the user asks about the data (e.g. "what's this about?", "describe the data", "what columns?"), look at the column names and table names in the [System] context and give a sharp 2-3 sentence description of what the dataset covers — what it measures, the key dimensions, the grain. Infer meaning from names: "occupazione" = employment, "geo" = geography, "anno" = year, etc.

When the user gives you a database URL (postgresql://, mysql://, snowflake://, bigquery://, sqlite://, redshift://, duckdb://), output this on its own line:
{"tool": "connect_source", "type": "postgresql", "connection_string": "...url..."}

When the user wants to name or rename the workspace, output this on its own line — this works at ANY time, even before data is connected:
{"tool": "set_workspace_name", "name": "Short Name"}

When the user has at least one data source connected AND has explicitly confirmed they are ready to open the workspace, output this on its own line:
{"tool": "finalize_setup"}

NEVER call finalize_setup automatically. Wait for the user to say something like "open workspace", "let's go", "I'm ready", "looks good", "open it", or similar explicit confirmation.
When files are uploaded, describe what the data contains based on the column and table names, then ask what to call the workspace — but always answer data questions first before asking for a name.
Only call connect_source for real DB URLs — never for plain text.

When the user wants to connect a SaaS platform (Shopify, Salesforce, Stripe, HubSpot, Google Analytics), ask for their API key or credentials and output:
{"tool": "connect_source", "type": "shopify", "connection_string": "shopify://storename?api_key=..."}
Supported SaaS URLs: shopify://, salesforce://, stripe://, hubspot://, ga4://, airbyte://"""


class SetupAgent:
    """Conversational agent for workspace creation.

    Instead of a form wizard, the user describes their data in natural language.
    The agent asks questions, collects connection details, and configures the workspace.
    """

    def __init__(self, llm: Any, connector_registry: Any = None):
        self._llm = llm
        self._registry = connector_registry
        self._conversations: dict[str, list[dict]] = {}  # workspace_id → messages

    async def chat(
        self,
        workspace_id: str,
        user_message: str,
        workspace_state: dict | None = None,
    ) -> AsyncIterator[SetupEvent]:
        """Process one user turn. Yields SetupEvent objects."""
        if workspace_id not in self._conversations:
            self._conversations[workspace_id] = [
                {"role": "system", "content": _SETUP_SYSTEM},
            ]

        self._conversations[workspace_id].append({"role": "user", "content": user_message})

        # Add workspace state context
        if workspace_state:
            state_msg = f"Current workspace state: {json.dumps(workspace_state)}"
            messages = self._conversations[workspace_id] + [
                {"role": "system", "content": state_msg}
            ]
        else:
            messages = self._conversations[workspace_id]

        resp = await self._llm.complete(messages)
        content = resp.content

        self._conversations[workspace_id].append({"role": "assistant", "content": content})

        # Known URL schemes for connect_source validation
        _VALID_URL_SCHEMES = (
            "postgresql://", "postgres://", "mysql://", "sqlite://",
            "snowflake://", "bigquery://", "redshift+psycopg2://",
            "duckdb://", "file://", "/", "./",
        )

        # Extract all JSON tool calls from the response
        import re as _re

        tool_calls = []
        for m in _re.finditer(r'\{[^{}]*"tool"[^{}]*\}', content):
            try:
                tool_calls.append(json.loads(m.group()))
            except json.JSONDecodeError:
                pass

        # Collect non-tool text
        text_only = _re.sub(r'\{[^{}]*"tool"[^{}]*\}', "", content).strip()
        if text_only:
            yield SetupEvent(event_type="message", data={"text": text_only})

        for tool_json in tool_calls:
            tool_name = tool_json.get("tool", "")

            if tool_name == "connect_source":
                conn_str = tool_json.get("connection_string", "")
                if not any(conn_str.startswith(s) for s in _VALID_URL_SCHEMES):
                    continue
                yield SetupEvent(
                    event_type="action",
                    data={"action": "connecting", "type": tool_json.get("type"), "url": conn_str},
                )
                try:
                    from sqlagent.connectors import ConnectorRegistry
                    conn = ConnectorRegistry.from_url(
                        source_id=f"src_{tool_json.get('type', 'unknown')}",
                        url=conn_str,
                    )
                    snap = await conn.introspect()
                    yield SetupEvent(
                        event_type="source_added",
                        data={
                            "source_id": conn.source_id,
                            "dialect": conn.dialect,
                            "table_count": snap.table_count,
                            "column_count": snap.column_count,
                        },
                    )
                except Exception as e:
                    yield SetupEvent(event_type="error", data={"error": str(e)})

            elif tool_name == "set_workspace_name":
                yield SetupEvent(
                    event_type="action",
                    data={"action": "naming", "name": tool_json.get("name", "")},
                )

            elif tool_name == "finalize_setup":
                yield SetupEvent(event_type="workspace_ready", data={})
