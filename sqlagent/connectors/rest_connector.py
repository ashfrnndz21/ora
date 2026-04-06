"""REST API Connector — base class for all SaaS/POS/ERP/CRM integrations.

Handles auth (OAuth2, API key, Bearer), pagination (cursor, offset, link-header),
rate limiting (token bucket with backoff), and schema inference. Data gets pulled
into DuckDB for SQL querying — every REST source becomes a queryable table.

Usage:
    class ShopifyConnector(RestConnector):
        dialect = "shopify"
        auth_type = "oauth2"
        base_url = "https://{store}.myshopify.com/admin/api/2024-01"
        endpoints = {"orders": "/orders.json", "customers": "/customers.json"}
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urljoin

import pandas as pd
import structlog

logger = structlog.get_logger()


@dataclass
class AuthConfig:
    """Authentication configuration for REST APIs."""
    auth_type: str = "api_key"  # api_key | bearer | oauth2 | basic
    api_key: str = ""
    api_key_header: str = "Authorization"  # or X-API-Key, etc.
    api_key_prefix: str = "Bearer"  # prefix before key value
    # OAuth2
    client_id: str = ""
    client_secret: str = ""
    token_url: str = ""
    access_token: str = ""
    refresh_token: str = ""
    token_expires_at: float = 0.0
    # Basic
    username: str = ""
    password: str = ""


@dataclass
class PaginationConfig:
    """Pagination strategy for REST APIs."""
    strategy: str = "none"  # none | cursor | offset | link_header | page_number
    cursor_param: str = "cursor"
    cursor_path: str = ""  # JSON path to next cursor in response
    offset_param: str = "offset"
    limit_param: str = "limit"
    page_param: str = "page"
    per_page: int = 100
    max_pages: int = 50  # safety cap


@dataclass
class RateLimitConfig:
    """Rate limiting configuration."""
    requests_per_second: float = 2.0
    burst_size: int = 5
    retry_after_header: str = "Retry-After"
    max_retries: int = 3
    backoff_factor: float = 2.0


@dataclass
class EndpointDef:
    """Definition of a single API endpoint → table."""
    path: str = ""
    method: str = "GET"
    params: dict = field(default_factory=dict)
    data_path: str = ""  # JSON path to the array of records (e.g. "orders", "data.items")
    table_name: str = ""  # override table name (default: derived from endpoint)


class RestConnector:
    """Base class for REST API connectors.

    Subclasses define endpoints and auth — this class handles the mechanics
    of pulling data, paginating, rate limiting, and loading into DuckDB.
    """

    # Subclasses override these
    dialect: str = "rest"
    base_url: str = ""
    endpoints: dict[str, str | EndpointDef] = {}
    default_auth_type: str = "api_key"
    default_pagination: str = "none"
    default_data_path: str = ""  # e.g. "data" if all responses wrap in {"data": [...]}

    def __init__(
        self,
        source_id: str,
        base_url: str = "",
        auth: AuthConfig | None = None,
        pagination: PaginationConfig | None = None,
        rate_limit: RateLimitConfig | None = None,
        config: dict | None = None,
    ):
        self._source_id = source_id
        self._base_url = base_url or self.base_url
        self._auth = auth or AuthConfig(auth_type=self.default_auth_type)
        self._pagination = pagination or PaginationConfig(strategy=self.default_pagination)
        self._rate_limit = rate_limit or RateLimitConfig()
        self._config = config or {}
        self._conn = None  # DuckDB connection for querying
        self._tables: dict[str, pd.DataFrame] = {}
        self._last_pull: dict[str, float] = {}
        self._token_bucket = self._rate_limit.burst_size
        self._last_request = 0.0

        # Apply config overrides (e.g. store name in URL template)
        if self._config:
            for key, val in self._config.items():
                self._base_url = self._base_url.replace(f"{{{key}}}", str(val))

    @property
    def source_id(self) -> str:
        return self._source_id

    # ── Auth ──────────────────────────────────────────────────────────

    def _build_headers(self) -> dict[str, str]:
        """Build request headers with authentication."""
        headers = {"Accept": "application/json", "Content-Type": "application/json"}
        auth = self._auth

        if auth.auth_type == "api_key":
            if auth.api_key_prefix:
                headers[auth.api_key_header] = f"{auth.api_key_prefix} {auth.api_key}"
            else:
                headers[auth.api_key_header] = auth.api_key
        elif auth.auth_type == "bearer":
            headers["Authorization"] = f"Bearer {auth.access_token or auth.api_key}"
        elif auth.auth_type == "oauth2":
            headers["Authorization"] = f"Bearer {auth.access_token}"
        elif auth.auth_type == "basic":
            import base64
            creds = base64.b64encode(f"{auth.username}:{auth.password}".encode()).decode()
            headers["Authorization"] = f"Basic {creds}"

        return headers

    async def _refresh_oauth2_token(self) -> None:
        """Refresh an expired OAuth2 access token."""
        if not self._auth.refresh_token or not self._auth.token_url:
            return
        if self._auth.token_expires_at > time.time() + 60:
            return  # not expired yet

        try:
            import httpx
            async with httpx.AsyncClient() as client:
                resp = await client.post(self._auth.token_url, data={
                    "grant_type": "refresh_token",
                    "refresh_token": self._auth.refresh_token,
                    "client_id": self._auth.client_id,
                    "client_secret": self._auth.client_secret,
                })
                data = resp.json()
                self._auth.access_token = data.get("access_token", "")
                self._auth.token_expires_at = time.time() + data.get("expires_in", 3600)
                if data.get("refresh_token"):
                    self._auth.refresh_token = data["refresh_token"]
                logger.info("rest.oauth2.refreshed", source_id=self._source_id)
        except Exception as exc:
            logger.warning("rest.oauth2.refresh_failed", error=str(exc))

    # ── Rate Limiting ─────────────────────────────────────────────────

    async def _wait_for_rate_limit(self) -> None:
        """Token bucket rate limiter with backoff."""
        now = time.monotonic()
        elapsed = now - self._last_request
        self._token_bucket = min(
            self._rate_limit.burst_size,
            self._token_bucket + elapsed * self._rate_limit.requests_per_second,
        )
        if self._token_bucket < 1.0:
            wait = (1.0 - self._token_bucket) / self._rate_limit.requests_per_second
            await asyncio.sleep(wait)
            self._token_bucket = 1.0
        self._token_bucket -= 1.0
        self._last_request = time.monotonic()

    # ── HTTP Request ──────────────────────────────────────────────────

    async def _request(self, method: str, url: str, params: dict | None = None) -> dict:
        """Make an authenticated, rate-limited HTTP request."""
        import httpx

        if self._auth.auth_type == "oauth2":
            await self._refresh_oauth2_token()

        await self._wait_for_rate_limit()
        headers = self._build_headers()

        for attempt in range(self._rate_limit.max_retries):
            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    resp = await client.request(method, url, headers=headers, params=params)

                    if resp.status_code == 429:
                        retry_after = float(resp.headers.get(
                            self._rate_limit.retry_after_header,
                            self._rate_limit.backoff_factor ** attempt
                        ))
                        logger.info("rest.rate_limited", wait=retry_after, attempt=attempt)
                        await asyncio.sleep(retry_after)
                        continue

                    resp.raise_for_status()
                    return resp.json()

            except Exception as exc:
                if attempt < self._rate_limit.max_retries - 1:
                    wait = self._rate_limit.backoff_factor ** attempt
                    await asyncio.sleep(wait)
                else:
                    raise

        return {}

    # ── Data Extraction ───────────────────────────────────────────────

    def _extract_data(self, response: dict, data_path: str = "") -> list[dict]:
        """Extract records array from API response using JSON path."""
        path = data_path or self.default_data_path
        if not path:
            if isinstance(response, list):
                return response
            return [response]

        obj = response
        for key in path.split("."):
            if isinstance(obj, dict):
                obj = obj.get(key, [])
            else:
                return []
        return obj if isinstance(obj, list) else [obj]

    def _get_next_cursor(self, response: dict, headers: dict | None = None) -> str | None:
        """Extract pagination cursor from response."""
        pg = self._pagination

        if pg.strategy == "cursor" and pg.cursor_path:
            obj = response
            for key in pg.cursor_path.split("."):
                if isinstance(obj, dict):
                    obj = obj.get(key)
                else:
                    return None
            return str(obj) if obj else None

        if pg.strategy == "link_header" and headers:
            link = headers.get("Link", "")
            for part in link.split(","):
                if 'rel="next"' in part:
                    url = part.split(";")[0].strip().strip("<>")
                    return url

        return None

    # ── Pull All Data ─────────────────────────────────────────────────

    async def _pull_endpoint(self, name: str, endpoint: str | EndpointDef) -> pd.DataFrame:
        """Pull all pages from a single endpoint → DataFrame."""
        if isinstance(endpoint, str):
            ep = EndpointDef(path=endpoint, data_path=self.default_data_path)
        else:
            ep = endpoint

        url = urljoin(self._base_url.rstrip("/") + "/", ep.path.lstrip("/"))
        params = dict(ep.params)
        all_records: list[dict] = []
        pg = self._pagination

        if pg.strategy in ("offset", "page_number"):
            params[pg.limit_param] = pg.per_page

        for page in range(pg.max_pages):
            resp = await self._request(ep.method, url, params)
            records = self._extract_data(resp, ep.data_path)
            if not records:
                break
            all_records.extend(records)

            # Next page logic
            if pg.strategy == "cursor":
                cursor = self._get_next_cursor(resp)
                if not cursor:
                    break
                params[pg.cursor_param] = cursor
            elif pg.strategy == "offset":
                if len(records) < pg.per_page:
                    break
                params[pg.offset_param] = params.get(pg.offset_param, 0) + pg.per_page
            elif pg.strategy == "page_number":
                if len(records) < pg.per_page:
                    break
                params[pg.page_param] = params.get(pg.page_param, 1) + 1
            elif pg.strategy == "link_header":
                next_url = self._get_next_cursor(resp, getattr(resp, 'headers', {}))
                if not next_url:
                    break
                url = next_url
                params = {}  # URL already contains params
            else:
                break  # no pagination

        if not all_records:
            return pd.DataFrame()

        df = pd.json_normalize(all_records, sep="_")
        logger.info("rest.pulled", endpoint=name, records=len(df), columns=len(df.columns))
        return df

    # ── Connector Protocol ────────────────────────────────────────────

    async def connect(self) -> None:
        """Pull data from all endpoints and load into DuckDB."""
        import duckdb
        self._conn = duckdb.connect(":memory:")

        for name, endpoint in self.endpoints.items():
            try:
                df = await self._pull_endpoint(name, endpoint)
                if not df.empty:
                    table_name = name.replace("/", "_").replace(".", "_").strip("_")
                    self._conn.register(table_name, df)
                    self._tables[table_name] = df
                    logger.info("rest.table_registered", table=table_name, rows=len(df))
            except Exception as exc:
                logger.warning("rest.endpoint_failed", endpoint=name, error=str(exc))

    async def execute(
        self, sql: str, timeout_s: float = 30.0, max_rows: int = 10_000
    ) -> pd.DataFrame:
        if self._conn is None:
            await self.connect()
        try:
            result = self._conn.execute(sql)
            return result.fetchdf().head(max_rows)
        except Exception as e:
            from sqlagent.exceptions import SQLExecutionFailed
            raise SQLExecutionFailed(sql=sql, error=str(e))

    async def introspect(self):
        """Build schema from loaded tables."""
        if self._conn is None:
            await self.connect()

        from sqlagent.models import SchemaSnapshot, SchemaTable, SchemaColumn

        tables = []
        for table_name, df in self._tables.items():
            columns = []
            for col in df.columns:
                dtype = str(df[col].dtype)
                examples = [str(v) for v in df[col].dropna().unique()[:6]]
                columns.append(SchemaColumn(
                    name=col, data_type=dtype, examples=examples,
                ))
            tables.append(SchemaTable(
                name=table_name, columns=columns,
                row_count_estimate=len(df),
            ))

        return SchemaSnapshot(tables=tables)

    async def sample(self, table: str, n: int = 5):
        from sqlagent.models import SampleData
        if table in self._tables:
            df = self._tables[table].head(n)
            return SampleData(table=table, rows=df.to_dict("records"), columns=list(df.columns))
        return SampleData(table=table, rows=[], columns=[])

    async def health_check(self) -> bool:
        try:
            headers = self._build_headers()
            import httpx
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(self._base_url, headers=headers)
                return resp.status_code < 500
        except Exception:
            return False
