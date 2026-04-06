"""Google Analytics 4 connector — events, sessions, users, conversions."""

from sqlagent.connectors.rest_connector import (
    RestConnector, AuthConfig, PaginationConfig, EndpointDef,
)


class GoogleAnalyticsConnector(RestConnector):
    """Google Analytics 4 Data API connector.

    Connection: google_analytics://{property_id}
    Auth: OAuth2 (service account or user credentials)

    Note: GA4 uses a reporting API (POST with JSON body), not a REST list endpoint.
    This connector pulls pre-defined reports into tables.
    """

    dialect = "google_analytics"
    base_url = "https://analyticsdata.googleapis.com/v1beta"
    default_auth_type = "oauth2"
    default_pagination = "offset"

    def __init__(self, source_id: str, property_id: str,
                 credentials_json: str = "", access_token: str = "", **kwargs):
        self._property_id = property_id
        # GA4 uses reporting API — define as endpoints that POST report requests
        self.endpoints = {
            "sessions": EndpointDef(
                path=f"/properties/{property_id}:runReport",
                method="POST",
                data_path="rows",
            ),
            "users": EndpointDef(
                path=f"/properties/{property_id}:runReport",
                method="POST",
                data_path="rows",
            ),
            "events": EndpointDef(
                path=f"/properties/{property_id}:runReport",
                method="POST",
                data_path="rows",
            ),
        }
        super().__init__(
            source_id=source_id,
            auth=AuthConfig(
                auth_type="oauth2",
                access_token=access_token,
                token_url="https://oauth2.googleapis.com/token",
            ),
            pagination=PaginationConfig(
                strategy="offset",
                per_page=10000,
                max_pages=10,
            ),
            **kwargs,
        )

    async def connect(self) -> None:
        """Pull GA4 reports into DuckDB tables.

        GA4 requires POST requests with report definitions — not simple GET.
        Each "endpoint" becomes a report → DataFrame → DuckDB table.
        """
        import duckdb
        import pandas as pd

        self._conn = duckdb.connect(":memory:")

        report_defs = {
            "sessions": {
                "dimensions": [{"name": "date"}, {"name": "sessionDefaultChannelGroup"}, {"name": "country"}],
                "metrics": [{"name": "sessions"}, {"name": "engagedSessions"}, {"name": "averageSessionDuration"}],
                "dateRanges": [{"startDate": "365daysAgo", "endDate": "today"}],
            },
            "users": {
                "dimensions": [{"name": "date"}, {"name": "country"}, {"name": "deviceCategory"}],
                "metrics": [{"name": "activeUsers"}, {"name": "newUsers"}, {"name": "totalUsers"}],
                "dateRanges": [{"startDate": "365daysAgo", "endDate": "today"}],
            },
            "events": {
                "dimensions": [{"name": "date"}, {"name": "eventName"}],
                "metrics": [{"name": "eventCount"}, {"name": "eventValue"}],
                "dateRanges": [{"startDate": "365daysAgo", "endDate": "today"}],
            },
        }

        headers = self._build_headers()

        for table_name, report_def in report_defs.items():
            try:
                import httpx
                url = f"{self._base_url}/properties/{self._property_id}:runReport"
                async with httpx.AsyncClient(timeout=30.0) as client:
                    resp = await client.post(url, headers=headers, json=report_def)
                    resp.raise_for_status()
                    data = resp.json()

                # Parse GA4 response format → flat rows
                dim_headers = [d["name"] for d in data.get("dimensionHeaders", [])]
                met_headers = [m["name"] for m in data.get("metricHeaders", [])]
                rows = []
                for row in data.get("rows", []):
                    r = {}
                    for i, dv in enumerate(row.get("dimensionValues", [])):
                        r[dim_headers[i]] = dv.get("value", "")
                    for i, mv in enumerate(row.get("metricValues", [])):
                        try:
                            r[met_headers[i]] = float(mv.get("value", 0))
                        except (ValueError, TypeError):
                            r[met_headers[i]] = mv.get("value", "")
                    rows.append(r)

                if rows:
                    df = pd.DataFrame(rows)
                    self._conn.register(table_name, df)
                    self._tables[table_name] = df

            except Exception as exc:
                import structlog
                structlog.get_logger().warning("ga4.report_failed", table=table_name, error=str(exc))
