"""Stripe connector — charges, customers, subscriptions, invoices, payouts."""

from sqlagent.connectors.rest_connector import (
    RestConnector, AuthConfig, PaginationConfig, EndpointDef,
)


class StripeConnector(RestConnector):
    """Stripe API connector.

    Connection: stripe://?api_key={secret_key}
    Auth: Bearer token (secret key)
    """

    dialect = "stripe"
    base_url = "https://api.stripe.com/v1"
    default_auth_type = "bearer"
    default_pagination = "cursor"
    default_data_path = "data"

    endpoints = {
        "charges": EndpointDef(
            path="/charges",
            params={"limit": "100"},
            data_path="data",
        ),
        "customers": EndpointDef(
            path="/customers",
            params={"limit": "100"},
            data_path="data",
        ),
        "subscriptions": EndpointDef(
            path="/subscriptions",
            params={"limit": "100"},
            data_path="data",
        ),
        "invoices": EndpointDef(
            path="/invoices",
            params={"limit": "100"},
            data_path="data",
        ),
        "payouts": EndpointDef(
            path="/payouts",
            params={"limit": "100"},
            data_path="data",
        ),
        "balance_transactions": EndpointDef(
            path="/balance_transactions",
            params={"limit": "100"},
            data_path="data",
        ),
    }

    def __init__(self, source_id: str, api_key: str, **kwargs):
        super().__init__(
            source_id=source_id,
            auth=AuthConfig(
                auth_type="bearer",
                access_token=api_key,
            ),
            pagination=PaginationConfig(
                strategy="cursor",
                cursor_param="starting_after",
                cursor_path="data.-1.id",  # last item's ID
                per_page=100,
                max_pages=50,
            ),
            **kwargs,
        )
