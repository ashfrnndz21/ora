"""Shopify connector — orders, customers, products, inventory."""

from sqlagent.connectors.rest_connector import (
    RestConnector, AuthConfig, PaginationConfig, EndpointDef,
)


class ShopifyConnector(RestConnector):
    """Shopify Admin API connector.

    Connection: shopify://{store_name}?api_key={key}&api_version=2024-01
    Auth: API key (X-Shopify-Access-Token header)
    """

    dialect = "shopify"
    default_auth_type = "api_key"
    default_pagination = "link_header"
    default_data_path = ""  # varies per endpoint

    endpoints = {
        "orders": EndpointDef(
            path="/orders.json",
            params={"status": "any", "limit": "250"},
            data_path="orders",
        ),
        "customers": EndpointDef(
            path="/customers.json",
            params={"limit": "250"},
            data_path="customers",
        ),
        "products": EndpointDef(
            path="/products.json",
            params={"limit": "250"},
            data_path="products",
        ),
        "inventory_levels": EndpointDef(
            path="/inventory_levels.json",
            params={"limit": "250"},
            data_path="inventory_levels",
        ),
        "collections": EndpointDef(
            path="/custom_collections.json",
            params={"limit": "250"},
            data_path="custom_collections",
        ),
    }

    def __init__(self, source_id: str, store_name: str, api_key: str,
                 api_version: str = "2024-01", **kwargs):
        super().__init__(
            source_id=source_id,
            base_url=f"https://{store_name}.myshopify.com/admin/api/{api_version}",
            auth=AuthConfig(
                auth_type="api_key",
                api_key=api_key,
                api_key_header="X-Shopify-Access-Token",
                api_key_prefix="",  # no prefix for Shopify
            ),
            pagination=PaginationConfig(
                strategy="link_header",
                per_page=250,
                max_pages=20,
            ),
            **kwargs,
        )
