"""HubSpot connector — contacts, companies, deals, tickets."""

from sqlagent.connectors.rest_connector import (
    RestConnector, AuthConfig, PaginationConfig, EndpointDef,
)


class HubSpotConnector(RestConnector):
    """HubSpot CRM API connector.

    Connection: hubspot://?api_key={private_app_token}
    Auth: Bearer token (private app token)
    """

    dialect = "hubspot"
    base_url = "https://api.hubapi.com"
    default_auth_type = "bearer"
    default_pagination = "cursor"
    default_data_path = "results"

    endpoints = {
        "contacts": EndpointDef(
            path="/crm/v3/objects/contacts",
            params={"limit": "100", "properties": "firstname,lastname,email,company,lifecyclestage,createdate"},
            data_path="results",
        ),
        "companies": EndpointDef(
            path="/crm/v3/objects/companies",
            params={"limit": "100", "properties": "name,industry,annualrevenue,numberofemployees,createdate"},
            data_path="results",
        ),
        "deals": EndpointDef(
            path="/crm/v3/objects/deals",
            params={"limit": "100", "properties": "dealname,amount,dealstage,closedate,pipeline,createdate"},
            data_path="results",
        ),
        "tickets": EndpointDef(
            path="/crm/v3/objects/tickets",
            params={"limit": "100", "properties": "subject,content,hs_pipeline_stage,hs_ticket_priority,createdate"},
            data_path="results",
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
                cursor_param="after",
                cursor_path="paging.next.after",
                per_page=100,
                max_pages=50,
            ),
            **kwargs,
        )
