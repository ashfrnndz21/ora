"""Salesforce connector — accounts, contacts, opportunities, leads, cases."""

from sqlagent.connectors.rest_connector import (
    RestConnector, AuthConfig, PaginationConfig, EndpointDef,
)


class SalesforceConnector(RestConnector):
    """Salesforce REST API connector.

    Connection: salesforce://{instance_url}
    Auth: OAuth2 (client credentials or refresh token)
    """

    dialect = "salesforce"
    default_auth_type = "oauth2"
    default_pagination = "cursor"
    default_data_path = "records"

    endpoints = {
        "accounts": EndpointDef(
            path="/services/data/v59.0/query",
            params={"q": "SELECT Id, Name, Industry, Type, AnnualRevenue, NumberOfEmployees, CreatedDate FROM Account LIMIT 10000"},
            data_path="records",
        ),
        "contacts": EndpointDef(
            path="/services/data/v59.0/query",
            params={"q": "SELECT Id, FirstName, LastName, Email, AccountId, Title, Department, CreatedDate FROM Contact LIMIT 10000"},
            data_path="records",
        ),
        "opportunities": EndpointDef(
            path="/services/data/v59.0/query",
            params={"q": "SELECT Id, Name, AccountId, Amount, StageName, CloseDate, Probability, Type, CreatedDate FROM Opportunity LIMIT 10000"},
            data_path="records",
        ),
        "leads": EndpointDef(
            path="/services/data/v59.0/query",
            params={"q": "SELECT Id, FirstName, LastName, Company, Email, Status, LeadSource, CreatedDate FROM Lead LIMIT 10000"},
            data_path="records",
        ),
        "cases": EndpointDef(
            path="/services/data/v59.0/query",
            params={"q": "SELECT Id, Subject, Status, Priority, AccountId, ContactId, CreatedDate, ClosedDate FROM Case LIMIT 10000"},
            data_path="records",
        ),
    }

    def __init__(self, source_id: str, instance_url: str,
                 client_id: str = "", client_secret: str = "",
                 access_token: str = "", refresh_token: str = "", **kwargs):
        token_url = f"{instance_url}/services/oauth2/token"
        super().__init__(
            source_id=source_id,
            base_url=instance_url,
            auth=AuthConfig(
                auth_type="oauth2",
                client_id=client_id,
                client_secret=client_secret,
                access_token=access_token,
                refresh_token=refresh_token,
                token_url=token_url,
            ),
            pagination=PaginationConfig(
                strategy="cursor",
                cursor_path="nextRecordsUrl",
                per_page=2000,
                max_pages=25,
            ),
            **kwargs,
        )
