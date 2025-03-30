from enum import Enum

toolkit_imports = {
    "ainetwork": "from langchain_community.agent_toolkits.ainetwork.toolkit import AINetworkToolkit",
    "amadeus": "from langchain_community.agent_toolkits.amadeus.toolkit import AmadeusToolkit",
    "azure_ai_services": "from langchain_community.agent_toolkits.azure_ai_services import AzureAiServicesToolkit",
    "azure_cognitive_services": "from langchain_community.agent_toolkits.azure_cognitive_services import AzureCognitiveServicesToolkit",
    "cassandra_database": "from langchain_community.agent_toolkits.cassandra_database.toolkit import CassandraDatabaseToolkit",
    "clickup": "from langchain_community.agent_toolkits.clickup.toolkit import ClickupToolkit",
    "cogniswitch": "from langchain_community.agent_toolkits.cogniswitch.toolkit import CogniswitchToolkit",
    "connery": "from langchain_community.agent_toolkits.connery.toolkit import ConneryToolkit",
    "file_management": "from langchain_community.agent_toolkits.file_management.toolkit import FileManagementToolkit",
    "financial_datasets": "from langchain_community.agent_toolkits.financial_datasets.toolkit import FinancialDatasetsToolkit",
    "github": "from langchain_community.agent_toolkits.github.toolkit import GitHubToolkit",
    "gitlab": "from langchain_community.agent_toolkits.gitlab.toolkit import GitLabToolkit",
    "gmail": "from langchain_community.agent_toolkits.gmail.toolkit import GmailToolkit",
    "jira": "from langchain_community.agent_toolkits.jira.toolkit import JiraToolkit",
    "json": "from langchain_community.agent_toolkits.json.toolkit import JsonToolkit",
    "multion": "from langchain_community.agent_toolkits.multion.toolkit import MultionToolkit",
    "nasa": "from langchain_community.agent_toolkits.nasa.toolkit import NasaToolkit",
    "nla": "from langchain_community.agent_toolkits.nla.toolkit import NLAToolkit",
    "office365": "from langchain_community.agent_toolkits.office365.toolkit import O365Toolkit",
    "openapi": "from langchain_community.agent_toolkits.openapi.toolkit import RequestsToolkit",
    "playwright": "from langchain_community.agent_toolkits.playwright.toolkit import PlayWrightBrowserToolkit",
    "polygon": "from langchain_community.agent_toolkits.polygon.toolkit import PolygonToolkit",
    "powerbi": "from langchain_community.agent_toolkits.powerbi.toolkit import PowerBIToolkit",
    "slack": "from langchain_community.agent_toolkits.slack.toolkit import SlackToolkit",
    "spark_sql": "from langchain_community.agent_toolkits.spark_sql.toolkit import SparkSQLToolkit",
    "sql": "from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit",
    "steam": "from langchain_community.agent_toolkits.steam.toolkit import SteamToolkit",
    "zapier": "from langchain_community.agent_toolkits.zapier.toolkit import ZapierToolkit"
}

class Toolkit(Enum):
    AINetwork = "ainetwork"
    Amadeus = "amadeus"
    AzureAiServices = "azure_ai_services"
    AzureCognitiveServices = "azure_cognitive_services"
    CassandraDatabase = "cassandra_database"
    Clickup = "clickup"
    Cogniswitch = "cogniswitch"
    Connery = "connery"
    FileManagement = "file_management"
    FinancialDatasets = "financial_datasets"
    GitHub = "github"
    GitLab = "gitlab"
    Gmail = "gmail"
    Jira = "jira"
    Json = "json"
    Multion = "multion"
    Nasa = "nasa"
    NLAToolkit = "nla"
    O365Toolkit = "office365"
    RequestsToolkit = "openapi"
    PlayWrightBrowserToolkit = "playwright"
    PolygonToolkit = "polygon"
    PowerBIToolkit = "powerbi"
    SlackToolkit = "slack"
    SparkSQLToolkit = "spark_sql"
    SQLDatabaseToolkit = "sql"
    SteamToolkit = "steam"
    ZapierToolkit = "zapier"