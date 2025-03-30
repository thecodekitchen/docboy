from enum import Enum

doc_loader_imports = {
    "acreom": "from langchain_community.document_loaders.acreom import AcreomLoader",
    "airbyte": "from langchain_community.document_loaders.airbyte import AirbyteZendeskSupportLoader",
    "airbyte_json": "from langchain_community.document_loaders.airbyte_json import AirbyteJSONLoader",
    "airtable": "from langchain_community.document_loaders.airtable import AirtableLoader",
    "apify_dataset": "from langchain_community.document_loaders.apify_dataset import ApifyDatasetLoader",
    "arcgis_loader": "from langchain_community.document_loaders.arcgis_loader import ArcGISLoader",
    "arxiv": "from langchain_community.document_loaders.arxiv import ArxivLoader",
    "assemblyai": "from langchain_community.document_loaders.assemblyai import AssemblyAIAudioTranscriptLoader",
    "astradb": "from langchain_community.document_loaders.astradb import AstraDBLoader",
    "async_html": "from langchain_community.document_loaders.async_html import AsyncHtmlLoader",
    "athena": "from langchain_community.document_loaders.athena import AthenaLoader",
    "azlyrics": "from langchain_community.document_loaders.azlyrics import WebBaseLoader",
    "azure_ai_data": "from langchain_community.document_loaders.azure_ai_data import UnstructuredFileIOLoader",
    "azure_blob_storage_container": "from langchain_community.document_loaders.azure_blob_storage_container import AzureBlobStorageFileLoader",
    "azure_blob_storage_file": "from langchain_community.document_loaders.azure_blob_storage_file import UnstructuredFileLoader",
    "baiducloud_bos_directory": "from langchain_community.document_loaders.baiducloud_bos_directory import BaiduBOSDirectoryLoader",
    "baiducloud_bos_file": "from langchain_community.document_loaders.baiducloud_bos_file import UnstructuredFileLoader",
    "base_o365": "from langchain_community.document_loaders.base_o365 import O365BaseLoader",
    "bibtex": "from langchain_community.document_loaders.bibtex import BibtexLoader",
    "bigquery": "from langchain_community.document_loaders.bigquery import BigQueryLoader",
    "bilibili": "from langchain_community.document_loaders.bilibili import BiliBiliLoader",
    "blackboard": "from langchain_community.document_loaders.blackboard import WebBaseLoader",
    "blockchain": "from langchain_community.document_loaders.blockchain import BlockchainDocumentLoader",
    "brave_search": "from langchain_community.document_loaders.brave_search import BraveSearchLoader",
    "browserbase": "from langchain_community.document_loaders.browserbase import BrowserbaseLoader",
    "browserless": "from langchain_community.document_loaders.browserless import BrowserlessLoader",
    "cassandra": "from langchain_community.document_loaders.cassandra import CassandraLoader",
    "chatgpt": "from langchain_community.document_loaders.chatgpt import ChatGPTLoader",
    "chm": "from langchain_community.document_loaders.chm import UnstructuredFileLoader",
    "chromium": "from langchain_community.document_loaders.chromium import AsyncChromiumLoader",
    "college_confidential": "from langchain_community.document_loaders.college_confidential import WebBaseLoader",
    "concurrent": "from langchain_community.document_loaders.concurrent import GenericLoader",
    "confluence": "from langchain_community.document_loaders.confluence import ConfluenceLoader",
    "conllu": "from langchain_community.document_loaders.conllu import CoNLLULoader",
    "couchbase": "from langchain_community.document_loaders.couchbase import CouchbaseLoader",
    "csv_loader": "from langchain_community.document_loaders.csv_loader import UnstructuredFileLoader",
    "cube_semantic": "from langchain_community.document_loaders.cube_semantic import CubeSemanticLoader",
    "datadog_logs": "from langchain_community.document_loaders.datadog_logs import DatadogLogsLoader",
    "dataframe": "from langchain_community.document_loaders.dataframe import DataFrameLoader",
    "dedoc": "from langchain_community.document_loaders.dedoc import DedocFileLoader",
    "diffbot": "from langchain_community.document_loaders.diffbot import DiffbotLoader",
    "directory": "from langchain_community.document_loaders.directory import UnstructuredFileLoader",
    "discord": "from langchain_community.document_loaders.discord import DiscordChatLoader",
    "doc_intelligence": "from langchain_community.document_loaders.doc_intelligence import AzureAIDocumentIntelligenceLoader",
    "docugami": "from langchain_community.document_loaders.docugami import DocugamiLoader",
    "docusaurus": "from langchain_community.document_loaders.docusaurus import SitemapLoader",
    "dropbox": "from langchain_community.document_loaders.dropbox import DropboxLoader",
    "duckdb_loader": "from langchain_community.document_loaders.duckdb_loader import DuckDBLoader",
    "email": "from langchain_community.document_loaders.email import UnstructuredFileLoader",
    "epub": "from langchain_community.document_loaders.epub import UnstructuredFileLoader",
    "etherscan": "from langchain_community.document_loaders.etherscan import EtherscanLoader",
    "evernote": "from langchain_community.document_loaders.evernote import EverNoteLoader",
    "excel": "from langchain_community.document_loaders.excel import UnstructuredFileLoader",
    "facebook_chat": "from langchain_community.document_loaders.facebook_chat import FacebookChatLoader",
    "fauna": "from langchain_community.document_loaders.fauna import FaunaLoader",
    "figma": "from langchain_community.document_loaders.figma import FigmaFileLoader",
    "firecrawl": "from langchain_community.document_loaders.firecrawl import FireCrawlLoader",
    "gcs_directory": "from langchain_community.document_loaders.gcs_directory import GCSFileLoader",
    "gcs_file": "from langchain_community.document_loaders.gcs_file import UnstructuredFileLoader",
    "generic": "from langchain_community.document_loaders.generic import GenericLoader",
    "geodataframe": "from langchain_community.document_loaders.geodataframe import GeoDataFrameLoader",
    "git": "from langchain_community.document_loaders.git import GitLoader",
    "gitbook": "from langchain_community.document_loaders.gitbook import WebBaseLoader",
    "github": "from langchain_community.document_loaders.github import GithubFileLoader",
    "glue_catalog": "from langchain_community.document_loaders.glue_catalog import GlueCatalogLoader",
    "google_speech_to_text": "from langchain_community.document_loaders.google_speech_to_text import GoogleSpeechToTextLoader",
    "googledrive": "from langchain_community.document_loaders.googledrive import GoogleDriveLoader",
    "gutenberg": "from langchain_community.document_loaders.gutenberg import GutenbergLoader",
    "hn": "from langchain_community.document_loaders.hn import WebBaseLoader",
    "html": "from langchain_community.document_loaders.html import UnstructuredHTMLLoader",
    "html_bs": "from langchain_community.document_loaders.html_bs import BSHTMLLoader",
    "hugging_face_dataset": "from langchain_community.document_loaders.hugging_face_dataset import HuggingFaceDatasetLoader",
    "hugging_face_model": "from langchain_community.document_loaders.hugging_face_model import HuggingFaceModelLoader",
    "ifixit": "from langchain_community.document_loaders.ifixit import WebBaseLoader",
    "image": "from langchain_community.document_loaders.image import UnstructuredImageLoader",
    "image_captions": "from langchain_community.document_loaders.image_captions import ImageCaptionLoader",
    "imsdb": "from langchain_community.document_loaders.imsdb import WebBaseLoader",
    "iugu": "from langchain_community.document_loaders.iugu import IuguLoader",
    "joplin": "from langchain_community.document_loaders.joplin import JoplinLoader",
    "json_loader": "from langchain_community.document_loaders.json_loader import JSONLoader",
    "kinetica_loader": "from langchain_community.document_loaders.kinetica_loader import KineticaLoader",
    "lakefs": "from langchain_community.document_loaders.lakefs import UnstructuredLakeFSLoader",
    "larksuite": "from langchain_community.document_loaders.larksuite import LarkSuiteWikiLoader",
    "llmsherpa": "from langchain_community.document_loaders.llmsherpa import LLMSherpaFileLoader",
    "markdown": "from langchain_community.document_loaders.markdown import UnstructuredMarkdownLoader",
    "mastodon": "from langchain_community.document_loaders.mastodon import MastodonTootsLoader",
    "max_compute": "from langchain_community.document_loaders.max_compute import MaxComputeLoader",
    "mediawikidump": "from langchain_community.document_loaders.mediawikidump import MWDumpLoader",
    "merge": "from langchain_community.document_loaders.merge import MergedDataLoader",
    "mhtml": "from langchain_community.document_loaders.mhtml import MHTMLLoader",
    "mintbase": "from langchain_community.document_loaders.mintbase import MintbaseDocumentLoader",
    "modern_treasury": "from langchain_community.document_loaders.modern_treasury import ModernTreasuryLoader",
    "mongodb": "from langchain_community.document_loaders.mongodb import MongodbLoader",
    "needle": "from langchain_community.document_loaders.needle import NeedleLoader",
    "news": "from langchain_community.document_loaders.news import NewsURLLoader",
    "notebook": "from langchain_community.document_loaders.notebook import NotebookLoader",
    "notion": "from langchain_community.document_loaders.notion import NotionDirectoryLoader",
    "notiondb": "from langchain_community.document_loaders.notiondb import NotionDBLoader",
    "nuclia": "from langchain_community.document_loaders.nuclia import NucliaLoader",
    "obs_directory": "from langchain_community.document_loaders.obs_directory import OBSFileLoader",
    "obs_file": "from langchain_community.document_loaders.obs_file import UnstructuredFileLoader",
    "obsidian": "from langchain_community.document_loaders.obsidian import ObsidianLoader",
    "odt": "from langchain_community.document_loaders.odt import UnstructuredODTLoader",
    "onedrive": "from langchain_community.document_loaders.onedrive import SharePointLoader",
    "onedrive_file": "from langchain_community.document_loaders.onedrive_file import UnstructuredFileLoader",
    "onenote": "from langchain_community.document_loaders.onenote import OneNoteLoader",
    "open_city_data": "from langchain_community.document_loaders.open_city_data import OpenCityDataLoader",
    "oracleadb_loader": "from langchain_community.document_loaders.oracleadb_loader import OracleAutonomousDatabaseLoader",
    "oracleai": "from langchain_community.document_loaders.oracleai import OracleDocLoader",
    "org_mode": "from langchain_community.document_loaders.org_mode import UnstructuredOrgModeLoader",
    "pdf": "from langchain_community.document_loaders.pdf import ZeroxPDFLoader",
    "pebblo": "from langchain_community.document_loaders.pebblo import PebbloTextLoader",
    "polars_dataframe": "from langchain_community.document_loaders.polars_dataframe import PolarsDataFrameLoader",
    "powerpoint": "from langchain_community.document_loaders.powerpoint import UnstructuredPowerPointLoader",
    "psychic": "from langchain_community.document_loaders.psychic import PsychicLoader",
    "pubmed": "from langchain_community.document_loaders.pubmed import PubMedLoader",
    "pyspark_dataframe": "from langchain_community.document_loaders.pyspark_dataframe import PySparkDataFrameLoader",
    "python": "from langchain_community.document_loaders.python import PythonLoader",
    "quip": "from langchain_community.document_loaders.quip import QuipLoader",
    "readthedocs": "from langchain_community.document_loaders.readthedocs import ReadTheDocsLoader",
    "recursive_url_loader": "from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader",
    "reddit": "from langchain_community.document_loaders.reddit import RedditPostsLoader",
    "roam": "from langchain_community.document_loaders.roam import RoamLoader",
    "rocksetdb": "from langchain_community.document_loaders.rocksetdb import RocksetLoader",
    "rspace": "from langchain_community.document_loaders.rspace import RSpaceLoader",
    "rss": "from langchain_community.document_loaders.rss import RSSFeedLoader",
    "rst": "from langchain_community.document_loaders.rst import UnstructuredRSTLoader",
    "rtf": "from langchain_community.document_loaders.rtf import UnstructuredRTFLoader",
    "s3_directory": "from langchain_community.document_loaders.s3_directory import S3FileLoader",
    "s3_file": "from langchain_community.document_loaders.s3_file import UnstructuredBaseLoader",
    "scrapfly": "from langchain_community.document_loaders.scrapfly import ScrapflyLoader",
    "scrapingant": "from langchain_community.document_loaders.scrapingant import ScrapingAntLoader",
    "sharepoint": "from langchain_community.document_loaders.sharepoint import SharePointLoader",
    "sitemap": "from langchain_community.document_loaders.sitemap import WebBaseLoader",
    "slack_directory": "from langchain_community.document_loaders.slack_directory import SlackDirectoryLoader",
    "snowflake_loader": "from langchain_community.document_loaders.snowflake_loader import SnowflakeLoader",
    "spider": "from langchain_community.document_loaders.spider import SpiderLoader",
    "spreedly": "from langchain_community.document_loaders.spreedly import SpreedlyLoader",
    "sql_database": "from langchain_community.document_loaders.sql_database import SQLDatabaseLoader",
    "srt": "from langchain_community.document_loaders.srt import SRTLoader",
    "stripe": "from langchain_community.document_loaders.stripe import StripeLoader",
    "surrealdb": "from langchain_community.document_loaders.surrealdb import SurrealDBLoader",
    "telegram": "from langchain_community.document_loaders.telegram import TelegramChatLoader",
    "tencent_cos_directory": "from langchain_community.document_loaders.tencent_cos_directory import TencentCOSFileLoader",
    "tencent_cos_file": "from langchain_community.document_loaders.tencent_cos_file import UnstructuredFileLoader",
    "tensorflow_datasets": "from langchain_community.document_loaders.tensorflow_datasets import TensorflowDatasetLoader",
    "text": "from langchain_community.document_loaders.text import TextLoader",
    "tidb": "from langchain_community.document_loaders.tidb import TiDBLoader",
    "tomarkdown": "from langchain_community.document_loaders.tomarkdown import ToMarkdownLoader",
    "toml": "from langchain_community.document_loaders.toml import TomlLoader",
    "trello": "from langchain_community.document_loaders.trello import TrelloLoader",
    "tsv": "from langchain_community.document_loaders.tsv import UnstructuredTSVLoader",
    "twitter": "from langchain_community.document_loaders.twitter import TwitterTweetLoader",
    "unstructured": "from langchain_community.document_loaders.unstructured import UnstructuredFileLoader",
    "url": "from langchain_community.document_loaders.url import UnstructuredURLLoader",
    "url_playwright": "from langchain_community.document_loaders.url_playwright import PlaywrightURLLoader",
    "url_selenium": "from langchain_community.document_loaders.url_selenium import SeleniumURLLoader",
    "vsdx": "from langchain_community.document_loaders.vsdx import VsdxLoader",
    "weather": "from langchain_community.document_loaders.weather import WeatherDataLoader",
    "web_base": "from langchain_community.document_loaders.web_base import WebBaseLoader",
    "whatsapp_chat": "from langchain_community.document_loaders.whatsapp_chat import WhatsAppChatLoader",
    "wikipedia": "from langchain_community.document_loaders.wikipedia import WikipediaLoader",
    "word_document": "from langchain_community.document_loaders.word_document import UnstructuredWordDocumentLoader",
    "xml": "from langchain_community.document_loaders.xml import UnstructuredXMLLoader",
    "xorbits": "from langchain_community.document_loaders.xorbits import XorbitsLoader",
    "youtube": "from langchain_community.document_loaders.youtube import YoutubeLoader",
    "yuque": "from langchain_community.document_loaders.yuque import YuqueLoader"
}

class DocumentLoaderType(Enum):
    ACREOM = "acreom"
    AIRBYTE = "airbyte"
    AIRBYTE_JSON = "airbyte_json"
    AIRTABLE = "airtable"
    APIFY_DATASET = "apify_dataset"
    ARCGIS_LOADER = "arcgis_loader"
    ARXIV = "arxiv"
    ASSEMBLYAI = "assemblyai"
    ASTRADB = "astradb"
    ASYNC_HTML = "async_html"
    ATHENA = "athena"
    AZLYRICS = "azlyrics"
    AZURE_AI_DATA = "azure_ai_data"
    AZURE_BLOB_STORAGE_CONTAINER = "azure_blob_storage_container"
    AZURE_BLOB_STORAGE_FILE = "azure_blob_storage_file"
    BAIDUCLOUD_BOS_DIRECTORY = "baiducloud_bos_directory"
    BAIDUCLOUD_BOS_FILE = "baiducloud_bos_file"
    BASE_O365 = "base_o365"
    BIBTEX = "bibtex"
    BIGQUERY = "bigquery"
    BILIBILI = "bilibili"
    BLACKBOARD = "blackboard"
    BLOCKCHAIN = "blockchain"
    BRAVE_SEARCH = "brave_search"
    BROWSERBASE = "browserbase"
    BROWSERLESS = "browserless"
    CASSANDRA = "cassandra"
    CHATGPT = "chatgpt"
    CHM = "chm"
    CHROMIUM = "chromium"
    COLLEGE_CONFIDENTIAL = "college_confidential"
    CONCURRENT = "concurrent"
    CONFLUENCE = "confluence"
    CONLLU = "conllu"
    COUCHBASE = "couchbase"
    CSV_LOADER = "csv_loader"
    CUBE_SEMANTIC = "cube_semantic"
    DATADOG_LOGS = "datadog_logs"
    DATAFRAME = "dataframe"
    DEDOC = "dedoc"
    DIFFBOT = "diffbot"
    DIRECTORY = "directory"
    DISCORD = "discord"
    DOC_INTELLIGENCE = "doc_intelligence"
    DOCUGAMI = "docugami"
    DOCUSAURUS = "docusaurus"
    DROPBOX = "dropbox"
    DUCKDB_LOADER = "duckdb_loader"
    EMAIL = "email"
    EPUB = "epub"
    ETHERSCAN = "etherscan"
    EVERNOTE = "evernote"
    EXCEL = "excel"
    FACEBOOK_CHAT = "facebook_chat"
    FAUNA = "fauna"
    FIGMA = "figma"
    FIRECRAWL = "firecrawl"
    GCS_DIRECTORY = "gcs_directory"
    GCS_FILE = "gcs_file"
    GENERIC = "generic"
    GEODATAFRAME = "geodataframe"
    GIT = "git"
    GITBOOK = "gitbook"
    GITHUB = "github"
    GLUE_CATALOG = "glue_catalog"
    GOOGLE_SPEECH_TO_TEXT = "google_speech_to_text"
    GOOGLEDRIVE = "googledrive"
    GUTENBERG = "gutenberg"
    HN = "hn"
    HTML = "html"
    HTML_BS = "html_bs"
    HUGGING_FACE_DATASET = "hugging_face_dataset"
    HUGGING_FACE_MODEL = "hugging_face_model"
    IFIXIT = "ifixit"
    IMAGE = "image"
    IMAGE_CAPTIONS = "image_captions"
    IMSDB = "imsdb"
    IUGU = "iugu"
    JOPLIN = "joplin"
    JSON_LOADER = "json_loader"
    KINETICA_LOADER = "kinetica_loader"
    LAKEFS = "lakefs"
    LARKSUITE = "larksuite"
    LLMSHERPA = "llmsherpa"
    MARKDOWN = "markdown"
    MASTODON = "mastodon"
    MAX_COMPUTE = "max_compute"
    MEDIAWIKIDUMP = "mediawikidump"
    MERGE = "merge"
    MHTML = "mhtml"
    MINTBASE = "mintbase"
    MODERN_TREASURY = "modern_treasury"
    MONGODB = "mongodb"
    NEEDLE = "needle"
    NEWS = "news"
    NOTEBOOK = "notebook"
    NOTION = "notion"
    NOTIONDB = "notiondb"
    NUCLIA = "nuclia"
    OBS_DIRECTORY = "obs_directory"
    OBS_FILE = "obs_file"
    OBSIDIAN = "obsidian"
    ODT = "odt"
    ONEDRIVE = "onedrive"
    ONEDRIVE_FILE = "onedrive_file"
    ONENOTE = "onenote"
    OPEN_CITY_DATA = "open_city_data"
    ORACLEADB_LOADER = "oracleadb_loader"
    ORACLEAI = "oracleai"
    ORG_MODE = "org_mode"
    PDF = "pdf"
    PEBBLO = "pebblo"
    POLARS_DATAFRAME = "polars_dataframe"
    POWERPOINT = "powerpoint"
    PSYCHIC = "psychic"
    PUBMED = "pubmed"
    PYSPARK_DATAFRAME = "pyspark_dataframe"
    PYTHON = "python"
    QUIP = "quip"
    READTHEDOCS = "readthedocs"
    RECURSIVE_URL_LOADER = "recursive_url_loader"
    REDDIT = "reddit"
    ROAM = "roam"
    ROCKSETDB = "rocksetdb"
    RSPACE = "rspace"
    RSS = "rss"
    RST = "rst"
    RTF = "rtf"
    S3_DIRECTORY = "s3_directory"
    S3_FILE = "s3_file"
    SCRAPFLY = "scrapfly"
    SCRAPINGANT = "scrapingant"
    SHAREPOINT = "sharepoint"
    SITEMAP = "sitemap"
    SLACK_DIRECTORY = "slack_directory"
    SNOWFLAKE_LOADER = "snowflake_loader"
    SPIDER = "spider"
    SPREEDLY = "spreedly"
    SQL_DATABASE = "sql_database"
    SRT = "srt"
    STRIPE = "stripe"
    SURREALDB = "surrealdb"
    TELEGRAM = "telegram"
    TENCENT_COS_DIRECTORY = "tencent_cos_directory"
    TENCENT_COS_FILE = "tencent_cos_file"
    TENSORFLOW_DATASETS = "tensorflow_datasets"
    TEXT = "text"
    TIDB = "tidb"
    TOMARKDOWN = "tomarkdown"
    TOML = "toml"
    TRELLO = "trello"
    TSV = "tsv"
    TWITTER = "twitter"
    UNSTRUCTURED = "unstructured"
    URL = "url"
    URL_PLAYWRIGHT = "url_playwright"
    URL_SELENIUM = "url_selenium"
    VSDX = "vsdx"
    WEATHER = "weather"
    WEB_BASE = "web_base"
    WHATSAPP_CHAT = "whatsapp_chat"
    WIKIPEDIA = "wikipedia"
    WORD_DOCUMENT = "word_document"
    XML = "xml"
    XORBITS = "xorbits"
    YOUTUBE = "youtube"
    YUQUE = "yuque"