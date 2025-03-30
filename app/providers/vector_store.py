from enum import Enum
vector_store_imports = {
    "aerospike": "from langchain_community.vectorstores.aerospike import Aerospike",
    "alibabacloud_opensearch": "from langchain_community.vectorstores.alibabacloud_opensearch import AlibabaCloudOpenSearch",
    "analyticdb": "from langchain_community.vectorstores.analyticdb import AnalyticDB",
    "annoy": "from langchain_community.vectorstores.annoy import Annoy",
    "apache_doris": "from langchain_community.vectorstores.apache_doris import ApacheDoris",
    "aperturedb": "from langchain_community.vectorstores.aperturedb import ApertureDB",
    "astradb": "from langchain_community.vectorstores.astradb import AstraDB",
    "atlas": "from langchain_community.vectorstores.atlas import AtlasDB",
    "awadb": "from langchain_community.vectorstores.awadb import AwaDB",
    "azure_cosmos_db": "from langchain_community.vectorstores.azure_cosmos_db import AzureCosmosDBVectorSearch",
    "azure_cosmos_db_no_sql": "from langchain_community.vectorstores.azure_cosmos_db_no_sql import AzureCosmosDBNoSqlVectorSearch",
    "azuresearch": "from langchain_community.vectorstores.azuresearch import AzureSearch",
    "bagel": "from langchain_community.vectorstores.bagel import Bagel",
    "bageldb": "from langchain_community.vectorstores.bageldb import Bagel",
    "baiducloud_vector_search": "from langchain_community.vectorstores.baiducloud_vector_search import BESVectorStore",
    "baiduvectordb": "from langchain_community.vectorstores.baiduvectordb import BaiduVectorDB",
    "bigquery_vector_search": "from langchain_community.vectorstores.bigquery_vector_search import BigQueryVectorSearch",
    "cassandra": "from langchain_community.vectorstores.cassandra import Cassandra",
    "chroma": "from langchain_community.vectorstores.chroma import Chroma",
    "clarifai": "from langchain_community.vectorstores.clarifai import Clarifai",
    "clickhouse": "from langchain_community.vectorstores.clickhouse import Clickhouse",
    "couchbase": "from langchain_community.vectorstores.couchbase import CouchbaseVectorStore",
    "dashvector": "from langchain_community.vectorstores.dashvector import DashVector",
    "databricks_vector_search": "from langchain_community.vectorstores.databricks_vector_search import DatabricksVectorSearch",
    "deeplake": "from langchain_community.vectorstores.deeplake import DeepLake",
    "dingo": "from langchain_community.vectorstores.dingo import Dingo",
    "docarray": "from langchain_community.vectorstores.docarray import DocArrayInMemorySearch",
    "documentdb": "from langchain_community.vectorstores.documentdb import DocumentDBVectorSearch",
    "duckdb": "from langchain_community.vectorstores.duckdb import DuckDB",
    "ecloud_vector_search": "from langchain_community.vectorstores.ecloud_vector_search import EcloudESVectorStore",
    "elastic_vector_search": "from langchain_community.vectorstores.elastic_vector_search import ElasticVectorSearch",
    "elasticsearch": "from langchain_community.vectorstores.elasticsearch import ElasticsearchStore",
    "epsilla": "from langchain_community.vectorstores.epsilla import Epsilla",
    "faiss": "from langchain_community.vectorstores.faiss import FAISS",
    "falkordb_vector": "from langchain_community.vectorstores.falkordb_vector import FalkorDBVector",
    "hanavector": "from langchain_community.vectorstores.hanavector import HanaDB",
    "hippo": "from langchain_community.vectorstores.hippo import Hippo",
    "hologres": "from langchain_community.vectorstores.hologres import Hologres",
    "infinispanvs": "from langchain_community.vectorstores.infinispanvs import InfinispanVS",
    "inmemory": "from langchain_community.vectorstores.inmemory import InMemoryVectorStore",
    "jaguar": "from langchain_community.vectorstores.jaguar import Jaguar",
    "kdbai": "from langchain_community.vectorstores.kdbai import KDBAI",
    "kinetica": "from langchain_community.vectorstores.kinetica import Kinetica",
    "lancedb": "from langchain_community.vectorstores.lancedb import LanceDB",
    "lantern": "from langchain_community.vectorstores.lantern import Lantern",
    "llm_rails": "from langchain_community.vectorstores.llm_rails import LLMRails",
    "manticore_search": "from langchain_community.vectorstores.manticore_search import ManticoreSearch",
    "marqo": "from langchain_community.vectorstores.marqo import Marqo",
    "matching_engine": "from langchain_community.vectorstores.matching_engine import MatchingEngine",
    "meilisearch": "from langchain_community.vectorstores.meilisearch import Meilisearch",
    "milvus": "from langchain_community.vectorstores.milvus import Milvus",
    "momento_vector_index": "from langchain_community.vectorstores.momento_vector_index import MomentoVectorIndex",
    "mongodb_atlas": "from langchain_community.vectorstores.mongodb_atlas import MongoDBAtlasVectorSearch",
    "myscale": "from langchain_community.vectorstores.myscale import MyScaleWithoutJSON",
    "neo4j_vector": "from langchain_community.vectorstores.neo4j_vector import Neo4jVector",
    "nucliadb": "from langchain_community.vectorstores.nucliadb import NucliaDB",
    "opensearch_vector_search": "from langchain_community.vectorstores.opensearch_vector_search import OpenSearchVectorSearch",
    "oraclevs": "from langchain_community.vectorstores.oraclevs import OracleVS",
    "pathway": "from langchain_community.vectorstores.pathway import PathwayVectorClient",
    "pgembedding": "from langchain_community.vectorstores.pgembedding import PGEmbedding",
    "pgvecto_rs": "from langchain_community.vectorstores.pgvecto_rs import PGVecto_rs",
    "pgvector": "from langchain_community.vectorstores.pgvector import PGVector",
    "pinecone": "from langchain_community.vectorstores.pinecone import Pinecone",
    "qdrant": "from langchain_community.vectorstores.qdrant import Qdrant",
    "redis": "from langchain_community.vectorstores.redis import Redis",
    "relyt": "from langchain_community.vectorstores.relyt import Relyt",
    "rocksetdb": "from langchain_community.vectorstores.rocksetdb import Rockset",
    "scann": "from langchain_community.vectorstores.scann import ScaNN",
    "semadb": "from langchain_community.vectorstores.semadb import SemaDB",
    "singlestoredb": "from langchain_community.vectorstores.singlestoredb import SingleStoreDB",
    "sklearn": "from langchain_community.vectorstores.sklearn import SKLearnVectorStore",
    "sqlitevec": "from langchain_community.vectorstores.sqlitevec import SQLiteVec",
    "sqlitevss": "from langchain_community.vectorstores.sqlitevss import SQLiteVSS",
    "starrocks": "from langchain_community.vectorstores.starrocks import StarRocks",
    "supabase": "from langchain_community.vectorstores.supabase import SupabaseVectorStore",
    "surrealdb": "from langchain_community.vectorstores.surrealdb import SurrealDBStore",
    "tablestore": "from langchain_community.vectorstores.tablestore import TablestoreVectorStore",
    "tair": "from langchain_community.vectorstores.tair import Tair",
    "tencentvectordb": "from langchain_community.vectorstores.tencentvectordb import TencentVectorDB",
    "thirdai_neuraldb": "from langchain_community.vectorstores.thirdai_neuraldb import NeuralDBVectorStore",
    "tidb_vector": "from langchain_community.vectorstores.tidb_vector import TiDBVectorStore",
    "tigris": "from langchain_community.vectorstores.tigris import Tigris",
    "tiledb": "from langchain_community.vectorstores.tiledb import TileDB",
    "timescalevector": "from langchain_community.vectorstores.timescalevector import TimescaleVector",
    "typesense": "from langchain_community.vectorstores.typesense import Typesense",
    "upstash": "from langchain_community.vectorstores.upstash import UpstashVectorStore",
    "usearch": "from langchain_community.vectorstores.usearch import USearch",
    "vald": "from langchain_community.vectorstores.vald import Vald",
    "vdms": "from langchain_community.vectorstores.vdms import VDMS",
    "vearch": "from langchain_community.vectorstores.vearch import Vearch",
    "vectara": "from langchain_community.vectorstores.vectara import Vectara",
    "vespa": "from langchain_community.vectorstores.vespa import VespaStore",
    "vikingdb": "from langchain_community.vectorstores.vikingdb import VikingDB",
    "vlite": "from langchain_community.vectorstores.vlite import VLite",
    "weaviate": "from langchain_community.vectorstores.weaviate import Weaviate",
    "xata": "from langchain_community.vectorstores.xata import XataVectorStore",
    "yellowbrick": "from langchain_community.vectorstores.yellowbrick import Yellowbrick",
    "zep": "from langchain_community.vectorstores.zep import ZepVectorStore",
    "zep_cloud": "from langchain_community.vectorstores.zep_cloud import ZepCloudVectorStore",
    "zilliz": "from langchain_community.vectorstores.zilliz import Zilliz"
}

class VectorStoreProvider(Enum):
    AEROSPIKE = "aerospike"
    ALIBABACLOUD_OPENSEARCH = "alibabacloud_opensearch"
    ANALYTICDB = "analyticdb"
    ANNOY = "annoy"
    APACHE_DORIS = "apache_doris"
    APERTUREDB = "aperturedb"
    ASTRADB = "astradb"
    ATLAS = "atlas"
    AWADB = "awadb"
    AZURE_COSMOS_DB = "azure_cosmos_db"
    AZURE_COSMOS_DB_NO_SQL = "azure_cosmos_db_no_sql"
    AZURESEARCH = "azuresearch"
    BAGEL = "bagel"
    BAGELDB = "bageldb"
    BAIDUCLOUD_VECTOR_SEARCH = "baiducloud_vector_search"
    BAIDUVECTORDB = "baiduvectordb"
    BIGQUERY_VECTOR_SEARCH = "bigquery_vector_search"
    CASSANDRA = "cassandra"
    CHROMA = "chroma"
    CLARIFAI = "clarifai"
    CLICKHOUSE = "clickhouse"
    COUCHBASE = "couchbase"
    DASHVECTOR = "dashvector"
    DATABRICKS_VECTOR_SEARCH = "databricks_vector_search"
    DEEPLAKE = "deeplake"
    DINGO = "dingo"
    DOCARRAY = "docarray"
    DOCUMENTDB = "documentdb"
    DUCKDB = "duckdb"
    ECLOUD_VECTOR_SEARCH = "ecloud_vector_search"
    ELASTIC_VECTOR_SEARCH = "elastic_vector_search"
    ELASTICSEARCH = "elasticsearch"
    EPSILLA = "epsilla"
    FAISS = "faiss"
    FALKORDB_VECTOR = "falkordb_vector"
    HANAVECTOR = "hanavector"
    HIPPO = "hippo"
    HOLOGRES = "hologres"
    INFINISPANVS = "infinispanvs"
    INMEMORY = "inmemory"
    JAGUAR = "jaguar"
    KDBAI = "kdbai"
    KINETICA = "kinetica"
    LANCEDB = "lancedb"
    LANTERN = "lantern"
    LLM_RAILS = "llm_rails"
    MANTICORE_SEARCH = "manticore_search"
    MARQO = "marqo"
    MATCHING_ENGINE = "matching_engine"
    MEILISEARCH = "meilisearch"
    MILVUS = "milvus"
    MOMENTO_VECTOR_INDEX = "momento_vector_index"
    MONGODB_ATLAS = "mongodb_atlas"
    MYSCALE = "myscale"
    NEO4J_VECTOR = "neo4j_vector"
    NUCLIADB = "nucliadb"
    OPENSEARCH_VECTOR_SEARCH = "opensearch_vector_search"
    ORACLEVS = "oraclevs"
    PATHWAY = "pathway"
    PGEMBEDDING = "pgembedding"
    PGVECTO_RS = "pgvecto_rs"
    PGVECTOR = "pgvector"
    PINECONE = "pinecone"
    QDRANT = "qdrant"
    REDIS = "redis"
    RELYT = "relyt"
    ROCKSETDB = "rocksetdb"
    SCANN = "scann"
    SEMADB = "semadb"
    SINGLESTOREDB = "singlestoredb"
    SKLEARN = "sklearn"
    SQLITEVEC = "sqlitevec"
    SQLITEVSS = "sqlitevss"
    STARROCKS = "starrocks"
    SUPABASE = "supabase"
    SURREALDB = "surrealdb"
    TABLESTORE = "tablestore"
    TAIR = "tair"
    TENCENTVECTORDB = "tencentvectordb"
    THIRDAI_NEURALDB = "thirdai_neuraldb"
    TIDB_VECTOR = "tidb_vector"
    TIGRIS = "tigris"
    TILEDB = "tiledb"
    TIMESCALEVECTOR = "timescalevector"
    TYPESENSE = "typesense"
    UPSTASH = "upstash"
    USEARCH = "usearch"
    VALD = "vald"
    VDMS = "vdms"
    VEARCH = "vearch"
    VECTARA = "vectara"
    VESPA = "vespa"
    VIKINGDB = "vikingdb"
    VLITE = "vlite"
    WEAVIATE = "weaviate"
    XATA = "xata"
    YELLOWBRICK = "yellowbrick"
    ZEP = "zep"
    ZEP_CLOUD = "zep_cloud"
    ZILLIZ = "zilliz"