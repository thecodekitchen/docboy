from enum import Enum

chain_store_imports = {
    "astradb": "from langchain_community.storage.astradb import AstraDBStore",
    "astradb_bytes": "from langchain_community.storage.astradb import AstraDBByteStore",
    "cassandra_bytes": "from langchain_community.storage.cassandra import CassandraByteStore",
    "mongodb_bytes": "from langchain_community.storage.mongodb import MongoDBByteStore",
    "mongodb": "from langchain_community.storage.mongodb import MongoDBStore",
    "redis": "from langchain_community.storage.redis import RedisStore",
    "sql": "from langchain_community.storage.sql import SQLStore",
    "upstash_redis_bytes": "from langchain_community.storage.upstash_redis import UpstashRedisByteStore",
    "upstash_redis": "from langchain_community.storage.upstash_redis import UpstashRedisStore"
}

class ChainStoreProvider(str, Enum):
    ASTRADB = "astradb"
    ASTRADB_BYTES = "astradb_bytes"
    CASSANDRA_BYTES = "cassandra_bytes"
    MONGODB = "mongodb"
    MONGODB_BYTES = "mongodb_bytes"
    REDIS = "redis"
    SQL = "sql"
    UPSTASH_REDIS = "upstash_redis"
    UPSTASH_REDIS_BYTES = "upstash_redis_bytes"
