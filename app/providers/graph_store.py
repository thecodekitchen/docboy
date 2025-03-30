from enum import Enum

graph_store_imports = {
    "memory": "from langgraph.store.memory import InMemoryStore",
    "postgres_async": "from langgraph.store.postgres import AsyncPostgresStore",
    "postgres": "from langgraph.store.postgres import PostgresStore"
}

class GraphStoreProvider(str, Enum):
    MEMORY = "memory"
    POSTGRES = "postgres"
    POSTGRES_ASYNC = "postgres_async"
