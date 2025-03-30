from enum import Enum

checkpoint_saver_imports = {
    "memory": "from langgraph.checkpoint.memory import InMemorySaver",
    "postgres": "from langgraph.checkpoint.postgres.base import PostgresSaver",
    "postgres_async": "from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver",
    "sqlite": "from langgraph.checkpoint.sqlite.base import SQLiteSaver",
    "sqlite_async": "from langgraph.checkpoint.sqlite.aio import AsyncSQLiteSaver",
}

class CheckpointSaverProvider(str, Enum):
    MEMORY = "memory"
    POSTGRES = "postgres"
    POSTGRES_ASYNC = "postgres_async"
    SQLITE = "sqlite"
    SQLITE_ASYNC = "sqlite_async"