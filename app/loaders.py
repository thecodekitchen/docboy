from langchain_text_splitters.base import TextSplitter
from langchain_core.documents import Document
from langchain_core.document_loaders import BaseLoader
from langgraph.graph import StateGraph
from langgraph.pregel import Pregel
from fastapi import BackgroundTasks, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Callable, Type
import os
import json
import logging

from utils import stream_callable
from providers import (
    get_vector_store_class, 
    VectorStoreProvider, 
    get_embeddings_class_instance, 
    EmbeddingsProvider,
    get_doc_loader_class,
    get_doc_loader_class_instance,
    DocumentLoaderType,
    get_text_splitter_class_instance,
    TextSplitterType
)

class DocLoaderState(BaseModel):
    files: list[str]
    file_types: dict = {}
    documents: list[Document] | None = None
    doc_ids: list[str] | None = None

def file_type_sort_node(state:DocLoaderState):
    for file in state.files:
        ext = os.path.splitext(file)[1]
        if ext not in state.file_types:
            state.file_types[ext] = []
        state.file_types[ext].append(file)
    return state

def default_writer_node(state: DocLoaderState):
    embeddings = get_embeddings_class_instance(
        EmbeddingsProvider.OLLAMA, 
        model="nomic-embed-text:latest"
    )
    DB_URI = os.getenv("DOC_DB_URI")
    store_class = get_vector_store_class(
        provider=VectorStoreProvider.PGVECTOR,
    )
    store = store_class(
        connection=DB_URI, 
        embedding=embeddings,
        collection_name=state.collection
    )
    store.add_documents(state.documents)
    state.doc_ids = []
    for doc in state.documents:
        state.doc_ids.append(doc.id)
    return state

def custom_writer_node(store_provider: VectorStoreProvider, **store_kwargs)->Callable[[DocLoaderState], DocLoaderState]:
    def node(state: DocLoaderState, config: dict):
        if config.get("configurable") and config["configurable"].get("collection"):
            store_kwargs["collection_name"] = config["configurable"]["collection"]
        store = get_vector_store_class(store_provider)(**store_kwargs)
        if config.get("configurable") and config["configurable"].get("user_id"):
            for document in state.documents:
                document.metadata["user_id"] = config["user_id"]
        store.add_documents(state.documents)
        state.doc_ids = []
        for doc in state.documents:
            state.doc_ids.append(doc.id)
        return state
    return node

def default_load_and_split_node(state: DocLoaderState):
    files = state.files
    documents = []
    for file in files:
        loader = get_doc_loader_class_instance(DocumentLoaderType.TEXT, file)
        splitter = get_text_splitter_class_instance(TextSplitterType.TextSplitter)
        for doc in loader.load_and_split(splitter):
            documents.append(doc)
    state.documents = documents
    return state

def custom_load_and_split_node(loader_class: Type[BaseLoader], splitter: TextSplitter)->Callable[[DocLoaderState], DocLoaderState]:
    """
    Custom load and split node for specific file types.
    Args:
        loader_class (Type[BaseLoader]): The loader class to use for loading documents.
        splitter (TextSplitter): The text splitter to use for splitting documents.
    Returns:
        Callable[[DocLoaderState], DocLoaderState]: A node function that takes a DocLoaderState and returns it.
    """
    # Only evaluate state schema for validation purposes, otherwise assume
    def node (state: DocLoaderState):
        files = state.files
        documents = []
        for file in files:
            loader = loader_class(file)
            loader.load_and_split(splitter)
        state.documents = documents
        return state
    return node

class DocLoaderSpec(BaseModel):
    supported_extensions: list[str] = [".txt"]
    loader: Callable[[DocLoaderState], DocLoaderState] = default_load_and_split_node
    writer: Callable[[DocLoaderState], DocLoaderState] = default_writer_node
    state_schema: Type[DocLoaderState]


def compile_loader_graph(spec:DocLoaderSpec):
    graph = StateGraph(spec.state_schema)
    graph.set_entry_point("file_type_sorter", file_type_sort_node)
    graph.add_node("loader", default_load_and_split_node)
    graph.add_node("writer", spec.writer)
    graph.add_edge("file_type_sorter", "loader")
    graph.add_edge("loader", "writer")
    graph.set_finish_point("writer")
    return graph.compile()

def assemble_loader_config(collection:str, user_id:str):
    return {
        "configurable": {
            "collection": collection,
            "user_id": user_id
        }
    }

def python_loader_graph():
    return compile_loader_graph(
        DocLoaderSpec(
            supported_extensions=[".py"],
            loader=custom_load_and_split_node(
                loader_class=get_doc_loader_class(DocumentLoaderType.PYTHON),
                splitter=get_text_splitter_class_instance(TextSplitterType.PythonCodeTextSplitter)
            ),
            writer=custom_writer_node(
                VectorStoreProvider.PGVECTOR,
                connection=os.getenv("DOC_DB_URI"),
                embedding=get_embeddings_class_instance(
                    EmbeddingsProvider.OLLAMA, 
                    model="nomic-embed-text:latest"
                )
            ),
            state_schema=DocLoaderState
        )
    )

def write_temporary_files(bg: BackgroundTasks, files: list[UploadFile]):
    paths = []
    for file in files:
        if file.filename.endswith(".py"):
            path = f"/tmp/{file.filename}"
            with open(path, "wb") as f:
                f.write(file.file.read())
            paths.append(path)
    def cleanup_temp_files(paths: list[str]):
        for path in paths:
            os.remove(path)
    bg.add_task(cleanup_temp_files, paths)
    return paths

async def generate_loader_stream(
        loader:Pregel, 
        state:DocLoaderState,
        config: dict,
        logger:logging.Logger
    ):
    
    try:
        async for s in stream_callable(
            cb=loader.astream,
            args=[state.model_dump()], 
            kwargs={"stream_mode": "updates", "config": config}, 
            logger=logger
        ):
            yield s
    except Exception as e:
        logger.error(f"Error in python loader: {str(e)}", exc_info=True)
        yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"

async def stream_loader(
        graph:Pregel, 
        config:dict, 
        user_id: str,
        files:list[UploadFile], 
        bg:BackgroundTasks, 
        logger:logging.Logger
    ):
    paths = write_temporary_files(bg, files)
    state = DocLoaderState(files=paths)
    return StreamingResponse(
        generate_loader_stream(graph, state, config, logger),
        media_type="text/event-stream"
    )