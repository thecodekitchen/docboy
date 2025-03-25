from langchain_text_splitters.python import PythonCodeTextSplitter
from langchain_core.documents import Document
from langchain_community.document_loaders import PythonLoader
from langchain_postgres.vectorstores import PGVector
from langchain_ollama import OllamaEmbeddings
from langgraph.graph import StateGraph
from langgraph.pregel import Pregel
from fastapi import BackgroundTasks, UploadFile
from fastapi.responses import StreamingResponse
import os
import json
import logging

from utils import stream_callable
from models import DocLoaderState

def postgres_writer_node(state: DocLoaderState):
    embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")
    DB_URI = os.getenv("DOC_DB_URI")
    store = PGVector.from_documents(
        collection_name=state.collection,
        documents=state.documents, 
        embedding=embeddings, 
        connection=DB_URI
    )
    state.doc_ids = []
    for doc in state.documents:
        state.doc_ids.append(doc.id)
    return state

def python_loader_node(state: DocLoaderState):
    files = state.files
    documents = []
    for file in files:
        loader = PythonLoader(file)
        splitter = PythonCodeTextSplitter()
        for doc in loader.load_and_split(splitter):
            documents.append(doc)
    state.documents = documents
    return state

def python_loader_graph():
    graph = StateGraph(DocLoaderState)
    graph.add_node("loader", python_loader_node)
    graph.add_node("writer", postgres_writer_node)
    graph.add_edge("loader", "writer")
    graph.set_entry_point("loader")
    return graph.compile()



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
        logger:logging.Logger
    ):
    
    try:
        async for s in stream_callable(
            cb=loader.astream,
            args=[state.model_dump()], 
            kwargs={"stream_mode": "updates"}, 
            logger=logger
        ):
            yield s
    except Exception as e:
        logger.error(f"Error in python loader: {str(e)}", exc_info=True)
        yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"

async def stream_loader(
        loader:Pregel, 
        collection:str, 
        files:list[UploadFile], 
        bg:BackgroundTasks, 
        logger:logging.Logger
    ):
    paths = write_temporary_files(bg, files)
    state = DocLoaderState(collection=collection, files=paths)
    return StreamingResponse(
        generate_loader_stream(loader, state, logger),
        media_type="text/event-stream"
    )