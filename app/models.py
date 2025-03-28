from pydantic import BaseModel
from langchain_core.documents import Document
from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages
from typing import Annotated

class ChatInput(BaseModel):
    message: str

class AgentState(BaseModel):
    messages: Annotated[list[AnyMessage], add_messages]
    scored_documents: list[tuple[Document,float]] | None = None

class DocLoaderState(BaseModel):
    collection: str
    files: list[str]
    documents: list[Document] | None = None
    doc_ids: list[str] | None = None
