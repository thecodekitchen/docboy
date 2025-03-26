from pydantic import BaseModel
from langchain_core.documents import Document
from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages
from typing import Annotated

class AuthTokenRequest(BaseModel):
    id_token: str
    code: str
    session_state: str

class RefreshRequest(BaseModel):
    id_token: str

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

class UserRecord(BaseModel):
    name: str
    email: str
    session_id: str
    session_expires: int