from pydantic import BaseModel
from langchain_core.documents import Document
from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages
from typing import Annotated

#API Request Types
class ChatInput(BaseModel):
    message: str

#State Schemas
class BaseAgentState(BaseModel):
    messages: Annotated[list[AnyMessage], add_messages]
    
class DocumentAgentState(BaseAgentState):
    scored_documents: list[tuple[Document,float]] | None = None

class UserProfile(BaseModel):
    user_id: str
    user_name: str
    profile: dict | None = None

class UserAgentState(BaseAgentState):
    user_profile: UserProfile | None = None

class DocLoaderState(BaseModel):
    collection: str
    files: list[str]
    documents: list[Document] | None = None
    doc_ids: list[str] | None = None

#Config Schemas
class AgentConfig(BaseModel):
    thread_id: str

class UserAgentConfig(AgentConfig):
    user_id: str
    user_name: str