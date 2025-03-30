from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.tools import BaseTool, StructuredTool, BaseToolkit
from langchain_core.language_models import BaseChatModel
from langgraph.graph import StateGraph
from langgraph.config import RunnableConfig
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.checkpoint.base import BaseCheckpointSaver
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from tools import math_tool, search_tool
from typing import Literal, AsyncGenerator, Type
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
from fastapi_server_session import Session
import json
import asyncio
import uuid
import dotenv
import os
import logging
from models import AgentState
from providers import (
    ChatModelProvider, 
    get_chat_model_class_instance,
    VectorStoreProvider, 
    get_vector_store_class_instance,
    EmbeddingsProvider,
    get_embeddings_class_instance
)

class ReactAgentSpec(BaseModel):
    name: str
    model_provider: ChatModelProvider = ChatModelProvider.OLLAMA
    embeddings_provider: EmbeddingsProvider = EmbeddingsProvider.OLLAMA
    vector_store_provider: VectorStoreProvider = VectorStoreProvider.PGVECTOR
    tools: list[BaseTool]
    toolkits: list[BaseToolkit]
    prompt: str
    config_schema: Type[BaseModel] | None = None
    state_schema: Type[BaseModel] | None = None

    def __init__(self, **data):
        super().__init__(**data)
        if len(self.toolkits)  > 0:
            for toolkit in self.toolkits:
                self.tools.extend(toolkit.get_tools())

user_nerd_system_prompt = """
You are a personalized assistant with access to a memory profile of the user. 
The memory profile contains details about the user's preferences, past interactions, and frequently asked questions. 
Use this memory profile to tailor your responses to the user's specific needs and context. 
If the memory profile provides relevant information, incorporate it into your answers. 
Always prioritize accuracy and relevance while maintaining a friendly and helpful tone.

Memory Profile: {profile}
"""

regular_nerd_system_prompt = "You are a nerd that is really good at finding documents to answer people's questions."
regular_nerd = ReactAgentSpec(name="nerd", model="llama3.2:latest", tools=[], prompt=regular_nerd_system_prompt)

math_nerd_system_prompt = "You are a nerd that does math good."
math_nerd = ReactAgentSpec(name="nerd", model="llama3.2:latest", tools=[math_tool], prompt=math_nerd_system_prompt)

web_nerd_prompt = """You are a chronicly online zoomer who drinks way too much caffeine and never stops looking at the internet.
You are a master of the internet and can find anything you need to know in seconds.
You obtain vital social validation from answering questions, so you tend to overexplain things.
If there is a document that appears to be relevant in the provided collection, you should prefer retrieving it over searching the web for an answer."""
web_nerd = ReactAgentSpec(name="web_nerd", model="llama3.2:latest", tools=[search_tool], prompt=web_nerd_prompt)

def assemble_thread_config(thread_id:str, session: Session)->RunnableConfig:
    if not session.get("user_id") or not session.get("user_name"):
        print("User not logged in")
        return
    return {
        "configurable": {
            "thread_id": thread_id,
            "user_id": session["user_id"],
            "user_name": session["user_name"]
        }
    }
dotenv.load_dotenv()
DB_URI = os.getenv("DOC_DB_URI")


def create_custom_react_agent(spec: ReactAgentSpec, collection:str=None, checkpointer:BaseCheckpointSaver[str]|None=None):
    # Setup a Postgres vector store with 
    embeddings = get_embeddings_class_instance(
        EmbeddingsProvider.OLLAMA, 
        model="nomic-embed-text:latest"
    )
    store = get_vector_store_class_instance(
        VectorStoreProvider.PGVECTOR,
        embeddings=embeddings,
        embedding_length=768,
        collection_name=collection,
        connection = DB_URI,
        create_extension=False
    )

    if spec.state_schema:
        graph = StateGraph(state_schema=spec.state_schema)
    else:
        graph = StateGraph(state_schema=AgentState)

    agent = get_chat_model_class_instance(
        spec.model_provider, 
        model=spec.model
    ).with_structured_output(spec.state_schema).bind_tools(spec.tools)
    
    graph.set_entry_point("llm")
    def agent_node(state: AgentState):
        return {"messages": [agent.invoke([spec.prompt] + state["messages"])]}

    graph.add_node("llm", agent_node)
    if collection:
        retriever_tool = StructuredTool.from_function(
            func=store.similarity_search_with_relevance_scores, 
            name="pgvector_search"
        )
        spec.tools.append(retriever_tool)
    if len(spec.tools)>0:
        graph.add_node("tools", ToolNode(spec.tools))
        graph.add_conditional_edges("llm", tools_condition)
        graph.add_edge("tools", "llm")
    return graph.compile(store=store, checkpointer=checkpointer)

async def stream_custom_agent(
        spec: ReactAgentSpec, 
        inputs: dict, 
        stream_mode: Literal["values", "updates"] = "values", 
        config:dict = None,
        collection:str|None=None
    ) -> AsyncGenerator[str, None]:
    """
    Stream responses from a LangGraph Ollama agent.
    
    Args:
        agent: The specifications for the agent to build
        inputs: Input dictionary with messages
        stream_mode: Either "values" (final values) or "updates" (incremental changes)
    
    Yields:
        String chunks of the agent's response
    """
    dotenv.load_dotenv()
    DB_URI = os.getenv("CONVERSATION_DB_URI")
    async with AsyncPostgresSaver.from_conn_string(DB_URI) as checkpointer:
        await checkpointer.setup()
        agent = create_custom_react_agent(spec=spec, collection=collection, checkpointer=checkpointer)

        async for s in agent.astream(inputs, stream_mode=stream_mode, config=config):
            try:
                # Get the latest message
                if "messages" in s:
                    message = s["messages"][-1]
                    print(str(message))
                    if isinstance(message, ToolMessage):
                        yield json.dumps({"type": "ToolMessage", "content": message.content, "thread": config["configurable"]["thread_id"]})
                    elif hasattr(message, "tool_calls") and len(message.tool_calls) > 0:
                        yield json.dumps({"type": "ToolCallMessage", "calls": message.tool_calls, "thread": config["configurable"]["thread_id"]})
                    elif isinstance(message, AIMessage):
                        yield json.dumps({"type": "AIMessage", "content": message.content, "thread": config["configurable"]["thread_id"]})
                    elif isinstance(message, HumanMessage):
                        yield json.dumps({"type": "HumanMessage", "content": message.content, "thread": config["configurable"]["thread_id"]})
            except Exception as e:
                yield json.dumps({"type": "error", "content": str(e)})

async def ollama_generate(agent, agent_input, logger, config, collection:str =None):
    try:
        counter = 0
        logger.info("Starting stream generation")
        
        async for chunk in stream_custom_agent(agent, agent_input, stream_mode="values", config=config, collection=collection):
            counter += 1
            logger.debug(f"Chunk {counter}: {chunk}")
            
            # Format for SSE
            yield f"data: {chunk}\n\n"

            # Add small delay for visibility in logs
            await asyncio.sleep(0.1)
            
        logger.info(f"Stream complete, sent {counter} chunks")
        
    except Exception as e:
        logger.error(f"Error in stream generation: {str(e)}", exc_info=True)
        yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"

def stream_agent(agent_spec:ReactAgentSpec, inputs:dict, logger:logging.Logger, config:dict, collection:str = None):
    logger.info(f"Received request with message: {inputs.message}")
    agent_input = {"messages": [{"content": inputs.message, "role": "user"}]}
    logger.debug(f"Agent input: {json.dumps(agent_input)}")
    return StreamingResponse(
        ollama_generate(agent_spec, agent_input, logger, config, collection),
        media_type="text/event-stream"
    )