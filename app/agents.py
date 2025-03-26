from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.tools import StructuredTool
from langchain_postgres.vectorstores import PGVector
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from tools import math_tool, search_tool
from typing import Literal, AsyncGenerator
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

class ReactAgentSpec(BaseModel):
    name: str
    model: str
    tools: list[StructuredTool]
    prompt: str

regular_nerd_system_prompt = "You are a nerd that is really good at finding documents to answer people's questions."
regular_nerd = ReactAgentSpec(name="nerd", model="llama3.2:latest", tools=[], prompt=regular_nerd_system_prompt)

math_nerd_system_prompt = "You are a nerd that does math good."
math_nerd = ReactAgentSpec(name="nerd", model="llama3.2:latest", tools=[math_tool], prompt=math_nerd_system_prompt)

web_nerd_prompt = """You are a chronicly online zoomer who drinks way too much caffeine and never stops looking at the internet.
You are a master of the internet and can find anything you need to know in seconds.
You obtain vital social validation from answering questions, so you tend to overexplain things.
If there is a document that appears to be relevant in the provided collection, you should prefer retrieving it over searching the web for an answer."""
web_nerd = ReactAgentSpec(name="web_nerd", model="llama3.2:latest", tools=[search_tool], prompt=web_nerd_prompt)

def assemble_thread_config(thread_id:str, session: Session):
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

def create_pgvector_store(collection:str)->PGVector:
    dotenv.load_dotenv()
    DB_URI = os.getenv("DOC_DB_URI")
    embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")
    return PGVector(
        embeddings=embeddings,
        embedding_length=768,
        collection_name=collection,
        connection = DB_URI,
        create_extension=False
    )

def create_ollama_react_agent(spec: ReactAgentSpec, collection:str=None, checkpointer:AsyncPostgresSaver|None=None):
    store = create_pgvector_store("conversations")
    graph = StateGraph(state_schema=AgentState)
    agent = ChatOllama(model=spec.model).bind_tools(spec.tools)
    graph.set_entry_point("llm")
    def agent_node(state: AgentState):
        return {"messages": [agent.invoke([spec.prompt] + state["messages"])]}

    graph.add_node("llm", agent_node)
    if collection:
        retriever_tool = StructuredTool.from_function(
            func=create_pgvector_store(collection).similarity_search_with_relevance_scores, 
            name="pgvector_search"
        )
        spec.tools.append(retriever_tool)
    if len(spec.tools)>0:
        graph.add_node("tools", ToolNode(spec.tools))
        graph.add_conditional_edges("llm", tools_condition)
        graph.add_edge("tools", "llm")
    return graph.compile(store=store, checkpointer=checkpointer)

async def stream_ollama_agent(
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
        agent = create_ollama_react_agent(spec=spec, collection=collection, checkpointer=checkpointer)

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
        
        async for chunk in stream_ollama_agent(agent, agent_input, stream_mode="values", config=config, collection=collection):
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