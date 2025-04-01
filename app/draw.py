from fastapi.responses import Response, JSONResponse
from langgraph.pregel import Pregel

from agents import math_nerd, web_nerd, regular_nerd, create_custom_react_agent, ReactAgentSpec
from loaders import python_loader_graph, DocLoaderSpec, compile_loader_graph

def draw_agent(spec:ReactAgentSpec):
    agent = create_custom_react_agent(spec)
    graph_bytes = agent.get_graph().draw_mermaid_png()
    return Response(content=graph_bytes, media_type="image/png", headers={"Content-Disposition": f"inline; filename={agent.name}.png"})

def draw_loader(spec: DocLoaderSpec):
    loader = compile_loader_graph(spec)
    graph_bytes = loader.get_graph().draw_mermaid_png()
    return Response(content=graph_bytes, media_type="image/png", headers={"Content-Disposition": f"inline; filename={loader.name}.png"})

def draw_pregel(pregel:Pregel):
    graph_bytes = pregel.get_graph().draw_mermaid_png()
    return Response(content=graph_bytes, media_type="image/png", headers={"Content-Disposition": f"inline; filename={pregel.name}.png"})
