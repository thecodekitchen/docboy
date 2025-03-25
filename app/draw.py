from fastapi.responses import Response, JSONResponse
from langgraph.pregel import Pregel

from agents import math_nerd, web_nerd, regular_nerd, create_ollama_react_agent, ReactAgentSpec
from loaders import python_loader_graph

def draw_agent(spec:ReactAgentSpec):
    agent = create_ollama_react_agent(spec)
    graph_bytes = agent.get_graph().draw_mermaid_png()
    return Response(content=graph_bytes, media_type="image/png", headers={"Content-Disposition": f"inline; filename={agent.name}.png"})

def draw_loader(compiled_graph:Pregel):
    graph_bytes = compiled_graph.get_graph().draw_mermaid_png()
    return Response(content=graph_bytes, media_type="image/png", headers={"Content-Disposition": f"inline; filename={compiled_graph.name}.png"})

def draw_pipeline(pipeline:str):
    match pipeline:
        case "math_nerd":
            return draw_agent(math_nerd)
        case "web_nerd":
            return draw_agent(web_nerd)
        case "regular_nerd":
            return draw_agent(regular_nerd)
        case "python_loader":
            return draw_loader(python_loader_graph())
        case _:
            return JSONResponse(
                status_code=404,
                content={"error": f"Graph {pipeline} not found"}
            )