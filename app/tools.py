from langchain_core.tools import StructuredTool
from tavily import TavilyClient
from typing import Literal
import dotenv
import os

def math(operands: tuple[int, int], operation: Literal["+","-","*","/"])-> str:
    """Performs basic math operations on two operands."""
    match operation:
        case "+":
            return str(sum(operands))
        case "-":
            return str(operands[0] - operands[1])
        case "*":
            return str(operands[0] * operands[1])
        case "/":
            return str(operands[0] / operands[1])
        case _:
            return "Invalid operation"
        
async def async_math(operands: tuple[int, int], operation: Literal["+","-","*","/"]) -> str:
    """Performs basic math operations on two operands."""
    match operation:
        case "+":
            return str(sum(operands))
        case "-":
            return str(operands[0] - operands[1])
        case "*":
            return str(operands[0] * operands[1])
        case "/":
            return str(operands[0] / operands[1])
        case _:
            return "Invalid operation"

def search(query: str) -> dict:
    """Searches the web for the query."""
    dotenv.load_dotenv()
    client = TavilyClient(os.getenv("TAVILY_API_KEY"))
    results = client.search(
        query=query,
        max_results=5,
        include_answer=True,
        include_raw_content=True,
        include_images=True
    )
    return results



math_tool = StructuredTool.from_function(name="math", func=math, coroutine=async_math)

search_tool = StructuredTool.from_function(func=search, name="search")

