from fastapi import FastAPI, Request, UploadFile, BackgroundTasks
from fastapi.responses import  JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import logging
import sys
import dotenv
from models import ChatInput
from agents import math_nerd, web_nerd, regular_nerd, stream_agent
from loaders import stream_loader, python_loader_graph
from draw import draw_pipeline

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("nerd-api")

app = FastAPI()
dotenv.load_dotenv()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/draw/{pipeline}")
async def draw_pipeline_graph(pipeline: str):
    draw_pipeline(pipeline)
    
@app.post("/regular_nerd")
async def regular_nerd_chat(
    inputs: ChatInput, 
    thread_id: str = None, 
    collection:str=None):
    return stream_agent(regular_nerd, inputs, logger, thread_id, collection)

@app.post("/math_nerd")
async def math_nerd_chat(
    inputs: ChatInput, 
    thread_id: str = None, 
    collection:str=None):
    return stream_agent(math_nerd, inputs, logger, thread_id, collection)

@app.post("/web_nerd") 
async def web_nerd_chat(
    inputs: ChatInput, 
    thread_id: str = None, 
    collection:str=None):
    return stream_agent(web_nerd, inputs, logger, thread_id, collection)

@app.post("/upload/python")
async def upload_python_file(
    files: list[UploadFile], 
    collection: str, 
    bg: BackgroundTasks):
    return stream_loader(python_loader_graph(), collection, files, bg, logger)
    
@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"Request path: {request.url.path}")
    try:
        response = await call_next(request)
        return response
    except Exception as e:
        logger.error(f"Request error: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)