from fastapi import FastAPI, Request, UploadFile, BackgroundTasks, Depends
from fastapi_server_session import SessionManager, RedisSessionInterface, Session
from fastapi.responses import  JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import logging
import sys
import dotenv
import redis
import os
import uuid
from models import ChatInput, AuthTokenRequest, RefreshRequest
from agents import math_nerd, web_nerd, regular_nerd, stream_agent, assemble_thread_config
from loaders import stream_loader, python_loader_graph, assemble_loader_config
from draw import draw_pipeline
from azure_oidc_auth import signup, check_session, refresh_session


logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("nerd-api")

app = FastAPI()
dotenv.load_dotenv()

session_manager = SessionManager(
    interface=RedisSessionInterface(redis.from_url(os.getenv("REDIS_URI")))
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

    

@app.post("/oauth_signup")
async def oauth_token(request:AuthTokenRequest, session: Session = Depends(session_manager.use_session)):
    return signup(request, session)

@app.post("/oauth_refresh")
async def oauth_refresh(request:RefreshRequest, session: Session = Depends(session_manager.use_session)):
    return refresh_session(session, request.id_token)

@app.get("/draw/{pipeline}")
async def draw_pipeline_graph(pipeline: str):
    draw_pipeline(pipeline)
    
@app.post("/regular_nerd")
async def regular_nerd_chat(
    inputs: ChatInput, 
    thread_id: str = None,
    collection:str=None,
    session: Session = Depends(session_manager.use_session)):
    session_invalid = check_session(session)
    if session_invalid:
        return session_invalid
    if not thread_id:
        thread_id = str(uuid.uuid4())
    config = assemble_thread_config(thread_id, session)
    if not config:
        return JSONResponse(
            status_code=403,
            content={"error": "User not logged in"}
        )
    return stream_agent(regular_nerd, inputs, logger, config, collection)

@app.post("/math_nerd")
async def math_nerd_chat(
    inputs: ChatInput, 
    thread_id: str = None,
    collection:str=None,
    session: Session = Depends(session_manager.use_session)):
    session_invalid = check_session(session)
    if session_invalid:
        return session_invalid
    if not thread_id:
        thread_id = str(uuid.uuid4())
    config = assemble_thread_config(thread_id, session)
    if not config:
        return JSONResponse(
            status_code=403,
            content={"error": "User not logged in"}
        )
    return stream_agent(math_nerd, inputs, logger, config, collection)

@app.post("/web_nerd") 
async def web_nerd_chat(
    inputs: ChatInput, 
    thread_id: str = None,
    collection:str=None,
    session: Session = Depends(session_manager.use_session)):
    session_invalid = check_session(session)
    if session_invalid:
        return session_invalid
    if not thread_id:
        thread_id = str(uuid.uuid4())
    config = assemble_thread_config(thread_id, session)
    if not config:
        return JSONResponse(
            status_code=403,
            content={"error": "User not logged in"}
        )
    return stream_agent(web_nerd, inputs, logger, config, collection)

@app.post("/upload/python")
async def upload_python_file(
    files: list[UploadFile], 
    collection: str, 
    bg: BackgroundTasks,
    session: Session = Depends(session_manager.use_session)):
    session_invalid = check_session(session)
    if session_invalid:
        return session_invalid
    config = assemble_loader_config(collection, session["user_id"])
    if not config:
        return JSONResponse(
            status_code=403,
            content={"error": "User not logged in"}
        )
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