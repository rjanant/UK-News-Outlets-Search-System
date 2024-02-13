from fastapi import FastAPI
import uvicorn
from routers.api import router as api_router
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
import os

load_dotenv()
os.chdir(os.path.dirname(os.path.abspath(__file__)))
app = FastAPI(dependencies=[])
app.include_router(api_router)
app.mount("/", StaticFiles(directory="react", html=True), name="static")