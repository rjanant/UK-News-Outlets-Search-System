from fastapi import FastAPI
import uvicorn
from routers.api import router as api_router
from dotenv import load_dotenv
import os
import sys

app = FastAPI(dependencies=[])
app.include_router(api_router)

if __name__ == "__main__":
    load_dotenv()
    
    args = sys.argv
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    if not os.path.exists(f"files"):
        os.makedirs(f"files")
    
    uvicorn.run("main:app", 
                host="127.0.0.1",
                port=8001,
                reload=True
                )
