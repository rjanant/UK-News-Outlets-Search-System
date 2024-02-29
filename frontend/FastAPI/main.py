from fastapi import FastAPI, HTTPException, Depends
from typing import Annotated,List
from sqlalchemy.orm import Session
from pydantic import BaseModel
from database import SessionLocal, engine
import models
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = [
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers = ['*']
)
#Validate the request from the react application
class FastCheckerBase(BaseModel):
    title: str
    description: str
    url: str
    truth: bool
    

class FactCheckerModel(FastCheckerBase):
    id: int

    class Config():
        orm_mode = True

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

db_dependency = Annotated[Session, Depends(get_db)]

models.Base.metadata.create_all(bind=engine)

@app.post("/factchecker/", response_model=FactCheckerModel)
async def create_factchecker(factchecker: FastCheckerBase, db: db_dependency):
    new_factchecker = models.FactChecker(**factchecker.dict())
    db.add(new_factchecker)
    db.commit()
    db.refresh(new_factchecker)
    return new_factchecker

@app.get("/factchecker/", response_model=list[FactCheckerModel])
async def read_factchecker(db: db_dependency, skip: int = 0, limit: int = 100):
    factchecker = db.query(models.FactChecker).offset(skip).limit(limit).all()
    return factchecker