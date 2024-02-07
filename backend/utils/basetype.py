from pydantic import BaseModel

class Result(BaseModel):
    id: str
    title: str
    description: str
    url: str
    score: float
    timestamp: int