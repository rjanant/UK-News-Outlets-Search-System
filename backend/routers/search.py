from fastapi import APIRouter, Depends, Query
from fastapi.responses import ORJSONResponse
from os.path import basename
from typing import Optional, Annotated
from pydantic import BaseModel, Field
from utils.basetype import Result
from os import getenv
from utils.query_engine import boolean_test, ranked_test

router = APIRouter(
    prefix=f"/{basename(__file__).replace('.py', '')}",
    tags=[basename(__file__).replace('.py', '')],
    dependencies=[],
    responses={404: {"description": "Not found"}}
)


class SearchResponse(BaseModel):
    results: list[Result]
    truth_value: float

@router.get("/")
async def search(
    q: str = Query(..., description="Search query", min_length=1, max_length=1024),
    year: Optional[int] = Query(None, description="Year of the result", ge=1900, le=2100),
    page: Optional[int] = Query(1, description="Page number", ge=1),
    limit: Optional[int] = Query(10, description="Results per page", ge=1, le=100)) -> SearchResponse:
    r'''
    Searching the results from the database.
    ```
        - q: query to search
        - page: page number
        - limit: results per page
    ```
    '''
    ## Search the results
    return ORJSONResponse(content={"results": ['123213'], "truth_value": 0.0})

class TestBody(BaseModel):
    field: str = Field(..., description="Test field", min_length=1, max_length=1024)
@router.post("/test")
async def test(body: TestBody):
    test_env = getenv("TESTING", "default")
    return ORJSONResponse(content={"field": body.field, "env": test_env})

@router.get("/boolean")
async def boolean_search(
    q: str = Query(..., description="Search query", min_length=1, max_length=1024),
    page: Optional[int] = Query(1, description="Page number", ge=1),
    limit: Optional[int] = Query(10, description="Results per page", ge=1, le=100)):
    r'''
    Searching the results from the database.
    ```
        - q: query to search (Must be a boolean query with AND, OR, NOT, brackets, proximity, exact match and word match)
        - page: page number (default: 1)
        - limit: results per page (default: 10)
    ```
    '''
    results = await boolean_test([q])
    if not results or len(results) > page*limit:
        return []
    return results[(page-1)*limit:page*limit]

@router.get("/tfidf")
async def tfidf_search(
    q: str = Query(..., description="Search query", min_length=1, max_length=1024),
    page: Optional[int] = Query(1, description="Page number", ge=1),
    limit: Optional[int] = Query(10, description="Results per page", ge=1, le=100)):
    r'''
    Searching the results from the database.
    ```
        - q: query to search (Treat every word as a seperated term)
        - page: page number
        - limit: results per page
    ```
    '''
    results = await ranked_test([q])
    if not results or len(results) > page*limit:
        return []
    return results[(page-1)*limit:page*limit]