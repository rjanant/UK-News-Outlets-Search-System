from fastapi import APIRouter, Depends, Query
from fastapi.responses import ORJSONResponse
from os.path import basename
from os import getenv
from typing import Optional, Annotated
from pydantic import BaseModel, Field
from utils.basetype import Result
from utils.query_engine import boolean_test, ranked_test
from utils.redis_utils import get_document_infos, caching_query_result, is_key_exists, get_json_value
from utils.basetype import RedisKeys
from math import ceil

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
    # uncomment this when the caching is ready
    # if await is_key_exists(RedisKeys.cache("boolean", q)):
    #     return await get_json_value(RedisKeys.cache("boolean", q))
    
    results = await boolean_test([q])
    if not results or len(results) > page*limit:
        return []
    
    # uncomment this if the document info is ready
    # for idx, doc_id_list in enumerate(results):
    #     results[idx] = await get_document_infos(doc_id_list)

    response = {
        "results": results[0][(page-1)*limit:page*limit],
        "total_pages": ceil(len(results[0])/limit)
    }
    
    # uncomment this when the caching is ready
    # await caching_query_result("boolean", q, response)
    
    return response

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
    # uncomment this when the caching is ready
    # if await is_key_exists(RedisKeys.cache("tfidf", q)):
    #     return await get_json_value(RedisKeys.cache("tfidf", q))
    
    results = await ranked_test([q])
    
    # uncomment this if the document info is ready
    # for idx, result in enumerate(results):
    #     doc_id_list = [t[0] for t in result]
    #     doc_info_list = await get_document_infos(doc_id_list)
    #     results[idx] = [(doc_info_list[i], t[1]) for i, t in enumerate(result)]
        
    if not results or len(results) > page*limit:
        return []

    response = {
        "results": results[0][(page-1)*limit:page*limit],
        "total_pages": ceil(len(results[0])/limit)
    }
    
    # uncomment this when the caching is ready
    # await caching_query_result("tfidf", q, response)
    
    return response