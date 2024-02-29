import orjson
import redis
import os
import asyncio
import aioredis
import time
from tqdm import tqdm
from basetype import InvertedIndex, RedisKeys, RedisDocKeys, NewsArticleData
from typing import List
from typing import Dict

BASEPATH = os.path.dirname(__file__)

redis_async_connection = None
redis_connection = None 
redis_config = None

def get_redis_config(env="prod", is_async=True, db=0):
    # pip install python-dotenv
    from dotenv import load_dotenv

    if env == "dev":
        cfg_path = os.path.join(BASEPATH, "redis_config_dev.env")
    else:
        cfg_path = os.path.join(BASEPATH, "redis_config_prod.env")

    load_dotenv(cfg_path)

    REDIS_HOST = os.environ.get("REDIS_HOST") or os.getenv("REDIS_HOST")
    REDIS_PORT = os.environ.get("REDIS_PORT") or os.getenv("REDIS_PORT")
    REDIS_PASSWORD = os.environ.get("REDIS_PASSWORD") or os.getenv("REDIS_PASSWORD")

    if is_async:
        config_redis = {'address': (REDIS_HOST, REDIS_PORT), "password": REDIS_PASSWORD, "db":db}
    else:
        config_redis = {"host": REDIS_HOST, "port": REDIS_PORT, "password": REDIS_PASSWORD, "db":db}

    return config_redis

# Redis Functions
def initialize_sync_redis():
    global redis_connection
    try:
        if not redis_connection:
            config_redis = get_redis_config(is_async=False)
            redis_connection = redis.StrictRedis(**config_redis)
        else:
            redis_connection.ping()
    except (redis.ConnectionError, redis.RedisError, Exception):
        print("Catched exception: ", Exception)
        config_redis = get_redis_config(is_async=False)
        redis_connection = redis.StrictRedis(**config_redis)

async def initialize_async_redis(db=0):
    global redis_async_connection
    try:
        if not redis_async_connection:
            config_redis = get_redis_config(db=db)
            redis_async_connection = await aioredis.create_redis_pool(**config_redis)
        else:
            await redis_async_connection.ping()
    except (aioredis.ConnectionClosedError, aioredis.RedisError, aioredis.ConnectionForcedCloseError, Exception):
        print("Catched exception: ", Exception)
        config_redis = get_redis_config(db=db)
        redis_async_connection = await aioredis.create_redis_pool(**config_redis)

# decorator to check if the redis connection is initialized
def check_redis_connection(func):
    def wrapper(*args, **kwargs):
        initialize_sync_redis()
        return func(*args, **kwargs)
    return wrapper

# decorator to check if the async redis connection is initialized
def do_check_async_redis_connection(db=0):
    def check_async_redis_connection(func):
        async def wrapper(*args, **kwargs):
            await initialize_async_redis(db=db)
            return await func(*args, **kwargs)
        return wrapper
    return check_async_redis_connection

@check_redis_connection
def update_doc_size(new_size, colname="meta:document_size"):
    doc_size = redis_connection.get(colname)

    if not (doc_size):
        doc_size = new_size
    else:
        doc_size = int(doc_size)
        doc_size += new_size

    redis_connection.set(colname, doc_size)
    return True

@do_check_async_redis_connection(db=0)
async def get_doc_size() -> int:
    doc_size = await redis_async_connection.get(RedisKeys.document_size)
    return int(doc_size)

@do_check_async_redis_connection(db=1)
async def get_doc_key(doc_id, key):
    result = await redis_async_connection.hget(
        RedisKeys.document(doc_id),
        key
    )
    return result.decode()

@do_check_async_redis_connection(db=1)
async def get_doc_data(doc_id):
    result = await redis_async_connection.hgetall(
        RedisKeys.document(doc_id)
    )
    return {key.decode('utf-8'): value.decode('utf-8') for key, value in result.items()}

@do_check_async_redis_connection(db=0)
async def get_doc_info() -> int:
    doc_size = await redis_async_connection.get(RedisKeys.document_size)
    return int(doc_size)

@do_check_async_redis_connection(db=0)
async def get_doc_ids_list() -> List[int]:
    doc_ids_list = await redis_async_connection.get(RedisKeys.doc_ids_list)
    return orjson.loads(doc_ids_list)

@check_redis_connection
def get_val(key):
    start_time = time.time()
    value = redis_connection.get(key)
    value = value.decode()
    value = eval(value)
    print(f"Time taken to get {key}: {time.time() - start_time}")
    return value



@do_check_async_redis_connection(db=0)
async def set_data(key, value):
    await redis_async_connection.set(key, value)

@do_check_async_redis_connection(db=0)
async def batch_push(batches):
    # Define keys and values to set
    for batch in tqdm(batches, desc="PUSH"):
        # Perform SET operations in parallel
        tasks = [set_data(key, value) for key, value in batch.items()]
        await asyncio.gather(*tasks)

@do_check_async_redis_connection(db=1)
async def set_news_data(article: NewsArticleData):
    # Get metadata
    doc_title = article.title
    doc_url = article.url
    doc_id = RedisKeys.document(article.doc_id)
    doc_date = article.date
    doc_sentiment = 'positive'
    doc_summary = ".".join(article.content.split('.')[:3])
    doc_source = article.url.split('.')[1]

    # Set the values
    lua_script = f"""
        redis.call('hset', ARGV[1], '{RedisDocKeys.url}', ARGV[2])
        redis.call('hset', ARGV[1], '{RedisDocKeys.title}', ARGV[3])
        redis.call('hset', ARGV[1], '{RedisDocKeys.date}', ARGV[4])
        redis.call('hset', ARGV[1], '{RedisDocKeys.sentiment}', ARGV[5])
        redis.call('hset', ARGV[1], '{RedisDocKeys.summary}', ARGV[6])
        redis.call('hset', ARGV[1], '{RedisDocKeys.source}', ARGV[7])
    """
    # Run the Lua script
    await redis_async_connection.eval(
        lua_script, 
        keys=[], 
        args=[
            doc_id, #1 
            doc_url, #2
            doc_title, #3
            doc_date, #4
            doc_sentiment, #5
            doc_summary, #6
            doc_source
            ]
        )

@do_check_async_redis_connection(db=1)
async def batch_push_news_data(news_batch):
    # Define keys and values to set
    tasks = []
    for _source in news_batch.fragments:
        if type(_source) != str:
            continue
        for fragment in news_batch.fragments[_source]:
            for article in fragment.articles:
                tasks.append(set_news_data(article))
    await asyncio.gather(*tasks)

@do_check_async_redis_connection(db=0)
async def update_index(inverted_index: InvertedIndex):
    tasks = []
    for term in inverted_index.index:
        tasks.append(update_index_term(term, inverted_index))
    await asyncio.gather(*tasks)
    
    doc_size = await redis_async_connection.get("meta:document_size")
    doc_ids_list = await redis_async_connection.get("meta:doc_ids_list")
    if not doc_size:
        doc_size = inverted_index.meta.document_size
    else:
        doc_size = int(doc_size) + inverted_index.meta.document_size

    
    if not doc_ids_list:
        doc_ids_list = inverted_index.meta.doc_ids_list
    else:
        doc_ids_list = orjson.loads(doc_ids_list)
        doc_ids_list.extend(inverted_index.meta.doc_ids_list)
    
    await redis_async_connection.mset({
        "meta:document_size": doc_size,
        "meta:doc_ids_list": orjson.dumps(doc_ids_list)
    })
            
async def update_index_term(term, inverted_index: InvertedIndex):
    db_value = await redis_async_connection.get(term)
    if not db_value:
        db_value = {}
    else:
        db_value = orjson.loads(db_value)
    
    for doc_id, pos in inverted_index.index[term].items():
        db_value[doc_id] = pos
    
    await redis_async_connection.set(f"w:{term}", orjson.dumps(db_value))
    
@do_check_async_redis_connection(db=0)
async def get_json_value(key: str) -> Dict:
    value = await redis_async_connection.get(key)
    return orjson.loads(value)

@do_check_async_redis_connection(db=0)
async def get_json_values(keys: List[str]) -> List[Dict]:
    values_list = await redis_async_connection.mget(*keys)
    values_list = [orjson.loads(value) for value in values_list]
    return values_list

@do_check_async_redis_connection(db=0)
async def get_idf_value(key: str) -> float:
    value = await redis_async_connection.get(key)
    return float(value)

@do_check_async_redis_connection(db=0)
async def get_document_infos(doc_ids: List[int]) -> List[Dict]:
    keys = [RedisKeys.document(doc_id) for doc_id in doc_ids]
    return await get_json_values(keys)

@check_redis_connection
def clear_redis():
    redis_connection.flushall()
    return True

@do_check_async_redis_connection(db=0)
async def is_key_exists(key):
    return await redis_async_connection.exists(key)

@do_check_async_redis_connection(db=0)
async def caching_query_result(method: str, query: str, result):
    key = f"{method}:{query}"
    # non-blocking set operation
    asyncio.create_task(set_data(key, orjson.dumps(result)))

async def test():
    # print(await get_values([RedisKeys.index('man'), RedisKeys.index("woman")]))
    print(await is_key_exists(RedisKeys.index('[]]')))

if __name__ == "__main__":
    # clear_redis()
    asyncio.run(test())
