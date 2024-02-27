import orjson
import redis
import os
import sys
import asyncio
import aioredis
import time
from tqdm import tqdm
from basetype import InvertedIndex

BASEPATH = os.path.dirname(__file__)

redis_async_connection = None
redis_connection = None 
redis_config = None

def get_redis_config(env="prod", is_async=True):
    # pip install python-dotenv
    from dotenv import load_dotenv

    if env == "dev":
        cfg_path = os.path.join(BASEPATH, "redis_config_dev.env")
    else:
        cfg_path = os.path.join(BASEPATH, "redis_config_prod.env")

    load_dotenv(cfg_path)

    REDIS_HOST = os.environ.get("REDIS_HOST")
    REDIS_PORT = os.environ.get("REDIS_PORT")
    REDIS_PASSWORD = os.environ.get("REDIS_PASSWORD")

    if is_async:
        config_redis = {'address': (REDIS_HOST, REDIS_PORT), "password": REDIS_PASSWORD}
    else:
        config_redis = {"host": REDIS_HOST, "port": REDIS_PORT, "password": REDIS_PASSWORD}

    return config_redis


def get_secret_value(project_id="652914548272", secret_id="redis", key="redis-test"):
    """To get the keys from Google Secret Manager.
    Note: Ask on discord to get the values.
    deprecated soon.
    """
    from google.cloud import secretmanager

    # Create the Secret Manager client.
    client = secretmanager.SecretManagerServiceClient()

    name = f"projects/{project_id}/secrets/{secret_id}/versions/latest"

    # Access the secret version.
    response = client.access_secret_version(request={"name": name})

    payload = response.payload.data.decode("UTF-8")
    configs = eval(payload)

    config_redis = configs[key]

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

async def initialize_async_redis():
    global redis_async_connection
    try:
        if not redis_async_connection:
            config_redis = get_redis_config()
            redis_async_connection = await aioredis.create_redis_pool(**config_redis)
        else:
            redis_async_connection.ping()
    except (aioredis.ConnectionClosedError, aioredis.RedisError, aioredis.ConnectionForcedCloseError, Exception):
        print("Catched exception: ", Exception)
        config_redis = get_redis_config()
        redis_async_connection = await aioredis.create_redis_pool(**config_redis)

# decorator to check if the redis connection is initialized
def check_redis_connection(func):
    def wrapper(*args, **kwargs):
        initialize_sync_redis()
        return func(*args, **kwargs)
    return wrapper

# decorator to check if the async redis connection is initialized
def check_async_redis_connection(func):
    async def wrapper(*args, **kwargs):
        await initialize_async_redis()
        return await func(*args, **kwargs)
    return wrapper

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

@check_redis_connection
def get_doc_size(colname="meta:document_size"):
    doc_size = redis_connection.get(colname)
    return int(doc_size)

@check_redis_connection
def get_val(key):
    start_time = time.time()
    value = redis_connection.get(key)
    value = value.decode()
    value = eval(value)
    return value

@check_async_redis_connection
async def set_data(key, value):
    await redis_async_connection.set(key, value)

@check_async_redis_connection
async def batch_push(batches):
    # Define keys and values to set
    for batch in tqdm(batches, desc="PUSH"):
        # Perform SET operations in parallel
        tasks = [set_data(key, value) for key, value in batch.items()]
        await asyncio.gather(*tasks)

@check_async_redis_connection
async def update_index(inverted_index: InvertedIndex):
    doc_size = await redis_async_connection.get("meta:document_size")
    doc_ids_list = await redis_async_connection.get("meta:doc_ids_list")
    if not doc_size:
        doc_size = inverted_index.meta.document_size
    else:
        doc_size = int(doc_size) + inverted_index.meta.document_size

    redis_async_connection.set("meta:document_size", doc_size)
    
    if not doc_ids_list:
        doc_ids_list = inverted_index.meta.doc_ids_list
    else:
        doc_ids_list = orjson.loads(doc_ids_list)
        doc_ids_list.extend(inverted_index.meta.doc_ids_list)
    redis_async_connection.set("meta:doc_ids_list", orjson.dumps(doc_ids_list))
    
    tasks = []
    for term in inverted_index.index:
        tasks.append(update_index_term(term, inverted_index))
    await asyncio.gather(*tasks)
    
    # print("Index updated")
        
async def update_index_term(term, inverted_index: InvertedIndex):
    db_value = await redis_async_connection.get(term)
    if not db_value:
        db_value = {}
    else:
        db_value = orjson.loads(db_value)
    
    for doc_id, pos in inverted_index.index[term].items():
        db_value[doc_id] = pos
    
    await redis_async_connection.set(term, orjson.dumps(db_value))
    

if __name__ == "__main__":
    print("Location:", BASEPATH)