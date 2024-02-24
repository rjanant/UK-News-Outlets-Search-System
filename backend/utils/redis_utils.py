import json
import redis
import os, sys
import asyncio
import aioredis

from tqdm import tqdm

def get_secret_value(project_id="652914548272", secret_id="redis", key="redis-test"):
    """To get the keys from Google Secret Manager.
    Note: Ask on discord to get the values.
    """
    from google.cloud import secretmanager

    # Create the Secret Manager client.
    client = secretmanager.SecretManagerServiceClient()

    name = f"projects/{project_id}/secrets/{secret_id}/versions/latest"

    # Access the secret version.
    response = client.access_secret_version(request={"name": name})

    payload = response.payload.data.decode("UTF-8")
    configs = eval(payload)

    config_key = configs[key]

    return config_key


# Redis Functions
def get_redis(config_redis):

    r = redis.Redis(
        host=config_redis["host"],
        port=config_redis["port"],
        password=config_redis["password"],
    )
    return r

def update_doc_size(r, new_size, colname='document_size'):
    doc_size = r.get(colname)
    
    if not (doc_size):
        doc_size = new_size
    else:
        doc_size = int(doc_size)
        doc_size += new_size

    r.set(colname, doc_size)
    return True

def get_doc_size(r, colname='document_size'):
    doc_size = r.get(colname)
    return int(doc_size)

def get_val(r, key):
    value = r.get(key)
    value = value.decode()
    value = eval(value)
    return value

async def set_data(redis, key, value):
    await redis.set(key, value)

async def batch_push(config_redis, batches):
    # Define Redis server configuration
    redis_config = {
        'address': (config_redis["host"], config_redis["port"]),  # IP address and port
        'password': config_redis["password"]  # Redis password
    }
    
    # Connect to Redis
    redis = await aioredis.create_redis_pool(**redis_config)

    # Define keys and values to set
    for batch in tqdm(batches, desc="PUSH"):
        # Perform SET operations in parallel
        tasks = [set_data(redis, key, value) for key, value in batch.items()]
        await asyncio.gather(*tasks)

    # Close Redis connection
    redis.close()
    await redis.wait_closed()