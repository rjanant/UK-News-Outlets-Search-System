import orjson
import os
import sys
import asyncio

from tqdm import tqdm

BASEPATH = os.path.dirname(__file__)
sys.path.append(BASEPATH)

# from redis_utils import get_redis_config, update_doc_size, batch_push
from common import read_file
from basetype import InvertedIndex
from redis_utils import initialize_async_redis, update_index, get_redis_config


def load_index(path_index="result/inverted_index.json"):
    with open(path_index, "r+") as f:
        data_json = orjson.loads(f.read())
    return data_json


def process_dict_in_batches(input_dict, batch_size, prefix="w:"):
    """Deprecated soon"""
    keys = list(input_dict.keys())
    num_keys = len(keys)
    batches = []
    for i in range(0, num_keys, batch_size):
        batch_keys = keys[i : i + batch_size]
        batch = {prefix + key: str(input_dict[key]) for key in batch_keys}
        batches.append(batch)
    return batches

# def do_push_index():

if __name__ == "__main__":
    config_redis = get_redis_config("")
    print(config_redis["address"])

    initialize_async_redis()

    INDEX_PATH = os.path.join(BASEPATH, "index", "child")
    files = os.listdir(INDEX_PATH)
    for f in tqdm(files):
        filepath = os.path.join(BASEPATH, "index", "child", f)
        inverted_index_str = read_file(filepath)
        inverted_index = InvertedIndex.model_validate_json(inverted_index_str)
        asyncio.run( update_index(inverted_index) )


    # minibatch = 15
    # filepath = "result/inverted_index.json"

    # # Ask on discord to get the hardcoded config
    # data_json = load_index(path_index=filepath)
    # batches = process_dict_in_batches(data_json["index"], minibatch)

    # asyncio.run(batch_push(batches))

    # update_doc_size(data_json["meta"]["document_size"])
