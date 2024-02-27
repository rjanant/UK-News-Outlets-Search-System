import orjson
import os
import sys
import asyncio

BASEPATH = os.path.dirname(__file__)
sys.path.append(BASEPATH)

from redis_utils import get_redis_config, update_doc_size, batch_push


def load_index(path_index="result/inverted_index.json"):
    with open(path_index, "r+") as f:
        data_json = orjson.loads(f.read())
    return data_json


def process_dict_in_batches(input_dict, batch_size, prefix="w:"):
    keys = list(input_dict.keys())
    num_keys = len(keys)
    batches = []
    for i in range(0, num_keys, batch_size):
        batch_keys = keys[i : i + batch_size]
        batch = {prefix + key: str(input_dict[key]) for key in batch_keys}
        batches.append(batch)
    return batches


if __name__ == "__main__":
    minibatch = 15
    filepath = "result/inverted_index.json"

    # Ask on discord to get the hardcoded config
    data_json = load_index(path_index=filepath)
    batches = process_dict_in_batches(data_json["index"], minibatch)

    asyncio.run(batch_push(batches))

    update_doc_size(data_json["meta"]["document_size"])
