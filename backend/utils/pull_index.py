import json
import os, sys
import asyncio

BASEPATH = os.path.dirname(__file__)
sys.path.append(BASEPATH)



from redis_utils import get_redis_config, get_redis, get_val, get_doc_size

if __name__ == "__main__":
    # Ask on discord to get the hardcoded config
    config_redis = get_redis_config("dev")

    r = get_redis(config_redis)

    word = "w:men"

    index_result = get_val(r, word)
    document_size = get_doc_size(r)

    doc_ids_list = list(range(document_size))

    # print(index_result)
    print(document_size)
