import os
import sys
import asyncio
from redis_utils import get_val, get_doc_size, get_doc_ids_list

if __name__ == "__main__":
    # Ask on discord to get the hardcoded config
    word = "men"

    # index_result = get_val(word)
    document_size = asyncio.run(get_doc_size())
    # document_size = get_doc_size()
    # doc_ids_list = get_doc_ids_list()

    # print(index_result)
    print(document_size)
    # print(len(doc_ids_list))
