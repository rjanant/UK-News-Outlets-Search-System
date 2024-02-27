import os
import sys

BASEPATH = os.path.dirname(__file__)
sys.path.append(BASEPATH)



from redis_utils import get_redis_config, get_val, get_doc_size

if __name__ == "__main__":
    # Ask on discord to get the hardcoded config
    word = "w:men"

    index_result = get_val(word)
    document_size = get_doc_size()

    doc_ids_list = list(range(document_size))

    # print(index_result)
    print(document_size)
