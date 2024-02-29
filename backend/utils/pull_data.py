
import asyncio
from redis_utils import get_doc_data, get_doc_key
from basetype import RedisDocKeys


if __name__ == "__main__":
    """To demonstrate collecting the data"""

    doc_id = 123

    # Pull data from doc id
    data_ = asyncio.run(get_doc_data(doc_id=doc_id))
    print(data_)


    # Pull url from doc id
    url_ = asyncio.run(get_doc_key(doc_id=doc_id, key=RedisDocKeys.url))
    print(url_)

    # Pull url from doc id
    title_ = asyncio.run(get_doc_key(doc_id=doc_id, key=RedisDocKeys.title))
    print(title_)