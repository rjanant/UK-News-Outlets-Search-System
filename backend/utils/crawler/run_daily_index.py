import pandas as pd
import os, sys
import asyncio

from datetime import datetime

FILENAME = os.path.basename(__file__)
BASEPATH = os.path.dirname(__file__)
UTILPATH = os.path.dirname(BASEPATH)

sys.path.append(UTILPATH)

from basetype import NewsArticlesFragment, NewsArticleData, NewsArticlesBatch
from build_index import positional_inverted_index, encode_index, save_json_file
from redis_utils import update_index, batch_push_news_data
from common import Logger


def load_data(filepath, date):
    news_fragment_list = []
    doc_ids = []

    df_all = pd.read_csv(filepath, chunksize=100)
    for df in df_all:
        df.fillna("", inplace=True)
        df["doc_id"] = df["doc_id"].astype(str)
        # convert the DataFrame to a list of dictionaries and change keys to lowercase
        news_article_list = []
        for row in df.itertuples(index=True):
            news_article = {k.lower(): v for k, v in row._asdict().items()}
            news_article_list.append(news_article)
            doc_ids.append(int(news_article["doc_id"]))

        # convert the list of dictionaries to NewsArticleData objects
        news_article_list = [
            NewsArticleData.model_validate(news_article)
            for news_article in news_article_list
        ]

        current_news_fragment = NewsArticlesFragment(
            source="daily", date=date, index=0, articles=news_article_list
        )

        news_fragment_list.append(current_news_fragment)

    return NewsArticlesBatch(
        doc_ids=sorted(doc_ids),
        indices={"daily": []},
        fragments={"daily": news_fragment_list},
    )


if __name__ == "__main__":
    logpath = os.path.join(UTILPATH, 'logger.log')
    logger = Logger(logpath)

    logger.log_event('info', f'{FILENAME} - Start script')

    today = datetime.now()
    today_str = today.strftime("%Y%m%d")
    today_str_dash = today.strftime("%Y-%m-%d")

    folder_path = os.path.join(UTILPATH, "data", today_str)

    inputfile = os.path.join(folder_path, f"data_{today_str}.csv")
    indexpath = f"data/{today_str}"
    indexname = 'index.json'

    logger.log_event('info', f'{FILENAME} - Loading Data')
    news_batch = load_data(inputfile, today_str_dash)

    logger.log_event('info', f'{FILENAME} - Creating Index')
    inverted_index = positional_inverted_index(news_batch)

    logger.log_event('info', f'{FILENAME} - Encoding')
    encode_index(inverted_index)

    logger.log_event('info', f'{FILENAME} - Saving JSON File')
    save_json_file(indexname, inverted_index.model_dump(), indexpath)

    logger.log_event('info', f'{FILENAME} - Pusing Index to Redis')
    asyncio.run(update_index(inverted_index))


    logger.log_event('info', f'{FILENAME} - Pusing Data to Redis')
    asyncio.run(batch_push_news_data(news_batch))

    logger.log_event('info', f'{FILENAME} - DONE')