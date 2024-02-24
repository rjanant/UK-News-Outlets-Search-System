import re
import orjson
import os
import pandas as pd
from nltk.stem import PorterStemmer
from xml.dom import minidom
from typing import List
from datetime import date
from constant import DATA_PATH, Source
from basetype import NewsArticlesFragment, NewsArticleData, NewsArticlesBatch
import numpy as np

STOP_WORDS_FILE = "ttds_2023_english_stop_words.txt"
CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))

def read_file(file_name: str, input_dir: str = "result") -> str:
    with open(os.path.join(CURRENT_DIR, input_dir, file_name), "r") as f:
        content = f.read()
    return content

def read_xml_file(file_name: str) -> minidom.Document:
    file = minidom.parse(file_name)
    return file

def get_stop_words(file_name: str = STOP_WORDS_FILE) -> list:
    assert os.path.exists(os.path.join(CURRENT_DIR, file_name)), f"File {file_name} does not exist"
    with open(os.path.join(CURRENT_DIR, file_name), "r") as f:
        stop_words = f.read()
    return stop_words.split("\n")

def remove_stop_words(tokens: list) -> list:
    assert os.path.exists(os.path.join(CURRENT_DIR, STOP_WORDS_FILE)), f"File {STOP_WORDS_FILE} does not exist"
    stop_words = get_stop_words(STOP_WORDS_FILE)
    return [token for token in tokens if token not in stop_words]

def tokenize(content: str) -> list:
    return re.findall(r"\w+", content)

def get_stemmed_words(tokens: list) -> list:
    # stemming
    stemmer = PorterStemmer()
    words = [stemmer.stem(token) for token in tokens]
    return words

def replace_non_word_characters(content: str) -> str:
    # replace non word characters with space
    return re.sub(r"[^\w\s]", " ", content)

def get_preprocessed_words(content: str, stopping: bool = True, stemming: bool = True) -> list:
    tokens = tokenize(content)
    tokens = [token.lower() for token in tokens]
    if stopping:
        tokens = remove_stop_words(tokens)
    if stemming:
        tokens = get_stemmed_words(tokens)
    return tokens

def save_json_file(file_name: str, data: dict, output_dir: str = "result"):
    
    if not os.path.exists(os.path.join(CURRENT_DIR, output_dir)):
        os.mkdir(os.path.join(CURRENT_DIR, output_dir))
    with open(os.path.join(CURRENT_DIR, output_dir, file_name), "wb") as f:
        f.write(orjson.dumps(data))
        
def load_json_file(file_name: str, input_dir: str = "result") -> dict:
    with open(os.path.join(CURRENT_DIR, input_dir, file_name), "rb") as f:
        data = f.read()
    return orjson.loads(data)
        
def get_indices_for_news_data(
    source_name: str,
    date: date,
    ) -> List[int]:
    # file name format: {source_name}_data_{date}_{number}.csv
    # date format: YYYYMMDD
    time_str = date.strftime("%Y%m%d")
    pattern = re.compile(f"{source_name}_data_{time_str}_([0-9]+).csv")
    
    data_path = os.path.join(DATA_PATH, source_name)
    print(data_path)
    assert os.path.exists(data_path), f"{data_path} does not exist"
    
    file_name_list = os.listdir(data_path)
    indices = []
    for file_name in file_name_list:
        match = pattern.match(file_name)
        if match:
            indices.append(int(match.group(1)))
    return sorted(indices)

def load_csv_from_news_source(
    source: Source,
    date: date,
    start_index: int = 0,
    end_index: int = -1,
    start_doc_id: int = 0,
    ) -> NewsArticlesBatch:
    
    indices = get_indices_for_news_data(source.value, date)
    assert start_index in indices, f"{start_index} is not in the indices list"
    
    end_index = len(indices) - 1 if end_index == -1 else end_index
    if end_index < start_index or end_index > len(indices):
        raise ValueError(f"Invalid end_index: {end_index}")
        
    
    indices = indices[indices.index(start_index) : indices.index(end_index) + 1]
    news_fragment_list = []
    current_doc_id = start_doc_id
    for index in indices:
        print(f"\r{' '*100}\rLoading {source.value} data {date.strftime('%Y%m%d')}_{index}.csv", end="")
        filename = f"{source.value}_data_{date.strftime('%Y%m%d')}_{index}.csv"
        filepath = os.path.join(DATA_PATH, source.value, filename)
        df = pd.read_csv(filepath)
        df.fillna("", inplace=True)
        # convert the DataFrame to a list of dictionaries and change keys to lowercase
        news_article_list = []
        for row in df.itertuples(index=True):
            news_article = {k.lower(): v for k, v in row._asdict().items()}
            news_article["doc_id"] = str(current_doc_id)
            current_doc_id += 1
            news_article_list.append(news_article)
        # convert the list of dictionaries to NewsArticleData objects
        news_article_list = [NewsArticleData.model_validate_json(orjson.dumps(article)) for article in news_article_list]
            
        current_news_fragment = NewsArticlesFragment(
            source=source.value,
            date=date,
            index=index,
            articles=news_article_list
        )
        
        news_fragment_list.append(current_news_fragment)
    
    # add an new line
    print()
        
    return NewsArticlesBatch(
        doc_ids=np.arange(current_doc_id).astype(str).tolist(),
        source_ids_map={source.value : np.arange(current_doc_id).astype(str).tolist()},
        indices={source.value : np.array(indices).astype(str).tolist()},
        fragments={source.value : news_fragment_list}
    )
    

def get_sources(datapath: str) -> List[str]:
    # check if the file exists
    if not os.path.exists(datapath):
        raise FileNotFoundError(f"{datapath} does not exist")
    
    # get the file list of current directory
    file_list = os.listdir(datapath)
    return file_list

if __name__ == "__main__":
    data = load_csv_from_news_source(Source.BBC, date(2024, 2, 17), 300)