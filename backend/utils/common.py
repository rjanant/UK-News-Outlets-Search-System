import re
from nltk.stem import PorterStemmer
from xml.dom import minidom
from bs4 import BeautifulSoup
import json
import os
from collections import defaultdict
from typing import DefaultDict, Dict


STOP_WORDS_FILE = "ttds_2023_english_stop_words.txt"
CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))

def read_file(file_name: str) -> str:
    with open(os.path.join(CURRENT_DIR, file_name), "r", encoding="utf8") as f:
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
        f.write(json.dumps(data).encode("utf8"))
        
