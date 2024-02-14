import re
from nltk.stem import PorterStemmer
from xml.dom import minidom
from bs4 import BeautifulSoup
import json
import traceback
import os
import time
import threading
from collections import defaultdict
from typing import DefaultDict, Dict
from .common import read_file, read_xml_file, get_stop_words, remove_stop_words, tokenize, get_stemmed_words, replace_non_word_characters, get_preprocessed_words, save_json_file

CURRENT_DIR = os.getcwd()
NUM_OF_CORES = os.cpu_count()
XML_FILES = ["sample.xml", "trec.sample.xml", "trec.5000.xml"]

lock = threading.Lock()

def index_docs(
    docs_batches: minidom.Document,
    stopping: bool = True,
    stemming: bool = True,
    escape_char: bool = False,
    headline: bool = False,
) -> DefaultDict[str, Dict[str, list]]:
    local_index = defaultdict(dict)
    try:
        for doc in docs_batches:
            doc_id = (
                doc.find("docno").text
                if not escape_char
                else doc.find("docno").decode_contents()
            )
            doc_text = (
                doc.find("text").text
                if not escape_char
                else doc.find("text").decode_contents()
            )

            text_words = get_preprocessed_words(doc_text, stopping, stemming)
            if headline:
                headline = (
                    doc.find("headline").text
                    if not escape_char
                    else doc.find("headline").decode_contents()
                )
                headline_words = get_preprocessed_words(headline, stopping, stemming)
                text_words = headline_words + text_words

            for position, word in enumerate(text_words):
                if doc_id not in local_index[word]:
                    local_index[word][doc_id] = []
                local_index[word][doc_id].append(position + 1)
    except:
        print("Error processing doc_id", doc_id)
        traceback.print_exc()
        exit()

    return local_index


def process_batch(
    docs_batch: list,
    pos_inverted_index: DefaultDict[str, Dict[str, list]],
    stopping: bool = True,
    stemming: bool = True,
    escape_char: bool = False,
    headline: bool = False,
):
    local_index = index_docs(docs_batch, stopping, stemming, escape_char, headline)
    try:
        lock.acquire()
        for word in local_index:
            for doc_id in local_index[word]:
                if (
                    word not in pos_inverted_index
                    or doc_id not in pos_inverted_index[word]
                ):
                    pos_inverted_index[word][doc_id] = []
                pos_inverted_index[word][doc_id] += local_index[word][doc_id]
    except:
        print("Error processing batch")
        traceback.print_exc()
        exit()
    finally:
        lock.release()


def positional_inverted_index(
    file_name: str,
    stopping: bool = True,
    stemming: bool = True,
    escape_char: bool = False,
    headline: bool = True,
) -> dict:
    assert os.path.exists(
        os.path.join(CURRENT_DIR, file_name)
    ), f"File {file_name} does not exist"
    xml_text = read_file(file_name)
    doc_ids_set = set()
    soup = BeautifulSoup(xml_text, "html.parser")
    docs = soup.find_all("doc")
    doc_nos = soup.find_all("docno")
    for doc_no in doc_nos:
        doc_ids_set.add(doc_no.text)
    document_size = len(docs)
    batch_size = document_size // NUM_OF_CORES
    remainder = document_size % NUM_OF_CORES
    pos_inverted_index = defaultdict(dict)
    pos_inverted_index["document_size"]["0"] = document_size
    pos_inverted_index["doc_ids_list"] = list(doc_ids_set)

    batches = [docs[i * batch_size : (i + 1) * batch_size] for i in range(NUM_OF_CORES)]
    if remainder != 0:
        # append the remainder to the last batch
        batches[-1] += docs[-remainder:]

    threads = []
    for batch in batches:
        thread = threading.Thread(
            target=process_batch,
            args=(batch, pos_inverted_index, stopping, stemming, escape_char, headline),
        )
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    return pos_inverted_index


# save as binary file
def save_index_file(
    file_name: str, index: DefaultDict[str, Dict[str, list]], output_dir: str = "binary_file"
):
    if not os.path.exists(os.path.join(CURRENT_DIR, output_dir)):
        os.mkdir(os.path.join(CURRENT_DIR, output_dir))
    # sort index by term and doc_id in int
    index_output = dict(sorted(index.items()))
    for term, record in index_output.items():
        if term == "document_size" or term == "doc_ids_list":
            continue
        index_output[term] = dict(sorted(record.items(), key=lambda x: int(x[0])))

    with open(os.path.join(CURRENT_DIR, output_dir, file_name), "wb") as f:
        for term, record in index_output.items():
            if term == "document_size" or term == "doc_ids_list":
                continue
            f.write(f"{term} {len(record)}\n".encode("utf8"))
            for doc_id, positions in record.items():
                f.write(
                    f"\t{doc_id}: {','.join([str(pos) for pos in positions])}\n".encode(
                        "utf8"
                    )
                )


def load_binary_index(file_name: str, output_dir: str = "binary_file") -> dict:
    with open(os.path.join(CURRENT_DIR, output_dir, file_name), "rb") as f:
        data = f.read().decode("utf8")
    return json.loads(data)


def add_two_inverted_indexes(index_old, index_new):
    index_being_updated = index_old

    # add words in index_new_small not in index_big_old
    keys_unique_to_index_new_small = set(index_old.keys()) - set(index_new.keys())

    for key in keys_unique_to_index_new_small:
        index_being_updated[key] = index_new[key]

    # Now, we have to check for the words that are in both indexes to
    # update the posting list for the same word in both indexes
    keys_in_both_indexes = {
        key
        for key in index_old.keys() & index_new.keys()
        if index_old[key]
        and index_new[key]
        and key != "document_size"
        and key != "doc_ids_list"
    }

    # the docID must be new!
    for key in keys_in_both_indexes:
        for doc_id in index_new[key]:
            if doc_id not in index_old[key]:
                try:
                    index_being_updated[key][str(doc_id)] = index_new[key][str(doc_id)]
                    print(f"Added {key} for doc_id {doc_id}")
                except:
                    print(f"Error updating {key} for doc_id {doc_id}")
            elif doc_id in index_old[key]:
                print(
                    "WARNING: Trying to add new documents under the same doc ID!",
                    key,
                    doc_id,
                )

    return index_being_updated



def delta_encode(positions):
    """Convert a list of positions into a delta-encoded list."""
    if not positions:
        return []
    # The first position remains the same, others are differences from the previous one
    delta_encoded = [positions[0]] + [positions[i] - positions[i-1] for i in range(1, len(positions))]
    return delta_encoded

def save_delta_index_file(file_name: str, index: DefaultDict[str, Dict[str, list]], output_dir: str = "binary_file"):
    if not os.path.exists(os.path.join(CURRENT_DIR, output_dir)):
        os.mkdir(os.path.join(CURRENT_DIR, output_dir))
    index_output = dict(sorted(index.items()))
    with open(os.path.join(CURRENT_DIR, output_dir, file_name), "wb") as f:
        for term, record in index_output.items():
            if term == "document_size" or term == "doc_ids_list":
                continue
            record = dict(sorted(record.items(), key=lambda x: int(x[0])))
            f.write(f"{term} {len(record)}\n".encode("utf8"))
            for doc_id, positions in record.items():
                # Apply delta encoding here
                delta_positions = delta_encode(positions)
                # Convert delta-encoded positions back to strings for storage
                positions_str = ','.join(str(pos) for pos in delta_positions)
                f.write(f"\t{doc_id}: {positions_str}\n".encode("utf8"))

def delta_decode(delta_encoded):
    """Reconstruct the original list of positions from a delta-encoded list."""
    positions = [delta_encoded[0]] if delta_encoded else []
    for delta in delta_encoded[1:]:
        positions.append(positions[-1] + delta)
    return positions

def decode_positions(data):
    """Recursively decode delta-encoded position lists in the index data."""
    if isinstance(data, dict):
        return {key: decode_positions(value) for key, value in data.items()}
    elif isinstance(data, list) and all(isinstance(x, int) for x in data):
        # Assuming the list is of integers, decode it if it's delta-encoded
        return delta_decode(data)
    else:
        return data

def load_delta_encoded_index(file_name: str, output_dir: str = "binary_file") -> dict:
    path = os.path.join(CURRENT_DIR, output_dir, file_name)
    with open(path, "rb") as f:
        data = json.loads(f.read().decode("utf8"))
    
    # Apply delta decoding to the loaded data
    index = decode_positions(data)
    return index