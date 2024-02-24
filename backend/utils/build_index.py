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
from typing import DefaultDict, Dict, List
from common import read_file, get_preprocessed_words, load_csv_from_news_source, save_json_file, load_json_file
from basetype import InvertedIndex, InvertedIndexMetadata, NewsArticleData, NewsArticlesFragment, NewsArticlesBatch, default_dict_list
from constant import Source
from datetime import date
from concurrent.futures import ThreadPoolExecutor

CURRENT_DIR = os.getcwd()
NUM_OF_CORES = os.cpu_count() or 1

lock = threading.Lock()

def process_batch(
    fragment_list: List[NewsArticlesFragment],
    inverted_index: InvertedIndex,
    stopping: bool = True,
    stemming: bool = True,
) -> None:
    local_index = defaultdict(dict)
    for fragment in fragment_list:
        for article in fragment.articles:
            doc_id = article.doc_id
            doc_text = article.title + "\n" + article.content
            text_words = get_preprocessed_words(doc_text, stopping, stemming)
            for position, word in enumerate(text_words):
                if doc_id not in local_index[word]:
                    local_index[word][doc_id] = []
                local_index[word][doc_id].append(position + 1)
    try:
        lock.acquire()
        for word in local_index:
            for doc_id in local_index[word]:
                if (
                    word not in inverted_index.index
                    or doc_id not in inverted_index.index[word]
                ):
                    inverted_index.index[word][doc_id] = []
                inverted_index.index[word][doc_id] += local_index[word][doc_id]
    except:
        print("Error processing batch")
        traceback.print_exc()
        exit()
    finally:
        lock.release()

def positional_inverted_index(
    news_batch: NewsArticlesBatch,
    stopping: bool = True,
    stemming: bool = True,
) -> InvertedIndex:
    doc_ids = news_batch.doc_ids
    document_size = len(doc_ids)
    source_ids_map = news_batch.source_ids_map
    inverted_index_meta = InvertedIndexMetadata(
        document_size=document_size, 
        doc_ids_list=doc_ids,
        source_doc_ids=source_ids_map)
    
    inverted_index = InvertedIndex(
        meta=inverted_index_meta,
        index=defaultdict(default_dict_list)
    )

    # cut the fragments into batches
    for source, fragments in news_batch.fragments.items():
        curr_time = time.time()
        batch_size = len(fragments) // NUM_OF_CORES
        remainder = len(fragments) % NUM_OF_CORES
        batches = [fragments[i * batch_size : (i + 1) * batch_size] for i in range(NUM_OF_CORES)]
        if remainder != 0:
            # append the remainder to the last batch
            batches[-1] += fragments[-remainder:]
        
        threads = []
        
        with ThreadPoolExecutor(max_workers=NUM_OF_CORES) as executor:
            futures = [executor.submit(process_batch, batch, inverted_index, stopping, stemming) for batch in batches]
        
        for future in futures:
            try:
                future.result()
            except Exception as e:
                print(f"Error processing batch: {e}")
                traceback.print_exc()
                exit()
        
        print(f"Time taken for processing {source}: {time.time() - curr_time:.2f} seconds")
        
    return inverted_index


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

if __name__ == "__main__":
    news_batch = load_csv_from_news_source(Source.BBC, date(2024, 2, 17), 300)
    inverted_index = positional_inverted_index(news_batch)
    save_json_file("inverted_index.json", inverted_index.model_dump())