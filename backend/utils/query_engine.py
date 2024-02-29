import re
import traceback
import os
import time
import math
import asyncio
import sys
sys.path.append(os.path.dirname(__file__))
from nltk.stem import PorterStemmer
from typing import DefaultDict, Dict, List, Tuple
from common import read_file, get_stop_words, get_preprocessed_words
from redis_utils import get_doc_size, get_doc_ids_list, get_json_values, is_key_exists, get_json_value
from build_index import delta_decode_list
from basetype import RedisKeys

STOP_WORDS_FILE = "ttds_2023_english_stop_words.txt"
CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
NUM_OF_CORES = os.cpu_count()
SPECIAL_PATTERN = {
    'proximity': re.compile(r"#(\d+)\((\w+),\s*(\w+)\)"),
    'exact': re.compile(r"\"[^\"]+\""),
    'spliter': re.compile(r"(AND|OR|NOT|#\d+\(\w+,\s*\w+\)|\"[^\"]+\"|\'[^\']+\'|\w+|\(|\))")
}

def preprocess_match(match: re.Match, stopping: bool = True, stemming: bool = True) -> str:
    word = match.group(0)
    if word in ["AND", "OR", "NOT"]:
        return word
    word = word.lower()
    
    stopwords = get_stop_words()
    if stopping and word in stopwords:
        return ""
    
    if stemming:
        stemmer = PorterStemmer()
        word = stemmer.stem(word)

    return word

def load_queries(file_name: str) -> list:
    query_lines = read_file(file_name).split("\n")
    queries = []
    for line in query_lines:
        if line == "":
            continue
        query_id, query_text = line.split(":")
        queries.append(query_text.strip())
    return queries

def handle_binary_operator(operator: str, left: list, right: list) -> list:
    # print("handle binary operator", operator, left, right)
    left = [] if left is None else left
    right = [] if right is None else right
    if operator == "AND":
        print("AND operation")
        return list(set(left) & set(right))
    elif operator == "OR":
        print("OR operation")
        return list(set(left) | set(right))

def handle_not_operator(operand: list, doc_ids_list: list) -> list:
    print("NOT operation")
    if operand is None:
        return doc_ids_list
    return list(set(doc_ids_list) - set(operand))

async def get_doc_ids_from_string(string: str) -> List[int]:
    # check if string is a phrase bounded by double quotes 
    if await is_key_exists(RedisKeys.index(string)):
        term_index = await get_json_value(RedisKeys.index(string))
        return list(term_index.keys())
    else:
        return []

async def get_doc_ids_from_pattern(pattern: str) -> List[int]:
    # pattern is of the form "A B"/"A B C" etc
    # retrieve words from the pattern
    doc_ids = []
    words = re.findall(r"\w+", pattern)
    # check if the word are in consecutive positions
    words_index = await get_json_values([RedisKeys.index(word) for word in words])
    for doc_id in words_index[0]:
        positions = delta_decode_list(words_index[0][doc_id])
        for pos in positions:
            try:
                if all([pos + i in words_index[i][doc_id] for i in range(1, len(words))]) and doc_id not in doc_ids:
                    doc_ids.append(doc_id)
            except:
                pass

    return doc_ids


def negate_doc_ids(doc_ids: list, doc_ids_list: list) -> list:
    return list(set(doc_ids_list) - set(doc_ids))

async def evaluate_proximity_pattern(n: int, w1: str, w2: str, doc_ids_list: list) -> List[int]:
    # find all the doc_ids for w1 and w2
    doc_ids_for_w1 = await get_doc_ids_from_string(w1)
    # find the doc_ids that satisfy the condition
    doc_ids = []
    for doc_id in doc_ids_for_w1:
        try:
            values = await get_json_values([RedisKeys.index(w1), RedisKeys.index(w2)])
            positions_for_w1 = delta_decode_list(values[0][doc_id])
            positions_for_w2 = delta_decode_list(values[1][doc_id])
            if any([abs(pos1 - pos2) <= int(n) for pos1 in positions_for_w1 for pos2 in positions_for_w2]):
                doc_ids.append(doc_id)
        except:
            pass
    return doc_ids

async def evaluate_subquery(subquery: str, doc_ids_list: List[int], special_patterns: dict[str, re.Pattern]) -> List[int]:
    
    proximity_match = re.match(special_patterns['proximity'], subquery)
    exact_match = re.match(special_patterns['exact'], subquery)
    if proximity_match:
        n = proximity_match.group(1)
        w1 = proximity_match.group(2)
        w2 = proximity_match.group(3)
        print("Handle proximity pattern", n, w1, w2)
        return await evaluate_proximity_pattern(n, w1, w2, doc_ids_list)
    else:
        # there is no NOT operator
        if exact_match:
            print("handle phrase", subquery[1:-1])
            return await get_doc_ids_from_pattern(subquery[1:-1])
        else:
            print("handle word(s)", subquery)
            return await get_doc_ids_from_string(subquery)

def read_boolean_queries(file_name: str) -> list:
    queries = []
    with open(os.path.join(CURRENT_DIR, file_name), "r") as f:
        for line in f.readlines():
            # split the query by the first space
            query_id, query = line.split(" ", 1)
            queries.append((query_id, query.strip()))
    return queries

def read_ranked_queries(file_name: str) -> list:
    queries = []
    with open(os.path.join(CURRENT_DIR, file_name), "r") as f:
        for line in f.readlines():
            # split the query by the first space
            query_id, query = line.split(" ", 1)
            queries.append((query_id, query.strip()))
        
    return queries

async def calculate_tf_idf(tokens: List, doc_id: str, docs_size: int) -> float:
    tf_idf_score = 0
    for token in tokens:
        # if token not in inverted_index or doc_id not in inverted_index[token]:
        #     continue
        if not await is_key_exists(RedisKeys.index(token)):
            continue
        
        token_index = await get_json_value(RedisKeys.index(token))
        if doc_id not in token_index:
            continue
        
        tf = 1 + math.log10(len(token_index[doc_id]))
        idf = math.log10(docs_size / len(token_index))
        tf_idf_score += tf * idf
    return tf_idf_score


# convert infix to postfix
def precedence(operator: str) -> int:
    if operator == "NOT":
        return 3
    elif operator == "AND" or operator == "OR":
        return 2
    elif operator == "(" or operator == ")":
        return 1
    else:
        return -1

def associativity(operator: str) -> str:
    if operator == "NOT":
        return "right"
    else:
        return "left"

def is_operator(token: str) -> bool:
    return token in ["AND", "OR", "NOT"]

def infix_to_postfix(query: str, spliter: re.Pattern) -> list:
    tokens = re.findall(spliter, query)
    stack = []
    postfix = []
    for token in tokens:
        if is_operator(token):
            while stack and is_operator(stack[-1]) and \
                ((associativity(token) == "left" and precedence(token) <= precedence(stack[-1])) or \
                (associativity(token) == "right" and precedence(token) < precedence(stack[-1]))):
                postfix.append(stack.pop())
            stack.append(token)
        elif token == '(':
            stack.append(token)
        elif token == ')':
            while stack and stack[-1] != '(':
                postfix.append(stack.pop())
            if stack and stack[-1] == '(':
                stack.pop()
        else:
            postfix.append(token)
    while stack:
        postfix.append(stack.pop())
    return postfix

def is_valid_query(query: str) -> bool:
    # check if the query is valid
    spliter = re.compile(r"(AND|OR|NOT|#\d+\(\w+,\s*\w+\)|\"[^\"]+\"|\'[^\']+\'|\w+|\(|\))")
    tokens = re.findall(spliter, query)
    prev_token = None
    parentheses_count = 0
    for token in tokens:
        if token == '(':
            parentheses_count += 1
        elif token == ')':
            parentheses_count -= 1
            if parentheses_count < 0:
                print("Parentheses count is less than 0")
                return False
        elif token == "NOT":
            if prev_token and (not is_operator(prev_token) and prev_token != '('):
                print("Invalid NOT position")
                return False
        elif is_operator(token):
            if prev_token and (prev_token == '(' or is_operator(prev_token)):
                print("Invalid operator position")
                return False
        else:
            # token is an operand
            if prev_token and prev_token == ')':
                print("Invalid operand position")
                return False
        prev_token = token
    if parentheses_count != 0:
        return False
    return True

async def evaluate_boolean_query(query: str,
                                 doc_ids_list: List[int],
                                 stopping: bool = True,
                                 stemming: bool = True,
                                 special_patterns: dict[str, re.Pattern] = SPECIAL_PATTERN) -> List:
    # query = " ".join([token.lower() if token not in ["AND", "OR", "NOT"] else token for token in query.split("\w+ ")])
    query = re.sub(r"(\w+)", lambda x: preprocess_match(x, stopping, stemming), query)
    print(query)
    if not is_valid_query(query):
        print("Invalid query: ", query)
        return []
    
    postfix = infix_to_postfix(query, special_patterns['spliter'])
    
    print("postfix", postfix)

    # evalute the value for the stuff first
    tasks = [evaluate_subquery(token, doc_ids_list, special_patterns) for token in postfix if not is_operator(token)]
    results = await asyncio.gather(*tasks)
    for idx, token in enumerate(postfix):
        if not is_operator(token):
            postfix[idx] = results.pop(0)
    print("postfix", postfix)
    
    try:
        stack = []
        for token in postfix:
            if is_operator(token):
                if token == "NOT":
                    right = stack.pop()
                    result = handle_not_operator(right, doc_ids_list)
                else:
                    right = stack.pop()
                    left = stack.pop()
                    result = handle_binary_operator(token, left, right)
                stack.append(result)
            else:
                # token is an operand
                stack.append(token)
        return stack.pop()

    except:
        # print the processing error term
        traceback.print_exc()
        exit()
        
async def evaluate_ranked_query(query: str, docs_size:int, stopping: bool = True, stemming:bool = True) -> List[Tuple[int, float]]:
    words = get_preprocessed_words(query, stopping, stemming)
    doc_ids = set()
    doc_ids_tasks = [get_json_value(RedisKeys.index(word)) for word in words if await is_key_exists(RedisKeys.index(word))]
    doc_ids_results = await asyncio.gather(*doc_ids_tasks)
    for result in doc_ids_results:
        doc_ids = doc_ids.union(result.keys())
    
    doc_ids = list(doc_ids)
    scores = []
    
    score_tasks = [calculate_tf_idf(words, doc_id, docs_size) for doc_id in doc_ids]
    score_results = await asyncio.gather(*score_tasks)
    for idx, doc_id in enumerate(doc_ids):
        scores.append((doc_id, score_results[idx]))
    # sort by the score and the doc_id
    scores.sort(key=lambda x: (-x[1], x[0]))

    return scores

async def boolean_test(boolean_queries: List[str] = ["\"Comic Relief\" AND (NOT wtf OR #1(Comic, Relief))"]) -> List[List[int]]:
    doc_ids_list = await get_doc_ids_list()
    start_time = time.time()
    results = []
    for query in boolean_queries:
        results.append(await evaluate_boolean_query(query, doc_ids_list))
    print("Time taken to process boolean queries", time.time() - start_time, len(results))
    return results

async def ranked_test(ranked_queries: List[str] = ["Comic Relief"]) -> List[List[Tuple[int, float]]]:
    doc_size = await get_doc_size()
    start_time = time.time()
    results = []
    for query in ranked_queries:
        results.append(await evaluate_ranked_query(query, doc_size))
    print("Time taken to process ranked queries", time.time() - start_time, len(results[0]))
    return results
    
async def main():
    await ranked_test()

if __name__ == "__main__":
    asyncio.run(main())
    
    # # ### processing ranked queries
    # ranked_queries = read_ranked_queries("queries.ranked.txt")
    # start = time.time()
    # ranked_results = evaluate_ranked_query(ranked_queries, index)
    # save_ranked_queries_result(ranked_results, '')
    # print("Time taken to process ranked queries", time.time() - start)