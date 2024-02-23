import re
from nltk.stem import PorterStemmer
import traceback
import os
import time
from collections import defaultdict
import math
from typing import DefaultDict, Dict
from basetype import InvertedIndex
from common import read_file, get_stop_words, save_json_file, get_preprocessed_words

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
    return list(set(doc_ids_list) - set(operand))

def get_doc_ids_from_string(string: str, inverted_index: Dict, doc_ids_list: list, negate: bool = False) -> list:
    # check if string is a phrase bounded by double quotes 
    if string in inverted_index:
        if negate:
            return negate_doc_ids(list(inverted_index[string].keys()), doc_ids_list)
        else:
            return list(inverted_index[string].keys()) if inverted_index[string] else []

def get_doc_ids_from_pattern(pattern: str, inverted_index: Dict, doc_ids_list: list, negate: bool = False):
    # pattern is of the form "A B"/"A B C" etc
    # retrieve words from the pattern
    doc_ids = []
    words = re.findall(r"\w+", pattern)
    # check if the word are in consecutive positions
    for doc_id in inverted_index[words[0]]:
        positions = inverted_index[words[0]][doc_id]
        for pos in positions:
            try:
                if all([pos + i in inverted_index[words[i]][doc_id] for i in range(1, len(words))]) and doc_id not in doc_ids:
                    doc_ids.append(doc_id)
            except:
                pass
    if negate:
        return negate_doc_ids(doc_ids, doc_ids_list)
    else:
        return doc_ids


def negate_doc_ids(doc_ids: list, doc_ids_list: list) -> list:
    return list(set(doc_ids_list) - set(doc_ids))

def evaluate_proximity_pattern(n: int, w1: str, w2: str, doc_ids_list: list, inverted_index: dict) -> list:
    # find all the doc_ids for w1 and w2
    doc_ids_for_w1 = get_doc_ids_from_string(w1, inverted_index, doc_ids_list)
    # find the doc_ids that satisfy the condition
    doc_ids = []
    for doc_id in doc_ids_for_w1:
        try:
            positions_for_w1 = inverted_index[w1][doc_id]
            positions_for_w2 = inverted_index[w2][doc_id]
            if any([abs(pos1 - pos2) <= int(n) for pos1 in positions_for_w1 for pos2 in positions_for_w2]):
                doc_ids.append(doc_id)
        except:
            pass
    return doc_ids

def evaluate_subquery(subquery: str, inverted_index: dict, doc_ids_list: list, special_patterns: dict[str, re.Pattern]) -> list:
    
    proximity_match = re.match(special_patterns['proximity'], subquery)
    exact_match = re.match(special_patterns['exact'], subquery)
    print("subquery", subquery)
    if proximity_match:
        n = proximity_match.group(1)
        w1 = proximity_match.group(2)
        w2 = proximity_match.group(3)
        print("Handle proximity pattern", n, w1, w2)
        return evaluate_proximity_pattern(n, w1, w2, doc_ids_list, inverted_index)
    else:
        # there is no NOT operator
        if exact_match:
            print("handle phrase", subquery[1:-1])
            return get_doc_ids_from_pattern(subquery[1:-1], inverted_index, doc_ids_list)
        else:
            print("handle word(s)", subquery)
            return get_doc_ids_from_string(subquery, inverted_index, doc_ids_list)

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

def calculate_tf_idf(inverted_index: dict, tokens: list, doc_id: str, docs_size: int) -> float:
    tf_idf_score = 0
    for token in tokens:
        if token not in inverted_index or doc_id not in inverted_index[token]:
            continue
        tf = 1 + math.log10(len(inverted_index[token][doc_id]))
        idf = math.log10(docs_size / len(inverted_index[token]))
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
            if prev_token and (not is_operator(prev_token) or prev_token == '('):
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

def evaluate_boolean_query(query: str, inverted_index: dict, doc_ids_list: list, stopping: bool = True, stemming: bool = True, special_patterns: dict[str, re.Pattern] = SPECIAL_PATTERN) -> list:
    query = re.sub(r"(\w+)", lambda x: preprocess_match(x, stopping, stemming), query)
    query = " ".join([token.lower() if token not in ["AND", "OR", "NOT"] else token for token in query.split(" ")])
    if not is_valid_query(query):
        print("Invalid query: ", query)
        return []
    postfix = infix_to_postfix(query, special_patterns['spliter'])
    
    for token in postfix:
        token = re.sub(r"(\w+)", lambda x: preprocess_match(x, stopping, stemming), token)
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
                result = evaluate_subquery(token, inverted_index, doc_ids_list, special_patterns)
                stack.append(result)
        return stack.pop()

    except:
        # print the processing error term
        traceback.print_exc()
        exit()
        
def evaluate_ranked_query(queries: list, index: DefaultDict, max_result: int = 150, stopping: bool = True, stemming:bool = True) -> list:
    results = []
    docs_size = int(index['document_size']['0'])
    for query_id, query in queries:
        words = get_preprocessed_words(query, stopping, stemming)
        print(words)
        doc_ids = set()
        for word in words:
            if word in index:
                doc_ids = doc_ids.union(set(index[word].keys()))
        
        doc_ids = list(doc_ids)
        scores = []
        for doc_id in doc_ids:
            scores.append((doc_id, calculate_tf_idf(index, words, doc_id, docs_size)))
        # sort by the score and the doc_id
        scores.sort(key=lambda x: (-x[1], x[0]))
        results.append((query_id, scores[:max_result]))
    return results

def save_boolean_queries_result(results: list, output_dir: str = "result"):
    if not os.path.exists(os.path.join(CURRENT_DIR, output_dir)):
        os.mkdir(os.path.join(CURRENT_DIR, output_dir))
    with open(os.path.join(CURRENT_DIR, output_dir, 'results.boolean.txt'), "w") as f:
        for query_id, result in results:
            for doc_id in result:
                f.write(f"{query_id},{doc_id}\n")
                
def save_ranked_queries_result(results: list, output_dir: str = "result"):
    if not os.path.exists(os.path.join(CURRENT_DIR, output_dir)):
        os.mkdir(os.path.join(CURRENT_DIR, output_dir))
    with open(os.path.join(CURRENT_DIR, output_dir, 'results.ranked.txt'), "w") as f:
        for query_id, result in results:
            for retrieved_doc_result in result:
                doc_id, score = retrieved_doc_result
                f.write(f"{query_id},{doc_id},{score:.4f}\n")

if __name__ == "__main__":
    ### loading index
    start_time = time.time()
    inverted_index_str = read_file("inverted_index.json")
    inverted_index = InvertedIndex.model_validate_json(inverted_index_str)
    print("Time taken to load index", time.time() - start_time)
    
    queries = ["\"Comic Relief\""]
    doc_ids_list = inverted_index.meta.doc_ids_list
    start_time = time.time()
    for query in queries:
        print(evaluate_boolean_query(query, inverted_index.index, doc_ids_list))
    print("Time taken to process boolean queries", time.time() - start_time)
    
    # # ### processing ranked queries
    # ranked_queries = read_ranked_queries("queries.ranked.txt")
    # start = time.time()
    # ranked_results = evaluate_ranked_query(ranked_queries, index)
    # save_ranked_queries_result(ranked_results, '')
    # print("Time taken to process ranked queries", time.time() - start)