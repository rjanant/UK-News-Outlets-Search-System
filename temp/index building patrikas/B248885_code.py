# -*- coding: utf-8 -*-
"""
Search Engine Script

Builds an inverted index from XML data, performs a Boolean search, and executes a ranked search based on TF-IDF scoring.

Functionalities:
----------------
- Building an Inverted Index:
    Creates and saves an inverted index using an XML file. The generated index will be saved as a pickled file and
    optionally can also be exported to CSV and text formats.
  
- Boolean Search:
    Reads a set of Boolean queries from a text file and performs Boolean search using an existing inverted index.
    The results are written to a text file.

- Ranked Search:
    Reads a set of queries from a text file and performs a ranked search using TF-IDF scoring on an existing
    inverted index. The results are written to a text file.

Example Usages:
---------------
1. Building the index:
    ```bash
    python code.py --mode build_index
    ```
2. Performing a Boolean search:
    ```bash
    python code.py --mode boolean_search --verbose True
    ```
3. Performing a Ranked Search:
    ```bash
    python code.py --mode ranked_search
    ```
4. Customizing parameters:
    ```bash
    python code.py --mode build_index --xml-path "custom.xml" --csv False
    ```

Parameters:
-----------
All options for each mode are provided as command line arguments. Run `python script_name.py --help`
for a full list of available options.

Notes:
------
- This script requires the following external Python libraries to be installed:
    - numpy (for TF-IDF scoring)
    - pandas (for CSV export)
    - nltk (for tokenization, stemming, and stopword removal)
    - tqdm (for progress bars)
    
- Standard Python libraries used include:
    - operator
    - pickle
    - re
    - time
    - xml.etree.ElementTree
    - collections (defaultdict, Counter)
    - argparse

- nltk may require additional downloads for stop words, tokenizers, and WordNet:
    ```python
    nltk.download('punkt')
    nltk.download('omw-1.4')
    nltk.download('wordnet')
    ```

Author: 
-------
Patrikas Vanagas
"""
import operator
import pickle
import re
import time
import xml.etree.ElementTree as ET
from collections import defaultdict, Counter
import argparse

import numpy as np
import pandas as pd
from nltk import word_tokenize, SnowballStemmer
from tqdm import tqdm

STEMMER = SnowballStemmer("english")
ENGLISH_STOPWORDS = open("english_stop_words.txt").read().split("\n")

# from https://github.com/dipanjanS/practical-machine-learning-with-python/blob/master/bonus%20content/nlp%20proven%20approach/contractions.py
"""
@author: DIP
"""

CONTRACTION_MAP = {
    "ain't": "is not",
    "aren't": "are not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'd've": "he would have",
    "he'll": "he will",
    "he'll've": "he he will have",
    "he's": "he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how is",
    "I'd": "I would",
    "I'd've": "I would have",
    "I'll": "I will",
    "I'll've": "I will have",
    "I'm": "I am",
    "I've": "I have",
    "i'd": "i would",
    "i'd've": "i would have",
    "i'll": "i will",
    "i'll've": "i will have",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it would",
    "it'd've": "it would have",
    "it'll": "it will",
    "it'll've": "it will have",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she would",
    "she'd've": "she would have",
    "she'll": "she will",
    "she'll've": "she will have",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so as",
    "that'd": "that would",
    "that'd've": "that would have",
    "that's": "that is",
    "there'd": "there would",
    "there'd've": "there would have",
    "there's": "there is",
    "they'd": "they would",
    "they'd've": "they would have",
    "they'll": "they will",
    "they'll've": "they will have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what'll've": "what will have",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "when's": "when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where is",
    "where've": "where have",
    "who'll": "who will",
    "who'll've": "who will have",
    "who's": "who is",
    "who've": "who have",
    "why's": "why is",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you would",
    "you'd've": "you would have",
    "you'll": "you will",
    "you'll've": "you will have",
    "you're": "you are",
    "you've": "you have",
}

# re.compile object to help to find document places with contractions
CONTRACTIONS_PATTERN = re.compile(
    "({})".format("|".join(CONTRACTION_MAP.keys())),
    flags=re.IGNORECASE | re.DOTALL,
)

def load_and_parse_XML(xml_path: str)->tuple:
    """
    Load and parse an XML file to extract documents and headlines.

    Parameters:
    -----------
    xml_path : str
        The file path to the XML file that needs to be loaded and parsed.

    Returns:
    --------
    documents_dict : dict
        A dictionary where the keys are document IDs (either integers or strings)
        and the values are the full text of each document (concatenation of headline and text).

    id_to_headline_dict : dict
        A dictionary mapping document IDs (either integers or strings) to their corresponding headlines.

    Notes:
    ------
    - The function uses the ElementTree (ET) library to parse the XML data.
    - Both the headline and text are stripped of leading/trailing whitespace and concatenated to form the full text.
    - In case the document ID cannot be converted to an integer, a warning is printed.

    Examples:
    ---------
    >>> documents_dict, id_to_headline_dict = load_and_parse_XML("path/to/xml/file")
    """
    # Initialize an empty dictionary to store the documents
    documents_dict = {}
    # Initialize an empty dictionary to map integers to headlines
    id_to_headline_dict = {}

    # Load and parse the XML data from the file
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Loop through each DOC element in the XML tree
    for doc in root.findall("DOC"):
        # Get the DOCNO element's text, removing any extra whitespace
        doc_id = doc.find("DOCNO").text.strip()

        # Convert the DOCNO to an integer, if possible
        try:
            doc_id = int(doc_id)
        except ValueError:
            # Handle the case where DOCNO is not convertible to an integer
            print(f"Warning: Could not convert DOCNO '{doc_id}' to an integer.")
        # Extract the headline and text, and replace '\n' with a space
        headline = (
            doc.find("HEADLINE").text.strip().replace("\n", " ").replace("'", "'")
            if doc.find("HEADLINE") is not None
            else ""
        )
        text = (
            doc.find("TEXT").text.strip().replace("\n", " ").replace("'", "'")
            if doc.find("TEXT") is not None
            else ""
        )

        # Concatenate the headline and text, replacing '\n' with a space
        full_text = f"{headline} {text}"  # Used space to concatenate

        # Store the full text in the dictionary, using the doc_id as the key
        documents_dict[doc_id] = full_text
        # Map the doc_id to the headline
        id_to_headline_dict[doc_id] = headline
    return documents_dict, id_to_headline_dict


def expand_match(contraction) -> str:
    """
    Expand a specific matched contraction within a text string.

    Parameters:
    ----------
    contraction : re.Match object
        The re.Match object containing the contraction to be expanded.

    Returns:
    -------
    str
        The expanded version of the matched contraction.

    Notes:
    -----
    - The function looks up the contraction from a pre-defined map (CONTRACTION_MAP) to find its expanded form.
    - The case of the first letter is preserved in the expansion.

    Examples:
    --------
    >>> expand_match(re.match(r"\bI'm\b", "I'm fine"))
    "I am"
    """
    # re.Match object, group(0) is the match str
    match = contraction.group(0)
    # in case it's lowercase
    first_char = match[0]
    # find the matching key and value from the map
    expanded_contraction = (
        CONTRACTION_MAP.get(match)
        if CONTRACTION_MAP.get(match)
        else CONTRACTION_MAP.get(match.lower())
    )
    expanded_contraction = first_char + expanded_contraction[1:]
    return expanded_contraction


def expand_contractions(text: str) -> str:
    """
    Expand all contractions in a given text string.

    Parameters:
    ----------
    text : str
        The text string containing contractions to be expanded.

    Returns:
    -------
    str
        The text string with all contractions expanded and apostrophes removed.

    Notes:
    -----
    - This function internally calls `expand_match` for each contraction found.
    - All remaining apostrophes in the text are replaced with spaces.

    Examples:
    --------
    >>> expand_contractions("I'm fine, aren't I?")
    "I am fine, are not I "
    """
    expanded_text = CONTRACTIONS_PATTERN.sub(expand_match, text)
    expanded_text = re.sub("'", " ", expanded_text)
    return expanded_text


def get_processed_tokens_from_string(
    document: str, contraction_expansion: bool = True, stopword_removal: bool = True
) -> list:
    """
    Tokenize and preprocess a given text document.

    Parameters:
    ----------
    document : str
        The text document to be tokenized and preprocessed.
        
    contraction_expansion : bool, optional (default=True)
        Whether to expand contractions in the text document.
        
    stopword_removal : bool, optional (default=True)
        Whether to remove stopwords from the tokenized text.

    Returns:
    -------
    list
        A list of processed tokens from the document.

    Notes:
    -----
    - Tokenization is performed using the `word_tokenize` function from the nltk library.
    - Text is converted to lowercase during processing.
    - Non-alphabetic characters are removed from the document.
    - If enabled, contractions are expanded using the `expand_contractions` function.
    - If enabled, stopwords are removed using a predefined list (ENGLISH_STOPWORDS).
    - Stemming is performed on each token using the Porter stemmer.

    Examples:
    --------
    >>> get_processed_tokens_from_string("I'm going to the store.")
    ['go', 'store']

    >>> get_processed_tokens_from_string("I'm going to the store.", stopword_removal=False)
    ['i', 'am', 'go', 'to', 'the', 'store']
    """
    if contraction_expansion:
        document = expand_contractions(document)
    document = re.sub(r"[^a-zA-Z\s]", " ", document).replace("[", " ").replace("]", " ")
    tokens = word_tokenize(document)
    # to lowercase
    tokens = [w.lower() for w in tokens]  # must be list, not set
    # remove stopwords
    if stopword_removal:
        tokens = list(filter(lambda token: token not in ENGLISH_STOPWORDS, tokens))
    # finally stem with Porter stemmer, by each TOKEN
    tokens = [STEMMER.stem(term) for term in tokens]
    return tokens


def create_index_corpus(
    document_dictionary: dict,
    contraction_expansion: bool = True,
    stopword_removal: bool = True,
) -> dict:
    """
    Preprocesses the documents and constructs an inverted index corpus.

    Parameters:
    ----------
    document_dictionary : dict
        Dictionary of documents to index. The keys are identifiers (article docNOs)
        and the values are the corresponding text articles.

    contraction_expansion : bool, optional (default=True)
        Whether to expand contractions in the text during preprocessing.

    stopword_removal : bool, optional (default=True)
        Whether to remove stopwords from the text during preprocessing.

    Returns:
    -------
    dict
        An inverted index dictionary. Each key is a unique term, and the value is a dictionary
        containing the Document Frequency and a list of Postings (document IDs and term positions).

    Notes:
    -----
    - The inverted index is sorted first by term and then by document ID.
    - The Document Frequency for each term is calculated.
    - The construction time for the index is printed.

    Examples:
    --------
    >>> create_index_corpus({'doc1': 'apple orange', 'doc2': 'orange banana'})
    {'apple': {'Document Frequency': 1, 'Postings': {'doc1': [0]}},
     'banana': {'Document Frequency': 1, 'Postings': {'doc2': [1]}},
     'orange': {'Document Frequency': 2, 'Postings': {'doc1': [1], 'doc2': [0]}}}
    """
    print("\nPreprocessing documents:")
    index_construction_start = time.time()
    # preprocess each string of an article
    for article in tqdm(document_dictionary):
        document_dictionary[article] = get_processed_tokens_from_string(
            document_dictionary[article], contraction_expansion, stopword_removal
        )
    token_sequence = []
    # Initialize a set for each term:
    term_doc_set = defaultdict(set)  # Step 1: Initialize a set for each term

    # for each article string
    for article in document_dictionary:
        # get the index of the term and the term
        for index, term in enumerate(document_dictionary[article]):
            # append the term, the episode, and the location number of that term/multi-term in an article
            # to a list of such entries for the whole corpus
            term_article_tuple = (
                term,
                article,
                index,  # ...so we substract the number of them we had previously in the document...
            )
            token_sequence.append(term_article_tuple)
            term_doc_set[term].add(article)  # Add the docID to the set
    print("\nSorting index...")
    index_sorting_start = time.time()
    # Sort first by term (alphabetically), then by docID
    token_sequence = sorted(
        token_sequence,
        key=lambda x: (x[0], x[1]),
    )
    print(
        "Index sorting took {0:4.2f} seconds.".format(
            (time.time() - index_sorting_start)
        )
    )
    inverted_index_dictionary = defaultdict(
        lambda: {"Document Frequency": 0, "Postings": defaultdict(list)}
    )
    # Construct the inverted index
    for entry in token_sequence:
        term, doc_id, location = entry
        inverted_index_dictionary[term]["Postings"][doc_id].append(location)
    # Set the frequency to the size of the set of docIDs
    for term in inverted_index_dictionary.keys():
        inverted_index_dictionary[term]["Document Frequency"] = len(term_doc_set[term])
    print(
        "Final index has {0} terms and took {1:4.2f} seconds to construct.".format(
            len(inverted_index_dictionary.keys()),
            time.time() - index_construction_start,
        )
    )
    return inverted_index_dictionary


def output_inverted_index_to_txt(inverted_index: dict, output_path: str) -> None:
    """
    Outputs the inverted index data to a text file.

    Parameters:
    ----------
    inverted_index : dict
        The inverted index dictionary to be written to file.

    output_path : str
        The path where the text file will be saved.
    """
    with open(output_path, "w") as f:
        for term, term_data in inverted_index.items():
            # Write the term and its document frequency
            f.write(f"{term}:{term_data['Document Frequency']}\n")
            for doc_id, positions in term_data["Postings"].items():
                # Write the document ID and positions
                f.write(f"   {doc_id}: {','.join(map(str, positions))}\n")


def create_and_save_index(
    xml_path: str,
    contraction_expansion: bool = True,
    stopword_removal: bool = True,
    csv: bool = True,
    txt: bool = True,
) -> None:
    """
    Creates and saves an inverted index from XML data to multiple file formats.

    Parameters:
    ----------
    xml_path : str
        The path to the XML file containing the documents to index.

    contraction_expansion : bool, optional (default=True)
        Whether to expand contractions in the text during preprocessing.

    stopword_removal : bool, optional (default=True)
        Whether to remove stopwords from the text during preprocessing.

    csv : bool, optional (default=True)
        Whether to save the index to a CSV file.

    txt : bool, optional (default=True)
        Whether to save the index to a text file.

    Notes:
    -----
    - Index and additional dictionary to match doc IDs to headlines are saved as pickled files.
    """
    document_dictionary, id_to_headline_dict = load_and_parse_XML(xml_path=xml_path)
    inverted_index_dictionary = create_index_corpus(
        document_dictionary=document_dictionary,
        contraction_expansion=contraction_expansion,
        stopword_removal=stopword_removal,
    )
    with open("inverted_index.pkl", "wb") as f:
        pickle.dump(dict(inverted_index_dictionary), f)
        print("Inverted index saved to file 'inverted_index.pkl'.")
    with open("id_to_headline_dict.pkl", "wb") as f:
        pickle.dump(id_to_headline_dict, f)
        print("ID to headline dictionary saved to file 'id_to_headline_dict.pkl'.")
    if csv:
        df = pd.DataFrame.from_dict(inverted_index_dictionary, orient="index")
        df.to_csv("inverted_index.csv")
        print("Inverted index saved to file 'inverted_index.csv'.")
    if txt:
        output_inverted_index_to_txt(inverted_index_dictionary, "inverted_index.txt")
        print("Inverted index saved to file 'inverted_index.txt'.")


def load_pickled_data(
    inverted_index_path: str = "inverted_index.pkl",
    id_to_headline_path="id_to_headline_dict.pkl",
)->tuple:
    """
    Loads pickled inverted index and ID-to-headline dictionary data from disk.

    Parameters:
    ----------
    inverted_index_path : str, optional (default="inverted_index.pkl")
        Path to the pickled file containing the inverted index.

    id_to_headline_path : str, optional (default="id_to_headline_dict.pkl")
        Path to the pickled file containing the ID-to-headline dictionary.

    Returns:
    -------
    tuple
        A tuple containing the loaded inverted index dictionary and the ID-to-headline dictionary.
    """
    with open(inverted_index_path, "rb") as f:
        loaded_dict = pickle.load(f)
    inverted_index_dict = defaultdict(
        lambda: {"Document Frequency": 0, "Postings": defaultdict(list)}, loaded_dict
    )

    with open(id_to_headline_path, "rb") as f:
        id_to_headline_dict = pickle.load(f)
    return inverted_index_dict, id_to_headline_dict


def parse_unranked_query(query: str) -> list:
    """
    Parses a single unranked boolean query and identifies its type.

    Parameters:
    ----------
    query : str
        The query string to parse.

    Returns:
    -------
    list
        A list containing the type of the query and its components. For example:
        - ("AND", ["Edinburgh", "SCOTLAND"]) for "Edinburgh AND SCOTLAND"
        - ("PHRASE", "income taxes") for "\"income taxes\""
        - ("PROXIMITY", (["income", "taxes"], 20)) for "#20(income, taxes)"
        - ("TERM", "Happiness") for "Happiness"

    Notes:
    -----
    - Supported query types are: AND, OR, NOT, PHRASE, PROXIMITY, TERM
    """
    # Identify the type of query: AND, OR, NOT, phrase, or proximity
    if " AND NOT " in query:
        return "AND NOT", query.split(" AND NOT ")
    elif " OR NOT " in query:
        return "OR NOT", query.split(" OR NOT ")
    elif " AND " in query:
        return "AND", query.split(" AND ")
    elif " OR " in query:
        return "OR", query.split(" OR ")
    elif query.startswith('"') and query.endswith('"'):
        return "PHRASE", query[1:-1]
    elif re.match(r"\s*#\d+\(.*\)", query):
        match = re.match(r"\s*#(\d+)\((.*),(.*)\)", query)
        distance = int(match[1])
        terms = match[2].strip(), match[3].strip()
        return "PROXIMITY", (terms, distance)
    else:
        if query != "":
            return "TERM", query


def get_boolean_query_list(query_file_path: str, chars_to_skip: int = 2) -> list:
    """
    Parses a text file containing boolean queries and returns them as a list.

    Parameters:
    ----------
    query_file_path : str
        The path to the text file containing boolean queries.

    chars_to_skip : int, optional (default=2)
        The number of characters to skip at the beginning of each line in the query file.

    Returns:
    -------
    list
        A list containing parsed boolean queries. Each element is the output from `parse_unranked_query`.

    Notes:
    -----
    - Each line in the text file should contain a single boolean query.
    - The function will remove None values from the list.

    Example query file content:
    ---------------------------
    1 Happiness
    2 Edinburgh AND SCOTLAND
    ...
    """
    queries = open(query_file_path).read().split("\n")
    query_list = [
        parse_unranked_query(queries[idx][chars_to_skip:])
        for idx, element in enumerate(queries)
    ]
    return [item for item in query_list if item is not None]


def execute_single_term_search(
    term: str,
    inverted_index: dict,
    verbose: bool = False,
    stopword_removal: bool = True,
) -> list:
    """
    Executes a single-term search on an inverted index and returns the document IDs containing the term.

    Parameters:
    ----------
    term : str
        The search term. This term will be processed before searching in the inverted index.

    inverted_index : dict
        The inverted index to search.

    verbose : bool, optional (default=False)
        If set to True, prints additional information about the search.

    stopword_removal : bool, optional (default=True)
        If set to True, stopwords are removed from the search term.

    Returns:
    -------
    list
        A list of document IDs where the term appears.

    Raises:
    ------
    AssertionError
        If the processed term contains other than 1 term, an AssertionError is raised.

    Notes:
    -----
    - The term will be preprocessed (e.g., stopwords removed, stemmed) before being searched in the index.
    - This function only supports single-term queries.
    """
    term = get_processed_tokens_from_string(term, stopword_removal=stopword_removal)
    assert (
        len(term) == 1
    ), f"Term search must contain exactly 1 term. Found {len(term)} term(s) instead."
    if verbose:
        print("Executing single term search on processed term '" + term[0] + "'.\n")
    return list((inverted_index[((term)[0])]["Postings"]).keys())


def execute_proximity_search(
    term_1: str,
    term_2: str,
    distance: int,
    inverted_index: dict,
    strictly_ordering_tokens: bool,
    verbose: bool = False,
    full_output: bool = False,
):
    """
    Executes a proximity search for two terms in an inverted index and returns the document IDs
    that satisfy the proximity constraint.

    Parameters:
    -----------
    term_1 : str
        The first search term.

    term_2 : str
        The second search term.

    distance : int
        The maximum allowable distance between the two terms.

    inverted_index : dict
        The inverted index in which to search.

    strictly_ordering_tokens : bool
        If True, maintains the order of tokens in proximity; otherwise, the order is not considered.

    verbose : bool, optional (default=False)
        If True, prints additional information during the operation.

    full_output : bool, optional (default=False)
        If True, returns a dictionary with detailed position information; otherwise, returns a list of doc_ids.

    Returns:
    --------
    dict or list
        If full_output is True, returns a dictionary where keys are document IDs and values are lists
        of tuple positions (term_1_position, term_2_position) (only used for debugging).
        Otherwise, returns a sorted list of unique document IDs where the terms meet the proximity constraint.

    Raises:
    ------
    AssertionError
        Raised if either of the terms does not condense to a single term after processing.
    """
    # Define functions for distance comparison
    def ordered_distance(pos1, pos2, distance):
        return 0 <= (pos2 - pos1) <= distance

    def unordered_distance(pos1, pos2, distance):
        return abs(pos1 - pos2) <= distance

    # Choose 1 of the 2 distance functions
    distance_fn = ordered_distance if strictly_ordering_tokens else unordered_distance

    term_1 = get_processed_tokens_from_string(term_1)
    term_2 = get_processed_tokens_from_string(term_2)

    assert (
        len(term_1) == 1 and len(term_2) == 1
    ), f"Term search must contain exactly 1 term. Found {len(term_1)} {len(term_2)} term(s) instead."

    # Get the actual terms from the lists
    term_1, term_2 = term_1[0], term_2[0]

    if verbose:
        print(
            "Executing proximity search on processed terms '"
            + str(term_1)
            + "' and '"
            + str(term_2)
            + "' with distance '"
            + str(distance)
            + " and strict_ordering = '"
            + str(strictly_ordering_tokens)
            + "'.\n"
        )
    
    # Sorted only for prettier output
    doc_ids_of_mutual_occurance = sorted(
        set(
            inverted_index[term_1]["Postings"].keys()
            & inverted_index[term_2]["Postings"].keys()
        )
    )

    # If full_output is True, return detailed position information of
    # both terms for each co-occurance satient doc_id
    if full_output:
        coexistence_dictionary = defaultdict(list)
        for doc_id in doc_ids_of_mutual_occurance:
            term_1_positions = inverted_index[term_1]["Postings"][doc_id]
            term_2_positions = inverted_index[term_2]["Postings"][doc_id]

            for term_1_position in term_1_positions:
                for term_2_position in term_2_positions:
                    if distance_fn(term_1_position, term_2_position, distance):
                        coexistence_dictionary[doc_id].append(
                            (term_1_position, term_2_position)
                        )
        return coexistence_dictionary
    # If full_output is False, return only the doc_ids - again, sorted for prettier output
    else:
        coexistence_list = []
        for doc_id in doc_ids_of_mutual_occurance:
            term_1_positions = inverted_index[term_1]["Postings"][doc_id]
            term_2_positions = inverted_index[term_2]["Postings"][doc_id]

            for term_1_position in term_1_positions:
                for term_2_position in term_2_positions:
                    if distance_fn(term_1_position, term_2_position, distance):
                        coexistence_list.append(doc_id)
        return list(sorted(set(coexistence_list)))


def execute_phrase_search(
    phrase: str,
    inverted_index: dict,
    verbose: bool = False,
):
    """
    Executes a phrase search for a given phrase in an inverted index and returns the document IDs
    where the phrase appears exactly as specified.

    Parameters:
    -----------
    phrase : str
        The search phrase, consisting of two terms that appear consecutively in the document.

    inverted_index : dict
        The inverted index in which to search.

    verbose : bool, optional (default=False)
        If True, prints additional information during the operation.

    Returns:
    --------
    list
        A list of document IDs where the phrase appears exactly as given.

    Raises:
    ------
    AssertionError
        Raised if the phrase does not condense to exactly two terms after processing.

    Notes:
    ------
    - The function uses execute_proximity_search() internally to find the documents where the terms
      appear with a distance of 1 and in the specified order.

    """
    terms = get_processed_tokens_from_string(phrase)
    assert (
        len(terms) == 2
    ), f"Term search must contain exactly 1 term. Found {len(terms)} term(s) instead."
    return execute_proximity_search(
        term_1=terms[0],
        term_2=terms[1],
        distance=1,
        inverted_index=inverted_index,
        strictly_ordering_tokens=True,
        verbose=verbose,
        full_output=False,
    )


def execute_logical_search(
    operator_name: str,
    term_1: str,
    term_2: str,
    inverted_index: dict,
    verbose: bool = False,
    contraction_expansion: bool = True,
    stopword_removal: bool = True,
) -> set:
    """
    Executes a Boolean search operation between two terms or phrases in an inverted index.

    Parameters:
    -----------
    operator_name : str
        The Boolean operator to use for the search. Must be one of the following: 'AND', 'OR', 'AND NOT', 'OR NOT'.

    term_1, term_2 : str
        The search terms or phrases.

    inverted_index : dict
        The inverted index in which to perform the search.

    verbose : bool, optional (default=False)
        If True, prints additional information during the operation.

    contraction_expansion : bool, optional (default=True)
        Whether or not to expand contractions in the search terms.

    stopword_removal : bool, optional (default=True)
        Whether or not to remove stopwords from the search terms.

    Returns:
    --------
    set
        A sorted set of document IDs where the Boolean condition between the terms/phrases is met.

    Raises:
    ------
    Exception
        Raised if the operator name is not valid or if the token number for any of the terms is incorrect.

    Notes:
    ------
    - The terms are preprocessed (e.g., stopwords removed, stemmed, contractions expanded) before the search.
    - Both single terms and two-term phrases can be used for term_1 and term_2.
    """
    # Determine operator and use operator module or lambda functions
    if operator_name == "AND":
        boolean_operator = operator.and_
    elif operator_name == "OR":
        boolean_operator = operator.or_
    elif operator_name == "AND NOT":
        boolean_operator = lambda a, b: a - b
    elif operator_name == "OR NOT":
        boolean_operator = lambda a, b: a.union(b) - a.intersection(b)
    else:
        raise Exception("Operator name is not valid")
    
    term_1 = get_processed_tokens_from_string(
        term_1, contraction_expansion, stopword_removal
    )
    term_2 = get_processed_tokens_from_string(
        term_2, contraction_expansion, stopword_removal
    )

    if verbose:
        print(
            "Executing "
            + operator_name
            + " search on processed terms '"
            + str(term_1)
            + "', '"
            + str(term_2)
            + "'.\n"
        )

    # Handle different combinations of single terms and phrases
    if len(term_1) == 1 and len(term_2) == 1:
        # Both are single words - sorting only for prettier output
        return list(
            sorted(
                boolean_operator(
                    set(
                        execute_single_term_search(
                            term=term_1[0],
                            inverted_index=inverted_index,
                            verbose=verbose,
                        )
                    ),
                    set(
                        execute_single_term_search(
                            term=term_2[0],
                            inverted_index=inverted_index,
                            verbose=verbose,
                        )
                    ),
                )
            )
        )
    elif len(term_1) == 2 and len(term_2) == 1:
        # First term is a phrase, second is a single-word term
        # Sorting only for prettier output
        return list(
            sorted(
                boolean_operator(
                    set(
                        execute_phrase_search(
                            phrase=" ".join(term_1),
                            inverted_index=inverted_index,
                            verbose=verbose,
                        )
                    ),
                    set(
                        execute_single_term_search(
                            term=term_2[0],
                            inverted_index=inverted_index,
                            verbose=verbose,
                        )
                    ),
                )
            )
        )
    elif len(term_1) == 1 and len(term_2) == 2:
        # First term is a single-word term, second is a phrase
        # Sorting only for prettier output
        return list(
            sorted(
                boolean_operator(
                    set(
                        execute_single_term_search(
                            term=term_1[0],
                            inverted_index=inverted_index,
                            verbose=verbose,
                        )
                    ),
                    set(
                        execute_phrase_search(
                            phrase=" ".join(term_2),
                            inverted_index=inverted_index,
                            verbose=verbose,
                        )
                    ),
                )
            )
        )
    elif len(term_1) == 2 and len(term_2) == 2:
        # Both terms are phrases
        # Sorting only for prettier output
        return list(
            sorted(
                boolean_operator(
                    set(
                        execute_phrase_search(
                            phrase=" ".join(term_1),
                            inverted_index=inverted_index,
                            verbose=verbose,
                        )
                    ),
                    set(
                        execute_phrase_search(
                            phrase=" ".join(term_2),
                            inverted_index=inverted_index,
                            verbose=verbose,
                        )
                    ),
                )
            )
        )
    else:
        raise Exception("Token number is wrong")



def write_boolean_search_results(
    inverted_index: dict,
    boolean_queries: list,
    file_name: str = "results.boolean.txt",
    verbose: bool = False,
    contraction_expansion: bool = True,
    stopword_removal: bool = True,
) -> None:
    """
    Executes a list of Boolean queries and writes the results to a file.

    Parameters:
    -----------
    inverted_index : dict
        The inverted index in which to perform the search.

    boolean_queries : list
        List of Boolean queries. Each query is a tuple where the first
        element specifies the type of query ('TERM', 'PROXIMITY', 'PHRASE',
        or Boolean operator), and the second element contains the query parameters.

    file_name : str, optional (default="results.boolean.txt")
        Name of the file where the search results will be written.

    verbose : bool, optional (default=False)
        If True, prints additional information during the operation.

    contraction_expansion : bool, optional (default=True)
        Whether or not to expand contractions in the search terms.

    stopword_removal : bool, optional (default=True)
        Whether or not to remove stopwords from the search terms.

    Returns:
    --------
    None

    Writes:
    -------
    A text file where each line represents the document IDs retrieved for each query.
    The format is "query_index,document_id\n".
    """
    # Open the file to write the results
    with open(file_name, "w") as file:
        # Loop through each boolean query
        for query_index, boolean_query in enumerate(
            boolean_queries, start=1
        ):  # start=1 to start query_index at 1
            # For verbose mode, print the current query
            if verbose:
                print(f"Executing query {query_index}: {boolean_query}")
            # Single term
            if boolean_query[0] == "TERM":
                current_term = boolean_query[1]
                retrieved_docids = execute_single_term_search(
                    term=current_term,
                    inverted_index=inverted_index,
                    verbose=verbose,
                )
            # Proximity search
            elif boolean_query[0] == "PROXIMITY":
                current_term_1, current_term_2, current_distance = (
                    boolean_query[1][0][0],
                    boolean_query[1][0][1],
                    boolean_query[1][1],
                )
                retrieved_docids = execute_proximity_search(
                    term_1=current_term_1,
                    term_2=current_term_2,
                    distance=current_distance,
                    inverted_index=inverted_index,
                    strictly_ordering_tokens=False,
                    verbose=verbose,
                    full_output=False,
                )
            # Phrase search
            elif boolean_query[0] == "PHRASE":
                current_phrase = boolean_query[1]
                retrieved_docids = execute_phrase_search(
                    phrase=current_phrase,
                    inverted_index=inverted_index,
                    verbose=verbose,
                )
            # Boolean search
            else:
                current_operator, current_term_1, current_term_2 = (
                    boolean_query[0],
                    boolean_query[1][0],
                    boolean_query[1][1],
                )
                retrieved_docids = execute_logical_search(
                    operator_name=current_operator,
                    term_1=current_term_1,
                    term_2=current_term_2,
                    inverted_index=inverted_index,
                    verbose=verbose,
                    contraction_expansion=contraction_expansion,
                    stopword_removal=stopword_removal,
                )
            # Write the retrieved doc IDs to the file
            for doc_id in retrieved_docids:
                file.write(f"{query_index},{doc_id}\n")

    print(f"Boolean search results written to file {file_name}.\n")


def execute_tfidf_search(
    query: str,
    inverted_index: dict,
    id_to_headline_dict: dict,
    simple_score: bool = True,
    contraction_expansion: bool = True,
    stopword_removal: bool = True,
) -> dict:
    """
    Executes a TF-IDF based search on a document collection, given a query, matching
    documents that contain *at least one* of the query terms. Either based on simple
    TF-IDF aggregation or cosine similarity.

    Parameters:
    - query (str): The search query as a string.
    - inverted_index (dict): Dictionary containing the inverted index. 
      The keys are terms and the values are dictionaries with 'Postings' and 'Document Frequency'.
    - id_to_headline_dict (dict): Dictionary mapping document IDs to their headlines/titles.
    - simple_score (bool, optional): Flag to determine the type of scoring. If True, a simple aggregated TF-IDF 
      score is calculated for each document (lect. 7, slide 15). If False, cosine similarity is used.
      Defaults to True.
    - contraction_expansion (bool, optional): Flag to determine if contractions should be expanded. Defaults to True.
    - stopword_removal (bool, optional): Flag to determine if stopwords should be removed. Defaults to True.

    Returns:
    - dict: A sorted dictionary where keys are document IDs and values are their respective scores, sorted in 
      descending order of scores.
    """
    # Get the number of documents in the collection
    collection_size = len(id_to_headline_dict.keys())

    query_token_list = get_processed_tokens_from_string(
        query, contraction_expansion, stopword_removal
    )

    # If the query consists of only one term
    if len(query_token_list) == 1:
        union_doc_ids = set(
            execute_single_term_search(
                query_token_list[0],
                inverted_index,
                verbose=False,
                stopword_removal=stopword_removal,
            )
        )
    # If the query has multiple terms
    else:
        # Initialize set of document IDs containing the first term
        union_doc_ids = set(
            execute_single_term_search(
                query_token_list[0],
                inverted_index,
                verbose=False,
                stopword_removal=stopword_removal,
            )
        )
        # Union with document IDs containing remaining terms
        for term in query_token_list[1:]:
            term_doc_ids = set(
                execute_single_term_search(
                    term,
                    inverted_index,
                    verbose=False,
                    stopword_removal=stopword_removal,
                )
            )
            union_doc_ids = operator.or_(union_doc_ids, term_doc_ids)
    # If using simple TF-IDF scoring
    if simple_score:
        # Initialize document score dictionary
        document_score_dictionary = {doc_id: 0.0 for doc_id in union_doc_ids}
        # Calculate simple TF-IDF score for each document
        for union_doc_id in union_doc_ids:
            for token in query_token_list:
                token_term_frequency = len(
                    inverted_index[token]["Postings"][union_doc_id]
                )
                document_frequency = inverted_index[token]["Document Frequency"]
                # If the term frequency is zero, then skip this term
                if token_term_frequency == 0:
                    continue
                # Update the document's TF-IDF score
                document_score_dictionary[union_doc_id] += (
                    1 + np.log10(token_term_frequency)
                ) * np.log10(collection_size / document_frequency)
        # Sort by score in descending order
        sorted_document_score_dictionary = {
            k: v
            for k, v in sorted(
                document_score_dictionary.items(),
                key=lambda item: item[1],
                reverse=True,
            )
        }
        return sorted_document_score_dictionary
    
    # If using cosine similarity scoring:

    # 1. Here I calculate the document magnitude with TF-IDF, 1:1 as in
    # page 125 of the Introduction to Information Retrieval book, as far
    # as I understand it.

    else:
        def compute_document_magnitude(doc_id, inverted_index, collection_size):
            magnitude = 0.0
            # Iterate over all terms in the document
            for term, posting_data in inverted_index.items():
                if doc_id in posting_data["Postings"]:
                    tf = len(posting_data["Postings"][doc_id])
                    idf = np.log10(collection_size / posting_data["Document Frequency"])
                    magnitude += (tf * idf) ** 2
            return np.sqrt(magnitude)
        
        # Initialize scores for all documents
        Scores = {doc_id: 0.0 for doc_id in union_doc_ids}
        Length = {doc_id: 0.0 for doc_id in union_doc_ids}
        
        query_term_frequencies = Counter(query_token_list)
        
        # For each query term, update the Scores array
        for token in query_token_list:
            # calculate wt_q, i.e. for query, and fetch postings list
            wt_q = (
                (1 + np.log10(query_term_frequencies[token]))
                * np.log10(collection_size / inverted_index[token]["Document Frequency"])
            )
            postings_list = list(inverted_index[token]["Postings"].keys())
            for doc_id in postings_list:
                if doc_id not in union_doc_ids:
                    continue
                token_term_frequency = len(inverted_index[token]["Postings"][doc_id])
                wft_d = (
                    (1 + np.log10(token_term_frequency))
                    * np.log10(collection_size / inverted_index[token]["Document Frequency"])
                )
                Scores[doc_id] += wft_d * wt_q
                Length[doc_id] += wft_d**2
        
        # Normalize the Scores by the vector length of each document
        for doc_id in union_doc_ids:
            Scores[doc_id] /= compute_document_magnitude(doc_id, inverted_index, collection_size)
        
        # Sort by score in descending order
        sorted_scores = {
            k: v
            for k, v in sorted(Scores.items(), key=lambda item: item[1], reverse=True)
        }
        
        return sorted_scores

    # 2. Here I just do it intuitively, NOT penalising the excessive
    # length of a document. I expected this to be give exactly the same results
    # simple_score=True. I don't know why that's not the case.

    # else:
    #     # Initialize variables for vector space model
    #     query_dimensionality = len(query_token_list)
    #     document_score_dictionary = {}

    #     query_term_frequencies = Counter(query_token_list)

    #     query_tfidf_vector = np.array(
    #         [
    #             (1 + np.log10(query_term_frequencies[token]))
    #             * np.log10(
    #                 collection_size / inverted_index[token]["Document Frequency"]
    #             )
    #             for token in query_token_list
    #         ]
    #     )

    #     query_tfidf_vector /= np.linalg.norm(query_tfidf_vector)

    #     # Calculate TF-IDF vector for each document
    #     for union_doc_id in union_doc_ids:
    #         # Initialize a zero vector for each document

    #         doc_tfidf_vector = np.zeros(query_dimensionality)

    #         for index, token in enumerate(query_token_list):
    #             token_term_frequency = len(
    #                 inverted_index[token]["Postings"][union_doc_id]
    #             )
    #             document_frequency = inverted_index[token]["Document Frequency"]

    #             # If the term frequency is zero, then skip this term
    #             if token_term_frequency == 0:
    #                 continue
    #             # Update the document's TF-IDF vector
    #             token_value_tfidf_in_document_vector = (
    #                 1 + np.log10(token_term_frequency)
    #             ) * np.log10(collection_size / document_frequency)

    #             doc_tfidf_vector[index] = token_value_tfidf_in_document_vector
    #         # Normalize the vector to have L2 norm = 1
    #         l2_norm = np.linalg.norm(doc_tfidf_vector)
    #         if l2_norm > 0:  # To avoid division by zero
    #             doc_tfidf_vector /= l2_norm
    #         # Store the normalized vector in the dictionary
    #         document_score_dictionary[union_doc_id] = doc_tfidf_vector
    #     # Calculate dot products for each document
    #     doc_id_to_dot_product = {
    #         doc_id: np.dot(query_tfidf_vector, tfidf_vector)
    #         for doc_id, tfidf_vector in document_score_dictionary.items()
    #     }

    #     # Sort the dictionary by dot product in descending order
    #     sorted_doc_id_to_dot_product = {
    #         k: v
    #         for k, v in sorted(
    #             doc_id_to_dot_product.items(), key=lambda item: item[1], reverse=True
    #         )
    #     }
    #     return sorted_doc_id_to_dot_product

    

def get_ranked_query_list(query_file_path: str, chars_to_skip: int = 2) -> list:
    """
    Reads a text file containing queries and returns a list of queries.

    Parameters:
    - query_file_path (str): The file path to the text file containing the queries.
    - chars_to_skip (int, optional): The number of characters to skip at the beginning 
      of each line to get to the query text. Defaults to 2.

    Returns:
    - list: A list of queries read from the file, with unwanted characters removed.

    Example:
    Given a text file ('queries.txt') with content like:
    1 income tax reduction
    2 stock market in Japan

    get_ranked_query_list('queries.txt', 2) will return ['income tax reduction', 'stock market in Japan'].
    """
    queries = open(query_file_path).read().split("\n")
    query_list = [
        (queries[idx][chars_to_skip:])
        for idx in range(len((queries)))
    ]
    return [item for item in query_list if item is not None and item != '']


def write_ranked_search_results(
    inverted_index: dict,
    id_to_headline_dict: dict,
    ranked_queries: list,
    file_name: str = "results.ranked.txt",
    simple_score: bool = True,
    contraction_expansion: bool = True,
    stopword_removal: bool = True,
)->None:
    """
    Executes TF-IDF based searches for each query in 'ranked_queries', and writes the 
    top 150 results to a text file.

    Parameters:
    - inverted_index (dict): Dictionary containing the inverted index.
    - id_to_headline_dict (dict): Dictionary mapping document IDs to their headlines/titles.
    - ranked_queries (list): List of preprocessed queries to search against the collection.
    - file_name (str, optional): The name of the file where the search results will be written. 
      Defaults to "results.ranked.txt".
    - simple_score (bool, optional): Flag to determine the type of scoring. Defaults to True.
    - contraction_expansion (bool, optional): Flag to determine if contractions should be expanded. 
      Defaults to True.
    - stopword_removal (bool, optional): Flag to determine if stopwords should be removed. Defaults to True.

    Returns:
    - None: The function writes to a file and does not return any value.

    Output Format:
    Writes to a text file where each line is of the form: <Query Index>,<Document ID>,<Score>
    For example, '1,doc1,0.1234\n'.
    """
    with open(file_name, "w") as out_file:
        # Loop through each query
        for index, current_ranked_query in enumerate(
            ranked_queries, start=1
        ):
            # Get the results of the TF-IDF search
            current_tfidf_results = execute_tfidf_search(
                current_ranked_query,
                inverted_index,
                id_to_headline_dict,
                simple_score=simple_score,
                contraction_expansion=contraction_expansion,
                stopword_removal=stopword_removal,
            )
            # Limit results to top 150
            for rank, (doc_id, score) in enumerate(current_tfidf_results.items()):
                if rank >= 150:
                    break
                # Round score to four decimal places
                rounded_score = round(score, 4)
                # Write to the file
                out_file.write(f"{index},{doc_id},{rounded_score}\n")
    print(f"Ranked search results written to file {file_name}.\n")


def perform_boolean_experiment(
    inverted_index_path: str = "inverted_index.pkl",
    id_to_headline_path: str = "id_to_headline_dict.pkl",
    query_file_path: str = "queries.boolean.txt",
    chars_to_skip: int = 2,
    verbose: bool = False,
    contraction_expansion: bool = True,
    stopword_removal: bool = True,
) -> None:
    """
    Performs a full Boolean search experiment using the given inverted index and query file.
    Writes the results to a text file.

    Parameters:

    - inverted_index_path (str): The file path to the pickled inverted index.
      Defaults to "inverted_index.pkl".
    - id_to_headline_path (str): The file path to the pickled id_to_headline dictionary.
      Defaults to "id_to_headline_dict.pkl".
    - query_file_path (str): The file path to the text file containing the queries.
      Defaults to "queries.boolean.txt".
    - chars_to_skip (int, optional): The number of characters to skip at the beginning
      of each line to get to the query text. Defaults to 2.
    - verbose (bool): If True, prints additional information during the operation. Defaults to False.
    - contraction_expansion (bool): Flag to determine if contractions should be expanded for queries.
      Defaults to True.
    - stopword_removal (bool): Flag to determine if stopwords should be removed. Defaults to True.

    Returns:
    - list: A list of queries read from the file, with unwanted characters removed.
    """

    current_inverted_index, _ = load_pickled_data(
        inverted_index_path=inverted_index_path,
        id_to_headline_path=id_to_headline_path,
    )

    current_boolean_query_list = get_boolean_query_list(
        query_file_path=query_file_path, chars_to_skip=chars_to_skip
    )

    write_boolean_search_results(
        current_inverted_index,
        current_boolean_query_list,
        verbose=verbose,
        contraction_expansion=contraction_expansion,
        stopword_removal=stopword_removal,
    )


def perform_ranked_experiment(
    inverted_index_path: str = "inverted_index.pkl",
    id_to_headline_path: str = "id_to_headline_dict.pkl",
    query_file_path: str = "queries.ranked.txt",
    chars_to_skip: int = 2,
    simple_score: bool = True,
    contraction_expansion: bool = True,
    stopword_removal: bool = True,
) -> None:
    """
    Performs a full TFIDF  search experiment using the given inverted index and query file.
    Writes the results to a text file.

    Parameters:

    - inverted_index_path (str): The file path to the pickled inverted index.
      Defaults to "inverted_index.pkl".
    - id_to_headline_path (str): The file path to the pickled id_to_headline dictionary.
      Defaults to "id_to_headline_dict.pkl".
    - query_file_path (str): The file path to the text file containing the queries.
      Defaults to "queries.ranked.txt".
    - chars_to_skip (int, optional): The number of characters to skip at the beginning
      of each line to get to the query text. Defaults to 2.
    - simple_score (bool, optional): Flag to determine the type of scoring. Defaults to True (aggregate TF-IDF).
    - contraction_expansion (bool): Flag to determine if contractions should be expanded for queries.
      Defaults to True.
    - stopword_removal (bool): Flag to determine if stopwords should be removed. Defaults to True.

    Returns:
    - list: A list of queries read from the file, with unwanted characters removed.
    """

    current_inverted_index, current_id_to_headline_dict = load_pickled_data(
        inverted_index_path=inverted_index_path,
        id_to_headline_path=id_to_headline_path,
    )

    current_ranked_query_list = get_ranked_query_list(
        query_file_path=query_file_path, chars_to_skip=chars_to_skip
    )

    write_ranked_search_results(
        current_inverted_index,
        current_id_to_headline_dict,
        current_ranked_query_list,
        simple_score=simple_score,
        contraction_expansion=contraction_expansion,
        stopword_removal=stopword_removal,
    )

def main():
    """
    Main of the search engine.

    This function parses command line arguments to run functionalities:
    building an index, performing a boolean search, or executing a ranked search.

    Example Usage:
        1. To build the index:
            python code.py --mode build_index
        2. To perform a boolean search:
            python code.py --mode boolean_search --verbose True
        3. To perform a ranked search with custom parameters:
            python code.py --mode ranked_search --verbose True

    Parameters can also be combined and/or changed. For example:
        python code.py --mode build_index --xml-path "custom.xml" --csv False
    """
    parser = argparse.ArgumentParser(description="Search Engine")

    parser.add_argument(
        "--mode",
        choices=["build_index", "boolean_search", "ranked_search"],
        help="build_index: create_and_save_index, boolean_search: perform_boolean_experiment, ranked_search: perform_ranked_experiment",
    )

    # Common options
    parser.add_argument(
        "--contraction-expansion",
        type=bool,
        default=True,
        help="Flag for every mode to determine if contractions should be expanded during preprocessing and index construction. Default is True.",
    )
    parser.add_argument(
        "--stopword-removal",
        type=bool,
        default=True,
        help="Flag for every mode determine if stopwords should be removed during preprocessing and index construction. Default is True.",
    )
    parser.add_argument(
        "--verbose",
        type=bool,
        default=False,
        help="Flag for search modes. If True, prints additional information during both boolean and ranked searches. Default is False.",
    )

    # Building options
    parser.add_argument(
        "--xml-path",
        type=str,
        default="trec.5000.xml",
        help="The path for build mode to the XML file containing the documents to index. Default is 'trec.5000.xml'.",
    )
    parser.add_argument(
        "--csv",
        type=bool,
        default=True,
        help="Whether to save the index to a CSV file in build mode. Default is True.",
    )
    parser.add_argument(
        "--txt",
        type=bool,
        default=True,
        help="Whether to save the index to a text file in build mode. Default is True.",
    )

    # Common search options
    parser.add_argument(
        "--inverted-index-path",
        type=str,
        default="inverted_index.pkl",
        help="The file path to the pickled inverted index for search modes. Default is 'inverted_index.pkl'.",
    )
    parser.add_argument(
        "--id-to-headline-path",
        type=str,
        default="id_to_headline_dict.pkl",
        help="The file path to the pickled id_to_headline dictionary for search modes. Default is 'id_to_headline_dict.pkl'.",
    )
    parser.add_argument(
        "--chars-to-skip",
        type=int,
        default=2,
        help="The number of characters for search modes to skip at the beginning of each line to get to the query text. Default is 2.",
    )

    # Boolean search options
    parser.add_argument(
        "--boolean-queries-path",
        type=str,
        default="queries.boolean.txt",
        help="The file path for boolean search mode to the text file containing the boolean queries. Default is 'queries.boolean.txt'.",
    )

    # Ranked search options
    parser.add_argument(
        "--ranked-queries-path",
        type=str,
        default="queries.ranked.txt",
        help="The file path for ranked search mode to the text file containing the ranked queries. Default is 'queries.ranked.txt'.",
    )

    args = parser.parse_args()

    if args.mode == "build_index":
        create_and_save_index(
            xml_path=args.xml_path,
            contraction_expansion=args.contraction_expansion,
            stopword_removal=args.stopword_removal,
            csv=args.csv,
            txt=args.txt,
        )
    elif args.mode == "boolean_search":
        perform_boolean_experiment(
            inverted_index_path=args.inverted_index_path,
            id_to_headline_path=args.id_to_headline_path,
            query_file_path=args.boolean_queries_path,
            chars_to_skip=args.chars_to_skip,
            verbose=args.verbose,
            contraction_expansion=args.contraction_expansion,
            stopword_removal=args.stopword_removal,
        )
    elif args.mode == "ranked_search":
        perform_ranked_experiment(
            inverted_index_path=args.inverted_index_path,
            id_to_headline_path=args.id_to_headline_path,
            query_file_path=args.ranked_queries_path,
            chars_to_skip=args.chars_to_skip,
            simple_score=True,
            contraction_expansion=args.contraction_expansion,
            stopword_removal=args.stopword_removal,
        )
    else:
        print("Invalid option chosen. Use '-h' or '--help' for usage information.")


if __name__ == "__main__":
    main()