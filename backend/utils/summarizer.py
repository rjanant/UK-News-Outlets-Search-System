import re
import pandas as pd
from common import get_preprocessed_words
import numpy as np
from collections import Counter
from math import log, sqrt


def compute_tf(text: str):
    """Calculate term frequency for a given text."""
    tf_text = Counter(text)
    for i in tf_text:
        tf_text[i] = tf_text[i] / float(len(text))
    return tf_text


# Helper function to calculate inverse document frequency
def compute_idf(word: str, corpus: list):
    """Calculate inverse document frequency for a given word in a corpus."""
    return log(len(corpus) / sum([1.0 for i in corpus if word in i]))


# Calculate cosine similarity between title and each sentence
def cosine_similarity(vector1, vector2):
    """Calculate cosine similarity between two vectors."""
    dot_product = np.dot(vector1, vector2)
    norm_a = np.linalg.norm(vector1)
    norm_b = np.linalg.norm(vector2)
    if norm_a == 0 or norm_b == 0:  # Avoid division by zero
        return 0.0
    return dot_product / (norm_a * norm_b)


def get_summary_sentence(
    title: str, content: str, number_of_initial_sentences_to_skip: int
):
    """Get the most relevant sentence from the article content setences based on the title."""
    split_regex = r"[.!?]"
    article_sentences = re.split(split_regex, content)
    article_sentences_lower = [x.lower() for x in article_sentences if x]

    # Preprocess title and sentences
    preprocessed_title = " ".join(get_preprocessed_words(title))
    preprocessed_sentences = [
        " ".join(get_preprocessed_words(sentence))
        for sentence in article_sentences_lower
    ]

    # Combine title and adjusted sentences for manual TF-IDF vectorization
    texts = [preprocessed_title] + preprocessed_sentences[
        number_of_initial_sentences_to_skip:
    ]

    # Create a set of all unique words
    vocabulary = set(word for text in texts for word in text.split())

    # Calculate TF for each document
    tfs = [compute_tf(text.split()) for text in texts]

    # Calculate IDF for each word in the vocabulary
    idfs = {word: compute_idf(word, texts) for word in vocabulary}

    # Calculate TF-IDF vectors
    tfidf_vectors = []
    for tf in tfs:
        tfidf_vectors.append(
            np.array([tf.get(word, 0) * idfs[word] for word in vocabulary])
        )

    # Compute similarities
    similarities = [
        cosine_similarity(tfidf_vectors[0], vec) for vec in tfidf_vectors[1:]
    ]

    # Find the most similar sentence index
    most_similar_sentence_index = (
        np.argmax(similarities) + number_of_initial_sentences_to_skip
    )
    return article_sentences[most_similar_sentence_index]
