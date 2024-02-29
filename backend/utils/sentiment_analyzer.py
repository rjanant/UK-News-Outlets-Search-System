### TO DO - MOVE THE MODEL, TOKENIZER, AND DEVICE DETERMINATION
# TO THE INITIALIZATION OF THE MODULE

import pandas as pd
from tqdm.notebook import tqdm
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)
MODEL = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)


def analyze_sentiment(text, model=MODEL, tokenizer=TOKENIZER, device=DEVICE):
    """
    Analyzes sentiment for a single piece of text and returns rounded sentiment probabilities.

    Parameters:
        text (str): The text to analyze.
        model: The pre-trained sentiment analysis model.
        tokenizer: The tokenizer for the model.
        device: The device to run the model on.

    Returns:
        list: A list containing the probabilities for [negative, neutral, positive] sentiments.
    """
    try:
        inputs = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512
        ).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        sentiments = (
            torch.nn.functional.softmax(outputs.logits, dim=-1)
            .to("cpu")[0]
            .float()
            .numpy()
        )
        # Round the first 2 elements
        rounded_sentiments = np.round(sentiments[:-1], decimals=2)
        # Ensure the sum equals 1 by adjusting the last element
        last_element = 1 - np.sum(rounded_sentiments)
        rounded_sentiments = list(rounded_sentiments) + [
            np.round(last_element, decimals=2)
        ]
        return rounded_sentiments
    except Exception as e:
        return None


def get_sentiment_dictionary_from_csv_path(
    csv_path,
    device=DEVICE,
    model=MODEL,
    tokenizer=TOKENIZER,
    csv_sentiment_dictionary=None,
):
    """
    Returns {doc_id: [prob_negative, prob_neutral, prob_positive]}.

    If csv_sentiment_dictionary is None, a new dictionary will be created.
    """
    csv_dataframe = pd.read_csv(csv_path)
    content_series = csv_dataframe["content"]
    doc_id_series = csv_dataframe["doc_id"]

    model.to(device)

    if csv_sentiment_dictionary is None:
        csv_sentiment_dictionary = {}

    for index, text in tqdm(enumerate(content_series), total=len(content_series)):
        sentiment_list = analyze_sentiment(text, model, tokenizer, device)
        doc_id = doc_id_series[index]
        csv_sentiment_dictionary[doc_id] = sentiment_list

    return csv_sentiment_dictionary