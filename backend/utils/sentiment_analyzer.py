import os
import pandas as pd
from tqdm.notebook import tqdm
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def analyze_sentiment_from_csv_path(
    csv_path, device, model, tokenizer, csv_sentiment_dictionary=None
):
    """[negative, neutral, positive]"""
    csv_dataframe = pd.read_csv(csv_path)
    content_series = csv_dataframe["content"]
    doc_id_series = csv_dataframe["doc_id"]

    model.to(device)

    if csv_sentiment_dictionary is None:
        print("Creating new dictionary!")
        csv_sentiment_dictionary = {}

    for index, text in enumerate(tqdm(content_series)):
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
            csv_sentiment_dictionary[doc_id_series[index]] = rounded_sentiments
        except:
            print(f"Error at index {index}, recording None.")
            csv_sentiment_dictionary[doc_id_series[index]] = None
