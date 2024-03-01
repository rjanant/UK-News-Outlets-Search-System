from typing import List
import os
import warnings
import pandas as pd
from tqdm import tqdm
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import orjson


class SentimentAnalyzer:
    """
    A class to analyze sentiment for a single piece of text or a directory of CSV files containing text.

    [negative, neutral, positive], to 2f precision.
    """

    def __init__(
        self,
        model_name: str = "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis",
    ) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)

    def analyze_sentiment(self, text: str) -> List[float]:
        """
        Analyzes sentiment for a single piece of text and returns rounded sentiment probabilities.
        """
        try:
            inputs = self.tokenizer(
                text, return_tensors="pt", truncation=True, max_length=512
            ).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
            sentiments = (
                torch.nn.functional.softmax(outputs.logits, dim=-1)
                .to("cpu")[0]
                .float()
                .numpy()
            )
            rounded_sentiments = [
                float(np.round(sentiment, 2)) for sentiment in sentiments
            ]
            diff = 1.0 - sum(rounded_sentiments)
            rounded_sentiments[-1] = float(np.round(rounded_sentiments[-1] + diff, 2))
            return rounded_sentiments
        except Exception as e:
            print(f"Error analyzing sentiment: {e}")
            return None

    def get_sentiment_dictionary_from_csv_path(
        self, csv_path: str, sentiment_dictionary: dict = None
    ) -> dict:
        """
        Processes a CSV file to return a dictionary with document IDs and their sentiment probabilities.
        """
        csv_dataframe = pd.read_csv(csv_path)
        content_series = csv_dataframe["content"]
        doc_id_series = csv_dataframe["doc_id"]

        if sentiment_dictionary is None:
            sentiment_dictionary = {}

        for index, text in enumerate(tqdm(content_series, desc="Processing CSV")):
            sentiment_list = self.analyze_sentiment(text)
            doc_id = doc_id_series[index]

            if str(doc_id) in sentiment_dictionary.keys():
                warnings.warn(
                    f"Duplicate doc_id found: {doc_id}. Overwriting the previous entry!"
                )

            sentiment_dictionary[str(doc_id)] = sentiment_list

        return sentiment_dictionary

    def process_directories_and_write(
        self,
        data_path: str,
        outlet_folders: List[str],
        output_file_path: str,
        sentiment_dictionary: dict = None,
    ) -> None:
        """
        Processes all CSV files in a given directory to return a dictionary with document IDs and their sentiment probabilities.
        Dumps the dictionary to a file at the given output file path.
        """
        if sentiment_dictionary is None:
            sentiment_dictionary = {}

        # Iterate over each outlet folder
        for outlet_folder in outlet_folders:
            # Construct the path to the current outlet folder
            folder_path = os.path.join(data_path, outlet_folder)
            # List all files in the current outlet folder
            all_file_paths = os.listdir(folder_path)

            # Iterate over each file in the current outlet folder
            for file_name in tqdm(all_file_paths, desc=outlet_folder):
                # Construct the full path to the current file
                file_path = os.path.join(folder_path, file_name)
                # Ensure the file is a CSV before attempting to read it
                if file_path.endswith(".csv"):
                    try:
                        # Read the current CSV file into a pandas DataFrame
                        sentiment_dictionary = (
                            self.get_sentiment_dictionary_from_csv_path(
                                csv_path=file_path,
                                sentiment_dictionary=sentiment_dictionary,
                            )
                        )
                    except Exception as e:
                        print(f"Error with {file_path}: {e}")

        with open(output_file_path, "wb") as file:
            file.write(orjson.dumps(sentiment_dictionary))


# if __name__ == "__main__":
#     sentiment_analyzer = SentimentAnalyzer()
#     sentiment_analyzer.analyze_sentiment("The market is rallying!")
#     sentiment_analyzer.get_sentiment_dictionary_from_csv_path(
#         "C:/Users/Asus/Desktop/ttds-proj/backend/data/bbc/bbc_data_20240217_0.csv"
#     )
#     sentiment_analyzer.process_directories_and_write(
#         "C:/Users/Asus/Desktop/ttds-proj/backend/data",
#         ["tele"],
#         "C:/Users/Asus/Desktop/ttds-proj/backend/sentiment_test.json",
#     )
