from symspellpy import SymSpell, Verbosity
import os
import pandas as pd
from tqdm import tqdm


class SpellChecker:
    def __init__(self, dictionary_path="spell_checking_files/symspell_dictionary.pkl"):

        self.sym_spell = SymSpell()
        if dictionary_path:
            self.sym_spell.load_pickle(dictionary_path)

    def create_spell_checking_txt(self, data_path, outlet_folders, output_corpus_path):
        """
        From txts with "content" and "doc_id" columns, create a txt file with all the content from the articles.

        data_path: str
            The path to the directory containing the outlet folders.
        outlet_folders: list
            A list of the outlet folder names.
        output_corpus_path: str
            Where to save the output corpus txt file containing all the articles.
        """
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
                    df = pd.read_csv(file_path)
                    content_series = df["content"]
                    for article in content_series:
                        try:
                            # Open the file in append mode and write the article string
                            with open(
                                output_corpus_path, "a", encoding="utf-8"
                            ) as file:
                                file.write(
                                    article + "\n"
                                )  # Add a newline to separate articles
                        except Exception as e:
                            pass

    def create_and_save_spellcheck_dictionary(
        self, corpus_txt_path, output_dictionary_path
    ):
        """
        Create and save a SymSpell dictionary from a corpus.

        corpus_txt_path: str
            The path to the corpus txt file.
        output_dictionary_path: str
            Where to save the output dictionary.

        """
        with open(corpus_txt_path, "r", encoding="utf-8") as file:
            corpus = file.read()
        self.sym_spell.create_dictionary(corpus)
        self.sym_spell.save_pickle(output_dictionary_path)

    def load_dictionary(self, dictionary_path):
        """Load the SymSpell dictionary from a pickle file."""
        self.sym_spell.load_pickle(dictionary_path)

    def correct_query(self, text, max_edit_distance=2, ignore_non_words=True):
        """
        Correct a query using the loaded SymSpell dictionary.

        text: str
            The query to correct.
        max_edit_distance: int
            The maximum edit distance to look for corrections.
        ignore_non_words: bool
            Whether to ignore non-words when correcting the query.
        
        """
        suggestions = self.sym_spell.lookup_compound(
            text, max_edit_distance=max_edit_distance, ignore_non_words=ignore_non_words
        )
        return suggestions[0].term if suggestions else text




# if __name__ == "__main__":
#     data_path = "C:/Users/Asus/Desktop/ttds-proj/backend/data/"
#     # outlet_folders = ["bbc", "gbn", "ind", "tele"]
#     # corpus_path = "C:/Users/Asus/Desktop/ttds-proj/backend/utils/corpus.txt"

#     # create_spell_checking_txt(data_path, outlet_folders, output_corpus_path)

#     # output_dictionary_path = "C:/Users/Asus/Desktop/ttds-proj/backend/utils/symspell_dictionary.pkl"

#     # create_spell_checker_dictionary(corpus_path, output_dictionary_path)

#     # pickle_file_path = "spell_checking_files/huge_symspell_dictionary.pkl"
#     # symspell_instance = load_sym_spell_instance("pickle_file_path")
#     # print(correct_query("helo", symspell_instance))
