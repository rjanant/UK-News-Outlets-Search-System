import os
from enum import Enum

DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
CHILD_INDEX_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "index", "child")
)
GLOBAL_INDEX_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "index", "global")
)
SPELLCHECK_AND_AUTOCORRECT_DICTIONARY_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "spell_checking_and_autocomplete_files",
        "symspell_dictionary.pkl",
    )
)

STOP_WORDS_FILE_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),  # The directory of the constants.py file
        "ttds_2023_english_stop_words.txt",  # Assuming it's directly under utils
    )
)

STOPWORDS_LIST_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "spell_checking_and_autocomplete_files",
        "stopwords.txt",
    )
)

class Source(Enum):
    BBC = "bbc"
    GBN = "gbn"
    IND = "ind"
    TELE = "tele"
