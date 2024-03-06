import os
from enum import Enum

DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
CHILD_INDEX_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "index", "child")
)
GLOBAL_INDEX_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "index", "global")
)
SPELL_CHECK_DICTIONARY_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "spell_checking_and_autocomplete_files",
        "symspell_dictionary.pkl",
    )
)

class Source(Enum):
    BBC = "bbc"
    GBN = "gbn"
    IND = "ind"
    TELE = "tele"
