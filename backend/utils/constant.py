import os
from enum import Enum

DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
class Source(Enum):
    BBC = "bbc"
    GBN = "gbn"
    IND = "ind"
    TELE = "tele"