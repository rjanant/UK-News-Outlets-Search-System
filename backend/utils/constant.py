import os
from enum import Enum

DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
CHILD_INDEX_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "index", "child"))
GLOBAL_INDEX_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "index", "global"))
class Source(Enum):
    BBC = "bbc"
    GBN = "gbn"
    IND = "ind"
    TELE = "tele"