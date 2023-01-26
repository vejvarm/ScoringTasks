from pathlib import Path
from enum import Enum

ROOT_PATH = Path(__file__).parent.absolute()


class DataPath(Enum):
    HYPOTHESIS = ROOT_PATH.joinpath("data/reference_val.txt")
    REFERENCE = ROOT_PATH.joinpath("data/reference_val.txt")
    TABLE = ROOT_PATH.joinpath("data/table_val.jsonl")


class ModelPath(Enum):
    BLEURT_20 = ROOT_PATH.joinpath("models/BLEURT-20")