from pathlib import Path
from enum import Enum

ROOT_PATH = Path(__file__).parent.absolute()


class DataPath(Enum):
    HYPOTHESIS = ROOT_PATH.joinpath("data/wb_predictions.txt")
    REFERENCE = ROOT_PATH.joinpath("data/wb_test_output.txt")
    TABLE = ROOT_PATH.joinpath("data/wb_test_tables.jl")

class ModelPath(Enum):
    BLEURT_20 = ROOT_PATH.joinpath("models/BLEURT-20")