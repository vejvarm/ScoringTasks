from pathlib import Path
from enum import Enum, auto

ROOT_PATH = Path(__file__).parent.absolute()
RESULTS_PATH = ROOT_PATH.joinpath('results')


class Partition(Enum):

    @staticmethod
    def _generate_next_value_(name, start, count, last_values):
        return name.lower()

    TEST = auto()
    TRAIN = auto()
    VAL = auto()


class LabelsAs(Enum):

    @staticmethod
    def _generate_next_value_(name, start, count, last_values):
        return name.upper()

    LABEL = auto()


class DataFolder(Enum):
    FINAL_SIMPLE = Path('data/final_simple')
    FINAL_SIMPLE_DIRECT = Path('data/final_simple_direct')


class HypothesisFileParser:

    def __init__(self, full_data_path: Path, partition: Partition, labels_as: LabelsAs):
        self.data_path = full_data_path
        self.partition = partition
        self.labels_as = labels_as

        self.T5_SMALL = self._generate_next_value_('T5_SMALL')
        self.T5_BASE = self._generate_next_value_('T5_BASE')
        self.T5_3B = self._generate_next_value_('T5_3B')
        self.T5_WHYN = self._generate_next_value_('T5_WHYN')

    def generate_enum(self):
        class HypothesisFiles(Enum):
            T5_SMALL = self.T5_SMALL
            T5_BASE = self.T5_BASE
            T5_3B = self.T5_3B
            T5_WHYN = self.T5_WHYN

        return HypothesisFiles

    def _generate_next_value_(self, name):
        return self.data_path.joinpath(f"hypothesis_{self.partition.value}-{name}-{self.labels_as.value}.txt")


class DataPath(Enum):
    HYPOTHESIS = ROOT_PATH.joinpath("data/final_simple_direct/hypothesis_val-T5_3B-LABEL.txt")
    REFERENCE = ROOT_PATH.joinpath("data/final_simple_direct/reference_val.txt")
    TABLE = ROOT_PATH.joinpath("data/final_simple_direct/table_val.jsonl")


def get_paths_to_data(data_folder: DataFolder, partition: Partition, labels_as: LabelsAs):
    full_data_path = ROOT_PATH.joinpath(data_folder.value)
    if not full_data_path.exists():
        raise FileNotFoundError(f"Data folder at path {full_data_path} doesn't exist.")
    if partition not in Partition:
        raise NotImplementedError(f'Partition {partition.name} is not supported.')

    hp = HypothesisFileParser(full_data_path=full_data_path, partition=partition, labels_as=labels_as)
    HypothesisFiles = hp.generate_enum()
    reference_file = full_data_path.joinpath(f'reference_{partition.value}.txt')
    table_file = full_data_path.joinpath(f'table_{partition.value}.jsonl')

    return HypothesisFiles, reference_file, table_file


class ModelPath(Enum):
    BLEURT_20 = ROOT_PATH.joinpath("models/BLEURT-20")


if __name__ == "__main__":
    file_paths = get_paths_to_data(DataFolder.FINAL_SIMPLE_DIRECT, Partition.VAL, LabelsAs.LABEL)
    print(file_paths[0].T5_WHYN.value)
