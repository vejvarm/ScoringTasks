import json
import logging
from flags import DataPath


LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.WARNING)

class DataLoader:

    def __init__(self,
                 path_to_reference_txt=DataPath.REFERENCE.value,
                 path_to_hypothesis_txt=DataPath.HYPOTHESIS.value):
        """ useful for following metrics:
         - BERTSCORE
         - BLEURT
         - MOVERSCORE

        :param path_to_reference_txt: (Path object/string) to text file with reference sentences (1 line = 1 sentence)
        :param path_to_hypothesis_txt: (Path object/string) to text file with hypothesis sentences (1 line = 1 sentence)
        """
        self.path_to_reference_txt = path_to_reference_txt
        self.path_to_hypothesis_txt = path_to_hypothesis_txt

    @staticmethod
    def _try_load(path_to_file):
        try:
            with open(path_to_file, mode="r", encoding='utf8') as f:
                return [line.strip() for line in f if line.strip()]
        except FileNotFoundError:
            LOGGER.warning(f"File {path_to_file.name} at path {path_to_file.parent} not found. Returning empty list.")
            return []

    def load(self):
        """ Load data from text files to sentence lists

        :return [reference], [hypothesis] lists of sentences
        """
        reference_list = self._try_load(self.path_to_reference_txt)
        hypothesis_list = self._try_load(self.path_to_hypothesis_txt)

        return reference_list, hypothesis_list


class TableDataLoader(DataLoader):

    def __init__(self, path_to_reference_txt=DataPath.REFERENCE.value,
                 path_to_hypothesis_txt=DataPath.HYPOTHESIS.value,
                 path_to_table_json=DataPath.TABLE.value):
        """ useful for following metrics:
         - PARENT

        :param path_to_reference_txt: (Path object/string) to text file with reference sentences (1 line = 1 sentence)
        :param path_to_hypothesis_txt: (Path object/string) to text file with hypothesis sentences (1 line = 1 sentence)
        :param path_to_table_json: (Path object/string) to json.ln file with line sepparated table entries
        """
        super().__init__(path_to_reference_txt, path_to_hypothesis_txt)
        self.path_to_table_json = path_to_table_json

    @staticmethod
    def _try_load_tokenized(path_to_file):
        try:
            with open(path_to_file, mode="r", encoding='utf8') as f:
                return [line.strip().split() for line in f if line.strip()]
        except FileNotFoundError:
            LOGGER.warning(f"File {path_to_file.name} at path {path_to_file.parent} not found. Returning empty list.")
            return []

    @staticmethod
    def _try_load_table(path_to_file):
        try:
            with open(path_to_file, mode="r", encoding='utf8') as f:
                return [json.loads(line) for line in f if line.strip()]
        except FileNotFoundError:
            LOGGER.warning(f"File {path_to_file.name} at path {path_to_file.parent} not found. Returning empty list.")
            return []

    def load_tokenized(self):
        """ Load data from text/json files to tokenized (word-level) lists

        :return [reference_tokenized], [hypothesis_tokenized], [table] lists of sentences and table entries
        """
        reference_tokenized_list = self._try_load_tokenized(self.path_to_reference_txt)
        hypothesis_tokenized_list = self._try_load_tokenized(self.path_to_hypothesis_txt)
        table_list = self._try_load_table(self.path_to_table_json)

        return reference_tokenized_list, hypothesis_tokenized_list, table_list
