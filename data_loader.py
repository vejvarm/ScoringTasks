import json

from flags import DataPath


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

    def load(self):
        """ Load data from text files to sentence lists

        :return [reference], [hypothesis] lists of sentences
        """
        with open(self.path_to_reference_txt, mode="r", encoding='utf8') as f:
            reference_list = [line.strip() for line in f if line.strip()]

        with open(self.path_to_hypothesis_txt, mode="r", encoding='utf8') as f:
            hypothesis_list = [line.strip() for line in f if line.strip()]

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

    def load_tokenized(self):
        """ Load data from text/json files to tokenized (word-level) lists

        :return [reference_tokenized], [hypothesis_tokenized], [table] lists of sentences and table entries
        """
        with open(self.path_to_reference_txt, mode="r", encoding='utf8') as f:
            reference_tokenized_list = [line.strip().split() for line in f if line.strip()]

        with open(self.path_to_hypothesis_txt, mode="r", encoding='utf8') as f:
            hypothesis_tokenized_list = [line.strip().split() for line in f if line.strip()]

        with open(self.path_to_table_json, mode="r", encoding='utf8') as f:
            table_list = [json.loads(line) for line in f if line.strip()]

        return reference_tokenized_list, hypothesis_tokenized_list, table_list
