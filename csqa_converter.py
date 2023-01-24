import logging
import json
from pathlib import Path
from typing import TextIO

# TODO:
#   Load json files with csqa Simple Question entries (final_simple)
#   Create a reference.txt file, where each row is (Q+A) reference sentence
#   For each QA2D model, create a hypothesis.txt file, where each row is a Declarative transformation of given Q+A.
#   (optional) do same for all entity replacement techniques (label->eid, label->placeholder, entity groups, etc. )

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


class CSQAConverter:

    def __init__(self, path_to_source_folder: Path, path_to_target_folder: Path):
        """ Do various conversion tasks on CSQA QA_*.json files contained in 'path_to_source_folder'

        :param path_to_source_folder:
        :param path_to_target_folder:
        """
        self.path_to_source_folder = path_to_source_folder
        self.path_to_target_folder = path_to_target_folder

        assert self.path_to_source_folder.exists(), "Path to source json file folder doesn't exists. Please provide valid path."

    @staticmethod
    def _load_csqa_json(pth: Path) -> list[dict]:
        """ load one csqa json file (corresponds to one conversation)"""
        with pth.open('r', encoding="utf8") as f:
            conversation = json.load(f)

        assert isinstance(conversation, list)
        assert isinstance(conversation[0], dict)

        return conversation

    def _append_to_reference_file(self, sentence: str, reference_file_name: str):
        with self.path_to_target_folder.joinpath(reference_file_name).open("a", encoding="utf8") as f:
            f.write(sentence+"\n")

    @staticmethod
    def _extract_and_concatenate_qa_from_conversation(conversation: list[dict]) -> list[str]:
        qa_list = []
        for i in range(0, len(conversation), 2):
            q = conversation[i]["utterance"].replace(" ?", "?").strip()
            a = conversation[i+1]["utterance"].strip()
            qa_list.append(f"{q} {a}")

        return qa_list

    def build_reference_file(self, reference_file_name="reference.txt"):
        for file_path in self.path_to_source_folder.glob("**/*.json"):
            conversation = self._load_csqa_json(file_path)
            qa_list = self._extract_and_concatenate_qa_from_conversation(conversation)
            for sentence in qa_list:
                self._append_to_reference_file(sentence, reference_file_name)
        LOGGER.info(f"Reference file built at path '{self.path_to_target_folder.joinpath(reference_file_name)}'")