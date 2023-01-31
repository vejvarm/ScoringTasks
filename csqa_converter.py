import logging
import json
from pathlib import Path

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

        assert self.path_to_source_folder.exists(), "Path to source json folder doesn't exists. Provide a valid path."

    @staticmethod
    def _load_csqa_json(pth: Path) -> list[dict]:
        """ load one csqa json file (corresponds to one conversation)"""
        with pth.open('r', encoding="utf8") as f:
            conversation = json.load(f)

        assert isinstance(conversation, list)
        if not conversation:
            return []

        assert isinstance(conversation[0], dict)

        return conversation

    def _append_to_file(self, line: str, target_file_name: str):
        with self.path_to_target_folder.joinpath(target_file_name).open("a", encoding="utf8") as f:
            f.write(line)
            f.write("\n")

    def _append_to_jsonl(self, line: list[str] or str, target_file_name: str):
        with self.path_to_target_folder.joinpath(target_file_name).open("a", encoding="utf8") as f:
            f.write(json.dumps(line))
            f.write("\n")

    @staticmethod
    def _extract_and_concatenate_qa_from_conversation(conversation: list[dict]) -> list[str]:
        qa_list = []
        for i in range(0, len(conversation), 2):
            q = conversation[i]["utterance"].replace(" ?", "?").strip()
            a = conversation[i+1]["utterance"].strip()
            qa_list.append(f"{q} {a}")

        return qa_list

    @staticmethod
    def _extract_table_format(conversation: list[dict]) -> list[str]:
        table_format_list = []
        for i in range(0, len(conversation), 2):
            sys = conversation[i+1]
            table_format_list.append(sys["table_format"])

        return table_format_list

    def build_reference_file(self, reference_file_name="reference.txt"):
        for file_path in self.path_to_source_folder.glob("**/*.json"):
            conversation = self._load_csqa_json(file_path)
            if not conversation:
                LOGGER.warning(f"File {file_path} is empty. Skipping")
                continue
            qa_list = self._extract_and_concatenate_qa_from_conversation(conversation)
            for sentence in qa_list:
                self._append_to_file(sentence, reference_file_name)
        LOGGER.info(f"Reference file built at path '{self.path_to_target_folder.joinpath(reference_file_name)}'")

    def print_active_set(self, min_len=1):
        coref_no_rel_field = 0
        coref_rel_field = 0
        for file_path in self.path_to_source_folder.glob("**/*.json"):
            conversation = self._load_csqa_json(file_path)
            if not conversation:
                LOGGER.warning(f"File {file_path} is empty. Skipping")
                continue
            for i in range(1, len(conversation), 2):
                q = conversation[i-1]
                a = conversation[i]
                active_set = a['active_set']
                turn_pos = q['turn_position']
                question_type = q['question-type']
                if len(active_set) < min_len:
                    continue
                if 'relations' in q.keys():
                    assert len(q['relations']) <= 1  # NOTE: All Questions have no more than 1 relation
                if question_type == "Simple Question (Ellipsis)":
                    assert len(q['relations']) == 0  # NOTE: All Ellipsis Questions have empty relations field
                if question_type == "Simple Question (Coreferenced)":
                    if 'relations' not in q.keys():
                        coref_no_rel_field += 1  # NOTE: "Yes/No, I meant" type of questions (preceded by Clarification)
                    else:
                        coref_rel_field += 1  # NOTE: And Who/Which/What ... type of questions
                        # print(q["sec_ques_type"], q["sec_ques_sub_type"], q["utterance"])
                        LOGGER.debug(
                            f'turn {turn_pos} in {file_path.parent.name}/{file_path.name} is Coref and has rel field'
                        )

                # print(f"{active_set}")

        print(f"# coref_no_rel: {coref_no_rel_field}\n# coref_rel: {coref_rel_field}")

    def build_table_file(self, table_file_name="table.txt"):
        for file_path in self.path_to_source_folder.glob("**/*.json"):
            conversation = self._load_csqa_json(file_path)
            if not conversation:
                LOGGER.warning(f"File {file_path} is empty. Skipping")
                continue
            table_format_list = self._extract_table_format(conversation)
            for line in table_format_list:
                self._append_to_jsonl(line, table_file_name)
        LOGGER.info(f"Reference file built at path '{self.path_to_target_folder.joinpath(table_file_name)}'")
