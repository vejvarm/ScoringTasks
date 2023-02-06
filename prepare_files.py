from pathlib import Path
from flags import ROOT_PATH
from csqa_converter import CSQAConverter


def make_partitioned_reference_files(source_folder: Path, partitions: tuple[str, ...], target_folder: Path):
    for partition in partitions:
        partition_source_folder = source_folder.joinpath(partition)
        csqa_conv = CSQAConverter(partition_source_folder, target_folder)

        csqa_conv.build_reference_file(f"reference_{partition}.txt")


def make_partitioned_table_files(source_folder: Path, partitions: tuple[str, ...], target_folder: Path):
    for partition in partitions:
        partition_source_folder = source_folder.joinpath(partition)
        csqa_conv = CSQAConverter(partition_source_folder, target_folder)

        csqa_conv.build_table_file(f"table_{partition}.jsonl")


def _print_active_sets(partitions: tuple[str], min_len=1):
    for partition in partitions:
        source_folder = ROOT_PATH.joinpath(f"source_data/final_simple/csqa/{partition}")
        csqa_conv = CSQAConverter(source_folder, ROOT_PATH)

        csqa_conv.print_active_set(min_len)


if __name__ == "__main__":
    source_folder = ROOT_PATH.joinpath(f"source_data/simple_direct_v3")
    target_folder = ROOT_PATH.joinpath("data/simple_direct_v3")
    partitions = ("test", "train", "val")[-1:]

    target_folder.mkdir(parents=True, exist_ok=True)

    make_partitioned_reference_files(source_folder, partitions, target_folder)
    make_partitioned_table_files(source_folder, partitions, target_folder)