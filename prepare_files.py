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
    source_folder = ROOT_PATH.joinpath(f"source_data/simple_direct")
    target_folder = ROOT_PATH.joinpath("data")
    partitions = ("test", "train", "val")

    make_partitioned_reference_files(source_folder, partitions, target_folder)
    make_partitioned_table_files(source_folder, partitions, target_folder)

    # _print_active_sets(partitions[1:2], min_len=1)
    # ['(Q213,P194,c(Q1752346))']
    # ['(Q19278,P47,c(Q15617994))', '(Q36,P47,c(Q15617994))']
    # ['(c(Q20667921),P6,Q82955)']
    # ['(c(Q15617994),P37,Q9056)', '(c(Q15617994),P37,Q8752)', '(c(Q15617994),P37,Q397)']
    # TODO: here we can just use transform_active_set.py from CARTON/annotate_csqa to create the table.jsonl file.
    #   !Or just use the new "table_format" entries from CARTON/data/final_simple after the operation finishes