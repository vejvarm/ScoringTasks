from flags import ROOT_PATH
from csqa_converter import CSQAConverter

if __name__ == "__main__":
    partitions = ("test", "train", "val")
    for partition in partitions:
        source_folder = ROOT_PATH.joinpath(f"source_data/final_simple/csqa/{partition}")
        target_folder = ROOT_PATH.joinpath("data")
        csqa_conv = CSQAConverter(source_folder, target_folder)

        csqa_conv.build_reference_file(f"reference_{partition}.txt")
