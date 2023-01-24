from flags import ROOT_PATH
from csqa_converter import CSQAConverter

if __name__ == "__main__":
    source_folder = ROOT_PATH.joinpath("source_data")
    target_folder = ROOT_PATH.joinpath("data")
    csqa_conv = CSQAConverter(source_folder, target_folder)

    csqa_conv.build_reference_file("reference_test.txt")
