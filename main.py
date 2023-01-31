from flags import get_paths_to_data, Partition, DataFolder, LabelsAs
from metrics import run_parent, run_bertscore

if __name__ == "__main__":
    data_folder = DataFolder.FINAL_SIMPLE_DIRECT
    partition = Partition.TEST
    labels_as = LabelsAs.LABEL
    HypothesisFilesEnum, reference_file, table_file = get_paths_to_data(data_folder, partition, labels_as)

    run_parent(data_folder, partition, labels_as, HypothesisFilesEnum, reference_file, table_file)
    run_bertscore(data_folder, partition, labels_as, HypothesisFilesEnum, reference_file)
    # TODO: do same for moverscore and bleurt

