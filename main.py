from flags import get_paths_to_data, Partition, DataFolder, LabelsAs
from metrics import run_parent, run_bertscore, run_moverscore, run_bleurt

# DONE: Load json files with csqa Simple Question entries (final_simple)
# DONE: Create a reference_{partition}.txt file, where each row is (Q+A) reference sentence
# DONE: For each QA2D model, create a hypothesis.txt file, where each row is a Declarative transformation of given Q+A.
# DONE: extract rdf information and create table.jsonl file with relevant data for PARENT metric
# TODO (optional): do same for all entity replacement techniques (label->eid, label->placeholder, entity groups, etc. )

if __name__ == "__main__":
    data_folder = DataFolder.FINAL_SIMPLE_DIRECT
    partition = Partition.VAL
    labels_as = LabelsAs.LABEL
    HypothesisFilesEnum, reference_file, table_file = get_paths_to_data(data_folder, partition, labels_as)

    run_parent(data_folder, partition, labels_as, HypothesisFilesEnum, reference_file, table_file)
    run_bertscore(data_folder, partition, labels_as, HypothesisFilesEnum, reference_file)
    run_moverscore(data_folder, partition, labels_as, HypothesisFilesEnum, reference_file)
    run_bleurt(data_folder, partition, labels_as, HypothesisFilesEnum, reference_file)

