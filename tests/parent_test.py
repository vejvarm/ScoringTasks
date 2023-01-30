import statistics
import time
import pandas as pd
from parent import parent

from flags import get_paths_to_data, RESULTS_PATH, Partition, DataFolder, LabelsAs
from data_loader import TableDataLoader

if __name__ == "__main__":
    df = pd.DataFrame()
    data_folder = DataFolder.FINAL_SIMPLE_DIRECT
    partition = Partition.VAL
    labels_as = LabelsAs.LABEL
    hypothesis_files_enum, reference_file, table_file = get_paths_to_data(data_folder, partition, labels_as)
    for hypothesis_txt_path in hypothesis_files_enum:
        print(hypothesis_txt_path.value.name)
        tdl = TableDataLoader(reference_file, hypothesis_txt_path.value, table_file)
        reference, hypothesis, table = tdl.load_tokenized()

        # print(len(reference))
        # print(len(hypothesis))
        # print(len(table))
        tic = time.perf_counter()
        precision, recall, f_score = parent(
            hypothesis,
            reference,
            table,
            avg_results=False,  # if False, returns list with individual scores for each (hyp, ref) pair
            n_jobs=32,
            use_tqdm=True,
        )

        p_mean = statistics.mean(precision)
        r_mean = statistics.mean(recall)
        f1_mean = statistics.mean(f_score)

        print(f"\ttime: {time.perf_counter() - tic}")
        # print(f"\tprecision: {p_mean:.3f}\n\trecall: {r_mean:.3f}\n\tf1_score: {f1_mean:.3f}")

        df = pd.concat([df, pd.Series({'precision': p_mean, 'recall': r_mean, 'f1_score': f1_mean}).rename(hypothesis_txt_path.name)], axis=1)

    print(df)
    df.to_csv(RESULTS_PATH.joinpath(f'PARENT_{data_folder.value.name}-{partition.value}-{labels_as.value}.csv').open('w', encoding='utf8'))
    # Computing PARENT: 100%|██████████| 72831/72831 [00:03<00:00, 23760.92it/s]
    # precision: 0.797
    # recall: 0.450
    # f1_score: 0.553

    # validation on reference-to-reference of SQ-CSQA
    # precision: 1.000
    # recall: 0.827
    # f1_score: 0.898

    # validation on ref-2-ref of SQ-CSQA-Direct
    # precision: 1.000
    # recall: 0.873
    # f1_score: 0.924

    # validation on hypothesis_val-T5-BASE-LABEL.txt
    # precision: 0.657
    # recall: 0.403
    # f1_score: 0.445

    # validation on hypothesis_val-T5-WHYN-LABEL.txt
    # precision: 0.626
    # recall: 0.389
    # f1_score: 0.429

    # validation on hypothesis_val-T5-SMALL-LABEL.txt
    # precision: 0.665
    # recall: 0.398
    # f1_score: 0.445

    # validation on hypothesis_val-T5-3B-LABEL.txt
    # precision: 0.605
    # recall: 0.330
    # f1_score: 0.377