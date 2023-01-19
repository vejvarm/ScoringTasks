# WORKING
import time
from parent import parent
import json

from flags import DataPath
from data_loader import TableDataLoader

if __name__ == "__main__":
    tdl = TableDataLoader()
    reference, hypothesis, table = tdl.load_tokenized()

    tic = time.perf_counter()
    precision, recall, f_score = parent(
        hypothesis,
        reference,
        table,
        avg_results=True,  # if False, returns list with individual scores for each (hyp, ref) pair
        n_jobs=32,
        use_tqdm=True,
    )

    print(f"time: {time.perf_counter() - tic}")
    print(f"precision: {precision:.3f}\nrecall: {recall:.3f}\nf1_score: {f_score:.3f}")

    # Computing PARENT: 100%|██████████| 72831/72831 [00:03<00:00, 23760.92it/s]
    # precision: 0.797
    # recall: 0.450
    # f1_score: 0.553
