# WORKING
from parent import parent
import json

from flags import DataPath
from data_loader import TableDataLoader

if __name__ == "__main__":
    tdl = TableDataLoader()
    reference, hypothesis, table = tdl.load_tokenized()

    precision, recall, f_score = parent(
        hypothesis,
        reference,
        table,
        avg_results=True,  # if False, returns list with individual scores for each (hyp, ref) pair
        n_jobs=32,
        use_tqdm=True,
    )

    print(f"precision: {precision:.3f}\nrecall: {recall:.3f}\nf_score: {f_score:.3f}")
