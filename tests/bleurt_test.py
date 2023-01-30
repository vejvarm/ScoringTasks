# WORKING
import time
import statistics
from bleurt import score

from flags import ModelPath
from data_loader import DataLoader

if __name__ == "__main__":
    dl = DataLoader()
    reference, hypothesis = dl.load()

    checkpoint = ModelPath.BLEURT_20.value

    tic = time.perf_counter()
    scorer = score.BleurtScorer(checkpoint)  # BLEURT-tiny if None # TODO: how about the bleurt/test_checkpoint from DART?
    scores = scorer.score(references=reference, candidates=hypothesis)
    assert isinstance(scores, list)

    print(f"time: {time.perf_counter() - tic} s")  # time: 3401.973938458017 s
    print(f"score (mean): {statistics.mean(scores)}")  # score (mean): 0.6011461934407719

    # validation on reference-to-reference of SQ-CSQA
    # score (mean): 0.9714486217386369

    # validation on simple_test hypothesis_val_T5_BASE-LABEL.txt
    # score (mean): 0.6491061105177953