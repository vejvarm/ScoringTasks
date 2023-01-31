# WORKING
import time
import statistics
from bleurt import score

from flags import ModelPath
from data_loader import DataLoader

if __name__ == "__main__":
    dl = DataLoader()
    reference, hypothesis = dl.load()

    print(f'ref len:{len(reference)}')
    print(f'hyp len:{len(hypothesis)}')

    if not reference or not hypothesis:
        raise ValueError('Either reference or hypothesis list is empty')

    checkpoint = ModelPath.BLEURT_20.value  # NOTE: how about the bleurt/test_checkpoint from DART?

    tic = time.perf_counter()
    scorer = score.BleurtScorer(checkpoint)  # BLEURT-tiny if None
    scores = scorer.score(references=reference, candidates=hypothesis)
    assert isinstance(scores, list)

    print(f"time: {time.perf_counter() - tic} s")  # time: 3401.973938458017 s
    print(f"score (mean): {statistics.mean(scores)}")  # score (mean): 0.6011461934407719
