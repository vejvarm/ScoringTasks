# WORKING, needs data loading
from bleurt import score

from flags import ModelPath
from data_loader import DataLoader

if __name__ == "__main__":
    dl = DataLoader()
    reference, hypothesis = dl.load()

    checkpoint = ModelPath.BLEURT_20.value

    scorer = score.BleurtScorer(checkpoint)  # BLEURT-tiny if None # TODO: how about the bleurt/test_checkpoint from DART?
    scores = scorer.score(references=reference, candidates=hypothesis)
    assert isinstance(scores, list)
    print(scores)
