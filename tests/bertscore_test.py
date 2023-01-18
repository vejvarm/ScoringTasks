from bert_score import score
import json

from flags import DataPath
from data_loader import DataLoader

if __name__ == "__main__":
    dl = DataLoader()
    reference, hypothesis = dl.load()

    print(reference[0])
    print(hypothesis[0])

    P, R, F1 = score(hypothesis, reference, model_type="roberta-large", lang='en', verbose=True, idf=True)

    print(P, R, F1)
