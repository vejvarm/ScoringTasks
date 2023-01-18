# WORKING
from bert_score import score

from data_loader import DataLoader

if __name__ == "__main__":
    dl = DataLoader()
    reference, hypothesis = dl.load()

    P, R, F1 = score(hypothesis, reference, model_type="roberta-large", lang='en', verbose=True, idf=True,
                     rescale_with_baseline=True)

    # print(P, R, F1)  # tensors for each metric (len of data)
    print(f"precision: {P.mean():.3f}\nrecall: {R.mean():.3f}\nf1_score: {F1.mean():.3f}")
    # 100%|██████████| 1138/1138 [00:06<00:00, 186.85it/s]
    # done in 280.88 seconds, 259.29 sentences/sec
    # P: 0.724
    # R: 0.482
    # F1: 0.597