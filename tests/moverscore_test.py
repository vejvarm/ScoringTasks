import time

from moverscore_v2 import get_idf_dict, word_mover_score
import statistics

from data_loader import DataLoader

if __name__ == '__main__':
    dl = DataLoader()
    reference, hypothesis = dl.load()

    print(f'ref len:{len(reference)}')
    print(f'hyp len:{len(hypothesis)}')

    if not reference or not hypothesis:
        raise ValueError('Either reference or hypothesis list is empty')

    tic = time.perf_counter()
    idf_dict_hyp = get_idf_dict(hypothesis)
    idf_dict_ref = get_idf_dict(reference)
    # reference is a list of reference sentences
    scores = word_mover_score(reference, hypothesis, idf_dict_ref, idf_dict_hyp, stop_words=[], n_gram=1, batch_size=64,
                              remove_subwords=True)

    print(f"time: {time.perf_counter() - tic}")
    print(f"score (mean): {statistics.mean(scores)}")  # 0.6850683179056607

    # validation on reference-to-reference of SQ-CSQA
    # score (mean): 0.9996483734041332

    # validation on simple_test hypothesis_val_T5_BASE-LABEL.txt
    # score (mean): 0.6699021602081112