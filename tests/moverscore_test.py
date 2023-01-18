from moverscore_v2 import get_idf_dict, word_mover_score
import statistics

from data_loader import DataLoader

if __name__ == '__main__':
    dl = DataLoader()
    reference, hypothesis = dl.load()

    print(f'ref len:{len(reference)}')
    print(f'hyp len:{len(hypothesis)}')
    idf_dict_hyp = get_idf_dict(hypothesis)
    idf_dict_ref = get_idf_dict(reference)
    # reference is a list of reference sentences
    scores = word_mover_score(reference, hypothesis, idf_dict_ref, idf_dict_hyp, stop_words=[], n_gram=1, batch_size=64,
                              remove_subwords=True)
    print(statistics.mean(scores))  # 0.6850683179056607
