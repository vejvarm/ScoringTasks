from pathlib import Path
from moverscore_v2 import get_idf_dict, word_mover_score
import statistics

from flags import DataPath

# TODO: finish test for individual metrics
#   - moverscore: working (gpu 30%, gpu memory expensive, medium time)
#   - bertscore: working (gpu 90%, slow)
#   - bleurt: wip (gpu 100%, slow)
#   - parent: working (efficient, fast)
# TODO: collectively load data once and then run all metrics on those ... no need to load all again right?

if __name__ =='__main__':
    root = Path('./data')
    references = [r.strip('\n') for r in  open(DataPath.REFERENCE.value()).readlines()]
    translations =[t.strip('\n') for t in  open(DataPath.HYPOTHESIS.value()).readlines()]
    print(f'ref len:{len(references)}')
    print(f'hyp len:{len(translations)}')
    idf_dict_hyp = get_idf_dict(translations)
    idf_dict_ref = get_idf_dict(references)
    #reference is a list of reference sentences
    scores = word_mover_score(references, translations, idf_dict_ref, idf_dict_hyp, stop_words=[], n_gram=1, remove_subwords=True)
    print(statistics.mean(scores))
