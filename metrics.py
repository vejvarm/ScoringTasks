import statistics
import time
import pandas as pd
from pathlib import Path
from typing import Iterable

from parent import parent
from bert_score import score as bertscore
from moverscore_v2 import get_idf_dict, word_mover_score
from bleurt import score as bleurtscore

from flags import RESULTS_PATH, Partition, DataFolder, LabelsAs, ModelPath
from data_loader import DataLoader, TableDataLoader


def run_parent(data_folder: DataFolder,
               partition: Partition,
               labels_as: LabelsAs,
               HypothesisFiles: Iterable,
               reference_file: Path,
               table_file: Path):
    print('Running PARENT metric:')
    df = pd.DataFrame()
    for hypothesis_txt_path in HypothesisFiles:
        print(hypothesis_txt_path.value.name)
        tdl = TableDataLoader(reference_file, hypothesis_txt_path.value, table_file)
        reference, hypothesis, table = tdl.load_tokenized()

        if not reference or not hypothesis or not table:
            continue

        tic = time.perf_counter()
        precision, recall, f_score = parent(
            hypothesis,
            reference,
            table,
            avg_results=False,  # if False, returns list with individual scores for each (hyp, ref) pair
            n_jobs=-1,
            use_tqdm=True,
        )

        p_mean = statistics.mean(precision)
        r_mean = statistics.mean(recall)
        f1_mean = statistics.mean(f_score)

        print(f"\ttime: {time.perf_counter() - tic}")
        # print(f"\tprecision: {p_mean:.3f}\n\trecall: {r_mean:.3f}\n\tf1_score: {f1_mean:.3f}")

        df = pd.concat([df, pd.Series({'precision': p_mean, 'recall': r_mean, 'f1_score': f1_mean}).rename(hypothesis_txt_path.name)], axis=1)

    result_file = RESULTS_PATH.joinpath(f'PARENT_{data_folder.value.name}-{partition.value}-{labels_as.value}.csv')
    df.to_csv(result_file.open('w', encoding='utf8'))
    print(f"PARENT results saved to '{result_file.parent.name}/{result_file.name}'")


def run_bertscore(data_folder: DataFolder,
                  partition: Partition,
                  labels_as: LabelsAs,
                  HypothesisFiles: Iterable,
                  reference_file: Path):
    print('Running BERTScore metric:')
    df = pd.DataFrame()
    for hypothesis_txt_path in HypothesisFiles:
        print(hypothesis_txt_path.value.name)
        dl = DataLoader(reference_file, hypothesis_txt_path.value)
        reference, hypothesis = dl.load()

        if not reference or not hypothesis:
            continue

        tic = time.perf_counter()
        precision, recall, f_score = bertscore(hypothesis, reference, model_type="roberta-large", lang='en', verbose=True, idf=True,
                         rescale_with_baseline=True)

        p_mean = precision.mean().cpu().numpy()
        r_mean = recall.mean().cpu().numpy()
        f1_mean = f_score.mean().cpu().numpy()
        print(f"\ttime: {time.perf_counter() - tic}")
        # print(f"\tprecision: {p_mean:.3f}\n\trecall: {r_mean:.3f}\n\tf1_score: {f1_mean:.3f}")

        df = pd.concat([df, pd.Series({'precision': p_mean, 'recall': r_mean, 'f1_score': f1_mean}).rename(hypothesis_txt_path.name)], axis=1)

    result_file = RESULTS_PATH.joinpath(f'BERTScore_{data_folder.value.name}-{partition.value}-{labels_as.value}.csv')
    df.to_csv(result_file.open('w', encoding='utf8'))
    print(f"BERTScore results saved to '{result_file.parent.name}/{result_file.name}'")


def run_moverscore(data_folder: DataFolder,
                   partition: Partition,
                   labels_as: LabelsAs,
                   HypothesisFiles: Iterable,
                   reference_file: Path):
    print('Running MoverScore metric:')
    df = pd.DataFrame()
    for hypothesis_txt_path in HypothesisFiles:
        print(hypothesis_txt_path.value.name)
        dl = DataLoader(reference_file, hypothesis_txt_path.value)
        reference, hypothesis = dl.load()

        if not reference or not hypothesis:
            continue

        tic = time.perf_counter()
        idf_dict_hyp = get_idf_dict(hypothesis)
        idf_dict_ref = get_idf_dict(reference)
        # reference is a list of reference sentences
        scores = word_mover_score(reference, hypothesis, idf_dict_ref, idf_dict_hyp, stop_words=[], n_gram=1,
                                  batch_size=64,
                                  remove_subwords=True)
        assert isinstance(scores, list)

        s_mean = statistics.mean(scores)
        print(f"\ttime: {time.perf_counter() - tic}")
        # print(f"\tscore (mean): {s_mean}")

        df = pd.concat([df, pd.Series({'Mscore': s_mean}).rename(hypothesis_txt_path.name)], axis=1)

    result_file = RESULTS_PATH.joinpath(f'MoverScore_{data_folder.value.name}-{partition.value}-{labels_as.value}.csv')
    df.to_csv(result_file.open('w', encoding='utf8'))
    print(f"MoverScore results saved to '{result_file.parent.name}/{result_file.name}'")


def run_bleurt(data_folder: DataFolder,
               partition: Partition,
               labels_as: LabelsAs,
               HypothesisFiles: Iterable,
               reference_file: Path):
    print('Running BLEURT metric:')
    df = pd.DataFrame()
    for hypothesis_txt_path in HypothesisFiles:
        print(hypothesis_txt_path.value.name)
        dl = DataLoader(reference_file, hypothesis_txt_path.value)
        reference, hypothesis = dl.load()

        if not reference or not hypothesis:
            continue

        checkpoint = ModelPath.BLEURT_20.value

        tic = time.perf_counter()
        scorer = bleurtscore.BleurtScorer(checkpoint)
        scores = scorer.score(references=reference, candidates=hypothesis)
        assert isinstance(scores, list)

        s_mean = statistics.mean(scores)
        print(f"\ttime: {time.perf_counter() - tic}")
        # print(f"\tscore (mean): {s_mean}")

        df = pd.concat([df, pd.Series({'BLscore': s_mean}).rename(hypothesis_txt_path.name)], axis=1)

    result_file = RESULTS_PATH.joinpath(f'BLEURT_{data_folder.value.name}-{partition.value}-{labels_as.value}.csv')
    df.to_csv(result_file.open('w', encoding='utf8'))
    print(f"BLEURT results saved to '{result_file.parent.name}/{result_file.name}'")
