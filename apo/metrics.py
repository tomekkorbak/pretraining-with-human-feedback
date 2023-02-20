import os
from typing import Any, Union
from abc import ABC
from collections import Counter
from multiprocessing import Pool

import numpy as np
from scipy.stats import entropy
from wandb import Table
from wandb.data_types import WBValue
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


MetricOutput = dict[str, Union[float, int, WBValue]]


class Metric(ABC):

    @classmethod
    def from_config(cls, config: dict[str, Any]):
        class_name = config.pop('class_name')
        return globals()[class_name](**config)

    def score_texts(self, texts: list[str]) -> MetricOutput:
        raise NotImplementedError('A subclass of Metric must implement score_texts')


class Length(Metric):
    name = 'length'

    def score_texts(self, texts: list[str]) -> MetricOutput:
        lenghts = [len(text) for text in texts]
        return {self.name: np.mean(lenghts)}


class NGramStats(Metric):

    def __init__(self, n: int, log_tables: bool = False):
        self.n = n
        self.tokenize = lambda x: x.split()
        self.log_tables = log_tables

    def score_texts(self, texts: list[str]) -> MetricOutput:
        batch_ngram_counts = Counter()
        distinct_ngrams_ratios = []
        for text in texts:
            ngrams_in_text = self._get_ngrams(self.tokenize(text))
            distinct_ngrams_ratio = len(set(ngrams_in_text)) / max(len(ngrams_in_text), 1)
            distinct_ngrams_ratios.append(distinct_ngrams_ratio)
            for ngram in ngrams_in_text:
                batch_ngram_counts[ngram] += 1
        ngram_entropy = entropy(list(batch_ngram_counts.values()))
        logs = {
            f'distinct-{self.n}-grams': sum(distinct_ngrams_ratios) / max(len(distinct_ngrams_ratios), 1),
            f'entropy-{self.n}-grams': ngram_entropy,
        }
        if self.log_tables:
            logs[f'distinct-{self.n}-grams_in_batch'] = len(batch_ngram_counts)
            logs[f'{self.n}-gram counts'] = Table(
                columns=['ngram', 'count', 'rank'],
                data=[(ngram, count, rank) for rank, (ngram, count) in enumerate(batch_ngram_counts.most_common())]
            )
        return logs

    def _get_ngrams(self, token_list: list[str]):
        return list(zip(*[token_list[i:] for i in range(self.n)]))


class SelfBlEU(Metric):

    def __init__(self, n=5):
        """
        Corpus level diversity metric. See https://arxiv.org/abs/1802.01886 for more details.
        """
        self.n = n
        self.name = f'Self-BLEU-{n}'
        self.weight = tuple((1. / self.n for _ in range(self.n)))

    def score_texts(self, texts: list[str]) -> MetricOutput:
        pool = Pool(os.cpu_count())
        results = list()
        for i in range(len(texts)):
            hypothesis = texts[i]
            references = texts[:i] + texts[i+1:]
            args = (
                references[:200],
                hypothesis,
                self.weight,
                SmoothingFunction().method1
            )
            results.append(pool.apply_async(self._score_fn, args))
        scores = [handle.get() for handle in results]
        scores = [score for score in scores if score is not None]
        pool.close()
        pool.join()
        if len(scores) > 0:
            return {self.name: sum(scores) / len(scores)}
        else:
            return {self.name: float('nan')}

    def _score_fn(self, references, hypothesis, weight, smoothing_fn):
        try:
            return sentence_bleu(references, hypothesis, weight, smoothing_fn)
        except:
            return None
