import re
from typing import Any, Generator
from functools import reduce
from operator import getitem
from pprint import pprint
import logging
from contextlib import contextmanager

import torch
from datasets import Dataset
from transformers.generation_logits_process import LogitsProcessor
import numpy as np


def override_config(config: dict[str, Any], params_to_override: str) -> None:
    for key_value_pair in params_to_override:
        key, value = key_value_pair.split('=')
        key_path = key.split('.')  # nested dict lookup
        value = value if bool(re.search(r"[^\.0-9 ]", value)) and value not in ["True","False", "None"] else eval(value)
        innermost_dict = reduce(getitem, key_path[:-1], config)
        innermost_dict[key_path[-1]] = value
    print(f'Configs after overriding:')
    pprint(config)


def unflatten_config(config: dict[str, Any]) -> dict[str, Any]:
    """
    Fix a bug in wandb's handling of nested configs in sweeps:
    https://github.com/wandb/client/issues/982
    """
    for key, value in config.items():
        if '.' in key:
            outer_key, inner_key = key.split('.')
            config[outer_key][inner_key] = value
    print(f'Configs for this sweep run:')
    pprint(config)
    return config


def merge_configs(config1: dict[str, Any], config2: dict[str, Any]) -> Generator[tuple[str, Any], None, None]:
    """
    If necessary, overrides config1 with config2.
    """

    for key in set(config1.keys()).union(config2.keys()):
        if key in config1 and key in config2:
            if isinstance(config1[key], dict) and isinstance(config2[key], dict):
                yield key, dict(merge_configs(config1[key], config2[key]))
            else:
                yield key, config2[key]
        elif key in config1:
            yield key, config1[key]
        else:
            yield key, config2[key]


def get_max_at_k(scores: list[int], k: int) -> np.ndarray:
    """
    Average maximum value of a k-element chunk of list `elements`. Useful for computing expected maximum toxicity as in
    RealToxicityPrompts (https://arxiv.org/pdf/2009.11462.pdf).
    """
    num_chunks = len(scores) // k
    chunked_scores = np.asarray(scores[:num_chunks*k]).reshape(k, -1)
    return np.max(chunked_scores, axis=0).mean()


@contextmanager
def all_logging_disabled(highest_level=logging.CRITICAL):
    """
    A context manager that will prevent any logging messages triggered during the body from being processed.
    Adapted from https://gist.github.com/simon-weber/7853144
    """
    previous_level = logging.root.manager.disable
    logging.disable(highest_level)
    try:
        yield
    finally:
        logging.disable(previous_level)


def print_dataset_stats(dataset: Dataset, threshold: float) -> None:
    df = dataset.to_pandas()
    df['length'] = df.text.apply(len)
    aligned_df = df[df.score <= threshold]
    misaligned_df = df[df.score > threshold]
    print('Loaded dataset with the following stats:')
    print(f'mean score: {df.score.mean():.3f}')
    print(f'mean score of aligned part ({len(aligned_df)} samples): '
          f'{(aligned_df.score * aligned_df.length).sum() / aligned_df.length.sum():.3f}')
    print(
        f'mean score of misaligned part ({len(misaligned_df)} samples): '
        f'{(misaligned_df.score * misaligned_df.length).sum() / misaligned_df.length.sum():.3f}')


def entropy_from_logits(logits):
    probs = torch.nn.functional.softmax(logits, dim=-1)
    entropy = torch.logsumexp(logits, axis=-1) - torch.sum(probs*logits, axis=-1)
    return entropy


def get_theoretical_loss(num_params, num_tokens):
    # loss as a function of params and data according to the Chinchilla scaling law
    # cf. eqn 10, https://arxiv.org/pdf/2203.15556.pdf
    return 1.69 + 406.4 / (num_params ** 0.34) + 410.7 / (num_tokens ** 0.28)


class CustomMinLengthLogitsProcessor(LogitsProcessor):
    def __init__(self, min_length: int, eos_token_id: int):
        self.min_length = min_length
        self.eos_token_id = eos_token_id
        self.prompt_lengths = None

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self.prompt_lengths is None:
            self.prompt_lengths = (input_ids == self.eos_token_id).sum(dim=1)
        cur_len = input_ids.shape[-1]
        for i in range(scores.shape[0]):
            if cur_len - self.prompt_lengths[i] < self.min_length:
                scores[i, self.eos_token_id] = -float("inf")
        return scores
