from typing import Any, Generator, Optional
import random

import torch
from torch.utils.data import IterableDataset
from torch.utils.data.datapipes.iter.combinatorics import ShufflerIterDataPipe
from datasets import load_dataset


class ConstantLengthDataset(IterableDataset):
    """
    Iterable dataset that returns constant length chunks of tokens from stream of text files.

    Based on https://github.com/huggingface/transformers/blob/main/examples/research_projects/codeparrot/scripts/codeparrot_training.py
    """

    def __init__(
        self,
        tokenizer,
        datasets: list[str],
        seq_length: int = 1024,
        num_of_sequences: int = 1024,
        chars_per_token: float = 3.6,
        is_split_by_sentences: bool = False,
        concat_token: Optional[str] = None,
        conditional_training_config: Optional[dict[str, Any]] = None,
        filter_threshold: Optional[float] = None,
        skip_tokens: int = 0,
    ):
        self.tokenizer = tokenizer
        self.concat_token = concat_token or tokenizer.eos_token
        self.filter_threshold = filter_threshold
        self.conditional_training = conditional_training_config is not None
        if self.conditional_training:
            self.conditional_training_threshold = conditional_training_config.get('threshold')
            self.aligned_prefix = conditional_training_config.get('aligned_prefix')
            print(f'Setting aligned prefix {self.aligned_prefix} ({self.tokenizer(self.aligned_prefix).input_ids})')
            self.misaligned_prefix = conditional_training_config.get('misaligned_prefix')
            print(f'Setting misaligned prefix {self.misaligned_prefix} '
                  f'({self.tokenizer(self.misaligned_prefix).input_ids})')
            self.drop_token_fraction = conditional_training_config.get('drop_token_fraction', 0)
        self.datasets = datasets
        self.seq_length = seq_length
        self.current_size = 0
        self.num_docs = 0
        self.is_split_by_sentences = is_split_by_sentences
        self.max_buffer_size = seq_length * chars_per_token * num_of_sequences
        self.skip_tokens = skip_tokens

    @property
    def tokens_used(self) -> int:
        return self.current_size * self.seq_length

    def __iter__(self):
        for dataset_name in self.datasets:
            print(f'Starting processing examples from dataset {dataset_name}')
            dataset = load_dataset(dataset_name, split='train', streaming=True)
            iterator = iter(dataset)
            more_examples = True
            while more_examples:
                text_buffer, score_buffer, buffer_len = [], [], 0
                while True:
                    if buffer_len >= self.max_buffer_size:
                        break
                    try:
                        document = next(iterator)
                        if not self._should_include(document):
                            continue
                        self.num_docs += 1
                        for raw_text, score in self._process_document(document):
                            text_buffer.append(raw_text)
                            score_buffer.append(score)
                            buffer_len += len(raw_text)
                    except StopIteration:
                        more_examples = False
                        break
                tokenized_inputs = self.tokenizer(text_buffer, truncation=False)["input_ids"]
                all_token_ids, all_token_scores = [], []
                for tokenized_input, score in zip(tokenized_inputs, score_buffer):
                    all_token_ids.extend(tokenized_input)
                    all_token_scores.extend([score] * len(tokenized_input))
                for i in range(0, len(all_token_ids), self.seq_length):
                    input_ids = all_token_ids[i : i + self.seq_length]
                    token_scores = all_token_scores[i : i + self.seq_length]
                    if len(input_ids) == self.seq_length:
                        self.current_size += 1
                        if self.skip_tokens > self.tokens_used:
                            if self.tokens_used % (self.seq_length * 1e5) == 0:
                                print(f'Skipping {self.tokens_used:2.4e} tokens')
                            continue
                        yield {
                            'input_ids': torch.tensor(input_ids),
                            'labels': torch.tensor(input_ids.copy()),
                            'token_scores': torch.tensor(token_scores),
                        }

    def _process_document(self, document: dict[str, Any]) -> Generator[tuple[str, float], None, None]:
        if self.is_split_by_sentences:
            for i, (sent, score) in enumerate(zip(document['texts'], document["scores"])):
                if i == 0:
                    # first sent of a document
                    text = self.concat_token + self._process_raw_text(sent, score)
                else:
                    text = self._process_raw_text(sent, score)
                yield text, score
        else:
            text = self.concat_token + document['text']
            yield text, document["score"]

    def _process_raw_text(self, text: str, score: float) -> str:
        if self.conditional_training and random.random() > self.drop_token_fraction:
            if score <= self.conditional_training_threshold:
                return self.aligned_prefix + text
            else:
                return self.misaligned_prefix + text
        else:
            return text

    def _should_include(self, document: dict[str, Any]) -> bool:
        if self.filter_threshold is None or self.skip_tokens > self.tokens_used:
            return True
        return document['avg_score'] <= self.filter_threshold

    def shuffle(self, buffer_size: int = 1000) -> ShufflerIterDataPipe:
        return ShufflerIterDataPipe(self, buffer_size=buffer_size)
