from abc import ABC
from typing import Optional, Any, Union, List, Dict
from dataclasses import dataclass
from time import sleep
from pathlib import Path
import os

import torch
import torch.nn.functional as F
import numpy as np
import openai
from transformers import PreTrainedModel, PreTrainedTokenizer, AutoModelForCausalLM, AutoTokenizer
from tqdm import trange
import srsly

CACHE_DIR = Path.home() / '.kl_gpt3/'


@dataclass
class Batch:
    model_name: str
    texts: List[str]
    logprobs: Optional[np.ndarray] = None
    token_logprobs: Optional[List[List[float]]] = None

    def __len__(self):
        return len(self.texts)

    def __add__(self, other):
        assert self.model_name == other.model_name
        if self.logprobs is not None and other.logprobs is not None:
            merged_logprobs = np.concatenate([self.logprobs, other.logprobs], axis=0)
        elif self.logprobs is None and other.logprobs is None:
            merged_logprobs = None
        else:
            raise TypeError()
        return Batch(
            texts=self.texts + other.texts,
            model_name=self.model_name,
            logprobs=merged_logprobs
        )

    def save_to_json(self, json_path: Union[str, Path]):
        content = {
            'model_name': self.model_name,
            'texts': self.texts,
            'logprobs': self.logprobs.tolist(),
        }
        srsly.write_json(json_path, content)

    @classmethod
    def load_from_json(cls, json_path: Union[str, Path]):
        content = srsly.read_json(json_path)
        content['logprobs'] = np.asarray(content['logprobs'])
        return cls(**content)


class LanguageModel(ABC):

    def get_logprobs(self: Batch) -> np.ndarray:
        pass

    def sample(self, num_samples: int = 32, save_logprobs: bool = True) -> Batch:
        pass


class GPT3(LanguageModel):
    model_name: str = "text-davinci-002"
    max_tokens: int = 16
    total_tokens_used: int = 0
    batch_size: 8

    def __init__(self, model_name: Optional[str] = "text-davinci-002", max_tokens: int = 16, batch_size: int = 8):
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.total_tokens_used = 0
        self.batch_size = batch_size
        if os.environ.get('OPENAI_API_KEY') is None:
            raise ValueError('Please set the OPENAI_API_KEY environment variable.')
        openai.api_key = os.getenv("OPENAI_API_KEY")

    def get_logprobs(self, batch: Batch) -> np.ndarray:
        assert all(len(text) > 0 for text in batch.texts)
        sequence_logprobs: List[np.ndarray] = []
        for i in trange(0, len(batch), self.batch_size):
            current_indices = slice(i, i + self.batch_size)
            response = openai.Completion.create(
                model=self.model_name,
                prompt=batch.texts[current_indices],
                max_tokens=0,
                temperature=1,
                logprobs=1,
                echo=True
            )
            self.total_tokens_used += response.usage.total_tokens
            token_logprobs = [response.choices[j].logprobs.token_logprobs[1:] for j in range(self.batch_size)]
            sequence_logprobs += [np.asarray(logprobs).sum() for logprobs in token_logprobs]
        return np.stack(sequence_logprobs, axis=0)

    def sample(self, num_samples: int = 32, save_logprobs: bool = True) -> Batch:
        batch = Batch(model_name=self.model_name, texts=[], logprobs=[] if save_logprobs else None)
        for _ in trange(num_samples // self.batch_size or 1):
            minibatch_size = min(self.batch_size, num_samples)
            while True:
                try:
                    response = openai.Completion.create(
                        model=self.model_name,
                        n=minibatch_size,
                        temperature=1,
                        logprobs=1 if save_logprobs else None,
                        echo=True,
                        max_tokens=self.max_tokens
                    )
                except openai.error.RateLimitError as exc:
                    sleep(30)
                    print(f'Sleeping because of rate limit error: {exc}')
                else:
                    break
            self.total_tokens_used += response.usage.total_tokens
            print(f'Total tokens used: {self.total_tokens_used}')
            texts = [response.choices[i].text for i in range(minibatch_size)]
            if save_logprobs:
                token_logprobs = [response.choices[i].logprobs.token_logprobs[1:] for i in range(minibatch_size)]
                sequence_logprobs = [np.asarray(logprobs).sum() for logprobs in token_logprobs]
                logprobs = np.stack(sequence_logprobs, axis=0)
            else:
                logprobs = None
                token_logprobs = None
            batch += Batch(
                model_name=self.model_name,
                texts=texts,
                logprobs=logprobs,
                token_logprobs=token_logprobs
            )
        return batch


class HFModel(LanguageModel):

    def __init__(
            self,
            hf_model: PreTrainedModel,
            hf_tokenizer: Optional[PreTrainedTokenizer] = None,
            model_name: Optional[str] = None,
            max_tokens: Optional[int] = 128,
            prefix: Optional[str] = None,
            should_insert_prefix: Optional[bool] = False,
            generate_batch_size: Optional[int] = 16,
            eval_batch_size: Optional[int] = 32,
            device: Optional[Union[str, torch.device]] = None,
    ):
        self.hf_model = hf_model
        self.model_name = model_name or hf_model.name_or_path
        self.hf_tokenizer = hf_tokenizer or AutoTokenizer.from_pretrained(self.model_name)
        self.max_tokens = max_tokens
        self.prefix = prefix
        self.should_insert_prefix = should_insert_prefix
        self.generate_batch_size = generate_batch_size
        self.eval_batch_size = eval_batch_size
        self.device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        self.total_tokens_used = 0
        self.hf_model.to(self.device)
        if self.hf_tokenizer.pad_token is None:
            self.hf_tokenizer.pad_token = self.hf_tokenizer.eos_token
            self.hf_tokenizer.pad_token_id = self.hf_tokenizer.eos_token_id

    @classmethod
    def from_pretrained(
            cls,
            model_name: str,
            tokenizer_name: Optional[str] = None,
            device: Optional[Union[str, torch.device]] = None,
            model_kwargs: Optional[Dict[str, Any]] = {},
            **kwargs
    ) -> 'HFModel':
        return HFModel(
            model_name=model_name,
            hf_model=AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs),
            hf_tokenizer=AutoTokenizer.from_pretrained(model_name or tokenizer_name),
            device=device,
            **kwargs
        )

    def sample(self, num_samples: int = 32, save_logprobs: bool = True) -> Batch:
        assert num_samples % self.generate_batch_size == 0 or num_samples < self.generate_batch_size
        batch = Batch(model_name=self.model_name, texts=[], logprobs=[] if save_logprobs else None)
        if self.prefix is not None:
            inputs = self.hf_tokenizer([self.hf_tokenizer.eos_token+self.prefix]*self.generate_batch_size, return_tensors='pt').to(self.device).input_ids
        else:
            inputs = None
        for _ in trange(num_samples // self.generate_batch_size or 1):
            output = self.hf_model.generate(
                inputs=inputs,
                do_sample=True,
                top_k=0,
                top_p=1,
                min_length=3,
                num_return_sequences=self.generate_batch_size if self.prefix is None else 1,
                max_length=self.max_tokens,
                return_dict_in_generate=True,
                output_scores=save_logprobs,
                pad_token_id=self.hf_tokenizer.pad_token_id
            )
            texts = self.hf_tokenizer.batch_decode(output.sequences, skip_special_tokens=False)
            if self.prefix is not None:
                texts = [text.replace(self.prefix, '') for text in texts]
            if save_logprobs:
                logits = torch.stack(output.scores, dim=1)
                attention_mask = output.sequences != self.hf_tokenizer.pad_token_id
                start_token_id = 1 if self.prefix is None else inputs.size(1)
                logprobs = self._get_logprobs_from_logits(
                    input_ids=output.sequences[:, start_token_id:, None],
                    logits=logits,
                    mask=attention_mask[:, start_token_id:]
                ).cpu().numpy()
            else:
                logprobs = None
            batch += Batch(model_name=self.model_name, texts=texts, logprobs=logprobs)
        return batch

    def get_logprobs(self, batch: Batch) -> np.ndarray:
        logprobs: List[np.ndarray] = []
        for i in trange(0, len(batch), self.eval_batch_size):
            current_indices = slice(i, i + self.eval_batch_size)
            if self.prefix is not None:
                if self.should_insert_prefix:
                    texts = [('\n'+self.prefix).join(text.split('\n'))
                             for text in batch.texts[current_indices]]
                else:
                    texts = batch.texts[current_indices]
                texts = [f'{self.hf_tokenizer.eos_token}{self.prefix}{text.removeprefix(self.hf_tokenizer.eos_token)}'
                         for text in texts]
            else:
                texts = batch.texts[current_indices]
            inputs = self.hf_tokenizer(
                text=texts,
                padding=True,
                max_length=self.max_tokens,
                return_tensors="pt"
            ).to(self.device)
            with torch.inference_mode():
                logits = self.hf_model.forward(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask
                ).logits
                mask = inputs.attention_mask
                # for token in self.hf_tokenizer.additional_special_tokens_ids:
                #     mask = torch.where(inputs.input_ids == token, torch.zeros_like(mask), mask)
                # mask = torch.where(inputs.input_ids == 199, torch.zeros_like(mask), mask)
                logprobs_minibatch = self._get_logprobs_from_logits(
                    input_ids=inputs.input_ids[:, 1:, None],
                    logits=logits[:, :-1],
                    mask=mask[:, :-1]
                ).cpu().numpy()
            logprobs.append(logprobs_minibatch)
        return np.concatenate(logprobs, axis=0)

    def _get_logprobs_from_logits(self, input_ids: torch.LongTensor, logits: torch.FloatTensor,
                                  mask: torch.LongTensor) -> torch.FloatTensor:
        log_probs = F.log_softmax(logits, dim=2)
        input_token_logprobs = log_probs.gather(2, input_ids).squeeze(dim=2)
        # masking out logprobs of padding tokens
        input_token_logprobs = torch.where(mask.bool(), input_token_logprobs, torch.zeros_like(input_token_logprobs))
        return input_token_logprobs.double().sum(dim=1)


def evaluate_forward_kl(
        hf_model: PreTrainedModel,
        hf_tokenizer: Optional[PreTrainedTokenizer] = None,
        hf_model_name: Optional[str] = None,
        hf_prefix: Optional[str] = None,
        should_insert_prefix: Optional[bool] = False,
        gpt3_batch: Optional[Batch] = None,
        num_samples: int = 1024,
        max_tokens: int = 32,
        use_cache: bool = True,
        gpt3_kwargs: Optional[Dict[str, Any]] = None,
):
    hf_model_wrapped = HFModel(
        hf_model=hf_model,
        hf_tokenizer=hf_tokenizer,
        model_name=hf_model_name,
        max_tokens=max_tokens,
        prefix=hf_prefix,
        should_insert_prefix=should_insert_prefix
    )
    gpt3 = GPT3(max_tokens=max_tokens, **(gpt3_kwargs or {}))
    if gpt3_batch is None:
        cache_file_name = CACHE_DIR / Path(f'{gpt3.model_name}_{gpt3.max_tokens}_tokens_cache.json')
        if use_cache and cache_file_name.exists():
            print(f'Loading GPT3 samples from cache {cache_file_name}')
            gpt3_batch = Batch.load_from_json(cache_file_name)
        else:
            print(f'Sampling {num_samples} sequences from GPT3')
            gpt3_batch = gpt3.sample(num_samples=num_samples, save_logprobs=True)
            cache_file_name.parent.mkdir(parents=True, exist_ok=True)
            print(f'Caching to {cache_file_name}')
            gpt3_batch.save_to_json(cache_file_name)
    hf_logprobs = hf_model_wrapped.get_logprobs(gpt3_batch)
    return (gpt3_batch.logprobs - hf_logprobs).mean()
