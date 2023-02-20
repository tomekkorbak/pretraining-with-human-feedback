from typing import Any, Optional
from dataclasses import dataclass, field
from random import choices
import time
import os

import numpy as np
import srsly
import wandb
from transformers import TrainerCallback, PreTrainedModel, PreTrainedTokenizer, TrainingArguments, TrainerState, \
    TrainerControl
from transformers.integrations import WandbCallback
from .kl_gpt3 import evaluate_forward_kl

from .scorers import Scorer, LMSamples
from .metrics import Metric
from .utils import get_max_at_k


@dataclass
class GenerationScenario:
    """
    A generation scenario encapsulates configuration a generation task, e.g. generation conditioned on prompts sampled
    from a particular set of prompts
    """
    name: str
    num_samples: int
    prompts: list[str] = None
    prefix: str = None
    token_type_id: int = None
    generate_kwargs: dict[str, Any] = field(default_factory=dict)
    num_hits_threshold: float = 0.5
    display_as_html: bool = False
    use_prompt_for_scoring: bool = False
    prompt_before_control: bool = False

    @classmethod
    def from_config(
            cls,
            name: str = None,
            prompts_path: str = None,
            num_samples: int = 32,
            prefix: str = None,
            token_type_id: int = None,
            generate_kwargs: dict[str, Any] = None,
            num_hits_threshold: float = 0.5,
            display_as_html: bool = False,
            use_prompt_for_scoring: bool = False,
            prompt_before_control: bool = False,
    ):
        if prompts_path is not None:
            prompts_data = srsly.read_jsonl(prompts_path)
            prompts = [prompt["text"] for prompt in prompts_data]
        else:
            prompts = None
        return cls(
            name=name,
            prompts=prompts,
            num_samples=num_samples,
            prefix=prefix,
            token_type_id=token_type_id,
            generate_kwargs=generate_kwargs,
            num_hits_threshold=num_hits_threshold,
            display_as_html=display_as_html,
            use_prompt_for_scoring=use_prompt_for_scoring,
            prompt_before_control=prompt_before_control,
        )


class CustomCallback(TrainerCallback):

    def __init__(self, *args, **kwargs):
        self.every_n_steps = kwargs.pop('every_n_steps', 1000)
        self.run_on_train_end = kwargs.pop('run_on_train_end', True)
        self.run_on_train_start = kwargs.pop('run_on_train_end', True)
        self.force_call_on = kwargs.pop('force_call_on', [])

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        **kwargs
    ):
        if self.run_on_train_start:
            self.run(args, state, control, model, tokenizer, **kwargs)

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        **kwargs
    ):
        if state.global_step % self.every_n_steps == 0 or state.global_step in self.force_call_on:
            self.run(args, state, control, model, tokenizer, **kwargs)

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        **kwargs
    ):
        if self.run_on_train_end:
            self.run(args, state, control, model, tokenizer, **kwargs)

    def run(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        **kwargs
    ):
        raise NotImplementedError


class SetupCallback(TrainerCallback):
    def on_train_begin(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            **kwargs
    ):
        assert not hasattr(state, 'tokens_seen')
        tokens_already_seen = kwargs.get('train_dataloader').dataset.datapipe.skip_tokens
        if len(state.log_history) > 0:
            assert tokens_already_seen > 0
            state.tokens_seen = state.log_history[-1]['tokens_seen']
            print(f'Found state.tokens_seen={state.tokens_seen:2.2e}')
        else:
            state.tokens_seen = tokens_already_seen
            print(f'Setting state.tokens_seen={state.tokens_seen:2.2e}')


class GenerateAndScoreCallback(CustomCallback):
    """
    A callback that generates samples from the model, scores them, and logs samples and scores to wandb
    """

    def __init__(self, scorer: Scorer, scenarios: list[GenerationScenario], metrics: list[Metric], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scorer = scorer
        self.scenarios = scenarios
        self.metrics = metrics
        self.batch_size = kwargs.pop('batch_size', 512)
        self.all_samples: dict[str, wandb.Table] = {}
        for scenario in self.scenarios:
            self.all_samples[f'generation/{scenario.name}/all_samples'] = wandb.Table(
                columns=['step', 'prompt', 'continuation', 'score']
            )

    def run(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        **kwargs
    ):
        self.generate_and_score(model, tokenizer, step=state.global_step)

    def generate_and_score(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        step: int = None
    ) -> dict[str, Any]:
        all_logs = {}
        for scenario in self.scenarios:
            start_time = time.time()
            samples = LMSamples()
            for i in range(scenario.num_samples // self.batch_size or 1):
                print(f'Generating samples, scenario {scenario.name}, batch {i+1} of '
                      f'{scenario.num_samples // self.batch_size}')
                samples += self.generate_and_score_for_scenario(model, tokenizer, scenario, num_samples=self.batch_size)
            prefix = f'generation/{scenario.name}'
            table = wandb.Table(
                columns=samples.column_names,
                data=list(samples if not scenario.display_as_html else samples.display_as_html())[:512]
            )
            logs = {
                f'{prefix}/current_samples': table,
                f'{prefix}/score': np.mean(samples.scores),
                f'{prefix}/score_max': np.max(samples.scores),
                f'{prefix}/score_max@25': get_max_at_k(samples.scores, k=25),
                f'{prefix}/num_hits': np.mean([score > scenario.num_hits_threshold for score in samples.scores]),
                f'{prefix}/samples_per_second': (len(samples) / (time.time() - start_time))
            }
            for metric in self.metrics:
                logs.update({
                    f'{prefix}/{name}': value
                    for name, value in metric.score_texts(texts=samples.continuations).items()
                })
            for sample_data in samples:
                self.all_samples[f'{prefix}/all_samples'].add_data(step, *sample_data)
            wandb.log(logs)
            all_logs.update(logs)
        return all_logs

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        **kwargs
    ):
        wandb.log(self.all_samples)

    def generate_and_score_for_scenario(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        scenario: GenerationScenario,
        num_samples: int
    ) -> LMSamples:
        # Step 1: prepare prompts
        if scenario.prompts is not None and scenario.prefix is not None:
            if scenario.prompt_before_control:
                prompts = [scenario.prefix + prompt for prompt in scenario.prompts]
            else:
                prompts = [prompt + scenario.prefix for prompt in scenario.prompts]
        elif scenario.prompts is not None:
            prompts = choices(scenario.prompts, k=num_samples)
        elif scenario.prefix is not None:
            prompts = [scenario.prefix] * num_samples
        else:
            prompts = [''] * num_samples
        tokenized_prompts = tokenizer(
            text=[tokenizer.eos_token + prompt for prompt in prompts],
            padding=True,
            truncation=True,
            max_length=tokenizer.model_max_length,
            return_tensors='pt'
        ).to(device=model.device)

        # Step 2: generate
        prompts_and_continuations = model.generate(
            inputs=tokenized_prompts['input_ids'],
            attention_mask=tokenized_prompts['attention_mask'],
            **scenario.generate_kwargs
        )
        prompts_and_continuations = tokenizer.batch_decode(prompts_and_continuations)
        continuations = [
            text.replace(tokenizer.eos_token, '').removeprefix(prompt)
            for prompt, text in zip(prompts, prompts_and_continuations)
        ]

        if tokenizer.aligned_prefix and tokenizer.misaligned_prefix:
            continuations = [ 
                text.replace(tokenizer.aligned_prefix, '').replace(tokenizer.misaligned_prefix, '') 
                for text in continuations
            ]    

        # Step 3: score
        lm_samples = LMSamples(prompts=prompts, continuations=continuations)
        lm_samples = self.scorer.score_samples(lm_samples, use_prompt_for_scoring=scenario.use_prompt_for_scoring)
        return lm_samples


class KLGPT3Callback(CustomCallback):

    def __init__(
            self,
            num_samples: int = 4096,
            max_tokens: int = 128,
            generate_batch_size: Optional[int] = 32,
            eval_batch_size: Optional[int] = 32,
            prefix: Optional[str] = None,
            should_insert_prefix: Optional[bool] = False,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.num_samples = num_samples
        self.max_tokens = max_tokens
        self.generate_batch_size = generate_batch_size
        self.eval_batch_size = eval_batch_size
        self.prefix = prefix
        self.should_insert_prefix = should_insert_prefix
        self.gpt3_kwargs = kwargs.get('gpt3_kwargs', {})
        if os.environ.get('OPENAI_API_KEY') is None:
            raise RuntimeError(
                'GenerateAndScoreCallback requires you to set OPENAI_API_KEY env variable. To obtain a token, go to '
                'https://beta.openai.com/account/api-keys'
            )

    def run(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        **kwargs
    ):
        was_in_training = model.training
        original_padding_side = tokenizer.padding_side
        model.eval()
        tokenizer.padding_side = 'right'
        forward_kl = evaluate_forward_kl(
            hf_model=model,
            hf_tokenizer=tokenizer,
            max_tokens=self.max_tokens,
            num_samples=self.num_samples,
            hf_prefix=self.prefix,
            should_insert_prefix=self.should_insert_prefix,
            gpt3_kwargs=self.gpt3_kwargs,
        )
        wandb.log({'KL/KL(GPT3, model)': forward_kl})
        print(({'KL/KL(GPT3, model)': forward_kl}))
        model.training = was_in_training
        tokenizer.padding_side = original_padding_side


class CustomWandbCallback(WandbCallback):
    """A thin wrapper around WandbCallback to disable logging gradients and storing model/trainer configs (we do that
    elsewhere more cleanly)"""

    def setup(self, args, state, model, **kwargs):
        self._initialized = True
        if state.is_world_process_zero:
            wandb.define_metric("train/tokens_seen")
            wandb.define_metric("*", step_metric="train/tokens_seen")
            wandb.define_metric("objective/eval/*", step_metric="objective/eval/tokens_seen_during_eval")
            wandb.log({'train/tokens_seen': state.tokens_seen})

    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        if not self._initialized:
            self.setup(args, state, model)
        if control.should_training_stop:
            return
        if state.is_world_process_zero:
            logs = {self._rename_key(k): v for k, v in logs.items()}
            logs['train/tokens_seen'] = state.tokens_seen
            self._wandb.log({**logs, "train/global_step": state.global_step})

    def _rename_key(self, key):
        key = key.replace('train_', 'train/', 1).replace('eval_', 'eval/', 1)
        if not '/' in key:
            key = 'train/' + key
        return key
