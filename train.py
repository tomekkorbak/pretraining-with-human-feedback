import os
from typing import Any, Optional
import argparse

import torch
from transformers import AutoConfig, AutoTokenizer, TrainingArguments, PreTrainedModel, PreTrainedTokenizer, set_seed
import wandb
import yaml

from apo.dataset_wrappers import ConstantLengthDataset
from apo.trainer import CustomObjectiveTrainer, ModelInputInspector
from apo.objectives import Objective
from apo.models import GPT2LMAndValueHeadModel
from apo.callbacks import GenerateAndScoreCallback, GenerationScenario, CustomWandbCallback, KLGPT3Callback, SetupCallback
from apo.scorers import Scorer
from apo.metrics import Metric
from apo.utils import override_config, unflatten_config, merge_configs


def prepare_tokenizer(path_or_name: str, special_tokens: list[str] = None) -> PreTrainedTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(path_or_name, use_fast=True)  # always using a pretrained tokenizer
    if special_tokens:
        tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
        print(f'Added control tokens: {tokenizer.additional_special_tokens} to tokenizer '
              f'with ids {tokenizer.additional_special_tokens_ids}')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'  # avoid issue with position embeddings for prompts in conditional generation
    tokenizer.aligned_prefix = special_tokens[0] if special_tokens else None
    tokenizer.misaligned_prefix = special_tokens[1] if special_tokens else None
    return tokenizer


def prepare_model(
    path_or_name: str,
    from_scratch: bool = True,
    num_additional_tokens: int = None,
    model_kwargs: dict[str, Any] = None,
    gpt2_config_kwargs: dict[str, Any] = None
) -> PreTrainedModel:
    model_kwargs = model_kwargs or {}
    if from_scratch:  # only using the config of a pretrained model
        config = AutoConfig.from_pretrained(path_or_name, **gpt2_config_kwargs)
        model = GPT2LMAndValueHeadModel(config, **model_kwargs)
        model.eval()
    else:
        model = GPT2LMAndValueHeadModel.from_pretrained(path_or_name, **model_kwargs)
    if num_additional_tokens:
        num_original_tokens = model.lm_head.weight.size(0)
        # Trick need to avoid initializing new embeddings to large values that'd cause oversampling
        # See https://nlp.stanford.edu//~johnhew//vocab-expansion.html
        model.resize_token_embeddings(num_original_tokens+num_additional_tokens)
        pre_expansion_embedding_mean = model.lm_head.weight.data[:num_original_tokens].mean(dim=0)
        noise = torch.randn_like(model.lm_head.weight.data[num_original_tokens:])
        model.lm_head.weight.data[num_original_tokens:] = pre_expansion_embedding_mean + noise * 0.01
        print(f'model.lm_head resized for additional {num_additional_tokens} token embeddings')
    if model_kwargs is not None and model_kwargs.get('q_value_head_config', {}).get('initialize_using_lm_head', False):
        model.q_value_head.head.weight.data = model.lm_head.weight.data.detach().clone()
        print('Initialising Q head using LM head\'s initial weights')
    return model


def prepare_trainer_arguments(**kwargs) -> TrainingArguments:
    num_tokens = kwargs.pop('num_tokens', None)
    effective_batch_size = kwargs.pop('effective_batch_size', None)
    tokens_already_seen = kwargs.pop('tokens_already_seen', 0)
    args = TrainingArguments(report_to=['none'], **kwargs)
    if effective_batch_size:
        if args.local_rank == -1:
            instantaneous_bsz = (args.per_device_train_batch_size * args.world_size * args.n_gpu)
            args.gradient_accumulation_steps = int(effective_batch_size // instantaneous_bsz)
            print(f'setting gradient_accumulation_steps={args.gradient_accumulation_steps} based on '
                  f'effective_batch_size={effective_batch_size} and instantaneous_bsz={instantaneous_bsz} '
                  f'(world_size={args.world_size}, n_gpu={args.n_gpu})')
            if args.gradient_accumulation_steps <= 0 or effective_batch_size % args.gradient_accumulation_steps != 0:
                raise ValueError("effective_batch_size is incompatible with per_device_train_batch_size and world_size")
        else:
            raise ValueError('effective_batch_size is not compatible with DDP')
    if num_tokens:
        num_tokens -= tokens_already_seen
        args.max_steps = int(num_tokens // (effective_batch_size * args.world_size * 1024))
        print(f'setting max_steps={args.max_steps} based on num_tokens={num_tokens:2.2e} '
              f'and tokens_already_seen={tokens_already_seen:2.2e}')
    return args


def prepare_generation_callback(
        scorer_config: dict[str, Any],
        scenario_configs: list[dict[str, Any]],
        metrics_configs: Optional[list[dict[str, Any]]],
        **kwargs: dict[str, Any]
) -> GenerateAndScoreCallback:
    scorer = Scorer.from_config(config=scorer_config)
    metrics = [Metric.from_config(config=metric_config) for metric_config in metrics_configs]
    scenarios = [GenerationScenario.from_config(**config) for config in scenario_configs]
    generation_callback = GenerateAndScoreCallback(scorer=scorer, scenarios=scenarios, metrics=metrics, **kwargs)
    return generation_callback


def train(checkpoint_path: str, config: dict[str, Any]):
    model = prepare_model(**config['model'])
    tokenizer = prepare_tokenizer(**config['tokenizer'])
    train_dataset = ConstantLengthDataset(tokenizer=tokenizer, **config['dataset']).shuffle(20_000)
    training_args = prepare_trainer_arguments(**config['training'])
    objective = Objective.from_config(**config['objective'])
    generation_callback = prepare_generation_callback(**config['generation'])
    callbacks = [
        SetupCallback(),
        CustomWandbCallback(),
        generation_callback
    ]
    if 'kl_gpt3_callback' in config:
        callbacks.append(KLGPT3Callback(**config['kl_gpt3_callback']))
    input_inspector = ModelInputInspector(
        tokenizer=tokenizer,
        scorer=generation_callback.scorer,
        metrics=generation_callback.metrics,
    )
    trainer = CustomObjectiveTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        objective=objective,
        input_inspector=input_inspector,
        callbacks=callbacks)
    if training_args.hub_model_id is not None:
        trainer.create_model_card(dataset_tags=config['dataset']['datasets'], wandb_run=wandb.run, full_config=config)
    trainer.train(resume_from_checkpoint=checkpoint_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', type=str, help='wandb run name', default=None)
    parser.add_argument('--group_name', type=str, help='wandb group name', default=None)
    parser.add_argument('--tags', nargs='+', help='wandb tags',  default=[])
    parser.add_argument('--task', type=str, help='a path to a YAML file with task configuration')
    parser.add_argument('--method', type=str, help='a path to a YAML file with method configuration')
    parser.add_argument('--checkpoint_path', type=str, help='a path to checkpoint to resume training', default=None)
    parser.add_argument('--override', nargs='+', type=str,
                        help='a list of params to override, e.g. model.from_scratch=True dataset.num_proc=16')
    args = parser.parse_args()
    task_config = yaml.full_load(open(args.task, 'r'))
    method_config = yaml.full_load(open(args.method, 'r'))
    config = dict(merge_configs(task_config, method_config))
    if args.override:  # override YAML config from command-line
        override_config(config, params_to_override=args.override)
    wandb.init(name=args.run_name, group=args.group_name, config=config, tags=args.tags,
               notes=os.environ.get('SLURM_JOB_ID', 'local'))
    if wandb.run.sweep_id is not None:
        config = unflatten_config(wandb.config)  # allow wandb to modify config for sweeps
    set_seed(config['training']['seed'])
    train(args.checkpoint_path, config=config)
