from typing import Any, Optional, Union
import os
from pprint import pformat

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import Trainer, PreTrainedModel, PreTrainedTokenizer
from transformers.modelcard import TrainingSummary
import wandb

from .metrics import Metric
from .scorers import Scorer
from .utils import get_theoretical_loss


class CustomObjectiveTrainer(Trainer):

    def __init__(self, *args, **kwargs):
        self.objective = kwargs.pop('objective', None)
        self.input_inspector = kwargs.pop('input_inspector', None)
        self.embedding_inspector = kwargs.pop('embedding_inspector', None)
        super().__init__(*args, **kwargs)
        assert self.label_smoother is None, 'CustomObjectiveTrainer does not support label smoothing'
        self.training_log_counter = 0
        self.eval_log_counter = 0
        self.num_params = self.model.num_parameters()

    def compute_loss(
        self,
        model: PreTrainedModel,
        inputs: dict[str, Any],
        return_outputs: bool = False
    ) -> torch.Tensor:
        microbatch_size = inputs.input_ids.numel()
        batch_size = inputs.input_ids.size(0)
        if model.training:
            self.training_log_counter += 1
            should_log = self.training_log_counter % (self.args.logging_steps * 100) == 0
            self.state.tokens_seen += microbatch_size
        else:
            self.eval_log_counter += 1
            should_log = self.eval_log_counter % (self.args.logging_steps * 100) == 0
            if not hasattr(self.state, 'eval_tokens_seen'):
                self.state.eval_tokens_seen = 0
            self.state.eval_tokens_seen += microbatch_size
        token_scores = inputs.pop('token_scores')
        outputs = model(**inputs)  # forward pass
        if self.objective is None:
            loss = outputs.loss  # just use the loss computed inside model.forward
        else:
            assert token_scores is not None, 'token_scores are required for a custom objective'
            # Prepare logits
            logits = outputs.logits
            shift_logits = logits[..., :-1, :].contiguous()  # Shift so that tokens < n predict n
            shift_logits = shift_logits.view(-1, shift_logits.size(-1))  # flatten tokens
            # Prepare labels
            labels = inputs['labels']
            shift_labels = labels[..., 1:].contiguous()
            shift_labels = shift_labels.view(-1)
            loss, logs = self.objective(
                logits=shift_logits,
                labels=shift_labels,
                token_scores=token_scores,
                values=outputs.values,
                q_values=outputs.q_values,
                input_ids=inputs['input_ids'],
                step=self.state.global_step,
                log_stats=should_log
            )
            # Additional logs
            if should_log and model.training:
                logs['original_loss'] = outputs.loss
                logs['instantaneous_microbatch_size'] = microbatch_size
                logs['instantaneous_batch_size'] = batch_size
                logs['theoretical_loss'] = get_theoretical_loss(
                    num_tokens=self.state.tokens_seen,
                    num_params=self.num_params
                )
                logs['tokens_used'] = (self.train_dataset.datapipe.tokens_used - self.train_dataset.buffer_size -
                                       self.train_dataset.datapipe.skip_tokens)
                logs['docs_used'] = self.train_dataset.datapipe.num_docs
                logs = {f'objective/train/{k}': v.mean().item() if isinstance(v, torch.Tensor) else v
                        for k, v in logs.items()}
                if self.input_inspector is not None and self.training_log_counter % self.input_inspector.freq == 0:
                    debugging_logs = self.input_inspector.inspect(
                        inputs,
                        token_scores=token_scores,
                        values=outputs.values
                    )
                    logs.update({k: v.mean().item() if isinstance(v, torch.Tensor) else v
                                 for k, v in debugging_logs.items()})
                self.log(logs)
            if should_log and not model.training:
                logs['original_loss'] = outputs.loss
                logs['tokens_seen_during_eval'] = self.state.eval_tokens_seen
                logs = {f'objective/eval/{k}': v.mean().item() if isinstance(v, torch.Tensor) else v
                        for k, v in logs.items()}
                self.log(logs)
        return (loss, outputs) if return_outputs else loss

    def log(self, logs: dict[str, float]) -> None:
        if self.state.epoch is not None:
            logs["epoch"] = round(self.state.epoch, 2)
        output = {
            **logs,
            "tokens_seen": self.state.tokens_seen,
            "theoretical_loss": get_theoretical_loss(num_tokens=self.state.tokens_seen, num_params=self.num_params)
        }
        self.state.log_history.append({k: v for k, v in output.items() if isinstance(v, (int, float))})
        self.control = self.callback_handler.on_log(self.args, self.state, self.control, output)

    def get_train_dataloader(self) -> DataLoader:
        if isinstance(self.train_dataset, torch.utils.data.IterableDataset) and self.args.world_size == 1:
            # fix for DP with multiple GPUs
            print(f'Setting train_dataloader.batch_size={self.args.train_batch_size}')
            return DataLoader(
                self._remove_unused_columns(self.train_dataset, description="training"),
                collate_fn=self.data_collator,
                batch_size=self.args.train_batch_size,
                num_workers=0,
                pin_memory=self.args.dataloader_pin_memory,
                shuffle=True,
            )
        else:
            return super().get_train_dataloader()

    def _push_from_checkpoint(self, checkpoint_folder: str) -> None:
        super()._push_from_checkpoint(checkpoint_folder)
        self.repo.add_tag(tag_name=str(self.state.tokens_seen))

    def create_model_card(
        self,
        language: Optional[str] = None,
        license: Optional[str] = None,
        tags: Optional[str] = None,
        model_name: Optional[str] = None,
        finetuned_from: Optional[str] = None,
        tasks: Optional[str] = None,
        dataset_tags: Optional[Union[str, list[str]]] = None,
        dataset: Optional[Union[str, list[str]]] = None,
        dataset_args: Optional[Union[str, list[str]]] = None,
        **kwargs
    ):
        if not self.is_world_process_zero():
            return

        training_summary = TrainingSummary.from_trainer(
            self,
            language='en',
            license=license,
            tags=tags,
            model_name=self.args.hub_model_id,
            tasks='text-generation',
            dataset_tags=dataset_tags,
            dataset=dataset,
            dataset_args=dataset_args,
        )
        training_summary.finetuned_from = None
        model_card = training_summary.to_model_card()
        model_card += '\n\n# Full config\n' + pformat(kwargs.get('full_config'))
        model_card += '\n\n# Wandb URL:\n' + wandb.run.url
        with open(os.path.join(self.args.output_dir, "README.md"), "w") as f:
            f.write(model_card)
        self.repo.push_to_hub(commit_message="update model card README.md", auto_lfs_prune=True)


class ModelInputInspector:
    """
    A class useful for inspecting the raw data seen by GPT2Model. Given a raw batch input_ids, ModelInputInspector
    first finds fixed-length segments starting with EOS token. This is to ensure a setup similar to generation. Then,
    segments are retokenized and scorer and metrics are run on the retokenized segments.
    """

    def __init__(
            self,
            tokenizer: PreTrainedTokenizer,
            metrics: list[Metric],
            scorer: Optional[Scorer] = None,
            segment_length: int = 128,
            freq: int = 10_000
    ):
        self.tokenizer = tokenizer
        self.scorer = scorer
        self.metrics = metrics
        self.segment_length = segment_length
        self.freq = freq

    def inspect(
        self,
        inputs: dict[str, Any],
        token_scores: torch.FloatTensor,
        values: torch.FloatTensor = None
    ) -> dict[str, Any]:
        segments = self.get_segments(inputs)
        scores = [self.scorer.score_text(text) for text in segments]
        segment_table = wandb.Table(columns=['segment', 'score'], data=list(zip(segments, scores)))
        raw_text = self.tokenizer.batch_decode(inputs['input_ids'])
        logs = {
            f'debugging/segments': segment_table,
            f'debugging/score': np.mean(scores),
            f'debugging/score_std': np.std(scores),
            f'debugging/num_segments': len(segments),
        }
        if token_scores is not None and values is not None:
            logs['debugging/raw_token_scores_avg'] = token_scores.mean()
            logs['debugging/raw_token_scores_std'] = token_scores.std()
            neg_scores = (-token_scores).reshape_as(inputs.input_ids).tolist()
            values = values.reshape_as(inputs.input_ids).tolist()
            logs['debugging/raw_text'] = wandb.Table(
                columns=['raw batch text', '-scores', 'values'],
                data=[(text, ' '.join(f'{s:.2f}' for s in score), ' '.join(f'{v:.2f}' for v in value))
                      for text, score, value in zip(raw_text, neg_scores, values)]
            )
        else:
            logs['debugging/raw_text'] = wandb.Table(columns=['raw batch text'], data=[(text,) for text in raw_text])

        for metric in self.metrics:
            metric_logs = {f'debugging/{name}': value for name, value in metric.score_texts(texts=segments).items()}
            logs.update(metric_logs)
        return logs

    def get_segments(self, input: torch.Tensor) -> list[str]:
        """
        Find segments from raw batch that start with EOS token and are  self.segment_length tokens long. The number of
        segments found is variable and independent of batch size.
        """
        segments = []
        batch_idx, position_idx = (input['input_ids'] == self.tokenizer.eos_token_id).nonzero(as_tuple=True)
        for batch_idx, position_idx in zip(batch_idx, position_idx):
            if position_idx + self.segment_length <= input['input_ids'].size(1):
                segment_input_ids = input['input_ids'][batch_idx, position_idx: position_idx + self.segment_length]
                segment_text = self.tokenizer.decode(segment_input_ids)
                segment_text = segment_text.split(self.tokenizer.eos_token)[1]  # discard whatever is after the next EOS
                segments.append(segment_text)
        return segments


class ModelEmbeddingInspector:

    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, tokens_to_track: list[str] = None):
        self.model = model
        self.tokenizer = tokenizer
        self.tokens_to_track = tokens_to_track

    def inspect(self) -> dict[str, Any]:
        if self.tokens_to_track is None:
            return {}
        logs = {}
        for token in self.tokens_to_track:
            token_id = self.tokenizer.convert_tokens_to_ids(token)
            logs[f'debugging/{token}_norm'] = self.model.lm_head.weight[token_id].norm()
        if len(self.tokens_to_track) > 1:
            token_a, token_b = self.tokens_to_track[0], self.tokens_to_track[-1]
            logs['debugging/control_tokens_similarity'] = F.cosine_similarity(
                self.model.lm_head.weight[self.tokenizer.convert_tokens_to_ids(token_a)],
                self.model.lm_head.weight[self.tokenizer.convert_tokens_to_ids(token_b)],
                dim=0
            )
        return logs
