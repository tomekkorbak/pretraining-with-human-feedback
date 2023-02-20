from typing import Optional, Tuple, Any

import torch
import torch.utils.checkpoint
from torch import nn
from transformers import GPT2LMHeadModel, GPT2Model, GPT2Config
from transformers.file_utils import ModelOutput

from apo.utils import CustomMinLengthLogitsProcessor


class CausalLMOutputWithCrossAttentionsAndValues(ModelOutput):
    """
    A custom variant of `CausalLMOutputWithCrossAttentions` that also stores the value predicted by a value head
    """
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    values: Optional[torch.FloatTensor] = None
    q_values: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None


class ValueHead(nn.Module):
    """
    A value head on top of a GPT2 model. Given hidden states, outputs a scalar value associated with each token.
    """
    def __init__(
        self,
        gpt2_config: GPT2Config,
        is_detached: bool = False,
        sigmoid: bool = True,
        **kwargs
    ):
        super().__init__()
        self.head = nn.Linear(gpt2_config.n_embd, 1, bias=False)
        self.head.weight.data.zero_()
        self.is_detached = is_detached
        self.sigmoid = sigmoid

    def forward(self, hidden_states: torch.FloatTensor) -> torch.FloatTensor:
        if self.is_detached:
            hidden_states = hidden_states.detach()
        if self.sigmoid:
            return -torch.sigmoid(self.head(hidden_states))
        else:
            return self.head(hidden_states)


class QValueHead(nn.Module):
    """
    A Q-value head on top of a GPT2 model. Given hidden states, outputs a vocabulary-sized vector of scores for each
     token.
    """
    def __init__(
        self,
        gpt2_config: GPT2Config,
        is_detached: bool = False,
        sigmoid: bool = True,
        **kwargs
    ):
        super().__init__()
        self.head = nn.Linear(gpt2_config.n_embd, gpt2_config.vocab_size, bias=False)
        self.head.weight.data.zero_()
        self.is_detached = is_detached
        self.sigmoid = sigmoid

    def forward(self, hidden_states: torch.FloatTensor) -> torch.FloatTensor:
        if self.is_detached:
            hidden_states = hidden_states.detach()
        if self.sigmoid:
            return -torch.sigmoid(self.head(hidden_states))
        else:
            return self.head(hidden_states)


class GPT2LMAndValueHeadModel(GPT2LMHeadModel):
    _keys_to_ignore_on_load_missing = [r"attn.masked_bias", r"attn.bias", r"lm_head.weight",
                                       'value_head.head.weight', 'value_head.head.bias']

    def __init__(
        self,
        config: GPT2Config,
        value_head_config: Optional[dict[str, Any]] = None,
        q_value_head_config: Optional[dict[str, Any]] = None
    ):
        super().__init__(config)
        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # Model parallel
        self.model_parallel = False
        self.device_map = None

        # Initialize weights and apply final processing
        self.post_init()

        # Add value heads which are initialised separately
        if value_head_config is not None:
            self.value_head = ValueHead(gpt2_config=config, **value_head_config)
        else:
            self.value_head = None

        if q_value_head_config is not None:
            self.q_value_head = QValueHead(gpt2_config=config, **q_value_head_config)
        else:
            self.q_value_head = None

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        logits_config: Optional[dict[str, Any]] = None
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)
        lm_logits = self.lm_head(hidden_states)
        values = self.value_head(hidden_states).squeeze(dim=2) if self.value_head is not None else None
        q_values = self.q_value_head(hidden_states) if self.q_value_head is not None else None

        # Standard loss computation kept for compatibility; the loss actually used is computed outside the model
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return CausalLMOutputWithCrossAttentionsAndValues(
            loss=loss,  # we will always compute loss outside this class
            logits=lm_logits,
            values=values,
            q_values=q_values,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )

    def _get_logits_processor(self, *args, **kwargs):
        logits_processors = super()._get_logits_processor(*args, **kwargs)
        min_length, eos_token_id = kwargs.get('min_length'), kwargs.get('eos_token_id')
        logits_processors.append(CustomMinLengthLogitsProcessor(min_length, eos_token_id))
        return logits_processors
