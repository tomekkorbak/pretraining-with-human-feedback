from abc import ABC, abstractmethod
from typing import NamedTuple, Optional

import wandb
import torch
from torch.nn import CrossEntropyLoss, MSELoss, Module
from torch.nn import functional as F
from scipy.stats import pearsonr


class LossOutput(NamedTuple):
    loss: torch.FloatTensor
    logs: dict[str, torch.Tensor]


class Objective(ABC, Module):
    """
    An objective is a PyTorch Module used for computing a differentiable loss based on LM-produced logits, ground truth
    labels and their token_scores
    """

    @classmethod
    def from_config(cls, name: str, **kwargs) -> 'Objective':
        objective_class = globals()[name]
        return objective_class(**kwargs)

    @abstractmethod
    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        token_scores: Optional[torch.Tensor],
        values: Optional[torch.Tensor] = None,
        q_values: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        step: int = None,
        log_stats: bool = False
    ) -> LossOutput:
        raise NotImplementedError()


class MLE(Objective):
    """
    A thin wrapper around PyTorch CrossEntropyLoss just making its .forward() signature compatible with Objective
    """

    def __init__(self):
        super().__init__()
        self.cross_entropy = CrossEntropyLoss()

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        **kwargs
    ) -> LossOutput:
        loss = self.cross_entropy(logits, labels)
        return LossOutput(loss=loss, logs={})


class AWR(Objective):
    """
    An implementation of advantage-weighted regression (AWR, https://arxiv.org/abs/1910.00177). The loss is computed as
    cross-entropy weighted at token-level by the advantage.
    """

    def __init__(
        self,
        alpha: float = 0.5,
        beta: int = 1,
        use_value_head: bool = True,
        fixed_value: Optional[float] = None,
        use_head_train_steps: int = 0,
        clip_value: bool = False,
        clip_weight_max: float = 1000,
        normalize_advantage: bool = False
    ):
        super().__init__()
        self.cross_entropy = CrossEntropyLoss(reduction='none')
        self.mse = MSELoss()
        self.alpha = alpha
        self.beta = beta
        self.use_value_head = use_value_head
        self.fixed_value = fixed_value
        self.use_head_train_steps = use_head_train_steps
        self.clip_value = clip_value
        self.clip_weight_max = clip_weight_max
        self.normalize_advantage = normalize_advantage

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        token_scores: Optional[torch.Tensor],
        values: Optional[torch.Tensor] = None,
        step: int = None,
        log_stats: bool = False,
        **kwargs
    ) -> LossOutput:
        token_scores_shifted = token_scores[..., 1:].contiguous().view(-1)
        values_shifted = values[..., :-1].contiguous().view(-1)
        if self.clip_value:
            values_shifted = torch.clamp(values_shifted, -1, 0)
        reward = -token_scores_shifted
        if self.fixed_value:
            advantage = reward - torch.tensor(self.fixed_value)
        elif self.use_value_head:
            advantage = reward - values_shifted.detach()
        else:
            advantage = reward
        if self.normalize_advantage:
            advantage = (advantage - advantage.mean())/(advantage.std() + 1e-10)
        lm_loss = self.cross_entropy(logits, labels)
        weights = torch.clamp(torch.exp((advantage/self.beta)), min=0, max=self.clip_weight_max)
        weighted_lm_loss = (lm_loss * weights).mean()
        value_loss = self.mse(reward, values_shifted)
        logs = {
            'lm_loss': lm_loss.mean(),
            'weighted_lm_loss': weighted_lm_loss,
            'value_loss': value_loss,
            'value_min': values_shifted.min(),
            'value_max': values_shifted.max(),
            'value_avg': values_shifted.mean(),
            'value_std': values_shifted.std(),
            'value_reward_corr': pearsonr(values_shifted.detach().cpu().numpy(), reward.cpu().numpy())[0],
            'value_dis': wandb.Histogram(values_shifted.detach().cpu().numpy()),
            'weight_avg': weights.mean(),
            'weights_min': weights.min(),
            'weights_max': weights.max(),
            'weight_dist': wandb.Histogram(weights.detach().cpu().numpy()),
            'advantage_avg': advantage.mean(),
        } if log_stats else {}
        loss = self.alpha * weighted_lm_loss + (1-self.alpha) * value_loss
        return LossOutput(loss=loss, logs=logs)


class Unlikelihood(Objective):
    """
    An implementation of token-level unlikelihood objective (https://arxiv.org/abs/1908.04319). Given token_scores and
    score_threshold, the likelihood of a ground-truth token is maximized (standard MLE) if its token_score is below the
    score_threshold. Otherwise, the unlikelihood (log (1-p(x)) of that token is maximized.
    """

    def __init__(self, score_threshold: float, alpha: float):
        super().__init__()
        self.score_threshold = score_threshold
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, labels: torch.Tensor, token_scores: torch.Tensor, **kwargs) -> LossOutput:
        # Adapted from:
        # https://github.com/facebookresearch/unlikelihood_training/blob/main/custom/gpt2/run_gpt2.py#L131-L143
        token_scores_shifted = token_scores[..., 1:].contiguous().view(-1)  # score of the token being predicted
        if self.score_threshold:
            is_token_aligned = (token_scores_shifted <= self.score_threshold).float()
        else:
            is_token_aligned = 1-token_scores_shifted  # no threshold, convex combination of MLE and UL will be applied
        log_probs = F.log_softmax(logits, dim=-1)
        likelihoods = log_probs.gather(1, labels.view(-1, 1)).view(-1)
        one_minus_probs = torch.clamp((1.0 - likelihoods.exp()), min=1e-20)
        unlikelihoods = one_minus_probs.log()
        likelihood_loss = -(is_token_aligned * likelihoods)
        unlikelihood_loss = -((1.0 - is_token_aligned) * unlikelihoods)
        loss = (likelihood_loss + self.alpha * unlikelihood_loss).mean()
        logs = {
            'avg_token_score': token_scores_shifted.mean(),
            'aligned_token_count': is_token_aligned.mean(),
            'likelihood_loss': likelihood_loss.mean(),
            'unlikelihood_loss': unlikelihood_loss.mean(),
        }
        return LossOutput(loss=loss, logs=logs)
