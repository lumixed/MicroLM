"""
finetuning/dpo_loss.py

Direct Preference Optimization (DPO) loss.

Paper: "Direct Preference Optimization: Your Language Model is Secretly a Reward Model"
       Rafailov et al., 2023 (https://arxiv.org/abs/2305.18290)

DPO Loss formula:
    L_DPO(π_θ) = -E[(x,y_w,y_l)~D] [
        log σ(
            β * log(π_θ(y_w|x) / π_ref(y_w|x))
          - β * log(π_θ(y_l|x) / π_ref(y_l|x))
        )
    ]

Where:
    π_θ   = policy model (being trained)
    π_ref = reference model (frozen SFT checkpoint)
    y_w   = chosen (preferred) response
    y_l   = rejected (dispreferred) response
    β     = temperature controlling deviation from reference (default 0.1)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class DPOLoss(nn.Module):
    """
    DPO loss module.

    Args:
        beta: KL penalty coefficient. Higher = stay closer to reference model.
              Typical range: 0.05 – 0.5. Default: 0.1.
    """

    def __init__(self, beta: float = 0.1):
        super().__init__()
        self.beta = beta

    def forward(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        ref_chosen_logps: torch.Tensor,
        ref_rejected_logps: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute DPO loss.

        Args:
            policy_chosen_logps:   Log-probs of chosen sequences under policy.   Shape: (B,)
            policy_rejected_logps: Log-probs of rejected sequences under policy. Shape: (B,)
            ref_chosen_logps:      Log-probs of chosen sequences under ref.      Shape: (B,)
            ref_rejected_logps:    Log-probs of rejected sequences under ref.    Shape: (B,)

        Returns:
            loss:           Scalar DPO loss.
            chosen_reward:  Mean implicit reward for chosen responses (for logging).
            rejected_reward:Mean implicit reward for rejected responses (for logging).
        """
        # Implicit rewards: β * log(π_θ / π_ref)
        chosen_reward = self.beta * (policy_chosen_logps - ref_chosen_logps)
        rejected_reward = self.beta * (policy_rejected_logps - ref_rejected_logps)

        # DPO loss: -log σ(r_w - r_l)
        loss = -F.logsigmoid(chosen_reward - rejected_reward).mean()

        return loss, chosen_reward.mean().detach(), rejected_reward.mean().detach()


@torch.no_grad()
def get_batch_logps(
    model: nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """
    Compute per-sequence log probabilities from a model.

    Args:
        model:     MicroLM (or any causal LM with forward → (logits, loss)).
        input_ids: Token IDs, shape (B, T).
        labels:    Target token IDs, shape (B, T). Use -100 for ignored positions.
        device:    Torch device.

    Returns:
        logps: Per-sequence sum of log-probs for non-masked positions. Shape: (B,)
    """
    model.eval()
    input_ids = input_ids.to(device)
    labels = labels.to(device)

    with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
        logits, _ = model(input_ids)  # (B, T, vocab_size)

    # Shift: predict token[t+1] from token[t]
    shift_logits = logits[:, :-1, :].contiguous()   # (B, T-1, V)
    shift_labels = labels[:, 1:].contiguous()        # (B, T-1)

    # Per-token log probabilities
    log_probs = F.log_softmax(shift_logits, dim=-1)  # (B, T-1, V)

    # Gather log-prob of the actual next token
    token_logps = log_probs.gather(
        dim=2, index=shift_labels.clamp(min=0).unsqueeze(-1)
    ).squeeze(-1)  # (B, T-1)

    # Mask out ignored positions (-100 in labels)
    mask = (shift_labels != -100).float()
    token_logps = token_logps * mask

    # Sum over sequence → scalar per example
    return token_logps.sum(dim=-1)  # (B,)
