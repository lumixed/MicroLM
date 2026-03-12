"""
microlm/model/feedforward.py

SwiGLU Feed-Forward Network — implemented from scratch.

Reference: Noam Shazeer 2020, "GLU Variants Improve Transformer"
https://arxiv.org/abs/2002.05202

Standard FFN:
    FFN(x) = activation(x @ W1) @ W2

SwiGLU (Gated Linear Unit with Swish activation):
    FFN_SwiGLU(x) = (Swish(x @ W_gate) ⊙ (x @ W_up)) @ W_down

Where Swish(x) = x * sigmoid(x) = x * σ(x)

Key properties:
    - Element-wise gate controls information flow through the FFN
    - More expressive than standard FFN for same parameter budget
    - Used in: Llama, PaLM, Gemma, Mistral
    - Note: requires 3 weight matrices instead of 2 (slight param overhead vs standard FFN)

Parameter note:
    To keep FFN params ~equal to standard FFN with d_ff = 4*d_model,
    the SwiGLU hidden dim is typically set to (2/3) * 4 * d_model ≈ 2.67 * d_model
    rounded to nearest multiple of 64 for hardware efficiency.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import MicroLMConfig


class SwiGLU(nn.Module):
    """
    SwiGLU Feed-Forward Network.

    Architecture:
        gate_proj: d_model → d_ff
        up_proj:   d_model → d_ff
        down_proj: d_ff    → d_model

    Forward:
        h = swish(gate_proj(x)) * up_proj(x)
        out = down_proj(h)

    Args:
        config: MicroLMConfig instance.
    """

    def __init__(self, config: MicroLMConfig) -> None:
        super().__init__()
        # Gate projection: produces the gate values
        self.gate_proj = nn.Linear(config.d_model, config.d_ff, bias=config.bias)
        # Up projection: produces the value to be gated
        self.up_proj = nn.Linear(config.d_model, config.d_ff, bias=config.bias)
        # Down projection: project back to d_model
        self.down_proj = nn.Linear(config.d_ff, config.d_model, bias=config.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, T, d_model)

        Returns:
            Output tensor of shape (B, T, d_model)
        """
        # Swish(gate) ⊙ up, then project down
        # F.silu is the Swish activation: silu(x) = x * sigmoid(x)
        gate = F.silu(self.gate_proj(x))  # (B, T, d_ff)
        up = self.up_proj(x)              # (B, T, d_ff)
        return self.down_proj(gate * up)  # (B, T, d_model)
