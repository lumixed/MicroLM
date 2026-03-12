"""
microlm/model/rmsnorm.py

RMSNorm — Root Mean Square Layer Normalization.

Reference: Zhang & Sennrich 2019, "Root Mean Square Layer Normalization"
https://arxiv.org/abs/1910.07467

Unlike LayerNorm, RMSNorm:
  - Normalizes by root-mean-square only (no mean subtraction)
  - Removes the 're-centering' bias term
  - ~10-15% faster than LayerNorm in practice
  - Used in: Llama, Gemma, PaLM, T5

Formula:
    RMSNorm(x) = x / RMS(x) * weight
    where RMS(x) = sqrt(mean(x^2) + eps)
"""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    Args:
        d_model (int): Model hidden dimension (dimension to normalize over).
        eps (float): Numerical stability epsilon. Default: 1e-6.
    """

    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        # Learnable per-dimension scale parameter (no bias in RMSNorm)
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (..., d_model)

        Returns:
            Normalized tensor of same shape as x.
        """
        # Compute RMS: sqrt(mean(x^2) + eps)
        # Using float32 for numerical stability even when x is bfloat16
        orig_dtype = x.dtype
        x = x.float()

        # rms: (..., 1)
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        x = x * rms

        return (x * self.weight.float()).to(orig_dtype)

    def extra_repr(self) -> str:
        return f"d_model={self.weight.shape[0]}, eps={self.eps}"
