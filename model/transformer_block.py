"""
microlm/model/transformer_block.py

Single Transformer decoder block (pre-norm architecture).

Pre-norm vs Post-norm:
    Post-norm (original Transformer): x = LayerNorm(x + sublayer(x))
    Pre-norm  (modern LLMs):          x = x + sublayer(LayerNorm(x))

    Pre-norm trains more stably at scale — the residual path stays clean
    throughout the network without normalization interfering.
    Used in: GPT-2+, Llama, Mistral, Gemma.

Block structure:
    x → RMSNorm → GQA Attention → + x → RMSNorm → SwiGLU FFN → + x
         (pre-attn norm)                   (pre-ffn norm)
"""

import torch
import torch.nn as nn

from .config import MicroLMConfig
from .rmsnorm import RMSNorm
from .attention import GroupedQueryAttention
from .feedforward import SwiGLU


class TransformerBlock(nn.Module):
    """
    A single pre-norm transformer decoder block.

    Args:
        config: MicroLMConfig instance.
        layer_idx: Layer index (used for debugging/logging).
    """

    def __init__(self, config: MicroLMConfig, layer_idx: int = 0) -> None:
        super().__init__()
        self.layer_idx = layer_idx

        # Pre-norm before attention
        self.attn_norm = RMSNorm(config.d_model)
        # Grouped Query Attention
        self.attn = GroupedQueryAttention(config)

        # Pre-norm before FFN
        self.ffn_norm = RMSNorm(config.d_model)
        # SwiGLU Feed-Forward Network
        self.ffn = SwiGLU(config)

        # Residual dropout (applied after each sublayer output, before adding residual)
        self.resid_dropout = nn.Dropout(config.residual_dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask=None,
        kv_cache=None,
        start_pos: int = 0,
    ) -> torch.Tensor:
        """
        Args:
            x:        (B, T, D) — input hidden states
            mask:     Optional causal mask
            kv_cache: Optional KV cache dict for inference
            start_pos: KV cache offset

        Returns:
            (B, T, D) — output hidden states
        """
        # ---- Attention sub-layer (pre-norm) ----
        # Normalize → attend → dropout → residual add
        attn_out = self.attn(self.attn_norm(x), mask=mask, kv_cache=kv_cache, start_pos=start_pos)
        x = x + self.resid_dropout(attn_out)

        # ---- FFN sub-layer (pre-norm) ----
        # Normalize → FFN → dropout → residual add
        ffn_out = self.ffn(self.ffn_norm(x))
        x = x + self.resid_dropout(ffn_out)

        return x
