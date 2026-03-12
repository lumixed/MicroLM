"""
microlm/model/attention.py

Grouped Query Attention (GQA) with RoPE and optional Flash Attention.

References:
    - Ainslie et al. 2023 "GQA: Training Generalized Multi-Query Transformer Models"
      https://arxiv.org/abs/2305.13245
    - Dao et al. 2022 "FlashAttention: Fast and Memory-Efficient Exact Attention"
      https://arxiv.org/abs/2205.14135
    - Vaswani et al. 2017 "Attention Is All You Need"
      https://arxiv.org/abs/1706.03762

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GQA vs MHA vs MQA:

  MHA (Multi-Head Attention):    n_heads Q-heads + n_heads K-heads + n_heads V-heads
  MQA (Multi-Query Attention):   n_heads Q-heads + 1       K-head  + 1       V-head
  GQA (Grouped Query Attention): n_heads Q-heads + n_kv_heads K-heads + n_kv_heads V-heads
                                 where 1 < n_kv_heads < n_heads

GQA reduces KV cache memory by a factor of (n_heads / n_kv_heads) while
maintaining most of the quality of MHA. Used in Llama 2/3, Mistral, Gemma.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

KV Cache:
    During inference, we cache past K,V tensors so we only compute attention
    for the new token rather than re-computing for the entire context.
    KV cache shape: (B, n_kv_heads, seq_len_so_far, d_head)
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import MicroLMConfig
from .rope import RotaryEmbedding


class GroupedQueryAttention(nn.Module):
    """
    Grouped Query Attention (GQA) with RoPE positional embeddings.

    Args:
        config: MicroLMConfig instance.

    Shapes (throughout):
        B = batch size
        T = sequence length (number of tokens)
        D = d_model (model hidden size)
        H = n_heads (query heads)
        Hkv = n_kv_heads (key/value heads)
        d = d_head = D / H (dimension per head)
        G = n_groups = H / Hkv (query heads per KV head)
    """

    def __init__(self, config: MicroLMConfig) -> None:
        super().__init__()
        self.config = config
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.n_groups = config.n_groups   # = n_heads // n_kv_heads
        self.d_head = config.d_head
        self.d_model = config.d_model
        self.use_flash = config.use_flash_attn

        # ---- Projections ----
        # Q projects to (n_heads * d_head)
        self.q_proj = nn.Linear(config.d_model, config.n_heads * config.d_head, bias=config.bias)
        # K, V project to (n_kv_heads * d_head) — smaller than Q
        self.k_proj = nn.Linear(config.d_model, config.n_kv_heads * config.d_head, bias=config.bias)
        self.v_proj = nn.Linear(config.d_model, config.n_kv_heads * config.d_head, bias=config.bias)
        # Output projection
        self.o_proj = nn.Linear(config.n_heads * config.d_head, config.d_model, bias=config.bias)

        # Dropout
        self.attn_dropout = nn.Dropout(config.attn_dropout)

        # RoPE module (precomputes frequency tables)
        self.rope = RotaryEmbedding(
            d_head=config.d_head,
            ctx_len=config.ctx_len,
            theta=config.rope_theta,
        )

        # Check if Flash Attention is available
        self._flash_available = False
        if self.use_flash:
            try:
                from flash_attn import flash_attn_func  # type: ignore
                self._flash_fn = flash_attn_func
                self._flash_available = True
            except ImportError:
                print("Warning: flash-attn not installed, falling back to standard attention.")

        self._scale = math.sqrt(self.d_head)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[dict] = None,
        start_pos: int = 0,
    ) -> torch.Tensor:
        """
        Args:
            x:        Input tensor of shape (B, T, D)
            mask:     Optional causal mask of shape (T, T) or (B, 1, T, T)
            kv_cache: Dict with 'k' and 'v' tensors for KV caching (inference)
            start_pos: Position offset for KV cache (inference only)

        Returns:
            Output tensor of shape (B, T, D)
        """
        B, T, D = x.shape

        # ---- Linear projections ----
        q = self.q_proj(x)   # (B, T, n_heads * d_head)
        k = self.k_proj(x)   # (B, T, n_kv_heads * d_head)
        v = self.v_proj(x)   # (B, T, n_kv_heads * d_head)

        # ---- Reshape to (B, T, n_heads/n_kv_heads, d_head) ----
        q = q.view(B, T, self.n_heads, self.d_head)
        k = k.view(B, T, self.n_kv_heads, self.d_head)
        v = v.view(B, T, self.n_kv_heads, self.d_head)

        # ---- Apply RoPE rotations to Q and K ----
        q, k = self.rope(q, k, start_pos=start_pos)

        # ---- KV Cache (inference) ----
        if kv_cache is not None:
            if "k" in kv_cache:
                k = torch.cat([kv_cache["k"], k], dim=1)  # (B, past+T, n_kv_heads, d_head)
                v = torch.cat([kv_cache["v"], v], dim=1)
            kv_cache["k"] = k
            kv_cache["v"] = v

        # ---- Transpose to (B, n_heads, T, d_head) for attention computation ----
        q = q.transpose(1, 2)           # (B, H, T, d_head)
        k = k.transpose(1, 2)           # (B, Hkv, T_total, d_head)
        v = v.transpose(1, 2)           # (B, Hkv, T_total, d_head)

        # ---- GQA: expand K,V to match n_heads by repeating each KV group ----
        # Each KV head serves n_groups query heads
        # k: (B, Hkv, T, d) → (B, H, T, d) by repeating each head n_groups times
        k = k.repeat_interleave(self.n_groups, dim=1)  # (B, H, T_total, d_head)
        v = v.repeat_interleave(self.n_groups, dim=1)  # (B, H, T_total, d_head)

        # ---- Scaled Dot-Product Attention ----
        if self._flash_available and self.use_flash:
            # Flash Attention expects (B, T, H, d_head) — retranspose
            q_fa = q.transpose(1, 2)
            k_fa = k.transpose(1, 2)
            v_fa = v.transpose(1, 2)
            # softmax_scale = 1/sqrt(d_head), causal=True for language modeling
            out = self._flash_fn(q_fa, k_fa, v_fa, causal=True)   # (B, T, H, d_head)
            out = out.reshape(B, T, -1)
        else:
            # Standard attention with PyTorch's scaled_dot_product_attention
            # (uses flash-like fused kernel when on CUDA, even without flash-attn package)
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=mask,
                dropout_p=self.config.attn_dropout if self.training else 0.0,
                is_causal=(mask is None),  # Use built-in causal masking if no explicit mask
            )  # (B, H, T, d_head)
            # Transpose back: (B, T, H * d_head)
            out = out.transpose(1, 2).contiguous().reshape(B, T, -1)

        # ---- Output projection ----
        return self.o_proj(out)   # (B, T, D)
