"""
microlm/model/rope.py

Rotary Position Embedding (RoPE) — implemented from scratch.

Reference: Su et al. 2021 "RoFormer: Enhanced Transformer with Rotary Position Embedding"
https://arxiv.org/abs/2104.09864

Key insight:
    Instead of adding position info to tokens, RoPE rotates query and key vectors
    by an angle proportional to their position. The dot product Q·K then naturally
    encodes *relative* position, which is more generalizable than absolute positions.

Math:
    For a vector x at position m, RoPE applies a rotation matrix R_m:
        x_rotated = R_m * x

    For position m, dimension 2i and 2i+1:
        theta_i = m / (base ** (2i / d_head))
        [x_{2i}   ]   [cos(theta_i)  -sin(theta_i)] [x_{2i}   ]
        [x_{2i+1} ] = [sin(theta_i)   cos(theta_i)] [x_{2i+1} ]

    This is implemented efficiently using complex number multiplication.

Properties:
    - Relative position encoding via dot-product invariance
    - No learned parameters (pure function of position)
    - Extrapolates better to longer sequences than absolute embeddings
    - Used in: Llama, Gemma, Mistral, Falcon, GPT-NeoX
"""

import torch
import torch.nn as nn


def precompute_freqs_cis(d_head: int, ctx_len: int, theta: float = 10_000.0) -> torch.Tensor:
    """
    Precompute the complex exponentials (cos + i*sin) for RoPE.

    This computes e^{i * m * theta_j} for all positions m and dimension pairs j.
    Stored as complex64 for efficient rotation via element-wise multiplication.

    Args:
        d_head (int): Dimension per attention head. Must be even.
        ctx_len (int): Maximum sequence length to precompute for.
        theta (float): RoPE base frequency. 10_000 is standard GPT-NeoX/Llama default.
                       Use 500_000 for extended context (Llama 3.1+).

    Returns:
        Tensor of shape (ctx_len, d_head // 2) with dtype=complex64.
    """
    assert d_head % 2 == 0, "d_head must be even for RoPE"

    # theta_i = 1 / (theta ^ (2i / d_head)) for i in [0, 1, ..., d_head/2 - 1]
    # Shape: (d_head // 2,)
    freqs = 1.0 / (theta ** (torch.arange(0, d_head, 2).float() / d_head))

    # Position indices: shape (ctx_len,)
    positions = torch.arange(ctx_len, dtype=torch.float32)

    # Outer product: shape (ctx_len, d_head // 2)
    # freqs_outer[m, j] = m * freqs[j]
    freqs_outer = torch.outer(positions, freqs)

    # Convert to complex: e^{i * theta} = cos(theta) + i*sin(theta)
    # Shape: (ctx_len, d_head // 2), dtype=complex64
    freqs_cis = torch.polar(torch.ones_like(freqs_outer), freqs_outer)

    return freqs_cis


def apply_rotary_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply RoPE rotations to query and key tensors.

    Args:
        q: Query tensor, shape (B, T, n_heads, d_head)
        k: Key tensor,   shape (B, T, n_kv_heads, d_head)
        freqs_cis: Precomputed complex freqs, shape (T, d_head // 2)

    Returns:
        Rotated (q, k) with same shapes as input.
    """
    # Reshape to complex: last dim pairs up → (B, T, n_heads, d_head//2) as complex64
    # View as complex by treating every 2 consecutive floats as real+imag
    q_complex = torch.view_as_complex(q.float().reshape(*q.shape[:-1], -1, 2))
    k_complex = torch.view_as_complex(k.float().reshape(*k.shape[:-1], -1, 2))

    # freqs_cis: (T, d_head//2) → (1, T, 1, d_head//2) for broadcasting
    freqs = freqs_cis[: q.shape[1]].unsqueeze(0).unsqueeze(2)

    # Rotate: element-wise complex multiplication applies the rotation
    q_rotated = torch.view_as_real(q_complex * freqs).flatten(start_dim=-2)
    k_rotated = torch.view_as_real(k_complex * freqs).flatten(start_dim=-2)

    return q_rotated.type_as(q), k_rotated.type_as(k)


class RotaryEmbedding(nn.Module):
    """
    Module wrapper around RoPE that precomputes and caches freq tensors.

    Usage:
        rope = RotaryEmbedding(d_head=64, ctx_len=1024)
        q_rot, k_rot = rope(q, k, position_ids)
    """

    def __init__(self, d_head: int, ctx_len: int, theta: float = 10_000.0) -> None:
        super().__init__()
        self.d_head = d_head
        self.ctx_len = ctx_len
        self.theta = theta

        # Register as non-parameter buffer (moves with .to(device) but not trained)
        freqs_cis = precompute_freqs_cis(d_head, ctx_len, theta)
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        start_pos: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            q: (B, T, n_heads, d_head)
            k: (B, T, n_kv_heads, d_head)
            start_pos: Starting position (for KV-cache offset during inference)

        Returns:
            Rotated (q, k)
        """
        seq_len = q.shape[1]
        freqs = self.freqs_cis[start_pos: start_pos + seq_len]
        return apply_rotary_emb(q, k, freqs)
