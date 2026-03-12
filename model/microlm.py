"""
microlm/model/microlm.py

MicroLM — Full GPT-style causal language model.

Architecture overview (pre-norm decoder-only transformer):
    Input tokens
        ↓
    Token Embedding (vocab_size → d_model)
        ↓
    N × TransformerBlock (GQA + RoPE + SwiGLU, pre-RMSNorm)
        ↓
    Final RMSNorm
        ↓
    LM Head (d_model → vocab_size) [optionally tied to embedding weights]
        ↓
    Logits / Loss

Key design choices:
    - No positional embedding table — RoPE handles position inside each attention layer
    - Pre-norm (RMSNorm before each sublayer) for training stability
    - Weight tying: embedding weights shared with LM head (halves embedding memory)
    - Gradient checkpointing compatible (set use_checkpoint=True in training)
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import MicroLMConfig
from .transformer_block import TransformerBlock
from .rmsnorm import RMSNorm


class MicroLM(nn.Module):
    """
    Full MicroLM decoder-only language model.

    Args:
        config: MicroLMConfig instance controlling model dimensions.

    Example:
        >>> config = MicroLMConfig()   # 125M config
        >>> model = MicroLM(config)
        >>> print(f"Parameters: {model.num_params/1e6:.1f}M")
        Parameters: 124.4M
        >>> tokens = torch.randint(0, config.vocab_size, (2, 64))
        >>> logits, loss = model(tokens, targets=tokens)
        >>> logits.shape
        torch.Size([2, 64, 32000])
    """

    def __init__(self, config: MicroLMConfig) -> None:
        super().__init__()
        self.config = config

        # ---- Token embedding ----
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)

        # ---- Transformer layers ----
        self.layers = nn.ModuleList(
            [TransformerBlock(config, layer_idx=i) for i in range(config.n_layers)]
        )

        # ---- Final normalization ----
        self.norm = RMSNorm(config.d_model)

        # ---- LM head (token prediction) ----
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # ---- Weight tying ----
        # The input embedding and output LM head share weights.
        # This saves ~vocab_size * d_model parameters and often improves quality.
        # Reference: Press & Wolf 2017 "Using the Output Embedding to Improve Language Models"
        if config.tie_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        # ---- Weight initialization ----
        self.apply(self._init_weights)
        # Scale residual projections by 1/sqrt(2 * n_layers) (GPT-2 trick)
        # This prevents activation variance from growing with depth
        scale = (2 * config.n_layers) ** -0.5
        for name, p in self.named_parameters():
            if name.endswith(("o_proj.weight", "down_proj.weight")):
                nn.init.normal_(p, mean=0.0, std=0.02 * scale)

    def _init_weights(self, module: nn.Module) -> None:
        """Standard weight initialization."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        kv_caches: Optional[list[dict]] = None,
        start_pos: int = 0,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.

        Args:
            input_ids: Token indices, shape (B, T)
            targets:   Target token indices for loss computation, shape (B, T).
                       If None, only logits are returned (inference mode).
            mask:      Optional explicit attention mask (B, 1, T, T).
                       If None, causal mask is applied automatically.
            kv_caches: List of per-layer KV cache dicts (for inference).
            start_pos: Position offset for KV cache.

        Returns:
            logits: Raw unnormalized token logits, shape (B, T, vocab_size)
            loss:   Cross-entropy loss (scalar) if targets provided, else None.
        """
        B, T = input_ids.shape
        assert T <= self.config.ctx_len, (
            f"Sequence length {T} exceeds context length {self.config.ctx_len}"
        )

        # ---- Embed tokens ----
        x = self.embed_tokens(input_ids)   # (B, T, D)

        # ---- Pass through transformer layers ----
        for i, layer in enumerate(self.layers):
            kv_cache = kv_caches[i] if kv_caches is not None else None
            x = layer(x, mask=mask, kv_cache=kv_cache, start_pos=start_pos)

        # ---- Final norm ----
        x = self.norm(x)   # (B, T, D)

        # ---- Compute logits ----
        if targets is not None:
            # Training: compute logits for all positions
            logits = self.lm_head(x)   # (B, T, vocab_size)
            # Cross-entropy loss — flatten B and T
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),   # (B*T, vocab_size)
                targets.view(-1),                   # (B*T,)
                ignore_index=-1,                    # Ignore padding positions
            )
        else:
            # Inference: only compute logits for the last token (efficiency)
            logits = self.lm_head(x[:, [-1], :])   # (B, 1, vocab_size)
            loss = None

        return logits, loss

    @property
    def num_params(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def init_kv_caches(self) -> list[dict]:
        """Initialize empty KV cache dicts for all layers (inference)."""
        return [{} for _ in range(self.config.n_layers)]

    def configure_optimizer(
        self,
        lr: float = 3e-4,
        weight_decay: float = 0.1,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        device: str = "cpu",
    ) -> torch.optim.AdamW:
        """
        Configure AdamW optimizer with weight decay applied only to ≥2D parameters.

        Following GPT-3 / Llama convention:
          - Embeddings, norms, and biases: NO weight decay (they're 1D or learned scales)
          - All other weights: weight decay applied
        """
        decay_params = [p for n, p in self.named_parameters() if p.dim() >= 2]
        nodecay_params = [p for n, p in self.named_parameters() if p.dim() < 2]

        param_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]

        n_decay = sum(p.numel() for p in decay_params)
        n_nodecay = sum(p.numel() for p in nodecay_params)
        print(f"Optimizer: {n_decay/1e6:.2f}M params with decay, {n_nodecay/1e6:.2f}M without")

        # Use fused AdamW if on CUDA (significant speedup)
        fused = "cuda" in device
        return torch.optim.AdamW(param_groups, lr=lr, betas=betas, eps=eps, fused=fused)

    def __repr__(self) -> str:
        return (
            f"MicroLM(\n"
            f"  config={self.config}\n"
            f"  params={self.num_params/1e6:.2f}M\n"
            f")"
        )
