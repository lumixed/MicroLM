"""
microlm/model/config.py

Model configuration dataclass.
All hyperparameters live here — change this to scale up or down.
"""

from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class MicroLMConfig:
    """
    Configuration for MicroLM transformer.

    Presets:
        tiny   — ~15M  params (fast for local dev/testing on MacBook)
        small  — ~125M params (target training config)
        medium — ~350M params (future scaling)
    """

    # ---- Model dimensions ----
    vocab_size: int = 32_000        # BPE vocabulary size
    d_model: int = 768              # Hidden dimension (embedding size)
    n_layers: int = 12              # Number of transformer layers
    n_heads: int = 12               # Number of query attention heads
    n_kv_heads: int = 4            # Number of key/value heads (GQA < n_heads)
    d_ff: int = 2048               # FFN hidden dimension (SwiGLU)
    ctx_len: int = 1024            # Maximum context length (sequence length)

    # ---- Architecture flags ----
    bias: bool = False              # Whether to use bias in linear layers (False = modern)
    tie_embeddings: bool = True     # Tie input embedding weights to output lm_head

    # ---- RoPE ----
    rope_theta: float = 10_000.0   # RoPE base frequency (10k = standard, 500k = long ctx)

    # ---- Dropout ----
    attn_dropout: float = 0.0      # Attention dropout (usually 0 for large models)
    residual_dropout: float = 0.0  # Residual stream dropout

    # ---- Flash Attention ----
    use_flash_attn: bool = False    # Enable Flash Attention (requires CUDA + flash-attn pkg)

    @property
    def d_head(self) -> int:
        """Dimension per attention head."""
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        return self.d_model // self.n_heads

    @property
    def n_groups(self) -> int:
        """GQA: number of query heads per KV head group."""
        assert self.n_heads % self.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        return self.n_heads // self.n_kv_heads

    def estimate_params(self) -> int:
        """Rough parameter count estimate."""
        embedding = self.vocab_size * self.d_model
        attention = self.n_layers * (
            self.d_model * (self.n_heads + 2 * self.n_kv_heads) * self.d_head  # QKV
            + self.d_model * self.d_model  # output proj
        )
        # SwiGLU FFN: 3 matrices (gate, up, down)
        ffn = self.n_layers * 3 * self.d_model * self.d_ff
        norm = self.n_layers * 2 * self.d_model + self.d_model  # pre-norm + final norm
        lm_head = 0 if self.tie_embeddings else self.vocab_size * self.d_model
        return embedding + attention + ffn + norm + lm_head

    def __repr__(self) -> str:
        n = self.estimate_params()
        return (
            f"MicroLMConfig("
            f"d_model={self.d_model}, n_layers={self.n_layers}, n_heads={self.n_heads}, "
            f"n_kv_heads={self.n_kv_heads}, d_ff={self.d_ff}, ctx_len={self.ctx_len}, "
            f"vocab_size={self.vocab_size}, "
            f"~{n/1e6:.1f}M params)"
        )


# ---------------------------------------------------------------------------
# Preset configurations
# ---------------------------------------------------------------------------

def tiny_config() -> MicroLMConfig:
    """~15M params — for local MacBook testing."""
    return MicroLMConfig(
        d_model=256,
        n_layers=6,
        n_heads=8,
        n_kv_heads=2,
        d_ff=512,
        ctx_len=512,
        vocab_size=8_000,
    )


def small_config() -> MicroLMConfig:
    """~125M params — main training target."""
    return MicroLMConfig()  # defaults are the 125M config


def medium_config() -> MicroLMConfig:
    """~350M params — future scale-up."""
    return MicroLMConfig(
        d_model=1024,
        n_layers=24,
        n_heads=16,
        n_kv_heads=4,
        d_ff=2816,
        ctx_len=2048,
        vocab_size=32_000,
    )
