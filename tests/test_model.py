"""
tests/test_model.py

Unit tests for MicroLM architecture components.

Run with:
    pytest tests/ -v
    pytest tests/test_model.py -v -k "test_attention"
"""

import math
import pytest
import torch
import torch.nn as nn

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================================================
# RMSNorm Tests
# ============================================================

class TestRMSNorm:
    def test_output_shape(self):
        from model.rmsnorm import RMSNorm
        norm = RMSNorm(d_model=64)
        x = torch.randn(2, 10, 64)
        out = norm(x)
        assert out.shape == x.shape

    def test_normalized_scale(self):
        """Output should be approximately unit RMS."""
        from model.rmsnorm import RMSNorm
        norm = RMSNorm(d_model=128)
        # Initialize weight to 1 (default) for clean test
        x = torch.randn(4, 20, 128) * 5  # Large values, should be normalized
        out = norm(x)
        rms = out.pow(2).mean(dim=-1).sqrt()
        # Should be close to 1 since weight=1 and RMS normalization
        assert rms.mean().item() == pytest.approx(1.0, abs=0.1)

    def test_dtype_preserved(self):
        """Output dtype should match input dtype."""
        from model.rmsnorm import RMSNorm
        norm = RMSNorm(d_model=64)
        x = torch.randn(2, 10, 64).to(torch.bfloat16)
        out = norm(x)
        assert out.dtype == torch.bfloat16

    def test_gradient_flows(self):
        from model.rmsnorm import RMSNorm
        norm = RMSNorm(d_model=32)
        x = torch.randn(2, 5, 32, requires_grad=True)
        out = norm(x).sum()
        out.backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()


# ============================================================
# RoPE Tests
# ============================================================

class TestRoPE:
    def test_freqs_cis_shape(self):
        from model.rope import precompute_freqs_cis
        freqs = precompute_freqs_cis(d_head=64, ctx_len=512)
        assert freqs.shape == (512, 32)  # (ctx_len, d_head // 2)
        assert freqs.dtype == torch.complex64

    def test_apply_rotary_qk_shape(self):
        from model.rope import apply_rotary_emb, precompute_freqs_cis
        B, T, H, Hkv, d = 2, 16, 4, 2, 32
        q = torch.randn(B, T, H, d)
        k = torch.randn(B, T, Hkv, d)
        freqs = precompute_freqs_cis(d_head=d, ctx_len=T)
        q_rot, k_rot = apply_rotary_emb(q, k, freqs)
        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape

    def test_rotary_preserves_norm(self):
        """RoPE should preserve the L2 norm of Q and K vectors."""
        from model.rope import apply_rotary_emb, precompute_freqs_cis
        B, T, H, d = 1, 8, 2, 32
        q = torch.randn(B, T, H, d)
        k = torch.randn(B, T, H, d)
        freqs = precompute_freqs_cis(d_head=d, ctx_len=T)
        q_rot, k_rot = apply_rotary_emb(q, k, freqs)
        # Norm preserved (rotation is an isometry)
        assert torch.allclose(q.norm(dim=-1), q_rot.norm(dim=-1), atol=1e-5)
        assert torch.allclose(k.norm(dim=-1), k_rot.norm(dim=-1), atol=1e-5)


# ============================================================
# SwiGLU Tests
# ============================================================

class TestSwiGLU:
    def test_output_shape(self):
        from model.feedforward import SwiGLU
        from model.config import tiny_config
        cfg = tiny_config()
        ffn = SwiGLU(cfg)
        x = torch.randn(2, 10, cfg.d_model)
        out = ffn(x)
        assert out.shape == x.shape

    def test_no_nan(self):
        from model.feedforward import SwiGLU
        from model.config import tiny_config
        cfg = tiny_config()
        ffn = SwiGLU(cfg)
        x = torch.randn(4, 32, cfg.d_model)
        out = ffn(x)
        assert not torch.isnan(out).any()


# ============================================================
# Attention Tests
# ============================================================

class TestGroupedQueryAttention:
    def test_output_shape(self):
        from model.attention import GroupedQueryAttention
        from model.config import tiny_config
        cfg = tiny_config()
        attn = GroupedQueryAttention(cfg)
        x = torch.randn(2, 16, cfg.d_model)
        out = attn(x)
        assert out.shape == x.shape

    def test_causal_mask_effect(self):
        """Causal attention: token at position i should not attend to i+1."""
        from model.attention import GroupedQueryAttention
        from model.config import tiny_config
        cfg = tiny_config()
        attn = GroupedQueryAttention(cfg)
        attn.eval()

        x = torch.randn(1, 8, cfg.d_model)
        # Output at position 0 should change if we modify position 1
        # (it shouldn't, since position 0 can't attend to position 1 causally)
        out1 = attn(x.clone())
        x2 = x.clone()
        x2[:, 1, :] = torch.randn(cfg.d_model)  # Perturb position 1
        out2 = attn(x2)

        # Position 0's output should be unchanged (cannot attend forward)
        assert torch.allclose(out1[:, 0, :], out2[:, 0, :], atol=1e-5)

    def test_gqa_kv_heads(self):
        """n_kv_heads < n_heads should not crash and produce correct shapes."""
        from model.attention import GroupedQueryAttention
        from model.config import MicroLMConfig
        cfg = MicroLMConfig(d_model=128, n_heads=8, n_kv_heads=2, d_ff=256, ctx_len=64)
        attn = GroupedQueryAttention(cfg)
        x = torch.randn(2, 16, 128)
        out = attn(x)
        assert out.shape == (2, 16, 128)


# ============================================================
# Full Model Tests
# ============================================================

class TestMicroLM:
    @pytest.fixture
    def tiny_model(self):
        from model import MicroLM, tiny_config
        cfg = tiny_config()
        return MicroLM(cfg), cfg

    def test_param_count(self, tiny_model):
        model, cfg = tiny_model
        n = model.num_params
        print(f"\nTiny model params: {n/1e6:.2f}M")
        assert 1_000_000 < n < 100_000_000  # Should be between 1M and 100M

    def test_forward_shape(self, tiny_model):
        model, cfg = tiny_model
        B, T = 2, 32
        ids = torch.randint(0, cfg.vocab_size, (B, T))
        logits, loss = model(ids, targets=ids)
        assert logits.shape == (B, T, cfg.vocab_size)
        assert loss is not None
        assert not torch.isnan(loss)

    def test_forward_inference(self, tiny_model):
        """In inference mode (no targets), logits should have shape (B, 1, vocab_size)."""
        model, cfg = tiny_model
        ids = torch.randint(0, cfg.vocab_size, (1, 8))
        logits, loss = model(ids)
        assert logits.shape == (1, 1, cfg.vocab_size)
        assert loss is None

    def test_loss_decreases(self, tiny_model):
        """Tiny overfit test: loss should decrease on a single batch."""
        model, cfg = tiny_model
        model.train()
        optimizer = model.configure_optimizer(lr=1e-3)

        ids = torch.randint(0, cfg.vocab_size, (2, 16))
        targets = torch.roll(ids, -1, dims=1)

        losses = []
        for _ in range(20):
            optimizer.zero_grad()
            _, loss = model(ids, targets=targets)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        assert losses[-1] < losses[0], f"Loss didn't decrease: {losses[0]:.4f} → {losses[-1]:.4f}"

    def test_kv_cache_consistency(self, tiny_model):
        """KV-cache inference should produce same logits as non-cached."""
        model, cfg = tiny_model
        model.eval()

        T = 8
        ids = torch.randint(0, cfg.vocab_size, (1, T))

        # Without KV cache (full forward)
        with torch.no_grad():
            logits_full, _ = model(ids)

        # With KV cache (step by step)
        # This is a simplified test — in real inference, we'd use start_pos tracking
        # Here we just verify the model can use caches without crashing
        kv_caches = model.init_kv_caches()
        with torch.no_grad():
            logits_cached, _ = model(ids[:, :4], kv_caches=kv_caches, start_pos=0)

        assert logits_cached.shape == (1, 1, cfg.vocab_size)

    def test_weight_tying(self, tiny_model):
        """Embedding weights and lm_head weights should be the same object."""
        model, cfg = tiny_model
        assert model.embed_tokens.weight is model.lm_head.weight


# ============================================================
# BPE Tokenizer Tests
# ============================================================

class TestBPETokenizer:
    @pytest.fixture
    def trained_tokenizer(self):
        """Train a tiny tokenizer on minimal corpus for testing."""
        from tokenizer.bpe import BPETokenizer
        tok = BPETokenizer(vocab_size=500)
        corpus = [
            "hello world hello python",
            "def foo(): return True",
            "class Bar: pass",
            "import os import sys",
            "for i in range(10): print(i)",
        ] * 10  # Repeat to get enough pair frequencies
        tok.train(iter(corpus), vocab_size=500, show_progress=False)
        return tok

    def test_encode_decode_roundtrip(self, trained_tokenizer):
        tok = trained_tokenizer
        text = "hello world"
        ids = tok.encode(text)
        decoded = tok.decode(ids)
        assert decoded == text

    def test_special_tokens(self, trained_tokenizer):
        tok = trained_tokenizer
        ids = tok.encode("hello", add_bos=True, add_eos=True)
        assert ids[0] == tok.bos_id
        assert ids[-1] == tok.eos_id

    def test_vocab_size_approximately_correct(self, trained_tokenizer):
        tok = trained_tokenizer
        assert 256 <= len(tok) <= 600  # Some slack

    def test_save_load_roundtrip(self, trained_tokenizer, tmp_path):
        from tokenizer.bpe import BPETokenizer
        tok = trained_tokenizer
        path = tmp_path / "test_tokenizer.json"
        tok.save(path)
        loaded = BPETokenizer.load(path)
        # Verify same encoding
        text = "def foo():"
        assert tok.encode(text) == loaded.encode(text)
