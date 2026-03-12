"""
microlm/inference/generate.py

Inference engine with KV-cache for efficient autoregressive text generation.

Key concepts:
    KV-Cache:
        During inference, at each step we need the Keys and Values from ALL
        previous tokens to compute attention for the NEW token.
        Naive approach: re-compute K,V for all past tokens every step → O(T²) compute.
        KV-cache: Store past K,V tensors → only compute K,V for the NEW token → O(T) compute.

    Sampling strategies:
        Greedy: Always pick the highest-probability token (deterministic, boring).
        Temperature: Divide logits by T before softmax. T→0=greedy, T→∞=uniform random.
        Top-K: Only sample from the K most likely tokens.
        Top-P (Nucleus): Only sample from tokens whose cumulative probability ≥ P.
            (Holtzman et al. 2020 "The Curious Case of Neural Text Degeneration")

Usage:
    python inference/generate.py \\
        --checkpoint checkpoints/best.pt \\
        --tokenizer tokenizer/tokenizer.json \\
        --prompt "def fibonacci(n):" \\
        --max-tokens 200 \\
        --temperature 0.8 \\
        --top-p 0.9
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from model import MicroLM, MicroLMConfig
from tokenizer import BPETokenizer


@torch.no_grad()
def generate(
    model: MicroLM,
    tokenizer: BPETokenizer,
    prompt: str,
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int | None = 50,
    top_p: float | None = 0.9,
    device: torch.device = torch.device("cpu"),
) -> str:
    """
    Autoregressive text generation with KV-cache.

    Args:
        model: Trained MicroLM model.
        tokenizer: Trained BPETokenizer.
        prompt: Text prompt to start generation from.
        max_new_tokens: Maximum number of new tokens to generate.
        temperature: Sampling temperature. 1.0=neutral, <1=more focused, >1=more random.
        top_k: If set, only sample from the top-K logits.
        top_p: If set, nucleus sampling threshold.
        device: Torch device.

    Returns:
        Generated text string (prompt + new tokens).
    """
    model.eval()
    model = model.to(device)

    # Encode prompt
    input_ids = tokenizer.encode(prompt, add_bos=True)
    tokens = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)  # (1, T)

    # Initialize KV cache for each layer
    kv_caches = model.init_kv_caches()

    # Pre-fill: process the full prompt in one forward pass to populate KV cache
    if tokens.shape[1] > 1:
        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
            _, _ = model(tokens[:, :-1], kv_caches=kv_caches, start_pos=0)
        start_pos = tokens.shape[1] - 1
    else:
        start_pos = 0

    # Generate new tokens one at a time
    generated_ids = list(input_ids)
    current_token = tokens[:, -1:]  # (1, 1) — last token of prompt

    for step in range(max_new_tokens):
        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
            logits, _ = model(
                current_token,
                kv_caches=kv_caches,
                start_pos=start_pos + step,
            )  # logits: (1, 1, vocab_size)

        logits = logits[:, -1, :]  # (1, vocab_size) — logits for next token

        # Apply temperature
        if temperature != 1.0:
            logits = logits / temperature

        # Top-K filtering
        if top_k is not None:
            top_k = min(top_k, logits.size(-1))
            topk_logits, _ = torch.topk(logits, top_k)
            threshold = topk_logits[:, -1].unsqueeze(-1)
            logits = logits.masked_fill(logits < threshold, float("-inf"))

        # Top-P (nucleus) filtering
        if top_p is not None:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            # Remove tokens with cumulative prob above the threshold
            sorted_indices_to_remove = cumulative_probs - F.softmax(sorted_logits, dim=-1) > top_p
            sorted_logits[sorted_indices_to_remove] = float("-inf")
            # Scatter back to original ordering
            logits = torch.zeros_like(logits).scatter_(1, sorted_indices, sorted_logits)

        # Sample from the distribution
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)  # (1, 1)

        token_id = next_token.item()
        generated_ids.append(token_id)

        # Stop at EOS
        if token_id == tokenizer.eos_id:
            break

        current_token = next_token

    return tokenizer.decode(generated_ids, skip_special_tokens=True)


def load_model_from_checkpoint(checkpoint_path: str, device: torch.device) -> tuple[MicroLM, MicroLMConfig]:
    """Load model and config from a checkpoint file."""
    ckpt = torch.load(checkpoint_path, map_location=device)
    config = ckpt["model_config"]
    model = MicroLM(config)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, config


def main():
    parser = argparse.ArgumentParser(description="Generate text with MicroLM")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pt checkpoint")
    parser.add_argument("--tokenizer", type=str, default="tokenizer/tokenizer.json")
    parser.add_argument("--prompt", type=str, default="def hello_world():")
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--num-samples", type=int, default=1)
    args = parser.parse_args()

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Load model
    print(f"Loading checkpoint: {args.checkpoint}")
    model, config = load_model_from_checkpoint(args.checkpoint, device)
    print(f"Model: {model.num_params/1e6:.1f}M params")

    # Load tokenizer
    tokenizer = BPETokenizer.load(args.tokenizer)
    print(f"Tokenizer: {tokenizer}")

    # Generate
    print(f"\nPrompt: {repr(args.prompt)}\n{'─'*60}")
    for i in range(args.num_samples):
        output = generate(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            device=device,
        )
        print(f"\n[Sample {i+1}]\n{output}")
        print("─" * 60)


if __name__ == "__main__":
    main()
