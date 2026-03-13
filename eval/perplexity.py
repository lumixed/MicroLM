"""
eval/perplexity.py

Evaluate MicroLM on a text file and report perplexity.

Perplexity = exp(average cross-entropy loss over all tokens).
Lower is better. A random model over a 32k vocab would score ~32,000.
Well-pretrained 125M models typically score 15-30 on WikiText-103.

Usage:
    python eval/perplexity.py \\
        --checkpoint checkpoints/microlm-125m-final.pt \\
        --tokenizer tokenizer/tokenizer.json \\
        --text data/raw/wikitext103.txt \\
        --ctx-len 1024
"""

from __future__ import annotations

import sys
import argparse
import math
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from model import MicroLM, MicroLMConfig
from tokenizer.bpe import BPETokenizer


@torch.no_grad()
def evaluate_perplexity(
    model: MicroLM,
    token_ids: list[int],
    ctx_len: int,
    batch_size: int,
    device: torch.device,
) -> float:
    """
    Compute perplexity over a flat list of token IDs using non-overlapping windows.

    Args:
        model:      MicroLM in eval mode.
        token_ids:  Flat list of token IDs (the whole corpus).
        ctx_len:    Context window length.
        batch_size: Batch size for evaluation.
        device:     Torch device.

    Returns:
        Perplexity (scalar float).
    """
    model.eval()

    # Build (input, target) windows
    windows = []
    for i in range(0, len(token_ids) - ctx_len, ctx_len):
        chunk = token_ids[i : i + ctx_len + 1]
        windows.append((chunk[:-1], chunk[1:]))

    if not windows:
        raise ValueError("Text too short for the given ctx_len.")

    total_loss = 0.0
    total_tokens = 0

    for i in range(0, len(windows), batch_size):
        batch = windows[i : i + batch_size]
        inputs = torch.tensor([w[0] for w in batch], dtype=torch.long).to(device)
        targets = torch.tensor([w[1] for w in batch], dtype=torch.long).to(device)

        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
            logits, _ = model(inputs)

        # Cross-entropy over all tokens in the batch
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            reduction="sum",
        )
        total_loss += loss.item()
        total_tokens += targets.numel()

        if (i // batch_size) % 10 == 0:
            running_ppl = math.exp(min(total_loss / total_tokens, 20))
            print(f"  [{i+len(batch)}/{len(windows)} windows] running ppl: {running_ppl:.2f}")

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(min(avg_loss, 20))
    return perplexity


def main():
    parser = argparse.ArgumentParser(description="Evaluate MicroLM perplexity")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, default="tokenizer/tokenizer.json")
    parser.add_argument("--text", type=str, required=True, help="Path to evaluation .txt file")
    parser.add_argument("--ctx-len", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=None, help="Cap evaluation tokens")
    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}")

    # Load tokenizer
    tokenizer = BPETokenizer.load(args.tokenizer)
    print(f"Tokenizer: {tokenizer.vocab_size} tokens")

    # Load model
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model_cfg = ckpt.get("model_config", MicroLMConfig())
    model = MicroLM(model_cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Model: {model.param_count():,} parameters")

    # Tokenize eval text
    print(f"Tokenizing: {args.text}")
    with open(args.text, "r", encoding="utf-8") as f:
        text = f.read()
    token_ids = tokenizer.encode(text)
    if args.max_tokens:
        token_ids = token_ids[: args.max_tokens]
    print(f"Eval tokens: {len(token_ids):,}")

    # Evaluate
    ppl = evaluate_perplexity(model, token_ids, args.ctx_len, args.batch_size, device)
    print(f"\n{'='*40}")
    print(f"  Perplexity: {ppl:.2f}")
    print(f"  NLL loss:   {math.log(ppl):.4f}")
    print(f"{'='*40}")


if __name__ == "__main__":
    main()
