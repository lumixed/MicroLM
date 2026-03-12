"""
microlm/data/preprocess.py

Tokenize raw code files into binary shards for training.

Usage:
    # Tokenize a directory of code files
    python data/preprocess.py \\
        --input data/raw/ \\
        --tokenizer tokenizer/tokenizer.json \\
        --output data/tokens/ \\
        --shard-size 10000000

This converts text files into flat uint16 arrays of token IDs,
split into shards of ~10M tokens each, stored as .bin files.
The training DataLoader reads these with memory-mapping.
"""

import sys
import argparse
import struct
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from tokenizer import BPETokenizer


def preprocess(
    input_path: str | Path,
    tokenizer_path: str | Path,
    output_dir: str | Path,
    shard_size: int = 10_000_000,
    extensions: list[str] | None = None,
) -> None:
    """
    Tokenize text files and write binary shards.

    Args:
        input_path: File or directory of source code.
        tokenizer_path: Path to trained tokenizer JSON.
        output_dir: Output directory for .bin shards.
        shard_size: Approximate tokens per shard file.
        extensions: File extensions to include.
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load tokenizer
    print(f"Loading tokenizer from {tokenizer_path}")
    tok = BPETokenizer.load(tokenizer_path)
    print(f"Tokenizer: {tok}")

    # Gather files
    exts = set(extensions or [".py", ".js", ".ts", ".java", ".c", ".cpp", ".go", ".rs", ".md"])
    if input_path.is_file():
        files = [input_path]
    else:
        files = [f for f in input_path.rglob("*") if f.suffix in exts]
    print(f"Found {len(files)} files to tokenize")

    # Process files and write shards
    shard_idx = 0
    shard_tokens: list[int] = []
    total_tokens = 0

    def write_shard(tokens: list[int], idx: int) -> None:
        arr = np.array(tokens, dtype=np.uint16)
        shard_path = output_dir / f"shard_{idx:05d}.bin"
        arr.tofile(shard_path)
        print(f"  Wrote shard {idx}: {len(tokens):,} tokens → {shard_path}")

    for fpath in tqdm(files, desc="Tokenizing"):
        try:
            text = fpath.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue

        # Tokenize with BOS/EOS markers between documents
        ids = tok.encode(text, add_bos=True, add_eos=True)
        shard_tokens.extend(ids)
        total_tokens += len(ids)

        # Flush shard when full
        if len(shard_tokens) >= shard_size:
            write_shard(shard_tokens, shard_idx)
            shard_idx += 1
            shard_tokens = []

    # Write remaining tokens
    if shard_tokens:
        write_shard(shard_tokens, shard_idx)
        shard_idx += 1

    print(f"\nDone! {total_tokens:,} total tokens across {shard_idx} shards.")
    print(f"Output: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Tokenize data into binary shards")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, default="tokenizer/tokenizer.json")
    parser.add_argument("--output", type=str, default="data/tokens")
    parser.add_argument("--shard-size", type=int, default=10_000_000)
    parser.add_argument("--ext", nargs="+", default=None)
    args = parser.parse_args()

    preprocess(
        input_path=args.input,
        tokenizer_path=args.tokenizer,
        output_dir=args.output,
        shard_size=args.shard_size,
        extensions=args.ext,
    )


if __name__ == "__main__":
    main()
