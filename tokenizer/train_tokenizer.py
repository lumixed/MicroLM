"""
microlm/tokenizer/train_tokenizer.py

Script to train the BPE tokenizer on a text corpus.

Usage:
    # Train on a single file
    python tokenizer/train_tokenizer.py --input data/sample.txt --vocab-size 1000

    # Train on a directory of .py files
    python tokenizer/train_tokenizer.py --input data/code/ --vocab-size 32000 --ext .py .js

    # Train on HuggingFace dataset (The Stack)
    python tokenizer/train_tokenizer.py --hf-dataset codeparrot/github-code --vocab-size 32000
"""

import argparse
import sys
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from tokenizer.bpe import BPETokenizer


def iter_file(path: Path, extensions: list[str] | None = None):
    """Yield text from a file or all files in a directory."""
    if path.is_file():
        print(f"Reading file: {path}")
        yield path.read_text(encoding="utf-8", errors="replace")
    elif path.is_dir():
        exts = set(extensions or [".py", ".js", ".ts", ".java", ".c", ".cpp", ".go", ".rs", ".txt"])
        files = [f for f in path.rglob("*") if f.suffix in exts]
        print(f"Found {len(files)} files in {path}")
        for f in files:
            try:
                yield f.read_text(encoding="utf-8", errors="replace")
            except Exception as e:
                print(f"Warning: could not read {f}: {e}")
    else:
        raise FileNotFoundError(f"Input path does not exist: {path}")


def iter_hf_dataset(dataset_name: str, split: str = "train", text_column: str = "code", limit: int | None = None):
    """Stream text from a HuggingFace dataset."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: 'datasets' package required. Run: pip install datasets")
        sys.exit(1)

    print(f"Loading HuggingFace dataset: {dataset_name} (split={split})")
    ds = load_dataset(dataset_name, split=split, streaming=True)
    for i, row in enumerate(ds):
        if limit and i >= limit:
            break
        text = row.get(text_column, "")
        if text:
            yield text


def main():
    parser = argparse.ArgumentParser(description="Train BPE tokenizer for MicroLM")
    parser.add_argument("--input", type=str, default=None, help="Input file or directory")
    parser.add_argument("--hf-dataset", type=str, default=None, help="HuggingFace dataset name")
    parser.add_argument("--hf-split", type=str, default="train", help="HF dataset split")
    parser.add_argument("--hf-column", type=str, default="code", help="HF dataset text column")
    parser.add_argument("--hf-limit", type=int, default=None, help="Max samples from HF dataset")
    parser.add_argument("--vocab-size", type=int, default=32_000, help="Target vocab size")
    parser.add_argument("--min-frequency", type=int, default=2, help="Min pair frequency for merge")
    parser.add_argument("--output", type=str, default="tokenizer/tokenizer.json", help="Output path")
    parser.add_argument("--ext", nargs="+", default=None, help="File extensions to include (for --input dir)")
    parser.add_argument("--split-pattern", choices=["code", "gpt2"], default="code")
    args = parser.parse_args()

    if args.input is None and args.hf_dataset is None:
        parser.error("Either --input or --hf-dataset must be specified.")

    # Build text iterator
    if args.input:
        texts = iter_file(Path(args.input), args.ext)
    else:
        texts = iter_hf_dataset(
            args.hf_dataset,
            split=args.hf_split,
            text_column=args.hf_column,
            limit=args.hf_limit,
        )

    # Train tokenizer
    tok = BPETokenizer(vocab_size=args.vocab_size)
    tok.train(
        texts=texts,
        min_frequency=args.min_frequency,
        show_progress=True,
        split_pattern=args.split_pattern,
    )

    # Save
    out_path = Path(args.output)
    tok.save(out_path)

    # Quick sanity check
    print("\n--- Sanity check ---")
    test_code = "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)"
    ids = tok.encode(test_code, add_bos=True, add_eos=True)
    decoded = tok.decode(ids)
    print(f"Input:   {repr(test_code[:60])}...")
    print(f"Encoded: {ids[:20]}...")
    print(f"Decoded: {repr(decoded[:60])}...")
    compression = len(test_code) / len(ids) if ids else 0
    print(f"Compression ratio: {compression:.2f} chars/token")
    print(f"Tokenizer: {tok}")
    print("Done!")


if __name__ == "__main__":
    main()
