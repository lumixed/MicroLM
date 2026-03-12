"""
microlm/tokenizer/bpe.py

Custom Byte-Pair Encoding (BPE) tokenizer — implemented from scratch.

BPE builds a vocabulary by iteratively merging the most frequent pair of tokens
in the training corpus. This is the same algorithm used by GPT-2/3/4 and Llama.

Key concepts:
  - We start with a byte-level vocabulary (256 bytes), so any text is representable
    without "unknown" tokens.
  - We train by counting pair frequencies and merging the top pair, repeating until
    we reach our target vocab size.
  - Encoding uses a trie/regex-based approach for efficiency.

References:
  - Sennrich et al. 2016 "Neural Machine Translation of Rare Words with Subword Units"
  - GPT-2 paper: Radford et al. 2019
  - tiktoken: https://github.com/openai/tiktoken
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Iterator

from tqdm import tqdm


# ---------------------------------------------------------------------------
# Byte-level pre-tokenization helpers
# ---------------------------------------------------------------------------

def _bytes_to_unicode() -> dict[int, str]:
    """
    Return a dict mapping every byte value (0-255) to a unique printable unicode
    character. This ensures every byte sequence can be represented as a string
    of 'characters' without collisions, following the GPT-2 approach.
    """
    # Start with the printable ASCII range + some latin-1 extras
    bs = list(range(ord("!"), ord("~") + 1))  # 33-126
    bs += list(range(ord("¡"), ord("¬") + 1))  # 161-172
    bs += list(range(ord("®"), ord("ÿ") + 1))  # 174-255

    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


# GPT-2 style regex for splitting text into pre-tokens before BPE
# This prevents merges from crossing word/space/punctuation boundaries
_GPT2_SPLIT_PATTERN = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?\w+| ?\d+| ?[^\s\w\d]+|\s+(?!\S)|\s+""",
    re.UNICODE,
)

# Code-aware split pattern — additionally keeps common code tokens together
_CODE_SPLIT_PATTERN = re.compile(
    r"""'(?:[sdmt]|ll|ve|re)|"""
    r"""(?:self|cls|None|True|False|return|import|from|class|def|if|else|elif|for|while|with|try|except|raise|pass|break|continue|lambda|yield|async|await)\b|"""
    r""" ?[A-Za-z_]\w*|"""   # identifiers
    r""" ?[0-9]+(?:\.[0-9]+)?|"""  # numbers
    r""" ?[^\s\w\d]+|"""     # operators/punctuation
    r"""\n+|\t+| +""",       # whitespace
    re.UNICODE,
)


# ---------------------------------------------------------------------------
# Core BPE implementation
# ---------------------------------------------------------------------------

class BPETokenizer:
    """
    Byte-Pair Encoding tokenizer built from scratch.

    Attributes:
        vocab_size (int): Target vocabulary size.
        merges (dict): Ordered merge rules: {(tok_a, tok_b): merged_tok}
        vocab (dict): token_id -> token_str mapping
        vocab_inv (dict): token_str -> token_id mapping
        special_tokens (dict): str -> token_id for special tokens like <|eos|>
    """

    # Special tokens
    PAD_TOKEN = "<|pad|>"
    BOS_TOKEN = "<|bos|>"
    EOS_TOKEN = "<|eos|>"
    UNK_TOKEN = "<|unk|>"
    SEP_TOKEN = "<|sep|>"

    DEFAULT_SPECIAL_TOKENS = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN, SEP_TOKEN]

    def __init__(self, vocab_size: int = 32_000) -> None:
        if vocab_size < 256:
            raise ValueError("vocab_size must be >= 256 (byte alphabet)")
        self.vocab_size = vocab_size
        self.merges: dict[tuple[str, str], str] = {}
        self.vocab: dict[int, str] = {}
        self.vocab_inv: dict[str, int] = {}
        self.special_tokens: dict[str, int] = {}
        self._byte_encoder = _bytes_to_unicode()
        self._byte_decoder = {v: k for k, v in self._byte_encoder.items()}

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        texts: Iterator[str],
        vocab_size: int | None = None,
        min_frequency: int = 2,
        show_progress: bool = True,
        special_tokens: list[str] | None = None,
        split_pattern: str = "code",
    ) -> None:
        """
        Train BPE merges on an iterator of text strings.

        Args:
            texts: Iterator of training strings (e.g., code files).
            vocab_size: Override self.vocab_size during training.
            min_frequency: Minimum pair frequency to merge.
            show_progress: Show tqdm progress bar.
            special_tokens: Extra special tokens to add to vocab.
            split_pattern: 'code' or 'gpt2' pre-tokenization strategy.
        """
        if vocab_size is not None:
            self.vocab_size = vocab_size

        pattern = _CODE_SPLIT_PATTERN if split_pattern == "code" else _GPT2_SPLIT_PATTERN

        # --- Step 1: Build initial word-frequency corpus ---
        # Each "word" is a tuple of byte-encoded characters + end-of-word marker
        word_freqs: dict[tuple[str, ...], int] = defaultdict(int)

        print("Building initial word frequencies...")
        for text in texts:
            # Encode to bytes, map bytes to unicode chars
            text_bytes = text.encode("utf-8", errors="replace")
            text_str = "".join(self._byte_encoder[b] for b in text_bytes)

            # Pre-tokenize with regex
            for token in pattern.findall(text_str):
                # Represent each pre-token as a tuple of characters (initially each char is a byte symbol)
                word = tuple(token)
                word_freqs[word] += 1

        # --- Step 2: Initialize base vocabulary with all 256 bytes ---
        # Build vocab from byte encoder values
        base_vocab: set[str] = set(self._byte_encoder.values())
        for word in word_freqs:
            for ch in word:
                base_vocab.add(ch)

        # Sort for deterministic ordering
        sorted_base = sorted(base_vocab)
        current_vocab: dict[str, int] = {tok: i for i, tok in enumerate(sorted_base)}

        # --- Step 3: Iteratively merge most frequent pairs ---
        n_merges = self.vocab_size - len(current_vocab) - len(
            special_tokens or self.DEFAULT_SPECIAL_TOKENS
        )
        print(f"Base vocab size: {len(current_vocab)}")
        print(f"Running {n_merges} merges to reach vocab_size={self.vocab_size}...")

        pbar = tqdm(total=n_merges, disable=not show_progress, desc="BPE merges")
        merges_done = 0

        while merges_done < n_merges:
            # Count all adjacent pairs weighted by word frequency
            pair_freqs = self._count_pairs(word_freqs)

            if not pair_freqs:
                break

            # Find most frequent pair
            best_pair = max(pair_freqs, key=lambda p: pair_freqs[p])
            best_freq = pair_freqs[best_pair]

            if best_freq < min_frequency:
                print(f"Stopping early: best pair frequency {best_freq} < min_frequency {min_frequency}")
                break

            # Merge the pair across the whole corpus
            merged_token = best_pair[0] + best_pair[1]
            word_freqs = self._merge_pair(best_pair, word_freqs)
            self.merges[best_pair] = merged_token
            current_vocab[merged_token] = len(current_vocab)

            pbar.update(1)
            merges_done += 1

        pbar.close()

        # --- Step 4: Build final vocab with special tokens at the end ---
        self.vocab = {i: tok for tok, i in current_vocab.items()}
        self.vocab_inv = {tok: i for i, tok in self.vocab.items()}

        # Add special tokens
        _specials = special_tokens or self.DEFAULT_SPECIAL_TOKENS
        for st in _specials:
            idx = len(self.vocab)
            self.vocab[idx] = st
            self.vocab_inv[st] = idx
            self.special_tokens[st] = idx

        print(f"Final vocab size: {len(self.vocab)}")

    # ------------------------------------------------------------------
    # Encoding / Decoding
    # ------------------------------------------------------------------

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
        split_pattern: str = "code",
    ) -> list[int]:
        """
        Encode a string into a list of token IDs.

        Args:
            text: Input text.
            add_bos: Prepend BOS token.
            add_eos: Append EOS token.
            split_pattern: 'code' or 'gpt2' pre-tokenization.

        Returns:
            List of integer token IDs.
        """
        if not self.vocab:
            raise RuntimeError("Tokenizer has not been trained or loaded. Call train() or load().")

        pattern = _CODE_SPLIT_PATTERN if split_pattern == "code" else _GPT2_SPLIT_PATTERN

        # Byte-encode text
        text_bytes = text.encode("utf-8", errors="replace")
        text_str = "".join(self._byte_encoder[b] for b in text_bytes)

        ids: list[int] = []

        if add_bos:
            ids.append(self.special_tokens[self.BOS_TOKEN])

        for pre_token in pattern.findall(text_str):
            # Apply BPE merges
            word = list(pre_token)
            word = self._apply_merges(word)
            for tok in word:
                ids.append(self.vocab_inv.get(tok, self.special_tokens.get(self.UNK_TOKEN, 0)))

        if add_eos:
            ids.append(self.special_tokens[self.EOS_TOKEN])

        return ids

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        """
        Decode a list of token IDs back to string.

        Args:
            ids: List of token IDs.
            skip_special_tokens: If True, strip special tokens from output.

        Returns:
            Decoded string.
        """
        if not self.vocab:
            raise RuntimeError("Tokenizer not trained or loaded.")

        special_ids = set(self.special_tokens.values())
        tokens = []
        for i in ids:
            if skip_special_tokens and i in special_ids:
                continue
            tok = self.vocab.get(i, "")
            tokens.append(tok)

        # Concatenate and byte-decode
        joined = "".join(tokens)
        # Map back from unicode characters to bytes
        byte_vals = bytearray()
        for ch in joined:
            if ch in self._byte_decoder:
                byte_vals.append(self._byte_decoder[ch])
            # Skip unknown chars silently (shouldn't happen with byte-level vocab)
        return byte_vals.decode("utf-8", errors="replace")

    def encode_batch(self, texts: list[str], **kwargs) -> list[list[int]]:
        """Encode a batch of texts."""
        return [self.encode(t, **kwargs) for t in texts]

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Serialize the tokenizer to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "vocab_size": self.vocab_size,
            "vocab": {str(k): v for k, v in self.vocab.items()},
            # Merges stored as list of [tok_a, tok_b] pairs (JSON-serializable)
            "merges": [[a, b] for (a, b) in self.merges.keys()],
            "special_tokens": self.special_tokens,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Tokenizer saved to {path}")

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        """Load a tokenizer from a JSON file."""
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        tok = cls(vocab_size=data["vocab_size"])
        tok.vocab = {int(k): v for k, v in data["vocab"].items()}
        tok.vocab_inv = {v: int(k) for k, v in data["vocab"].items()}
        tok.merges = {(a, b): a + b for a, b in data["merges"]}
        tok.special_tokens = data["special_tokens"]
        return tok

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def pad_id(self) -> int:
        return self.special_tokens.get(self.PAD_TOKEN, 0)

    @property
    def bos_id(self) -> int:
        return self.special_tokens.get(self.BOS_TOKEN, 1)

    @property
    def eos_id(self) -> int:
        return self.special_tokens.get(self.EOS_TOKEN, 2)

    @property
    def unk_id(self) -> int:
        return self.special_tokens.get(self.UNK_TOKEN, 3)

    def __len__(self) -> int:
        return len(self.vocab)

    def __repr__(self) -> str:
        return (
            f"BPETokenizer(vocab_size={len(self.vocab)}, "
            f"n_merges={len(self.merges)}, "
            f"special_tokens={list(self.special_tokens.keys())})"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _count_pairs(
        word_freqs: dict[tuple[str, ...], int],
    ) -> dict[tuple[str, str], int]:
        """
        Count the frequency of all adjacent token pairs across the corpus.

        Complexity: O(sum of word lengths * number of unique words)
        """
        pair_freqs: dict[tuple[str, str], int] = defaultdict(int)
        for word, freq in word_freqs.items():
            for i in range(len(word) - 1):
                pair_freqs[(word[i], word[i + 1])] += freq
        return pair_freqs

    @staticmethod
    def _merge_pair(
        pair: tuple[str, str],
        word_freqs: dict[tuple[str, ...], int],
    ) -> dict[tuple[str, ...], int]:
        """
        Merge all occurrences of `pair` in the word_freqs corpus.
        Returns a new dict with the merged tokens applied.
        """
        new_word_freqs: dict[tuple[str, ...], int] = {}
        merged = pair[0] + pair[1]

        for word, freq in word_freqs.items():
            # Scan word and replace pair with merged token
            new_word: list[str] = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == pair[0] and word[i + 1] == pair[1]:
                    new_word.append(merged)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word_freqs[tuple(new_word)] = freq

        return new_word_freqs

    def _apply_merges(self, word: list[str]) -> list[str]:
        """
        Apply all learned BPE merge rules to a single pre-tokenized word.
        This is the inference-time encoding step.

        Uses a greedy O(n * merges) approach. For production, a trie would be faster.
        """
        if len(word) == 1:
            return word

        # While we can still apply merges, find and apply the highest-priority one
        while len(word) > 1:
            # Find which adjacent pair in `word` has the highest-priority merge
            best_pair: tuple[str, str] | None = None
            best_rank = float("inf")

            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                if pair in self.merges:
                    # Use merge insertion order as priority (lower index = higher priority)
                    rank = list(self.merges.keys()).index(pair)
                    if rank < best_rank:
                        best_rank = rank
                        best_pair = pair

            if best_pair is None:
                break  # No more applicable merges

            # Apply the best merge
            merged = best_pair[0] + best_pair[1]
            new_word: list[str] = []
            i = 0
            while i < len(word):
                if (
                    i < len(word) - 1
                    and word[i] == best_pair[0]
                    and word[i + 1] == best_pair[1]
                ):
                    new_word.append(merged)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = new_word

        return word
