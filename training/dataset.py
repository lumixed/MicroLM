"""
microlm/training/dataset.py

Token dataset for language model training.

We pre-tokenize the corpus into binary files of uint16 token IDs,
then stream them with a memory-mapped array during training.
This avoids loading the entire dataset into RAM.

Sharding strategy:
    The corpus is split into .bin shard files, each containing a flat
    array of token IDs (uint16). The DataLoader reads random windows
    of ctx_len+1 tokens from a randomly chosen shard.

File format: each shard is a NumPy uint16 binary with a 1-token header:
    [vocab_size (uint32)] [token_id, token_id, ...]
"""

from __future__ import annotations

import os
import struct
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class TokenDataset(Dataset):
    """
    Language modeling dataset that loads from pre-tokenized binary shards.

    Args:
        data_dir: Directory containing .bin shard files.
        ctx_len: Sequence length per sample.
        split: 'train' or 'val' (uses different shards).
        val_shards: Number of shards to reserve for validation.
    """

    def __init__(
        self,
        data_dir: str | Path,
        ctx_len: int,
        split: str = "train",
        val_shards: int = 1,
    ) -> None:
        self.ctx_len = ctx_len
        self.split = split

        data_dir = Path(data_dir)
        shards = sorted(data_dir.glob("*.bin"))
        if not shards:
            raise FileNotFoundError(f"No .bin shard files found in {data_dir}")

        # Split shards: last val_shards for validation, rest for training
        if split == "val":
            self.shards = shards[-val_shards:]
        else:
            self.shards = shards[:-val_shards] if len(shards) > val_shards else shards

        if not self.shards:
            raise ValueError(f"No shards available for split='{split}'")

        print(f"Dataset [{split}]: {len(self.shards)} shards")

        # Preload shard sizes (number of tokens in each shard)
        self._shard_tokens: list[np.ndarray] = []
        self._cumulative_sizes: list[int] = []
        total = 0

        for shard in self.shards:
            arr = np.memmap(shard, dtype=np.uint16, mode="r")
            # Number of complete (ctx_len+1) windows we can extract
            n_windows = max(0, len(arr) - ctx_len)
            self._shard_tokens.append(arr)
            total += n_windows
            self._cumulative_sizes.append(total)

        self._total_windows = total
        print(f"  Total windows: {total:,} (ctx_len={ctx_len})")

    def __len__(self) -> int:
        return self._total_windows

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (input_ids, targets) where targets = input_ids shifted by 1.
        (Standard causal LM training: predict next token at each position.)
        """
        # Find which shard this index falls in
        shard_idx = 0
        offset = idx
        for i, cumulative in enumerate(self._cumulative_sizes):
            if idx < cumulative:
                shard_idx = i
                if i > 0:
                    offset = idx - self._cumulative_sizes[i - 1]
                break

        arr = self._shard_tokens[shard_idx]
        chunk = arr[offset: offset + self.ctx_len + 1].astype(np.int64)

        input_ids = torch.from_numpy(chunk[:-1])  # (ctx_len,)
        targets = torch.from_numpy(chunk[1:])      # (ctx_len,) — shifted by 1

        return input_ids, targets


def build_dataloaders(
    data_dir: str | Path,
    ctx_len: int,
    batch_size: int,
    num_workers: int = 2,
    val_shards: int = 1,
) -> tuple[DataLoader, DataLoader]:
    """
    Build train and validation DataLoaders.

    Args:
        data_dir: Directory with .bin shards.
        ctx_len: Sequence length.
        batch_size: Batch size per GPU.
        num_workers: DataLoader worker processes.
        val_shards: Number of shards reserved for validation.

    Returns:
        (train_loader, val_loader) tuple.
    """
    train_ds = TokenDataset(data_dir, ctx_len, split="train", val_shards=val_shards)
    val_ds = TokenDataset(data_dir, ctx_len, split="val", val_shards=val_shards)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return train_loader, val_loader
