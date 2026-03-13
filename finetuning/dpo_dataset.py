"""
finetuning/dpo_dataset.py

DPO (Direct Preference Optimization) dataset.

Uses Anthropic's HH-RLHF dataset which contains human preference pairs:
  - chosen: the preferred response
  - rejected: the dispreferred response

Each example: (prompt, chosen_response, rejected_response)
"""

from __future__ import annotations

import torch
from torch.utils.data import Dataset, DataLoader


class DPODataset(Dataset):
    """
    DPO preference dataset from Anthropic HH-RLHF.

    Args:
        tokenizer: Trained BPETokenizer instance.
        max_length: Max tokens per sequence.
        split: 'train' or 'test'.
        max_examples: Cap the dataset size (None = use all).
    """

    def __init__(
        self,
        tokenizer,
        max_length: int = 512,
        split: str = "train",
        max_examples: Optional[int] = 5000,
    ):
        from datasets import load_dataset

        self.tokenizer = tokenizer
        self.max_length = max_length

        raw = load_dataset("Anthropic/hh-rlhf", split=split)
        if max_examples:
            raw = raw.select(range(min(max_examples, len(raw))))

        self.examples = self._process(raw)
        print(f"DPODataset: {len(self.examples)} preference pairs ({split})")

    def _process(self, raw_dataset) -> list[dict]:
        examples = []
        for item in raw_dataset:
            chosen = item.get("chosen", "").strip()
            rejected = item.get("rejected", "").strip()
            if not chosen or not rejected:
                continue

            chosen_ids = self._tokenize(chosen)
            rejected_ids = self._tokenize(rejected)

            examples.append(
                {
                    "chosen_ids": chosen_ids,
                    "rejected_ids": rejected_ids,
                }
            )
        return examples

    def _tokenize(self, text: str) -> list[int]:
        ids = self.tokenizer.encode(text)
        if len(ids) > self.max_length:
            ids = ids[: self.max_length]
        # Pad to max_length
        ids = ids + [0] * (self.max_length - len(ids))
        return ids

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        ex = self.examples[idx]
        return {
            "chosen_ids": torch.tensor(ex["chosen_ids"], dtype=torch.long),
            "rejected_ids": torch.tensor(ex["rejected_ids"], dtype=torch.long),
        }


def build_dpo_dataloader(
    tokenizer,
    max_length: int = 512,
    batch_size: int = 2,
    split: str = "train",
    max_examples: int = 5000,
) -> DataLoader:
    dataset = DPODataset(
        tokenizer,
        max_length=max_length,
        split=split,
        max_examples=max_examples,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        pin_memory=True,
    )


# Allow Optional import at module level
from typing import Optional
