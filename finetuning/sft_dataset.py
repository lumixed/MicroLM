"""
finetuning/sft_dataset.py

Supervised Fine-Tuning dataset for MicroLM using CodeAlpaca-20K.

Format (Alpaca-style):
    ### Instruction:
    Write a Python function that returns the nth Fibonacci number.

    ### Response:
    def fibonacci(n): ...

Only the Response tokens contribute to the loss; Instruction tokens are masked.
"""

from __future__ import annotations

import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional


class SFTDataset(Dataset):
    """
    Loads CodeAlpaca-style instruction-response pairs and tokenizes them.

    Args:
        tokenizer: Trained BPETokenizer instance.
        max_length: Maximum sequence length (pad/truncate to this).
        split: 'train' or 'test'.
        hf_dataset_name: HuggingFace dataset identifier.
    """

    PROMPT_TEMPLATE = (
        "### Instruction:\n{instruction}\n\n### Response:\n"
    )

    def __init__(
        self,
        tokenizer,
        max_length: int = 512,
        split: str = "train",
        hf_dataset_name: str = "sahil2801/CodeAlpaca-20k",
    ):
        from datasets import load_dataset

        self.tokenizer = tokenizer
        self.max_length = max_length

        raw = load_dataset(hf_dataset_name, split=split)
        self.examples = self._process(raw)
        print(f"SFTDataset: loaded {len(self.examples)} examples ({split} split)")

    def _process(self, raw_dataset) -> list[dict]:
        examples = []
        for item in raw_dataset:
            instruction = item.get("instruction", "").strip()
            output = item.get("output", "").strip()
            if not instruction or not output:
                continue

            prompt = self.PROMPT_TEMPLATE.format(instruction=instruction)
            full_text = prompt + output

            prompt_ids = self.tokenizer.encode(prompt)
            full_ids = self.tokenizer.encode(full_text)

            # Truncate to max_length
            if len(full_ids) > self.max_length:
                full_ids = full_ids[: self.max_length]

            # Build loss mask: -100 for prompt tokens (ignored by cross-entropy)
            prompt_len = min(len(prompt_ids), len(full_ids))
            labels = [-100] * prompt_len + full_ids[prompt_len:]

            # Pad to max_length
            pad_len = self.max_length - len(full_ids)
            input_ids = full_ids + [0] * pad_len
            labels = labels + [-100] * pad_len

            examples.append(
                {
                    "input_ids": input_ids[: self.max_length],
                    "labels": labels[: self.max_length],
                }
            )

        return examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        ex = self.examples[idx]
        return {
            "input_ids": torch.tensor(ex["input_ids"], dtype=torch.long),
            "labels": torch.tensor(ex["labels"], dtype=torch.long),
        }


def build_sft_dataloader(
    tokenizer,
    max_length: int = 512,
    batch_size: int = 4,
    split: str = "train",
    num_workers: int = 2,
) -> DataLoader:
    dataset = SFTDataset(tokenizer, max_length=max_length, split=split)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=True,
    )
