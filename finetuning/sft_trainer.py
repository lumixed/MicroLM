"""
finetuning/sft_trainer.py

Supervised Fine-Tuning (SFT) trainer for MicroLM.

Loads a pretrained checkpoint and fine-tunes on instruction-response pairs.
Only the response tokens contribute to the loss (prompt tokens are masked with -100).

Usage:
    python finetuning/finetune.py --mode sft \\
        --checkpoint checkpoints/microlm-125m-final.pt \\
        --output checkpoints/sft/
"""

from __future__ import annotations

import os
import math
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@dataclass
class SFTConfig:
    """Hyperparameters for SFT fine-tuning."""

    run_name: str = "microlm-sft"
    out_dir: str = "checkpoints/sft"

    # Data
    max_length: int = 512              # Max tokens per example
    dataset: str = "sahil2801/CodeAlpaca-20k"

    # Optimization — lower LR than pretraining
    max_lr: float = 1e-5
    min_lr: float = 1e-6
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    betas: tuple = (0.9, 0.95)

    # Schedule
    num_epochs: int = 3
    warmup_ratio: float = 0.03        # 3% of total steps as warmup

    # Batch
    batch_size: int = 4
    grad_accum_steps: int = 4

    # Precision
    dtype: str = "bfloat16"

    # Logging
    log_every: int = 10
    save_every: int = 200
    use_wandb: bool = False
    wandb_project: str = "microlm-sft"


class SFTTrainer:
    """
    Fine-tunes MicroLM with supervised learning on instruction-response pairs.

    Args:
        model:      MicroLM model loaded from a pretrained checkpoint.
        tokenizer:  Trained BPETokenizer instance.
        config:     SFTConfig with hyperparameters.
        train_loader: DataLoader for training split.
        val_loader:  Optional DataLoader for validation.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        config: SFTConfig,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.cfg = config
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.device = self._get_device()
        self.model = self.model.to(self.device)

        total_steps = len(train_loader) * config.num_epochs // config.grad_accum_steps
        warmup_steps = int(total_steps * config.warmup_ratio)

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.max_lr,
            weight_decay=config.weight_decay,
            betas=config.betas,
        )

        # Cosine schedule with warmup
        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return step / max(warmup_steps, 1)
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            return config.min_lr / config.max_lr + 0.5 * (1 - config.min_lr / config.max_lr) * (
                1 + math.cos(math.pi * progress)
            )

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        self.step = 0
        self.amp_dtype = torch.bfloat16 if config.dtype == "bfloat16" else torch.float32
        os.makedirs(config.out_dir, exist_ok=True)

        if config.use_wandb:
            self._init_wandb()

    def train(self) -> None:
        """Run SFT training loop."""
        print(f"\n{'='*60}")
        print(f"SFT Fine-tuning: {self.cfg.run_name}")
        print(f"Epochs: {self.cfg.num_epochs} | LR: {self.cfg.max_lr:.1e}")
        print(f"{'='*60}\n")

        self.model.train()
        best_val_loss = float("inf")

        for epoch in range(self.cfg.num_epochs):
            accum_loss = 0.0
            t0 = time.time()

            self.optimizer.zero_grad(set_to_none=True)
            micro_step = 0

            for batch in self.train_loader:
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)

                with torch.amp.autocast(device_type=self.device.type, dtype=self.amp_dtype):
                    logits, _ = self.model(input_ids)
                    # Shift for next-token prediction
                    shift_logits = logits[:, :-1, :].contiguous()
                    shift_labels = labels[:, 1:].contiguous()
                    loss = torch.nn.functional.cross_entropy(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1),
                        ignore_index=-100,
                    )
                    loss = loss / self.cfg.grad_accum_steps

                loss.backward()
                accum_loss += loss.item()
                micro_step += 1

                if micro_step % self.cfg.grad_accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.step += 1

                    if self.step % self.cfg.log_every == 0:
                        avg_loss = accum_loss * self.cfg.grad_accum_steps / self.cfg.log_every
                        lr = self.scheduler.get_last_lr()[0]
                        print(
                            f"epoch {epoch+1}/{self.cfg.num_epochs} | "
                            f"step {self.step} | loss {avg_loss:.4f} | "
                            f"ppl {math.exp(min(avg_loss, 20)):.2f} | lr {lr:.2e}"
                        )
                        accum_loss = 0.0

                    if self.step % self.cfg.save_every == 0:
                        self._save_checkpoint(f"step_{self.step:06d}")

            # End of epoch
            if self.val_loader:
                val_loss = self._evaluate()
                print(f"\n[Epoch {epoch+1}] val_loss={val_loss:.4f}")
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self._save_checkpoint("best")
                    print("  ⭐ New best checkpoint saved!")
                self.model.train()

        self._save_checkpoint("final")
        print("\nSFT fine-tuning complete!")

    @torch.no_grad()
    def _evaluate(self, max_batches: int = 50) -> float:
        self.model.eval()
        total_loss, n = 0.0, 0
        for i, batch in enumerate(self.val_loader):
            if i >= max_batches:
                break
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)
            with torch.amp.autocast(device_type=self.device.type, dtype=self.amp_dtype):
                logits, _ = self.model(input_ids)
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()
                loss = torch.nn.functional.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100,
                )
            total_loss += loss.item()
            n += 1
        return total_loss / max(n, 1)

    def _save_checkpoint(self, tag: str) -> None:
        path = Path(self.cfg.out_dir) / f"{self.cfg.run_name}_{tag}.pt"
        torch.save({
            "step": self.step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.cfg,
        }, path)
        print(f"  Checkpoint saved: {path}")

    def _get_device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _init_wandb(self) -> None:
        try:
            import wandb
            wandb.init(
                project=self.cfg.wandb_project,
                name=self.cfg.run_name,
                config=vars(self.cfg),
            )
        except ImportError:
            print("wandb not installed — skipping.")
            self.cfg.use_wandb = False
