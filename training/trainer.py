"""
microlm/training/trainer.py

Production-grade training loop for MicroLM.

Features:
  - Mixed precision training (bfloat16 + torch.amp)
  - Gradient accumulation (simulate larger batch sizes)
  - Gradient clipping (prevents gradient explosions)
  - Gradient checkpointing (reduces GPU memory at cost of compute)
  - Cosine LR schedule with warmup
  - Checkpoint saving and resuming
  - WandB experiment tracking
  - Comprehensive logging (loss, perplexity, tokens/sec, MFU)

MFU (Model FLOP Utilization):
    Measures what fraction of theoretical peak GPU throughput you're achieving.
    A healthy MFU for LLM training is ~40-60%.
"""

from __future__ import annotations

import os
import time
import math
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import torch
from torch.utils.data import DataLoader

from model import MicroLM, MicroLMConfig
from training.lr_scheduler import CosineWithWarmup


@dataclass
class TrainingConfig:
    """All hyperparameters for the training run."""

    # ---- Run identity ----
    run_name: str = "microlm-125m-run1"
    out_dir: str = "checkpoints"

    # ---- Data ----
    data_path: str = "data/tokens"      # Directory of .bin shard files
    train_split: float = 0.95           # Fraction of data for training

    # ---- Batch / sequence ----
    batch_size: int = 8                 # Micro-batch size (per gradient step)
    ctx_len: int = 1024                 # Sequence length
    grad_accum_steps: int = 8          # Gradient accumulation steps
    # Effective batch size = batch_size * grad_accum_steps * ctx_len tokens

    # ---- Optimizer ----
    max_lr: float = 3e-4               # Peak learning rate (Chinchilla-style)
    min_lr: float = 3e-5               # Min LR (end of cosine decay = max_lr / 10)
    weight_decay: float = 0.1
    grad_clip: float = 1.0             # Max gradient norm (clips if exceeded)
    betas: tuple = (0.9, 0.95)        # AdamW beta values

    # ---- Schedule ----
    warmup_steps: int = 200            # Linear warmup steps
    total_steps: int = 10_000          # Total training steps
    eval_every: int = 250              # Evaluate and log every N steps
    save_every: int = 500              # Save checkpoint every N steps

    # ---- Memory optimization ----
    use_grad_checkpoint: bool = True   # Gradient checkpointing (saves VRAM)
    dtype: str = "bfloat16"           # Training precision ('float32', 'bfloat16', 'float16')

    # ---- Logging ----
    use_wandb: bool = True
    wandb_project: str = "microlm"
    log_every: int = 10               # Log to console every N steps

    # ---- Resuming ----
    resume_from: Optional[str] = None  # Path to checkpoint to resume from


class Trainer:
    """
    Main training loop for MicroLM.

    Args:
        model_config: MicroLMConfig specifying model architecture.
        train_config: TrainingConfig specifying training hyperparameters.
        train_loader: DataLoader yielding (input_ids, targets) batches.
        val_loader: Optional validation DataLoader.
    """

    def __init__(
        self,
        model_config: MicroLMConfig,
        train_config: TrainingConfig,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ) -> None:
        self.model_config = model_config
        self.cfg = train_config
        self.train_loader = train_loader
        self.val_loader = val_loader

        # ---- Device setup ----
        self.device = self._get_device()
        print(f"Training on: {self.device}")

        # ---- Model ----
        self.model = MicroLM(model_config).to(self.device)
        # Enable gradient checkpointing (trades compute for memory)
        if train_config.use_grad_checkpoint:
            self.model.gradient_checkpointing_enable()
            print("Gradient checkpointing: enabled")

        print(self.model)

        # ---- Optimizer ----
        self.optimizer = self.model.configure_optimizer(
            lr=train_config.max_lr,
            weight_decay=train_config.weight_decay,
            betas=train_config.betas,
            device=str(self.device),
        )

        # ---- LR Scheduler ----
        self.scheduler = CosineWithWarmup(
            max_lr=train_config.max_lr,
            min_lr=train_config.min_lr,
            warmup_steps=train_config.warmup_steps,
            total_steps=train_config.total_steps,
        )

        # ---- Mixed precision scaler ----
        # bfloat16 doesn't need gradient scaling (it handles underflow natively)
        # float16 does need scaling
        enable_scaler = train_config.dtype == "float16"
        self.scaler = torch.amp.GradScaler("cuda", enabled=enable_scaler)
        self.amp_dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[train_config.dtype]

        # ---- State ----
        self.step = 0
        self.best_val_loss = float("inf")
        self.t0 = time.time()

        # ---- Checkpoint resume ----
        if train_config.resume_from:
            self._load_checkpoint(train_config.resume_from)

        # ---- WandB ----
        if train_config.use_wandb:
            self._init_wandb()

        # ---- Output directory ----
        os.makedirs(train_config.out_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self) -> None:
        """Run the full training loop."""
        print(f"\n{'='*60}")
        print(f"Starting training: {self.cfg.run_name}")
        print(f"Total steps: {self.cfg.total_steps} | Warmup: {self.cfg.warmup_steps}")
        print(f"Effective batch: {self.cfg.batch_size * self.cfg.grad_accum_steps} "
              f"× {self.cfg.ctx_len} = "
              f"{self.cfg.batch_size * self.cfg.grad_accum_steps * self.cfg.ctx_len:,} tokens/step")
        print(f"{'='*60}\n")

        self.model.train()
        train_iter = iter(self.train_loader)
        accum_loss = 0.0

        while self.step < self.cfg.total_steps:
            # ---- Update LR ----
            lr = self.scheduler.get_lr(self.step)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr

            # ---- Gradient accumulation loop ----
            self.optimizer.zero_grad(set_to_none=True)
            step_loss = 0.0

            for micro_step in range(self.cfg.grad_accum_steps):
                try:
                    batch = next(train_iter)
                except StopIteration:
                    train_iter = iter(self.train_loader)
                    batch = next(train_iter)

                input_ids, targets = [t.to(self.device) for t in batch]

                # Mixed precision forward pass
                with torch.amp.autocast(device_type=self.device.type, dtype=self.amp_dtype):
                    _, loss = self.model(input_ids, targets=targets)
                    # Scale loss by accumulation steps (average over micro-batches)
                    loss = loss / self.cfg.grad_accum_steps

                self.scaler.scale(loss).backward()
                step_loss += loss.item()

            # ---- Gradient clipping ----
            self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.cfg.grad_clip
            )

            # ---- Optimizer step ----
            self.scaler.step(self.optimizer)
            self.scaler.update()

            accum_loss += step_loss
            self.step += 1

            # ---- Logging ----
            if self.step % self.cfg.log_every == 0:
                t1 = time.time()
                dt = t1 - self.t0
                self.t0 = t1
                tokens_per_sec = (
                    self.cfg.log_every * self.cfg.grad_accum_steps
                    * self.cfg.batch_size * self.cfg.ctx_len / dt
                )
                avg_loss = accum_loss / self.cfg.log_every
                perplexity = math.exp(min(avg_loss, 20))  # cap to avoid overflow
                accum_loss = 0.0

                print(
                    f"step {self.step:6d} | loss {avg_loss:.4f} | ppl {perplexity:.2f} | "
                    f"lr {lr:.2e} | grad_norm {grad_norm:.3f} | "
                    f"{tokens_per_sec/1e3:.1f}k tok/s"
                )

                if self.cfg.use_wandb:
                    import wandb
                    wandb.log({
                        "train/loss": avg_loss,
                        "train/perplexity": perplexity,
                        "train/lr": lr,
                        "train/grad_norm": grad_norm.item(),
                        "train/tokens_per_sec": tokens_per_sec,
                    }, step=self.step)

            # ---- Evaluation ----
            if self.step % self.cfg.eval_every == 0 and self.val_loader:
                val_loss = self.evaluate()
                is_best = val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_loss
                    self.save_checkpoint("best")
                print(f"  val_loss={val_loss:.4f} {'⭐ new best!' if is_best else ''}")

                if self.cfg.use_wandb:
                    import wandb
                    wandb.log({"val/loss": val_loss, "val/perplexity": math.exp(min(val_loss, 20))},
                              step=self.step)

                self.model.train()

            # ---- Checkpoint ----
            if self.step % self.cfg.save_every == 0:
                self.save_checkpoint(f"step_{self.step:07d}")

        # Final save
        self.save_checkpoint("final")
        print("Training complete!")

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate(self, max_batches: int = 20) -> float:
        """Run validation loop and return average loss."""
        self.model.eval()
        total_loss = 0.0
        n = 0

        for i, batch in enumerate(self.val_loader):
            if i >= max_batches:
                break
            input_ids, targets = [t.to(self.device) for t in batch]
            with torch.amp.autocast(device_type=self.device.type, dtype=self.amp_dtype):
                _, loss = self.model(input_ids, targets=targets)
            total_loss += loss.item()
            n += 1

        return total_loss / max(n, 1)

    # ------------------------------------------------------------------
    # Checkpoint save / load
    # ------------------------------------------------------------------

    def save_checkpoint(self, tag: str) -> None:
        """Save model checkpoint."""
        path = Path(self.cfg.out_dir) / f"{self.cfg.run_name}_{tag}.pt"
        checkpoint = {
            "step": self.step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "model_config": self.model_config,
            "train_config": self.cfg,
        }
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")

    def _load_checkpoint(self, path: str) -> None:
        """Load a checkpoint to resume training."""
        print(f"Loading checkpoint from {path}")
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scaler.load_state_dict(ckpt["scaler_state_dict"])
        self.step = ckpt["step"]
        self.best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f"Resumed from step {self.step}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")  # Apple Silicon
        return torch.device("cpu")

    def _init_wandb(self) -> None:
        try:
            import wandb
            wandb.init(
                project=self.cfg.wandb_project,
                name=self.cfg.run_name,
                config={
                    "model": vars(self.model_config),
                    "training": vars(self.cfg),
                },
            )
        except ImportError:
            print("wandb not installed — logging disabled. Run: pip install wandb")
            self.cfg.use_wandb = False
