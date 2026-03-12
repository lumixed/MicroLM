"""
microlm/training/lr_scheduler.py

Cosine annealing learning rate scheduler with linear warmup.

This is the standard training schedule for LLMs:
  1. Linear warmup: LR ramps from 0 → max_lr over `warmup_steps` steps
  2. Cosine decay: LR decays from max_lr → min_lr over the remaining steps

Formula:
    if step < warmup_steps:
        lr = max_lr * (step / warmup_steps)
    else:
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + cos(π * progress))

References:
    - Loshchilov & Hutter 2016 "SGDR: Stochastic Gradient Descent with Warm Restarts"
    - Chinchilla (Hoffmann et al. 2022) training schedule
"""

import math


class CosineWithWarmup:
    """
    Cosine LR schedule with linear warmup.

    Usage:
        scheduler = CosineWithWarmup(max_lr=3e-4, min_lr=3e-5,
                                     warmup_steps=100, total_steps=10_000)
        for step in range(total_steps):
            lr = scheduler.get_lr(step)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
    """

    def __init__(
        self,
        max_lr: float,
        min_lr: float,
        warmup_steps: int,
        total_steps: int,
    ) -> None:
        """
        Args:
            max_lr: Peak learning rate (reached after warmup).
            min_lr: Minimum LR at end of cosine decay. Typically max_lr / 10.
            warmup_steps: Number of linear warmup steps.
            total_steps: Total training steps.
        """
        assert warmup_steps < total_steps, "warmup_steps must be < total_steps"
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps

    def get_lr(self, step: int) -> float:
        """
        Compute learning rate for a given step.

        Args:
            step: Current training step (0-indexed).

        Returns:
            Learning rate as a float.
        """
        # Phase 1: linear warmup
        if step < self.warmup_steps:
            return self.max_lr * (step + 1) / self.warmup_steps

        # Phase 2: after training (shouldn't happen, but clamp)
        if step >= self.total_steps:
            return self.min_lr

        # Phase 3: cosine decay
        decay_ratio = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        assert 0.0 <= decay_ratio <= 1.0
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return self.min_lr + coeff * (self.max_lr - self.min_lr)
