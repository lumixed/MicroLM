"""
finetuning/finetune.py

Unified CLI entry point for SFT and DPO fine-tuning.

Usage:
    # Stage 1: SFT on CodeAlpaca
    python finetuning/finetune.py --mode sft \\
        --checkpoint checkpoints/microlm-125m-final.pt \\
        --output checkpoints/sft/

    # Stage 2: DPO alignment
    python finetuning/finetune.py --mode dpo \\
        --checkpoint checkpoints/sft/microlm-sft-best.pt \\
        --output checkpoints/dpo/
"""

from __future__ import annotations

import sys
import argparse
import copy
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from model import MicroLM, MicroLMConfig
from tokenizer.bpe import BPETokenizer


def load_model_from_checkpoint(checkpoint_path: str):
    """Load MicroLM and config from a .pt checkpoint file."""
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    model_cfg = ckpt.get("model_config", MicroLMConfig())
    model = MicroLM(model_cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"  Model loaded: {model.param_count():,} parameters")
    return model, model_cfg


def run_sft(args):
    from finetuning.sft_trainer import SFTTrainer, SFTConfig
    from finetuning.sft_dataset import build_sft_dataloader

    tokenizer = BPETokenizer.load(args.tokenizer)
    model, _ = load_model_from_checkpoint(args.checkpoint)

    cfg = SFTConfig(
        run_name="microlm-sft",
        out_dir=args.output,
        max_lr=args.lr,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        use_wandb=args.wandb,
    )

    train_loader = build_sft_dataloader(
        tokenizer, max_length=cfg.max_length, batch_size=cfg.batch_size, split="train"
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        config=cfg,
        train_loader=train_loader,
    )
    trainer.train()


def run_dpo(args):
    from finetuning.dpo_loss import DPOLoss, get_batch_logps
    from finetuning.dpo_dataset import build_dpo_dataloader

    tokenizer = BPETokenizer.load(args.tokenizer)

    # Policy model (being trained)
    policy_model, _ = load_model_from_checkpoint(args.checkpoint)

    # Reference model (frozen copy of SFT model)
    ref_model = copy.deepcopy(policy_model)
    for p in ref_model.parameters():
        p.requires_grad_(False)
    ref_model.eval()

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    policy_model = policy_model.to(device)
    ref_model = ref_model.to(device)

    train_loader = build_dpo_dataloader(
        tokenizer, max_length=512, batch_size=2, split="train"
    )

    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=args.lr, weight_decay=0.01)
    dpo_loss_fn = DPOLoss(beta=0.1)

    import os
    os.makedirs(args.output, exist_ok=True)

    print(f"\n{'='*60}\nDPO Alignment\n{'='*60}")
    policy_model.train()

    for step, batch in enumerate(train_loader):
        chosen_ids = batch["chosen_ids"].to(device)
        rejected_ids = batch["rejected_ids"].to(device)

        labels = chosen_ids.clone()  # For DPO we use all tokens
        policy_chosen_logps = get_batch_logps(policy_model, chosen_ids, labels, device)
        policy_rejected_logps = get_batch_logps(policy_model, rejected_ids, labels, device)

        with torch.no_grad():
            ref_chosen_logps = get_batch_logps(ref_model, chosen_ids, labels, device)
            ref_rejected_logps = get_batch_logps(ref_model, rejected_ids, labels, device)

        policy_model.train()
        # Recompute with grad
        policy_chosen_logps = get_batch_logps.__wrapped__(
            policy_model, chosen_ids, labels, device
        ) if hasattr(get_batch_logps, '__wrapped__') else _logps_with_grad(
            policy_model, chosen_ids, labels, device
        )

        loss, chosen_r, rejected_r = dpo_loss_fn(
            policy_chosen_logps, policy_rejected_logps,
            ref_chosen_logps, ref_rejected_logps
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 1.0)
        optimizer.step()

        if step % 10 == 0:
            print(
                f"step {step:4d} | dpo_loss {loss.item():.4f} | "
                f"chosen_r {chosen_r.item():.3f} | rejected_r {rejected_r.item():.3f}"
            )

        if step % 200 == 0 and step > 0:
            ckpt_path = Path(args.output) / f"dpo_step_{step:06d}.pt"
            torch.save({"step": step, "model_state_dict": policy_model.state_dict()}, ckpt_path)
            print(f"  Checkpoint: {ckpt_path}")

        if step >= args.max_steps:
            break

    final_path = Path(args.output) / "dpo_final.pt"
    torch.save({"model_state_dict": policy_model.state_dict()}, final_path)
    print(f"DPO complete! Final checkpoint: {final_path}")


def _logps_with_grad(model, input_ids, labels, device):
    """get_batch_logps but with gradients enabled (for DPO policy update)."""
    import torch.nn.functional as F
    input_ids = input_ids.to(device)
    labels = labels.to(device)
    logits, _ = model(input_ids)
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_logps = log_probs.gather(2, shift_labels.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    mask = (shift_labels != -100).float()
    return (token_logps * mask).sum(dim=-1)


def main():
    parser = argparse.ArgumentParser(description="MicroLM Fine-tuning (SFT / DPO)")
    parser.add_argument("--mode", choices=["sft", "dpo"], required=True)
    parser.add_argument("--checkpoint", type=str, required=True, help="Pretrained .pt checkpoint")
    parser.add_argument("--tokenizer", type=str, default="tokenizer/tokenizer.json")
    parser.add_argument("--output", type=str, default="checkpoints/finetune")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=3, help="SFT epochs")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=1000, help="Max DPO steps")
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()

    if args.mode == "sft":
        run_sft(args)
    elif args.mode == "dpo":
        run_dpo(args)


if __name__ == "__main__":
    main()
