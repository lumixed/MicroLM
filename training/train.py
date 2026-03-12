"""
microlm/training/train.py

Main training entry point. Run this on Kaggle or Google Colab:

    python training/train.py --config training/configs/125m_config.yaml

Or for a quick local test on MacBook (tiny model):
    python training/train.py --config training/configs/tiny_config.yaml
"""

import sys
import argparse
import yaml
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from model import MicroLMConfig, tiny_config, small_config
from training.trainer import Trainer, TrainingConfig
from training.dataset import build_dataloaders


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Train MicroLM")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint to resume from")
    parser.add_argument("--tiny", action="store_true", help="Use tiny model for quick tests")
    args = parser.parse_args()

    cfg_dict = load_config(args.config)
    if args.resume:
        cfg_dict["resume_from"] = args.resume

    # Build configs
    train_cfg = TrainingConfig(**{k: v for k, v in cfg_dict.items()
                                   if k in TrainingConfig.__dataclass_fields__})
    model_cfg = tiny_config() if args.tiny else small_config()

    # Build data loaders
    train_loader, val_loader = build_dataloaders(
        data_dir=train_cfg.data_path,
        ctx_len=train_cfg.ctx_len,
        batch_size=train_cfg.batch_size,
    )

    # Train!
    trainer = Trainer(
        model_config=model_cfg,
        train_config=train_cfg,
        train_loader=train_loader,
        val_loader=val_loader,
    )
    trainer.train()


if __name__ == "__main__":
    main()
