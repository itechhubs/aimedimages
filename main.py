"""
Lung X-Ray Disease Detection – Training Entry Point
====================================================

Multi-task deep-learning pipeline for the NIH Chest X-Ray 14 dataset.

* 14-class multi-label classification (diseases)
* Per-class bounding-box regression (disease localization)
* Patient metadata fusion (age, gender, follow-up, view position)

Usage
-----
    python main.py                          # train with defaults
    python main.py --epochs 100 --batch_size 24
    python main.py --resume outputs/checkpoints/epoch_010.pth
"""

import argparse
import logging
import sys

import torch

from src.config import Config
from src.dataset import (
    build_loaders,
    compute_class_alpha,
    compute_class_weights,
    compute_follow_up_stats,
    load_and_prepare_data,
    split_by_patient,
)
from src.model import LungDiseaseNet
from src.trainer import Trainer
from src.utils import seed_everything, setup_logging

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train Lung X-Ray Disease Detection Model")

    # Paths
    p.add_argument("--image_dir", type=str, default=None,
                   help="Override image directory")
    p.add_argument("--output_dir", type=str, default=None)

    # Training
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--learning_rate", type=float, default=None)
    p.add_argument("--grad_accum_steps", type=int, default=None)

    # Model
    p.add_argument("--backbone", type=str, default=None,
                   help="timm model name for backbone")
    p.add_argument("--image_size", type=int, default=None)

    # Hardware
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--no_amp", action="store_true",
                   help="Disable mixed-precision training")

    # Misc
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--resume", type=str, default=None,
                   help="Path to specific checkpoint to resume from")

    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Config()

    # Apply CLI overrides
    for field_name in ("image_dir", "output_dir", "epochs", "batch_size",
                       "learning_rate", "grad_accum_steps", "backbone",
                       "image_size", "num_workers", "seed"):
        val = getattr(args, field_name, None)
        if val is not None:
            setattr(cfg, field_name, val)
    if args.no_amp:
        cfg.mixed_precision = False
    if args.output_dir:
        cfg.checkpoint_dir = f"{args.output_dir}/checkpoints"
        cfg.log_dir = f"{args.output_dir}/logs"
        cfg.tensorboard_dir = f"{args.output_dir}/tensorboard"

    cfg.ensure_dirs()

    # ── Logging ─────────────────────────────────────────────────────────
    setup_logging(cfg)
    logger.info("Configuration: %s", vars(cfg))

    # ── Seed ────────────────────────────────────────────────────────────
    seed_everything(cfg.seed)

    # ── Device info ─────────────────────────────────────────────────────
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_properties(0)
        logger.info("GPU: %s  |  VRAM: %.1f GB  |  CUDA: %s",
                     gpu.name, gpu.total_memory / 1024**3, torch.version.cuda)
    else:
        logger.warning("No CUDA GPU detected - training on CPU (will be slow)")
        cfg.device = "cpu"
        cfg.mixed_precision = False

    # ── Data ────────────────────────────────────────────────────────────
    logger.info("Loading data...")
    df, bbox_dict = load_and_prepare_data(cfg)
    df = compute_follow_up_stats(df)
    train_df, val_df = split_by_patient(df, cfg)

    class_weights = compute_class_weights(train_df, cfg)
    class_alpha = compute_class_alpha(train_df, cfg)
    train_loader, val_loader = build_loaders(train_df, val_df, bbox_dict, cfg)

    logger.info("Train batches: %d  |  Val batches: %d",
                 len(train_loader), len(val_loader))

    # ── Model ───────────────────────────────────────────────────────────
    logger.info("Building model (backbone=%s)...", cfg.backbone)
    model = LungDiseaseNet(cfg)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    logger.info("Parameters: %.1f M total, %.1f M trainable", n_params, n_train)

    if cfg.compile_model and hasattr(torch, "compile"):
        logger.info("Compiling model with torch.compile...")
        model = torch.compile(model)

    # ── Trainer ─────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
        pos_weight=class_weights,
        class_alpha=class_alpha,
    )

    # ── Train ───────────────────────────────────────────────────────────
    try:
        trainer.fit()
    except KeyboardInterrupt:
        logger.info("Training interrupted - saving emergency checkpoint...")
        trainer._save(trainer.start_epoch, is_best=False)
        sys.exit(0)

    logger.info("Done.")


if __name__ == "__main__":
    main()
