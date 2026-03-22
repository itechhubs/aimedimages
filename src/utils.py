"""Utility functions: losses, metrics, logging, seeding."""

import logging
import os
import random
import sys
from logging.handlers import RotatingFileHandler
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from .config import Config


# ── EMA (Exponential Moving Average) ───────────────────────────────────────


class EMAModel:
    """Maintains an exponential moving average of model parameters.

    Use ``apply_shadow`` before validation to swap in the averaged weights,
    and ``restore`` afterwards to put the training weights back.
    """

    def __init__(self, model: torch.nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        self.shadow: dict[str, torch.Tensor] = {}
        self.backup: dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model: torch.nn.Module) -> None:
        """Update shadow params with current model params."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                if name in self.shadow:
                    self.shadow[name].mul_(self.decay).add_(
                        param.data, alpha=1.0 - self.decay)
                else:
                    self.shadow[name] = param.data.clone()

    def apply_shadow(self, model: torch.nn.Module) -> None:
        """Swap model weights with shadow (EMA) weights for evaluation."""
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model: torch.nn.Module) -> None:
        """Restore the original training weights after evaluation."""
        for name, param in model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}

    def state_dict(self) -> dict:
        return {"shadow": self.shadow, "decay": self.decay}

    def load_state_dict(self, state: dict) -> None:
        self.shadow = state["shadow"]
        self.decay = state.get("decay", self.decay)


# ── Seeding ─────────────────────────────────────────────────────────────────


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False   # keep True only if exact repro needed
    torch.backends.cudnn.benchmark = True         # faster conv algorithms for fixed sizes


# ── Logging ─────────────────────────────────────────────────────────────────


def setup_logging(cfg: Config) -> logging.Logger:
    """Configure root logger with console + rotating file handlers."""
    cfg.ensure_dirs()

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console – force UTF-8 to avoid cp1252 UnicodeEncodeError on Windows
    console_stream = open(sys.stdout.fileno(), mode="w", encoding="utf-8",
                          closefd=False)
    ch = logging.StreamHandler(console_stream)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    root_logger.addHandler(ch)

    # Rotating file (10 MB x 5 backups) – explicit UTF-8
    log_path = os.path.join(cfg.log_dir, "training.log")
    fh = RotatingFileHandler(log_path, maxBytes=10_000_000, backupCount=5,
                             encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    root_logger.addHandler(fh)

    return root_logger


# ── Loss functions ──────────────────────────────────────────────────────────


class AsymmetricLoss(nn.Module):
    """Asymmetric Loss for multi-label classification.

    Uses different focusing strengths for positive vs negative examples,
    which is more effective than symmetric Focal Loss for imbalanced
    multi-label tasks like chest X-ray diagnosis.

    Reference: Ben-Baruch et al., "Asymmetric Loss For Multi-Label
    Classification" (https://arxiv.org/abs/2009.14119)
    """

    def __init__(self, gamma_neg: float = 4.0, gamma_pos: float = 1.0,
                 clip: float = 0.05, label_smoothing: float = 0.0,
                 pos_weight: Optional[torch.Tensor] = None) -> None:
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.label_smoothing = label_smoothing
        self.register_buffer("pos_weight", pos_weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.label_smoothing > 0:
            targets = targets * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing

        xs_pos = torch.sigmoid(logits)
        xs_neg = 1.0 - xs_pos

        # Asymmetric probability clipping (shift negatives toward 0)
        if self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1.0)

        # Per-sample BCE components
        loss_pos = targets * torch.log(xs_pos.clamp(min=1e-8))
        loss_neg = (1.0 - targets) * torch.log(xs_neg.clamp(min=1e-8))

        # Apply pos_weight to positive term if provided
        if self.pos_weight is not None:
            loss = loss_pos * self.pos_weight.unsqueeze(0) + loss_neg
        else:
            loss = loss_pos + loss_neg

        # Asymmetric focal weighting (detached for stable gradients)
        pt_pos = xs_pos * targets
        pt_neg = xs_neg * (1.0 - targets)
        pt = pt_pos + pt_neg
        gamma = self.gamma_pos * targets + self.gamma_neg * (1.0 - targets)
        focal_weight = torch.pow(1.0 - pt, gamma).detach()
        loss = loss * focal_weight

        return -loss.mean()


class CombinedLoss(nn.Module):
    """Combined classification (focal) + detection (smooth-L1) loss."""

    def __init__(self, cfg: Config,
                 pos_weight: Optional[torch.Tensor] = None) -> None:
        super().__init__()
        self.cls_weight = cfg.cls_loss_weight
        self.bbox_weight = cfg.bbox_loss_weight

        self.cls_loss_fn = AsymmetricLoss(
            gamma_neg=cfg.asl_gamma_neg,
            gamma_pos=cfg.asl_gamma_pos,
            clip=cfg.asl_clip,
            label_smoothing=cfg.label_smoothing,
            pos_weight=pos_weight,
        )
        self.smooth_l1 = nn.SmoothL1Loss(reduction="none", beta=0.1)

    def forward(
        self,
        logits: torch.Tensor,
        bbox_pred: torch.Tensor,
        labels: torch.Tensor,
        bbox_targets: torch.Tensor,
        bbox_mask: torch.Tensor,
    ) -> dict:
        cls_loss = self.cls_loss_fn(logits, labels)

        # BBox loss only where annotations exist
        if bbox_mask.sum() > 0:
            # Expand mask: [B, C] → [B, C, 4]
            mask_4d = bbox_mask.unsqueeze(-1).expand_as(bbox_pred)
            raw_bbox_loss = self.smooth_l1(bbox_pred, bbox_targets)
            bbox_loss = (raw_bbox_loss * mask_4d).sum() / mask_4d.sum().clamp(min=1.0)
        else:
            bbox_loss = torch.tensor(0.0, device=logits.device)

        total = self.cls_weight * cls_loss + self.bbox_weight * bbox_loss
        return {
            "total": total,
            "cls_loss": cls_loss.detach(),
            "bbox_loss": bbox_loss.detach(),
        }


# ── Metrics ─────────────────────────────────────────────────────────────────


def compute_auc(
    all_logits: np.ndarray,
    all_labels: np.ndarray,
    disease_classes: list,
) -> dict:
    """Per-class and mean AUC-ROC.  Returns dict with per-class & mean AUC."""
    # Clip logits to avoid overflow in exp() for large negative values
    clipped = np.clip(all_logits, -500, 500)
    probs = 1.0 / (1.0 + np.exp(-clipped))   # sigmoid
    aucs = {}
    valid_aucs = []
    for i, name in enumerate(disease_classes):
        y_true = all_labels[:, i]
        if y_true.sum() == 0 or y_true.sum() == len(y_true):
            aucs[name] = float("nan")
            continue
        auc_val = roc_auc_score(y_true, probs[:, i])
        aucs[name] = auc_val
        valid_aucs.append(auc_val)

    aucs["mean_AUC"] = float(np.mean(valid_aucs)) if valid_aucs else 0.0
    return aucs


# ── Checkpoint helpers ──────────────────────────────────────────────────────


def save_checkpoint(
    state: dict, path: str, is_best: bool = False
) -> None:
    torch.save(state, path)
    if is_best:
        best_path = os.path.join(os.path.dirname(path), "best_model.pth")
        torch.save(state, best_path)


def load_checkpoint(path: str, device: str = "cpu") -> dict:
    return torch.load(path, map_location=device, weights_only=False)
