"""Configuration for Lung X-Ray Disease Detection Training."""

import os
from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    # ── Data paths ──────────────────────────────────────────────────────
    data_csv: str = os.path.join("data", "Data_Entry_2017_v2020.csv")
    bbox_csv: str = os.path.join("data", "BBox_List_2017.csv")
    image_dir: str = r"C:\projects\ai-training\images"
    output_dir: str = "outputs"
    checkpoint_dir: str = os.path.join("outputs", "checkpoints")
    log_dir: str = os.path.join("outputs", "logs")
    tensorboard_dir: str = os.path.join("outputs", "tensorboard")

    # ── Disease classes (ordered, excluding "No Finding") ───────────────
    disease_classes: List[str] = field(default_factory=lambda: [
        "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
        "Effusion", "Emphysema", "Fibrosis", "Hernia",
        "Infiltration", "Mass", "Nodule", "Pleural_Thickening",
        "Pneumonia", "Pneumothorax",
    ])

    # BBox CSV uses "Infiltrate" while main CSV uses "Infiltration"
    bbox_label_mapping: dict = field(default_factory=lambda: {
        "Infiltrate": "Infiltration",
    })

    # ── Image handling ───────────────────────────────────────────────────
    # X-ray images are single-channel grayscale; we standardize to 8-bit
    # [0,255] grayscale range, then normalize with medical-imaging stats.
    # Grayscale mean/std computed from NIH CXR-14 (approx.)
    grayscale_mean: float = 0.5024
    grayscale_std: float = 0.2898
    # The single-channel image is replicated to 3 channels for the
    # pretrained backbone, so we duplicate the stats to 3-channel.
    in_channels: int = 3                   # backbone expects 3ch

    # ── Model ───────────────────────────────────────────────────────────
    backbone: str = "convnext_base.fb_in22k_ft_in1k_384"
    backbone_features: int = 1024          # last-stage channel dim
    image_size: int = 1024
    num_classes: int = 14
    metadata_dim: int = 5                  # age, gender, followup, view_pos, has_followups
    meta_hidden: int = 128
    cls_hidden: int = 512
    det_hidden: int = 256

    # ── Training ────────────────────────────────────────────────────────
    epochs: int = 50
    batch_size: int = 8
    grad_accum_steps: int = 4              # effective batch = 32
    learning_rate: float = 3e-5            # backbone LR (low to preserve pretrained features)
    head_lr_multiplier: float = 10.0       # head gets 3e-4
    weight_decay: float = 2e-4
    freeze_backbone_epochs: int = 3        # Phase 1: train heads only
    warmup_epochs: int = 2                 # linear warmup after backbone unfreeze
    min_lr: float = 1e-7

    # ── LR scheduler (ReduceLROnPlateau) ────────────────────────────────
    lr_reduce_factor: float = 0.5          # halve LR on plateau
    lr_reduce_patience: int = 4            # epochs to wait before reducing

    # ── EMA (Exponential Moving Average) ────────────────────────────────
    use_ema: bool = True
    ema_decay: float = 0.999

    # ── Hardware (HP Z840 + RTX 3090) ───────────────────────────────────
    num_workers: int = 8
    mixed_precision: bool = True
    pin_memory: bool = True
    device: str = "cuda"
    compile_model: bool = False            # torch.compile (experimental)

    # ── Regularization ──────────────────────────────────────────────────
    drop_rate: float = 0.4
    drop_path_rate: float = 0.3            # stochastic depth in backbone
    label_smoothing: float = 0.1
    gradient_clip: float = 1.0

    # ── ASL (Asymmetric Loss) – arxiv 2009.14119 ────────────────────────
    asl_gamma_neg: float = 3.0             # down-weight easy negatives
    asl_gamma_pos: float = 0.0             # no focusing on positives (standard BCE behavior)
    asl_clip: float = 0.05                 # probability margin shift for negatives

    # ── Loss weights ────────────────────────────────────────────────────
    cls_loss_weight: float = 1.0
    bbox_loss_weight: float = 5.0

    # ── Checkpointing / early stopping ──────────────────────────────────
    save_every_epochs: int = 5
    patience: int = 15

    # ── Data splitting ──────────────────────────────────────────────────
    val_ratio: float = 0.2
    seed: int = 42

    # ── Derived ─────────────────────────────────────────────────────────
    def ensure_dirs(self) -> None:
        for d in (self.output_dir, self.checkpoint_dir, self.log_dir,
                  self.tensorboard_dir):
            os.makedirs(d, exist_ok=True)
