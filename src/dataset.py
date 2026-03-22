"""Dataset and DataLoader for NIH Chest X-Ray 14 with bounding-box annotations.

Grayscale handling
~~~~~~~~~~~~~~~~~~
Chest X-ray images are **single-channel grayscale**.  The pipeline:
1.  Load as Pillow mode ``"L"`` (8-bit grayscale, 0-255).
2.  Apply histogram equalization to standardize contrast across scanners.
3.  Resize to ``cfg.image_size`` with high-quality resampling.
4.  Convert to a float [0,1] tensor and normalize with dataset mean/std.
5.  Replicate to 3 identical channels so that pretrained ImageNet backbones
    (which expect 3-channel input) work out of the box.
"""

import logging
import os
import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .config import Config

logger = logging.getLogger(__name__)


# ── Data preparation ────────────────────────────────────────────────────────


def load_and_prepare_data(cfg: Config) -> Tuple[pd.DataFrame, Dict]:
    """Load CSVs, parse columns, and build bbox lookup.

    Returns
    -------
    df : pd.DataFrame
        Cleaned main dataframe (one row per image).
    bbox_dict : dict
        Mapping  image_id -> {disease_label: [x, y, w, h]}  (pixel coords).
    """
    # ── Main CSV ────────────────────────────────────────────────────────
    logger.info("Reading main CSV: %s", cfg.data_csv)
    df = pd.read_csv(cfg.data_csv)
    logger.debug("Raw CSV columns: %s", df.columns.tolist())

    # Fix mangled column names caused by commas inside brackets
    col_map = {
        "OriginalImage[Width": "OriginalImageWidth",
        "Height]": "OriginalImageHeight",
        "OriginalImagePixelSpacing[x": "PixelSpacingX",
        "y]": "PixelSpacingY",
    }
    df.rename(columns=col_map, inplace=True)
    logger.debug("Cleaned CSV columns: %s", df.columns.tolist())

    # Parse patient age (handles both "58" and "058Y" formats)
    df["AgeYears"] = df["Patient Age"].apply(_parse_age)

    n_images = len(df)
    n_patients = df["Patient ID"].nunique()
    n_male = (df["Patient Gender"].str.strip() == "M").sum()
    n_female = (df["Patient Gender"].str.strip() == "F").sum()
    age_stats = df["AgeYears"]
    logger.info("Main CSV loaded: %d images, %d unique patients", n_images, n_patients)
    logger.info("  Gender distribution: Male=%d  Female=%d  Other=%d",
                n_male, n_female, n_images - n_male - n_female)
    logger.info("  Age range: %.0f - %.0f years  (mean=%.1f, median=%.1f)",
                age_stats.min(), age_stats.max(), age_stats.mean(), age_stats.median())
    logger.info("  Follow-up range: %d - %d",
                df["Follow-up #"].min(), df["Follow-up #"].max())

    # Log per-disease counts
    disease_counts = defaultdict(int)
    for labels_str in df["Finding Labels"]:
        for lbl in labels_str.split("|"):
            disease_counts[lbl.strip()] += 1
    logger.info("  Disease distribution in full dataset:")
    for disease, count in sorted(disease_counts.items(), key=lambda x: -x[1]):
        logger.info("    %-25s %6d  (%.1f%%)", disease, count, 100.0 * count / n_images)

    # ── BBox CSV ────────────────────────────────────────────────────────
    logger.info("Reading bbox CSV: %s", cfg.bbox_csv)
    bbox_df = pd.read_csv(cfg.bbox_csv)
    bbox_col_map = {"Bbox [x": "bbox_x", "y": "bbox_y", "w": "bbox_w", "h]": "bbox_h"}
    bbox_df.rename(columns=bbox_col_map, inplace=True)

    # Apply label mapping (e.g. Infiltrate -> Infiltration)
    bbox_df["Finding Label"] = bbox_df["Finding Label"].replace(cfg.bbox_label_mapping)

    bbox_dict: Dict[str, Dict[str, List[float]]] = defaultdict(dict)
    for _, row in bbox_df.iterrows():
        img_id = row["Image Index"]
        label = row["Finding Label"]
        bbox_dict[img_id][label] = [
            float(row["bbox_x"]),
            float(row["bbox_y"]),
            float(row["bbox_w"]),
            float(row["bbox_h"]),
        ]

    logger.info("BBox CSV loaded: %d annotations across %d unique images",
                len(bbox_df), len(bbox_dict))

    # Log per-disease bbox annotation counts
    bbox_disease_counts = bbox_df["Finding Label"].value_counts()
    logger.info("  BBox annotation counts by disease:")
    for disease, count in bbox_disease_counts.items():
        logger.info("    %-25s %4d annotations", disease, count)

    # Log bbox size statistics
    bbox_areas = bbox_df["bbox_w"] * bbox_df["bbox_h"]
    logger.info("  BBox area (pixels^2): min=%.0f  max=%.0f  mean=%.0f  median=%.0f",
                bbox_areas.min(), bbox_areas.max(), bbox_areas.mean(), bbox_areas.median())

    return df, dict(bbox_dict)


def _parse_age(val) -> float:
    """Convert age value to float years."""
    s = str(val).strip().upper()
    if s.endswith("Y"):
        s = s[:-1]
    try:
        return float(s)
    except ValueError:
        return 0.0


def compute_follow_up_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Add per-patient follow-up count column."""
    follow_counts = df.groupby("Patient ID")["Follow-up #"].transform("max")
    df["MaxFollowUp"] = follow_counts.astype(float)
    return df


def split_by_patient(
    df: pd.DataFrame, cfg: Config
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split into train / val ensuring no patient leakage."""
    logger.info("Splitting data by patient (val_ratio=%.2f, seed=%d)...",
                cfg.val_ratio, cfg.seed)
    gss = GroupShuffleSplit(n_splits=1, test_size=cfg.val_ratio,
                           random_state=cfg.seed)
    train_idx, val_idx = next(gss.split(df, groups=df["Patient ID"]))
    train_df = df.iloc[train_idx].copy()
    val_df = df.iloc[val_idx].copy()
    logger.info("Train split: %d images from %d patients",
                len(train_df), train_df["Patient ID"].nunique())
    logger.info("Val   split: %d images from %d patients",
                len(val_df), val_df["Patient ID"].nunique())
    # Verify no patient leakage
    overlap = set(train_df["Patient ID"]) & set(val_df["Patient ID"])
    if overlap:
        logger.error("PATIENT LEAKAGE DETECTED: %d patients in both splits!", len(overlap))
    else:
        logger.info("  [OK] No patient leakage between train/val splits")
    return train_df, val_df


def compute_class_weights(train_df: pd.DataFrame, cfg: Config) -> torch.Tensor:
    """Compute positive-class weights from label frequencies (for focal loss)."""
    counts = np.zeros(cfg.num_classes, dtype=np.float64)
    for labels_str in train_df["Finding Labels"]:
        for lbl in labels_str.split("|"):
            lbl = lbl.strip()
            if lbl in cfg.disease_classes:
                idx = cfg.disease_classes.index(lbl)
                counts[idx] += 1
    total = len(train_df)
    # pos_weight = (total - count) / max(count, 1)
    weights = (total - counts) / np.maximum(counts, 1.0)
    logger.info("Class weights: %s", dict(zip(cfg.disease_classes,
                                               [f"{w:.1f}" for w in weights])))
    return torch.tensor(weights, dtype=torch.float32)


# ── Grayscale standardization ───────────────────────────────────────────────


def standardize_grayscale(image: Image.Image) -> Image.Image:
    """Standardize a grayscale X-ray image using CLAHE.

    Steps:
        1. Convert to mode ``"L"`` (8-bit grayscale) if not already.
        2. Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
           to normalize contrast while preserving local detail.
        3. Result is a clean 8-bit [0, 255] grayscale image.

    CLAHE is preferred over global histogram equalization for medical
    imaging because it preserves local contrast variations that carry
    diagnostic information (e.g., subtle opacities, nodule edges).
    """
    if image.mode != "L":
        image = image.convert("L")

    img_array = np.array(image)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_array = clahe.apply(img_array)

    return Image.fromarray(img_array, mode="L")


def grayscale_to_3ch(image: Image.Image) -> Image.Image:
    """Replicate a grayscale ``"L"`` image to 3 identical channels.

    Pretrained backbones (ConvNeXt, ResNet, etc.) expect 3-channel input.
    Replicating the same grayscale channel 3 times is the medically
    correct approach—**no** synthetic color information is introduced.
    """
    return Image.merge("RGB", (image, image, image))


# ── Transforms ──────────────────────────────────────────────────────────────


def get_train_transforms(cfg: Config) -> transforms.Compose:
    """Training transforms for 3-channel replicated grayscale images.

    Uses the *same* grayscale mean/std for each of the 3 identical
    channels.  This preserves the true intensity distribution rather
    than imposing ImageNet colour statistics on monochrome data.
    """
    m = cfg.grayscale_mean
    s = cfg.grayscale_std
    return transforms.Compose([
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1),
                                scale=(0.85, 1.15)),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.ToTensor(),          # -> [3, H, W] float [0, 1]
        transforms.Normalize(mean=[m, m, m], std=[s, s, s]),
        transforms.RandomErasing(p=0.15, scale=(0.02, 0.15)),
    ])


def get_val_transforms(cfg: Config) -> transforms.Compose:
    """Validation transforms – deterministic, no augmentation."""
    m = cfg.grayscale_mean
    s = cfg.grayscale_std
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[m, m, m], std=[s, s, s]),
    ])


# ── Dataset ─────────────────────────────────────────────────────────────────


class ChestXrayDataset(Dataset):
    """Yields (image, labels, metadata, bbox_targets, bbox_mask) tuples."""

    def __init__(
        self,
        df: pd.DataFrame,
        bbox_dict: Dict,
        cfg: Config,
        transform: Optional[transforms.Compose] = None,
        is_training: bool = True,
    ) -> None:
        self.cfg = cfg
        self.bbox_dict = bbox_dict
        self.transform = transform
        self.is_training = is_training

        # Filter to images that actually exist on disk
        exists_mask = df["Image Index"].apply(
            lambda x: os.path.isfile(os.path.join(cfg.image_dir, x))
        )
        if not exists_mask.all():
            n_missing = (~exists_mask).sum()
            logger.warning(
                "%d / %d images not found in %s – skipping them",
                n_missing, len(df), cfg.image_dir,
            )
        self.df = df[exists_mask].reset_index(drop=True)
        logger.info("Dataset size after filtering: %d images", len(self.df))

        # Pre-compute label vectors
        self.labels = self._encode_labels()
        # Pre-compute metadata vectors
        self.metadata = self._encode_metadata()

    # ── label encoding ──────────────────────────────────────────────────

    def _encode_labels(self) -> torch.Tensor:
        """Multi-hot encode Finding Labels."""
        n = len(self.df)
        labels = torch.zeros(n, self.cfg.num_classes, dtype=torch.float32)
        for i, findings_str in enumerate(self.df["Finding Labels"]):
            for lbl in findings_str.split("|"):
                lbl = lbl.strip()
                if lbl in self.cfg.disease_classes:
                    labels[i, self.cfg.disease_classes.index(lbl)] = 1.0
        return labels

    # ── metadata encoding ───────────────────────────────────────────────

    def _encode_metadata(self) -> torch.Tensor:
        """Encode patient age, gender, followup, view position."""
        n = len(self.df)
        meta = torch.zeros(n, self.cfg.metadata_dim, dtype=torch.float32)
        max_followup = max(self.df["Follow-up #"].max(), 1.0)
        max_age = max(self.df["AgeYears"].max(), 1.0)
        for i, row in self.df.iterrows():
            meta[i, 0] = float(row["AgeYears"]) / max_age                      # normalized age
            meta[i, 1] = 1.0 if str(row["Patient Gender"]).strip() == "F" else 0.0  # gender
            meta[i, 2] = float(row["Follow-up #"]) / max_followup              # normalized followup
            meta[i, 3] = 1.0 if str(row["View Position"]).strip() == "AP" else 0.0  # view pos
            meta[i, 4] = 1.0 if float(row.get("MaxFollowUp", 0)) > 0 else 0.0 # has followups
        return meta

    # ── __getitem__ ─────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        image_id = row["Image Index"]

        # ── Load & standardize grayscale X-ray ─────────────────────────
        img_path = os.path.join(self.cfg.image_dir, image_id)
        image = Image.open(img_path)

        # Convert to standardized 8-bit grayscale
        image = standardize_grayscale(image)
        orig_w, orig_h = image.size

        # Resize with high-quality Lanczos resampling
        image = image.resize(
            (self.cfg.image_size, self.cfg.image_size), Image.LANCZOS
        )

        # Replicate to 3 identical channels for pretrained backbone
        image = grayscale_to_3ch(image)

        # ── Bounding boxes (normalized to [0, 1]) ──────────────────────
        bbox_targets = torch.zeros(self.cfg.num_classes, 4, dtype=torch.float32)
        bbox_mask = torch.zeros(self.cfg.num_classes, dtype=torch.float32)

        if image_id in self.bbox_dict:
            for disease_lbl, (bx, by, bw, bh) in self.bbox_dict[image_id].items():
                if disease_lbl in self.cfg.disease_classes:
                    ci = self.cfg.disease_classes.index(disease_lbl)
                    bbox_targets[ci] = torch.tensor([
                        bx / orig_w, by / orig_h,
                        bw / orig_w, bh / orig_h,
                    ])
                    bbox_mask[ci] = 1.0

        # ── Random horizontal flip (adjust bbox accordingly) ───────────
        do_flip = self.is_training and random.random() < 0.5
        if do_flip:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            for ci in range(self.cfg.num_classes):
                if bbox_mask[ci] > 0:
                    x, y, w, h = bbox_targets[ci].tolist()
                    bbox_targets[ci, 0] = 1.0 - x - w   # flip x

        # Apply torchvision transforms (brightness/contrast jitter, affine, normalize)
        if self.transform:
            image = self.transform(image)

        return {
            "image": image,
            "labels": self.labels[idx],
            "metadata": self.metadata[idx],
            "bbox_targets": bbox_targets,
            "bbox_mask": bbox_mask,
        }


# ── DataLoader helpers ──────────────────────────────────────────────────────


def build_loaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    bbox_dict: Dict,
    cfg: Config,
) -> Tuple[DataLoader, DataLoader]:
    """Create training and validation DataLoaders."""
    train_ds = ChestXrayDataset(
        train_df, bbox_dict, cfg,
        transform=get_train_transforms(cfg),
        is_training=True,
    )
    val_ds = ChestXrayDataset(
        val_df, bbox_dict, cfg,
        transform=get_val_transforms(cfg),
        is_training=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=True,
        persistent_workers=cfg.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size * 2,   # no grads → can double batch
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        persistent_workers=cfg.num_workers > 0,
    )
    return train_loader, val_loader
