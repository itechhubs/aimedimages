"""
Lung Disease Detection Model – Multi-task (Classification + BBox Regression).

Architecture
~~~~~~~~~~~~
* **Backbone**: ConvNeXt V2 Base (ImageNet-22k → 1k pretrained) via *timm*.
* **Metadata encoder**: small MLP that fuses patient age, gender, follow-up
  count, view position, and has-followups flag.
* **Classification head**: multi-label (14 diseases) with dropout.
* **Detection head**: per-class bounding-box regression ([x, y, w, h] in 0-1).
"""

import logging
from typing import Tuple

import timm
import torch
import torch.nn as nn

from .config import Config

logger = logging.getLogger(__name__)


class LungDiseaseNet(nn.Module):
    """Multi-task chest X-ray model: classification + localization."""

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.num_classes = cfg.num_classes

        # ── Backbone (ConvNeXt V2 via timm) ─────────────────────────────
        self.backbone = self._build_backbone(cfg)
        backbone_dim = cfg.backbone_features

        # ── Global Average Pooling ──────────────────────────────────────
        self.gap = nn.AdaptiveAvgPool2d(1)

        # ── Metadata encoder ────────────────────────────────────────────
        self.meta_encoder = nn.Sequential(
            nn.Linear(cfg.metadata_dim, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, cfg.meta_hidden),
            nn.GELU(),
        )

        fusion_dim = backbone_dim + cfg.meta_hidden

        # ── Classification head (multi-label) ───────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, cfg.cls_hidden),
            nn.GELU(),
            nn.Dropout(cfg.drop_rate),
            nn.Linear(cfg.cls_hidden, cfg.num_classes),
        )

        # ── Detection head (per-class bbox [x,y,w,h] in 0-1) ──────────
        self.detector = nn.Sequential(
            nn.Linear(fusion_dim, cfg.det_hidden),
            nn.GELU(),
            nn.Dropout(cfg.drop_rate * 0.5),
            nn.Linear(cfg.det_hidden, cfg.num_classes * 4),
        )

        # ── Spatial attention for detection (learns where to look) ──────
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(backbone_dim, backbone_dim // 4, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(backbone_dim // 4, cfg.num_classes, kernel_size=1),
            nn.Sigmoid(),
        )
        self.spatial_pool = nn.AdaptiveAvgPool2d(1)

        # ── Detection refinement head (spatial + fused features) ────────
        self.det_refine = nn.Sequential(
            nn.Linear(cfg.num_classes + fusion_dim, cfg.det_hidden),
            nn.GELU(),
            nn.Dropout(cfg.drop_rate * 0.5),
            nn.Linear(cfg.det_hidden, cfg.num_classes * 4),
        )

        self._init_weights()

    # ── backbone builder ────────────────────────────────────────────────

    @staticmethod
    def _build_backbone(cfg: Config):
        """Load a timm backbone with ``features_only=True``."""
        model_name = cfg.backbone
        fallbacks = [
            "convnext_base.fb_in22k_ft_in1k_384",
            "convnext_base.fb_in22k_ft_in1k",
            "convnext_base",
        ]
        for name in [model_name] + fallbacks:
            try:
                backbone = timm.create_model(
                    name,
                    pretrained=True,
                    features_only=True,
                    out_indices=(3,),       # last stage only
                    in_chans=3,
                    drop_path_rate=cfg.drop_path_rate,
                )
                logger.info("Loaded backbone: %s", name)
                return backbone
            except Exception:
                logger.warning("Backbone %s not available, trying fallback...", name)
        raise RuntimeError("No compatible timm backbone found. "
                           "Install timm >= 0.9 with: pip install timm")

    # ── weight init ─────────────────────────────────────────────────────

    def _init_weights(self) -> None:
        """Kaiming init for head layers (backbone already pretrained)."""
        for module in [self.meta_encoder, self.classifier,
                       self.detector, self.spatial_attn, self.det_refine]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, nonlinearity="linear")
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                            nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    # ── forward ─────────────────────────────────────────────────────────

    def forward(
        self,
        images: torch.Tensor,
        metadata: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        images   : [B, 3, H, W]
        metadata : [B, metadata_dim]

        Returns
        -------
        logits    : [B, num_classes]          classification logits
        bbox_pred : [B, num_classes, 4]       predicted normalized bboxes
        """
        # Backbone → spatial feature map
        feat_map = self.backbone(images)[-1]                 # [B, C, h, w]

        # Global features for classification
        pooled = self.gap(feat_map).flatten(1)               # [B, C]

        # Metadata fusion
        meta_feat = self.meta_encoder(metadata)              # [B, meta_hidden]
        fused = torch.cat([pooled, meta_feat], dim=1)        # [B, C+meta_hidden]

        # ── Classification ──────────────────────────────────────────────
        logits = self.classifier(fused)                      # [B, num_classes]

        # ── Detection with spatial attention ────────────────────────────
        attn_maps = self.spatial_attn(feat_map)              # [B, num_classes, h, w]
        attn_pooled = self.spatial_pool(attn_maps).flatten(1)  # [B, num_classes]

        det_input = torch.cat([fused, attn_pooled], dim=1)   # [B, fusion+num_classes]
        bbox_pred = self.det_refine(det_input)               # [B, num_classes*4]
        bbox_pred = bbox_pred.view(-1, self.num_classes, 4)  # [B, num_classes, 4]
        bbox_pred = torch.sigmoid(bbox_pred)                 # normalize to [0, 1]

        return logits, bbox_pred
