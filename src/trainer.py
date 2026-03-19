"""Training pipeline with phased training (freeze/unfreeze backbone),
ReduceLROnPlateau, EMA, mixed-precision, gradient accumulation,
checkpoint resume, early stopping, and TensorBoard logging."""

import logging
import os
import time

import numpy as np
import torch
import torchvision.transforms.functional as TF
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .config import Config
from .utils import CombinedLoss, EMAModel, compute_auc, load_checkpoint, save_checkpoint

logger = logging.getLogger(__name__)


class Trainer:
    """Handles the full training/validation lifecycle with 3-phase progressive fine-tuning.

    Phase 1 -- backbone frozen, train classification/detection heads only.
    Phase 2 -- last backbone stage unfrozen (high-level feature adaptation).
    Phase 3 -- full backbone unfrozen with low LR (preserve pretrained knowledge).
    """

    def __init__(
        self,
        model: torch.nn.Module,
        train_loader,
        val_loader,
        cfg: Config,
        pos_weight: torch.Tensor | None = None,
        class_alpha: torch.Tensor | None = None,
    ) -> None:
        self.cfg = cfg
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)

        # ── Loss (with per-class alpha for class imbalance) ─────────────
        pw = pos_weight.to(self.device) if pos_weight is not None else None
        ca = class_alpha.to(self.device) if class_alpha is not None else None
        self.criterion = CombinedLoss(cfg, pos_weight=pw, class_alpha=ca).to(self.device)

        # ── Phase management (3-phase progressive fine-tuning) ──────────
        if cfg.freeze_backbone_epochs > 0:
            self.phase = 1
        elif cfg.partial_unfreeze_epochs > 0:
            self.phase = 2
        else:
            self.phase = 3

        if self.phase == 1:
            self._set_backbone_frozen("frozen")
            logger.info("Phase 1: backbone frozen, training heads for %d epochs",
                        cfg.freeze_backbone_epochs)
        elif self.phase == 2:
            self._set_backbone_frozen("partial")
            logger.info("Phase 2: last backbone stage unfrozen for %d epochs",
                        cfg.partial_unfreeze_epochs)

        # ── Optimizer + Scheduler (depends on phase) ────────────────────
        self._setup_optimizer_and_scheduler()

        # ── Mixed precision ─────────────────────────────────────────────
        self.scaler = torch.amp.GradScaler("cuda", enabled=cfg.mixed_precision)

        # ── EMA (created at phase 2+ for meaningful averages) ───────────
        self.ema: EMAModel | None = None
        if cfg.use_ema and self.phase >= 2:
            self.ema = EMAModel(self.model, decay=cfg.ema_decay)

        # ── Data ────────────────────────────────────────────────────────
        self.train_loader = train_loader
        self.val_loader = val_loader

        # ── TensorBoard ─────────────────────────────────────────────────
        self.writer = SummaryWriter(log_dir=cfg.tensorboard_dir)

        # ── State ───────────────────────────────────────────────────────
        self.start_epoch = 0
        self.global_step = 0
        self.best_auc = 0.0
        self.epochs_no_improve = 0
        self._current_phase_start_epoch = 0
        self._current_phase_base_lrs: list[float] = []

    # ── Backbone freeze / unfreeze ──────────────────────────────────────

    def _set_backbone_frozen(self, mode: str) -> None:
        """Set backbone freeze mode: 'frozen', 'partial', or 'unfrozen'.

        'partial' unfreezes only the last backbone stage (stage 3) for
        Phase 2 progressive fine-tuning.
        """
        for name, p in self.model.named_parameters():
            if "backbone" not in name:
                continue
            if mode == "frozen":
                p.requires_grad = False
            elif mode == "partial":
                is_last_stage = ("stages.3" in name or "stages_3" in name
                                 or "norm3" in name)
                p.requires_grad = is_last_stage
            else:  # unfrozen
                p.requires_grad = True
        n_train = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self.model.parameters())
        logger.info("Backbone %s: %.1fM / %.1fM params trainable",
                     mode.upper(), n_train / 1e6, n_total / 1e6)

    # ── Optimizer + Scheduler creation ──────────────────────────────────

    def _setup_optimizer_and_scheduler(self) -> None:
        cfg = self.cfg
        if self.phase == 1:
            # Phase 1: only head parameters (backbone frozen)
            params = [p for p in self.model.parameters() if p.requires_grad]
            self.optimizer = torch.optim.AdamW(
                params,
                lr=cfg.learning_rate * cfg.head_lr_multiplier,
                weight_decay=cfg.weight_decay,
            )
        elif self.phase == 2:
            # Phase 2: last backbone stage + heads (differential LR)
            last_stage_params = []
            head_params = []
            for n, p in self.model.named_parameters():
                if not p.requires_grad:
                    continue
                if "backbone" in n:
                    last_stage_params.append(p)
                else:
                    head_params.append(p)
            self.optimizer = torch.optim.AdamW([
                {"params": last_stage_params, "lr": cfg.learning_rate},
                {"params": head_params,       "lr": cfg.learning_rate * cfg.head_lr_multiplier},
            ], weight_decay=cfg.weight_decay)
        else:
            # Phase 3: full backbone (3 groups: early stages, last stage, heads)
            early_backbone = []
            late_backbone = []
            head_params = []
            for n, p in self.model.named_parameters():
                if "backbone" in n:
                    if "stages.3" in n or "stages_3" in n or "norm3" in n:
                        late_backbone.append(p)
                    else:
                        early_backbone.append(p)
                else:
                    head_params.append(p)
            self.optimizer = torch.optim.AdamW([
                {"params": early_backbone, "lr": cfg.learning_rate * 0.1},
                {"params": late_backbone,  "lr": cfg.learning_rate},
                {"params": head_params,    "lr": cfg.learning_rate * cfg.head_lr_multiplier},
            ], weight_decay=cfg.weight_decay)

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="max", factor=cfg.lr_reduce_factor,
            patience=cfg.lr_reduce_patience, min_lr=cfg.min_lr,
        )

    # ── Phase transitions ─────────────────────────────────────────────────

    def _transition_to_phase2(self, epoch: int) -> None:
        logger.info("=" * 70)
        logger.info("PHASE 2: Partially unfreezing backbone (last stage) at epoch %d", epoch)
        self.phase = 2
        self._set_backbone_frozen("partial")
        self._setup_optimizer_and_scheduler()
        self._current_phase_start_epoch = epoch
        self._current_phase_base_lrs = [pg["lr"] for pg in self.optimizer.param_groups]
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.cfg.mixed_precision)
        if self.cfg.use_ema:
            self.ema = EMAModel(self.model, decay=self.cfg.ema_decay)
        logger.info("  Last-stage LR=%.2e  Head LR=%.2e  Warmup=%d epochs",
                     self._current_phase_base_lrs[0], self._current_phase_base_lrs[-1],
                     self.cfg.warmup_epochs)
        logger.info("=" * 70)

    def _transition_to_phase3(self, epoch: int) -> None:
        logger.info("=" * 70)
        logger.info("PHASE 3: Full backbone unfreeze (low LR) at epoch %d", epoch)
        self.phase = 3
        self._set_backbone_frozen("unfrozen")
        self._setup_optimizer_and_scheduler()
        self._current_phase_start_epoch = epoch
        self._current_phase_base_lrs = [pg["lr"] for pg in self.optimizer.param_groups]
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.cfg.mixed_precision)
        if self.ema is not None:
            self.ema = EMAModel(self.model, decay=self.cfg.ema_decay)
        logger.info("  Early-bb LR=%.2e  Late-bb LR=%.2e  Head LR=%.2e  Warmup=%d ep",
                     self._current_phase_base_lrs[0], self._current_phase_base_lrs[1],
                     self._current_phase_base_lrs[-1], self.cfg.warmup_epochs)
        logger.info("=" * 70)

    # ── Warmup LR adjustment ───────────────────────────────────────────

    def _apply_warmup_lr(self, epoch: int) -> None:
        """Linear warmup for the first few epochs after each phase transition."""
        if self.phase == 1 or not self._current_phase_base_lrs:
            return
        phase_epoch = epoch - self._current_phase_start_epoch
        if 0 <= phase_epoch < self.cfg.warmup_epochs:
            factor = (phase_epoch + 1) / self.cfg.warmup_epochs
            for pg, base_lr in zip(self.optimizer.param_groups, self._current_phase_base_lrs):
                pg["lr"] = base_lr * factor
            logger.info("  Warmup: factor=%.3f  backbone_lr=%.2e  head_lr=%.2e",
                        factor, self.optimizer.param_groups[0]["lr"],
                        self.optimizer.param_groups[-1]["lr"])

    # ── Auto-resume ─────────────────────────────────────────────────────

    def auto_resume(self) -> None:
        """Load the latest checkpoint if one exists."""
        ckpt_dir = self.cfg.checkpoint_dir
        if not os.path.isdir(ckpt_dir):
            return

        ckpts = sorted(
            [f for f in os.listdir(ckpt_dir) if f.startswith("epoch_") and f.endswith(".pth")],
            key=lambda x: int(x.split("_")[1].split(".")[0]),
        )
        if not ckpts:
            best_path = os.path.join(ckpt_dir, "best_model.pth")
            if os.path.isfile(best_path):
                ckpts = ["best_model.pth"]
            else:
                return

        latest = os.path.join(ckpt_dir, ckpts[-1])
        logger.info("Resuming from checkpoint: %s", latest)
        ckpt = load_checkpoint(latest, device=str(self.device))

        # Restore phase state (transition phases as needed before loading optimizer)
        saved_phase = ckpt.get("phase", 3)
        if saved_phase >= 2 and self.phase == 1:
            self._transition_to_phase2(
                ckpt.get("current_phase_start_epoch",
                         ckpt.get("phase2_start_epoch", self.cfg.freeze_backbone_epochs)))
        if saved_phase >= 3 and self.phase == 2:
            phase3_start = self.cfg.freeze_backbone_epochs + self.cfg.partial_unfreeze_epochs
            self._transition_to_phase3(
                ckpt.get("current_phase_start_epoch", phase3_start))

        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.scaler.load_state_dict(ckpt["scaler_state_dict"])
        self.start_epoch = ckpt["epoch"] + 1
        self.global_step = ckpt.get("global_step", 0)
        self.best_auc = ckpt.get("best_auc", 0.0)
        self.epochs_no_improve = ckpt.get("epochs_no_improve", 0)
        self._current_phase_start_epoch = ckpt.get(
            "current_phase_start_epoch",
            ckpt.get("phase2_start_epoch", self.cfg.freeze_backbone_epochs))
        self._current_phase_base_lrs = ckpt.get(
            "current_phase_base_lrs",
            ckpt.get("phase2_base_lrs", []))

        # Restore EMA
        if self.ema is not None and "ema_state" in ckpt:
            self.ema.load_state_dict(ckpt["ema_state"])

        logger.info("Resumed at epoch %d  (phase %d, best AUC = %.4f)",
                     self.start_epoch, self.phase, self.best_auc)

    # ── Save ────────────────────────────────────────────────────────────

    def _save(self, epoch: int, is_best: bool = False) -> None:
        state = {
            "epoch": epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "best_auc": self.best_auc,
            "epochs_no_improve": self.epochs_no_improve,
            "phase": self.phase,
            "current_phase_start_epoch": self._current_phase_start_epoch,
            "current_phase_base_lrs": self._current_phase_base_lrs,
            "config": vars(self.cfg),
        }
        if self.ema is not None:
            state["ema_state"] = self.ema.state_dict()
        path = os.path.join(self.cfg.checkpoint_dir, f"epoch_{epoch:03d}.pth")
        save_checkpoint(state, path, is_best=is_best)
        logger.info("Saved checkpoint -> %s%s", path,
                     " (* best)" if is_best else "")

    # ── Train one epoch ─────────────────────────────────────────────────

    def _train_epoch(self, epoch: int) -> dict:
        self.model.train()
        running = {"total": 0.0, "cls_loss": 0.0, "bbox_loss": 0.0}
        n_batches = 0
        total_batches = len(self.train_loader)
        log_interval = max(total_batches // 10, 1)  # log ~10 times per epoch

        # Track min/max losses for the epoch
        epoch_loss_min = float("inf")
        epoch_loss_max = float("-inf")

        logger.debug("-- Train epoch %02d  |  %d batches  |  batch_size=%d  "
                    "|  grad_accum=%d --",
                    epoch, total_batches, self.cfg.batch_size,
                    self.cfg.grad_accum_steps)

        self.optimizer.zero_grad(set_to_none=True)
        pbar = tqdm(self.train_loader, desc=f"Train E{epoch:02d}",
                    leave=False, dynamic_ncols=True)

        for batch_idx, batch in enumerate(pbar):
            images = batch["image"].to(self.device, non_blocking=True)
            labels = batch["labels"].to(self.device, non_blocking=True)
            metadata = batch["metadata"].to(self.device, non_blocking=True)
            bbox_targets = batch["bbox_targets"].to(self.device, non_blocking=True)
            bbox_mask = batch["bbox_mask"].to(self.device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=self.cfg.mixed_precision):
                logits, bbox_pred = self.model(images, metadata)
                losses = self.criterion(logits, bbox_pred, labels,
                                        bbox_targets, bbox_mask)
                loss = losses["total"] / self.cfg.grad_accum_steps

            self.scaler.scale(loss).backward()

            grad_norm = 0.0
            if (batch_idx + 1) % self.cfg.grad_accum_steps == 0:
                self.scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg.gradient_clip).item()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                self.global_step += 1

                # EMA update
                if self.ema is not None:
                    self.ema.update(self.model)

                # TensorBoard per-step scalars
                self.writer.add_scalar("step/train_loss", losses["total"].item(), self.global_step)
                self.writer.add_scalar("step/cls_loss", losses["cls_loss"].item(), self.global_step)
                self.writer.add_scalar("step/bbox_loss", losses["bbox_loss"].item(), self.global_step)
                self.writer.add_scalar("step/grad_norm", grad_norm, self.global_step)
                self.writer.add_scalar("step/lr_backbone", self.optimizer.param_groups[0]["lr"], self.global_step)
                self.writer.add_scalar("step/lr_heads", self.optimizer.param_groups[-1]["lr"], self.global_step)
                self.writer.add_scalar("step/scaler_scale", self.scaler.get_scale(), self.global_step)

            batch_loss = losses["total"].item()
            epoch_loss_min = min(epoch_loss_min, batch_loss)
            epoch_loss_max = max(epoch_loss_max, batch_loss)

            for k in running:
                running[k] += losses[k].item()
            n_batches += 1

            # Detailed periodic log
            if (batch_idx + 1) % log_interval == 0 or batch_idx == total_batches - 1:
                avg_total = running["total"] / n_batches
                avg_cls = running["cls_loss"] / n_batches
                avg_bbox = running["bbox_loss"] / n_batches
                backbone_lr = self.optimizer.param_groups[0]["lr"]
                head_lr = self.optimizer.param_groups[-1]["lr"]
                bbox_samples = bbox_mask.sum().item()
                scaler_scale = self.scaler.get_scale()

                gpu_mem_mb = torch.cuda.memory_allocated(self.device) / 1024**2
                gpu_reserved_mb = torch.cuda.memory_reserved(self.device) / 1024**2

                logger.debug(
                    "  [Epoch %02d] Batch %4d/%d  "
                    "loss=%.4f (cls=%.4f bbox=%.4f)  "
                    "grad_norm=%.3f  lr_bb=%.2e lr_hd=%.2e  "
                    "scaler=%.0f  bbox_samples=%d  "
                    "GPU=%.0fMB/%.0fMB",
                    epoch, batch_idx + 1, total_batches,
                    avg_total, avg_cls, avg_bbox,
                    grad_norm, backbone_lr, head_lr,
                    scaler_scale, int(bbox_samples),
                    gpu_mem_mb, gpu_reserved_mb,
                )

            pbar.set_postfix(
                loss=f"{running['total']/n_batches:.4f}",
                cls=f"{running['cls_loss']/n_batches:.4f}",
                bbox=f"{running['bbox_loss']/n_batches:.4f}",
                lr=f"{self.optimizer.param_groups[0]['lr']:.2e}",
                gn=f"{grad_norm:.2f}",
            )

        avg = {k: v / max(n_batches, 1) for k, v in running.items()}
        avg["loss_min"] = epoch_loss_min
        avg["loss_max"] = epoch_loss_max

        logger.debug("  [Epoch %02d] Train summary:  avg_loss=%.4f  "
                    "min_loss=%.4f  max_loss=%.4f  "
                    "avg_cls=%.4f  avg_bbox=%.4f  "
                    "batches=%d  global_step=%d",
                    epoch, avg["total"], epoch_loss_min, epoch_loss_max,
                    avg["cls_loss"], avg["bbox_loss"],
                    n_batches, self.global_step)

        return avg

    # ── TTA (Test Time Augmentation) ───────────────────────────────────────

    def _tta_augmentations(self, images: torch.Tensor) -> list:
        """Generate TTA augmented image batches (excluding original).

        Returns up to (tta_augmentations - 1) augmented versions.
        """
        augmented = []
        n = self.cfg.tta_augmentations

        # 1. Horizontal flip
        if n >= 2:
            augmented.append(torch.flip(images, dims=[-1]))
        # 2. Slight rotation (+5 degrees)
        if n >= 3:
            augmented.append(TF.rotate(images, 5))
        # 3. Slight rotation (-5 degrees)
        if n >= 4:
            augmented.append(TF.rotate(images, -5))
        # 4. Horizontal flip + slight rotation
        if n >= 5:
            augmented.append(TF.rotate(torch.flip(images, dims=[-1]), 5))

        return augmented

    # ── Validate ────────────────────────────────────────────────────────

    @torch.no_grad()
    def _validate(self, epoch: int) -> dict:
        self.model.eval()
        all_logits, all_labels = [], []
        running = {"total": 0.0, "cls_loss": 0.0, "bbox_loss": 0.0}
        n_batches = 0
        total_val_batches = len(self.val_loader)
        total_bbox_samples = 0
        use_tta = self.cfg.use_tta

        logger.debug("-- Val   epoch %02d  |  %d batches  |  TTA=%s --",
                     epoch, total_val_batches,
                     f"{self.cfg.tta_augmentations} views" if use_tta else "off")

        pbar = tqdm(self.val_loader, desc=f"Val   E{epoch:02d}",
                    leave=False, dynamic_ncols=True)

        for batch in pbar:
            images = batch["image"].to(self.device, non_blocking=True)
            labels = batch["labels"].to(self.device, non_blocking=True)
            metadata = batch["metadata"].to(self.device, non_blocking=True)
            bbox_targets = batch["bbox_targets"].to(self.device, non_blocking=True)
            bbox_mask = batch["bbox_mask"].to(self.device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=self.cfg.mixed_precision):
                logits, bbox_pred = self.model(images, metadata)
                losses = self.criterion(logits, bbox_pred, labels,
                                        bbox_targets, bbox_mask)

            for k in running:
                running[k] += losses[k].item()
            n_batches += 1
            total_bbox_samples += int(bbox_mask.sum().item())

            # TTA: average logits across augmented views for AUC
            if use_tta:
                tta_logits = [logits]
                for aug_images in self._tta_augmentations(images):
                    with torch.amp.autocast("cuda", enabled=self.cfg.mixed_precision):
                        aug_logits, _ = self.model(aug_images, metadata)
                    tta_logits.append(aug_logits)
                avg_logits = torch.stack(tta_logits).mean(dim=0)
                all_logits.append(avg_logits.cpu().numpy())
            else:
                all_logits.append(logits.cpu().numpy())

            all_labels.append(labels.cpu().numpy())

        avg = {k: v / max(n_batches, 1) for k, v in running.items()}

        # AUC metrics
        all_logits_np = np.concatenate(all_logits, axis=0)
        all_labels_np = np.concatenate(all_labels, axis=0)
        auc_dict = compute_auc(all_logits_np, all_labels_np,
                               self.cfg.disease_classes)
        avg.update(auc_dict)

        # Detailed validation log
        n_val_images = all_labels_np.shape[0]
        logger.debug("  [Epoch %02d] Val summary:  %d images  |  "
                    "avg_loss=%.4f  cls_loss=%.4f  bbox_loss=%.4f  |  "
                    "bbox_annotated_samples=%d",
                    epoch, n_val_images,
                    avg["total"], avg["cls_loss"], avg["bbox_loss"],
                    total_bbox_samples)

        # Per-class AUC table
        logger.debug("  [Epoch %02d] Per-class AUC-ROC:", epoch)
        logger.debug("  %-25s  %8s  %8s  %8s", "Disease", "AUC", "#Pos", "#Neg")
        logger.debug("  %s", "-" * 55)
        for i, name in enumerate(self.cfg.disease_classes):
            auc_val = auc_dict.get(name, float("nan"))
            n_pos = int(all_labels_np[:, i].sum())
            n_neg = n_val_images - n_pos
            auc_str = f"{auc_val:.4f}" if auc_val == auc_val else "  N/A"
            logger.debug("  %-25s  %8s  %8d  %8d", name, auc_str, n_pos, n_neg)
        logger.debug("  %s", "-" * 55)
        logger.debug("  %-25s  %8.4f", "MEAN AUC", auc_dict.get("mean_AUC", 0.0))

        return avg

    # ── Main training loop ──────────────────────────────────────────────

    def fit(self) -> None:
        """Run full training with phased training, auto-resume, EMA, and early stopping."""
        self.auto_resume()

        logger.info("=" * 70)
        logger.info("Starting training  |  epochs %d->%d  |  device %s",
                     self.start_epoch, self.cfg.epochs - 1, self.device)
        logger.info("Effective batch size: %d  (batch %d x accum %d)",
                     self.cfg.batch_size * self.cfg.grad_accum_steps,
                     self.cfg.batch_size, self.cfg.grad_accum_steps)
        logger.info("Phase %d  |  phase1=%d  phase2=%d  phase3=%d+  |  warmup=%d",
                     self.phase, self.cfg.freeze_backbone_epochs,
                     self.cfg.partial_unfreeze_epochs,
                     max(0, self.cfg.epochs - self.cfg.freeze_backbone_epochs - self.cfg.partial_unfreeze_epochs),
                     self.cfg.warmup_epochs)
        if self.cfg.use_ema:
            logger.info("EMA enabled  (decay=%.4f)", self.cfg.ema_decay)
        if self.cfg.use_tta:
            logger.info("TTA enabled  (%d augmentation views)", self.cfg.tta_augmentations)
        logger.info("=" * 70)

        for epoch in range(self.start_epoch, self.cfg.epochs):
            t0 = time.time()

            # ── Phase transitions (3-phase progressive fine-tuning) ───
            if (self.phase == 1
                    and epoch >= self.cfg.freeze_backbone_epochs):
                self._transition_to_phase2(epoch)
            elif (self.phase == 2
                    and epoch >= self.cfg.freeze_backbone_epochs + self.cfg.partial_unfreeze_epochs):
                self._transition_to_phase3(epoch)

            # ── Warmup LR for early phase-2/3 epochs ───────────────────
            self._apply_warmup_lr(epoch)

            # Train
            train_metrics = self._train_epoch(epoch)

            # Validate (with EMA weights if available)
            if self.ema is not None:
                self.ema.apply_shadow(self.model)
            val_metrics = self._validate(epoch)
            if self.ema is not None:
                self.ema.restore(self.model)

            elapsed = time.time() - t0
            mean_auc = val_metrics.get("mean_AUC", 0.0)

            # ── Step ReduceLROnPlateau (after warmup) ───────────────────
            phase_epoch = epoch - self._current_phase_start_epoch if self.phase >= 2 else -1
            if phase_epoch >= self.cfg.warmup_epochs:
                old_lr = self.optimizer.param_groups[0]["lr"]
                self.scheduler.step(mean_auc)
                new_lr = self.optimizer.param_groups[0]["lr"]
                if new_lr < old_lr:
                    logger.info("  ReduceLROnPlateau: LR reduced %.2e -> %.2e",
                                old_lr, new_lr)

            # ── Epoch summary log ───────────────────────────────────────
            logger.info("=" * 70)
            logger.info(
                "EPOCH %02d/%02d COMPLETE  [%.0fs = %.1f min]  (Phase %d%s)",
                epoch, self.cfg.epochs - 1, elapsed, elapsed / 60.0,
                self.phase, ", EMA" if self.ema is not None else "",
            )
            logger.info(
                "  Train  > total_loss=%.4f  cls_loss=%.4f  bbox_loss=%.4f  "
                "loss_range=[%.4f, %.4f]",
                train_metrics["total"], train_metrics["cls_loss"],
                train_metrics["bbox_loss"],
                train_metrics.get("loss_min", 0), train_metrics.get("loss_max", 0),
            )
            logger.info(
                "  Val    > total_loss=%.4f  cls_loss=%.4f  bbox_loss=%.4f  "
                "mean_AUC=%.4f",
                val_metrics["total"], val_metrics["cls_loss"],
                val_metrics["bbox_loss"], mean_auc,
            )
            logger.info(
                "  LR     > backbone=%.2e  heads=%.2e  |  "
                "scaler_scale=%.0f  global_step=%d",
                self.optimizer.param_groups[0]["lr"],
                self.optimizer.param_groups[-1]["lr"],
                self.scaler.get_scale(), self.global_step,
            )

            # GPU memory snapshot
            if torch.cuda.is_available():
                alloc_gb = torch.cuda.memory_allocated(self.device) / 1024**3
                reserved_gb = torch.cuda.memory_reserved(self.device) / 1024**3
                peak_gb = torch.cuda.max_memory_allocated(self.device) / 1024**3
                logger.info(
                    "  GPU    > allocated=%.2f GB  reserved=%.2f GB  "
                    "peak=%.2f GB",
                    alloc_gb, reserved_gb, peak_gb,
                )
            logger.info("=" * 70)

            # TensorBoard epoch-level scalars
            self.writer.add_scalar("epoch/train_loss", train_metrics["total"], epoch)
            self.writer.add_scalar("epoch/train_cls_loss", train_metrics["cls_loss"], epoch)
            self.writer.add_scalar("epoch/train_bbox_loss", train_metrics["bbox_loss"], epoch)
            self.writer.add_scalar("epoch/val_loss", val_metrics["total"], epoch)
            self.writer.add_scalar("epoch/val_cls_loss", val_metrics["cls_loss"], epoch)
            self.writer.add_scalar("epoch/val_bbox_loss", val_metrics["bbox_loss"], epoch)
            self.writer.add_scalar("epoch/val_mean_AUC", mean_auc, epoch)
            self.writer.add_scalar("epoch/lr_backbone", self.optimizer.param_groups[0]["lr"], epoch)
            self.writer.add_scalar("epoch/lr_heads", self.optimizer.param_groups[-1]["lr"], epoch)
            self.writer.add_scalar("epoch/phase", self.phase, epoch)
            for name in self.cfg.disease_classes:
                auc_val = val_metrics.get(name, 0.0)
                if auc_val == auc_val:  # skip NaN
                    self.writer.add_scalar(f"val_auc/{name}", auc_val, epoch)
            self.writer.flush()

            # ── Checkpointing ───────────────────────────────────────────
            is_best = mean_auc > self.best_auc
            if is_best:
                prev_best = self.best_auc
                self.best_auc = mean_auc
                self.epochs_no_improve = 0
                logger.info("* New best AUC: %.4f -> %.4f  (improved by %.4f)",
                            prev_best, mean_auc, mean_auc - prev_best)
            else:
                self.epochs_no_improve += 1
                logger.info("  No improvement for %d epoch(s)  (best=%.4f, "
                            "patience=%d/%d)",
                            self.epochs_no_improve, self.best_auc,
                            self.epochs_no_improve, self.cfg.patience)

            if is_best or (epoch + 1) % self.cfg.save_every_epochs == 0:
                self._save(epoch, is_best=is_best)

            # ── Early stopping ──────────────────────────────────────────
            if self.epochs_no_improve >= self.cfg.patience:
                logger.info("Early stopping triggered after %d epochs without "
                            "improvement (best AUC=%.4f)", self.cfg.patience,
                            self.best_auc)
                self._save(epoch, is_best=False)
                break

        self.writer.close()
        logger.info("Training complete.  Best validation mean AUC = %.4f",
                     self.best_auc)
