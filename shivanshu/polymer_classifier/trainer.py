"""
Training engine for Polymer Classification.
Features mixed precision (AMP), differential learning rate scheduling with warmup,
class-balanced tracking, and checkpointing based on Validation Macro-F1.
"""

import os
import time
import json
from typing import Dict, List, Any, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import ClassifierConfig, get_device
from .models import PolymerClassifier
from .losses import get_criterion
from .utils import compute_classification_metrics, format_classification_table


class PolymerTrainer:
    """
    Orchestrates the training, validation, metric evaluation, and checkpoint persistence.
    """

    def __init__(
        self,
        model: PolymerClassifier,
        config: ClassifierConfig,
        train_loader: DataLoader,
        val_loader: DataLoader,
        class_counts: List[int],
        classes: List[str],
        device: torch.device,
    ):
        self.model = model.to(device)
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.class_counts = class_counts
        self.classes = classes
        self.device = device

        # Loss Criterion
        self.criterion = get_criterion(
            loss_type=config.loss_type,
            class_counts=class_counts,
            gamma=config.focal_gamma,
            label_smoothing=config.label_smoothing,
            device=device,
        )

        # Optimizer with differential learning rates
        param_groups = self.model.get_parameter_groups(
            base_lr=config.base_lr,
            head_lr=config.head_lr,
            weight_decay=config.weight_decay,
        )
        self.optimizer = torch.optim.AdamW(param_groups)

        # Cosine Annealing with Warmup Scheduler
        total_steps = config.epochs
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=max(1, total_steps - config.warmup_epochs),
            eta_min=config.min_lr,
        )

        # Automatic Mixed Precision
        self.scaler = torch.amp.GradScaler("cuda", enabled=config.use_amp and device.type == "cuda")

        self.history: List[Dict[str, Any]] = []
        self.best_macro_f1 = 0.0
        self.best_epoch = 0

    def _train_epoch(self, epoch: int) -> Tuple[float, float]:
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        # Linear LR warmup for epoch <= warmup_epochs
        if epoch <= self.config.warmup_epochs:
            warmup_factor = float(epoch) / float(max(1, self.config.warmup_epochs))
            for i, group in enumerate(self.optimizer.param_groups):
                target_lr = self.config.head_lr if i == 2 else self.config.base_lr
                group["lr"] = target_lr * warmup_factor

        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            inputs, targets = inputs.to(self.device, non_blocking=True), targets.to(self.device, non_blocking=True)
            self.optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=self.config.use_amp and self.device.type == "cuda"):
                logits = self.model(inputs)
                loss = self.criterion(logits, targets)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            batch_size = inputs.size(0)
            running_loss += loss.item() * batch_size
            _, preds = torch.max(logits, 1)
            correct += (preds == targets).sum().item()
            total += batch_size

        if epoch > self.config.warmup_epochs:
            self.scheduler.step()

        epoch_loss = running_loss / max(1, total)
        epoch_acc = correct / max(1, total)
        return epoch_loss, epoch_acc

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Tuple[float, Dict[str, Any]]:
        self.model.eval()
        running_loss = 0.0
        total = 0
        all_targets = []
        all_preds = []
        all_probs = []

        for inputs, targets in loader:
            inputs, targets = inputs.to(self.device, non_blocking=True), targets.to(self.device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=self.config.use_amp and self.device.type == "cuda"):
                logits = self.model(inputs)
                loss = self.criterion(logits, targets)

            probs = torch.softmax(logits, dim=1)
            _, preds = torch.max(logits, 1)

            running_loss += loss.item() * inputs.size(0)
            total += inputs.size(0)

            all_targets.extend(targets.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())
            all_probs.append(probs.cpu().numpy())

        eval_loss = running_loss / max(1, total)
        eval_probs = np.vstack(all_probs) if all_probs else None
        metrics = compute_classification_metrics(all_targets, all_preds, self.classes, eval_probs)
        return eval_loss, metrics

    def train(self) -> Dict[str, Any]:
        os.makedirs(self.config.save_dir, exist_ok=True)
        print("=" * 80)
        print(f"🚀 Launching Polymer Classification Training Pipeline")
        print(f"Backbone: {self.config.model_name} | Device: {self.device}")
        print(f"Loss: {self.config.loss_type} (γ={self.config.focal_gamma}) | Sampler: {self.config.sampler}")
        print(f"Epochs: {self.config.epochs} | Batch Size: {self.config.batch_size} | Head LR: {self.config.head_lr}")
        print("=" * 80)

        for epoch in range(1, self.config.epochs + 1):
            t0 = time.time()
            train_loss, train_acc = self._train_epoch(epoch)
            elapsed = time.time() - t0

            val_loss, val_metrics = 0.0, {}
            if self.val_loader:
                val_loss, val_metrics = self.evaluate(self.val_loader)

            val_acc = val_metrics.get("accuracy", 0.0)
            val_macro_f1 = val_metrics.get("macro_f1", 0.0)
            val_bal_acc = val_metrics.get("balanced_accuracy", 0.0)

            # Per-class recall summary string
            per_class_recalls = []
            if "per_class" in val_metrics:
                for c in self.classes:
                    rec = val_metrics["per_class"].get(c, {}).get("recall", 0.0)
                    per_class_recalls.append(f"{c}: {rec*100:.1f}%")
            pc_str = " | ".join(per_class_recalls)

            print(
                f"Epoch [{epoch:>2}/{self.config.epochs:>2}] "
                f"Train Loss: {train_loss:.4f} Acc: {train_acc*100:.2f}% | "
                f"Val Loss: {val_loss:.4f} Acc: {val_acc*100:.2f}% | "
                f"Macro F1: {val_macro_f1*100:.2f}% | Time: {elapsed:.1f}s"
            )
            if pc_str:
                print(f"   ↳ Per-Class Recall: {pc_str}")

            record = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_macro_f1": val_macro_f1,
                "val_balanced_acc": val_bal_acc,
                "elapsed_sec": elapsed,
            }
            self.history.append(record)

            # Checkpoint saved based on best Validation Macro F1 to preserve minority classes
            if val_macro_f1 > self.best_macro_f1 or epoch == 1:
                self.best_macro_f1 = val_macro_f1
                self.best_epoch = epoch
                save_payload = {
                    "epoch": epoch,
                    "model_name": self.config.model_name,
                    "model_state_dict": self.model.state_dict(),
                    "classes": self.classes,
                    "val_macro_f1": val_macro_f1,
                    "val_acc": val_acc,
                    "val_balanced_acc": val_bal_acc,
                    "val_metrics": val_metrics,
                    "config": vars(self.config),
                }
                torch.save(save_payload, self.config.checkpoint_path)
                print(f"   ★ Best model checkpoint saved! (Macro F1: {val_macro_f1*100:.2f}%)")

        # Save training history JSON
        history_path = os.path.join(self.config.save_dir, "training_history.json")
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)

        print("\n" + "=" * 80)
        print(f"🏁 Training Finished! Best Epoch: {self.best_epoch} with Val Macro F1: {self.best_macro_f1*100:.2f}%")
        print(f"Saved Checkpoint: {self.config.checkpoint_path}")
        print("=" * 80)

        return {
            "best_epoch": self.best_epoch,
            "best_macro_f1": self.best_macro_f1,
            "history": self.history,
        }
