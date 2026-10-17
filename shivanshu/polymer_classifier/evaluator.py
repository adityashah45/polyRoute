"""
Comprehensive evaluation engine for the Polymer Classification model.
Generates publication-quality performance reports, per-class breakdowns,
ASCII confusion matrices, and PNG heatmaps.
"""

import os
import json
from typing import Dict, Any, Optional
import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import ClassifierConfig, get_device, DEFAULT_CLASSES
from .models import PolymerClassifier, load_classifier_checkpoint
from .dataset import create_dataloaders, get_transforms
from .utils import (
    compute_classification_metrics,
    format_classification_table,
    format_confusion_matrix_ascii,
    save_confusion_matrix_figure,
)


class PolymerEvaluator:
    """
    Evaluator for rigorous benchmark verification on held-out test splits.
    """

    def __init__(
        self,
        model: PolymerClassifier,
        classes: Optional[list] = None,
        device: Optional[torch.device] = None,
    ):
        self.device = device or get_device()
        self.model = model.to(self.device)
        self.model.eval()
        self.classes = classes or list(DEFAULT_CLASSES)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        device: Optional[torch.device] = None,
    ) -> "PolymerEvaluator":
        device = device or get_device()
        model, checkpoint = load_classifier_checkpoint(checkpoint_path, device)
        classes = checkpoint.get("classes", DEFAULT_CLASSES)
        print(f"Loaded checkpoint from {checkpoint_path} (trained on {len(classes)} classes)")
        return cls(model=model, classes=classes, device=device)

    @torch.no_grad()
    def evaluate_loader(self, loader: DataLoader) -> Dict[str, Any]:
        all_targets = []
        all_preds = []
        all_probs = []

        for inputs, targets in loader:
            inputs = inputs.to(self.device, non_blocking=True)
            logits = self.model(inputs)
            probs = torch.softmax(logits, dim=1)
            _, preds = torch.max(logits, 1)

            all_targets.extend(targets.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())
            all_probs.append(probs.cpu().numpy())

        y_prob = np.vstack(all_probs) if all_probs else None
        metrics = compute_classification_metrics(all_targets, all_preds, self.classes, y_prob)
        return metrics

    def run_benchmark(
        self,
        test_loader: DataLoader,
        output_dir: str = "models",
        save_chart: bool = True,
    ) -> Dict[str, Any]:
        """
        Runs complete benchmark on test data, prints detailed reports,
        and saves metric artifacts.
        """
        os.makedirs(output_dir, exist_ok=True)
        print("\n" + "=" * 80)
        print(f"🔬 RUNNING POLYMER CLASSIFICATION BENCHMARK ON HELD-OUT TEST SPLIT")
        print("=" * 80)

        metrics = self.evaluate_loader(test_loader)

        # Print classification table
        table_str = format_classification_table(metrics, self.classes)
        print("\n" + table_str)

        # Print ASCII confusion matrix
        ascii_cm = format_confusion_matrix_ascii(metrics["confusion_matrix"], self.classes)
        print("\nConfusion Matrix (Rows: Ground Truth, Cols: Predicted):")
        print(ascii_cm)

        # Save metrics JSON
        metrics_json_path = os.path.join(output_dir, "test_metrics.json")
        with open(metrics_json_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\n📊 Metrics saved to: {metrics_json_path}")

        # Save Confusion Matrix Heatmap PNG
        if save_chart:
            fig_path = os.path.join(output_dir, "test_confusion_matrix.png")
            cm_np = np.array(metrics["confusion_matrix"])
            save_confusion_matrix_figure(
                cm_np,
                self.classes,
                fig_path,
                title=f"Polymer Classification Confusion Matrix (Test Split: {metrics['total_samples']} samples)",
            )
            print(f"🖼️ Confusion matrix plot saved to: {fig_path}")

        return metrics
