#!/usr/bin/env python3
"""
Side-by-Side Benchmark Comparison Script.
Evaluates both the Baseline MobileNetV2 and the New ConvNeXt-Tiny model
on the exact same held-out test split (491 images).
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader

from polymer_classifier.config import BASE_DIR, get_device, ClassifierConfig
from polymer_classifier.evaluator import PolymerEvaluator
from polymer_classifier.utils import compute_classification_metrics


def evaluate_baseline_mobilenet(checkpoint_path: str, test_loader: DataLoader, device: torch.device):
    """Evaluates the original MobileNetV2 checkpoint."""
    with open(os.path.join(BASE_DIR, "class_mapping.json")) as f:
        mapping = json.load(f)["idx_to_class"]
    classes = [mapping[str(i)] for i in range(len(mapping))]

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = models.mobilenet_v2()
    model.classifier[1] = nn.Linear(model.last_channel, len(classes))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    all_targets = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_targets.extend(targets.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())
            all_probs.append(probs.cpu().numpy())

    y_prob = np.vstack(all_probs)
    return compute_classification_metrics(all_targets, all_preds, classes, y_prob), classes


def main():
    device = get_device()
    test_dir = os.path.join(BASE_DIR, "dataset_crops", "test")
    if not os.path.exists(test_dir):
        print(f"Error: {test_dir} not found")
        return

    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    test_dataset = datasets.ImageFolder(test_dir, transform=eval_transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    baseline_path = os.path.join(BASE_DIR, "models", "baseline_mobilenet.pth")
    new_model_path = os.path.join(BASE_DIR, "models", "best_polymer_model.pth")

    print("\n" + "=" * 80)
    print("      POLYMER CLASSIFICATION MODEL BENCHMARK COMPARISON      ")
    print(f"Testing on held-out test split: {len(test_dataset)} images")
    print("=" * 80)

    # 1. Baseline Evaluation
    if os.path.exists(baseline_path):
        print("\n[1/2] Evaluating Baseline MobileNetV2...")
        base_metrics, classes = evaluate_baseline_mobilenet(baseline_path, test_loader, device)
    else:
        print("Baseline checkpoint not found.")
        return

    # 2. New Model Evaluation
    if os.path.exists(new_model_path):
        print("\n[2/2] Evaluating New ConvNeXt-Tiny Model...")
        evaluator = PolymerEvaluator.from_checkpoint(new_model_path, device=device)
        new_metrics = evaluator.evaluate_loader(test_loader)
    else:
        print(f"New model checkpoint not found at {new_model_path}. Train it first using python train_best_model.py!")
        return

    # Side-by-side Table
    print("\n" + "=" * 80)
    print("                       HEAD-TO-HEAD RESULTS")
    print("=" * 80)
    print(f"{'Metric':<25}{'Baseline MobileNetV2':<25}{'New ConvNeXt-Tiny':<25}{'Delta':<10}")
    print("-" * 80)

    metrics_to_compare = [
        ("Overall Accuracy", "accuracy"),
        ("Balanced Accuracy", "balanced_accuracy"),
        ("Macro F1-Score", "macro_f1"),
        ("Weighted F1-Score", "weighted_f1"),
        ("Macro Precision", "macro_precision"),
        ("Macro Recall", "macro_recall"),
    ]

    for label, key in metrics_to_compare:
        v1 = base_metrics[key] * 100
        v2 = new_metrics[key] * 100
        delta = v2 - v1
        sign = "+" if delta >= 0 else ""
        print(f"{label:<25}{v1:>18.2f}%{v2:>23.2f}%{sign + f'{delta:.2f}%':>13}")

    print("\n" + "-" * 80)
    print("PER-CLASS RECALL COMPARISON (True Positive Detection Rate):")
    print(f"{'Class':<12}{'Baseline Recall':<22}{'New Model Recall':<22}{'Delta':<12}")
    print("-" * 80)
    for c in classes:
        r1 = base_metrics["per_class"][c]["recall"] * 100
        r2 = new_metrics["per_class"][c]["recall"] * 100
        delta = r2 - r1
        sign = "+" if delta >= 0 else ""
        print(f"{c:<12}{r1:>18.2f}%{r2:>20.2f}%{sign + f'{delta:.2f}%':>12}")

    print("=" * 80 + "\n")

    # Save comparison to file
    comparison_summary = {
        "baseline": base_metrics,
        "new_model": new_metrics,
    }
    with open(os.path.join(BASE_DIR, "models", "comparison_summary.json"), "w") as f:
        json.dump(comparison_summary, f, indent=2)


if __name__ == "__main__":
    main()
