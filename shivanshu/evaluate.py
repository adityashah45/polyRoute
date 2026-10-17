#!/usr/bin/env python3
"""
Master Evaluation Script.
Evaluates a trained checkpoint on the held-out test split (or custom folder)
and outputs a comprehensive performance breakdown with confusion matrix.
"""

import os
import argparse
import torch

from polymer_classifier.config import ClassifierConfig, get_device, BASE_DIR
from polymer_classifier.dataset import create_dataloaders
from polymer_classifier.evaluator import PolymerEvaluator


def main():
    parser = argparse.ArgumentParser(description="Evaluate Polymer Classifier on Test Set")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=os.path.join(BASE_DIR, "models", "best_polymer_model.pth"),
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(BASE_DIR, "models"),
        help="Directory to save evaluation reports and confusion matrix heatmap",
    )
    args = parser.parse_args()

    # Fallback to root checkpoint if models/ doesn't exist yet
    if not os.path.exists(args.checkpoint):
        alt_path = os.path.join(BASE_DIR, "best_polymer_model.pth")
        if os.path.exists(alt_path):
            args.checkpoint = alt_path

    if not os.path.exists(args.checkpoint):
        print(f"❌ Error: Checkpoint file not found: {args.checkpoint}")
        return

    device = get_device()
    print(f"Device: {device}")
    print(f"Evaluating checkpoint: {args.checkpoint}")

    config = ClassifierConfig()
    _, _, test_loader, _, _ = create_dataloaders(config)

    if test_loader is None:
        print("❌ Error: Test split not found in dataset_crops/test")
        return

    evaluator = PolymerEvaluator.from_checkpoint(args.checkpoint, device=device)
    metrics = evaluator.run_benchmark(test_loader, output_dir=args.output_dir, save_chart=True)


if __name__ == "__main__":
    main()
