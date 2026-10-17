#!/usr/bin/env python3
"""
Master Training Script for Best Polymer Classifier.
Trains ConvNeXt-Tiny with Class-Balanced Focal Loss, balanced sampling,
and Cosine Annealing with warmup.
"""

import os
import sys
import argparse

from polymer_classifier.config import ClassifierConfig, get_device, BASE_DIR
from polymer_classifier.dataset import create_dataloaders
from polymer_classifier.models import build_classifier
from polymer_classifier.trainer import PolymerTrainer
from polymer_classifier.evaluator import PolymerEvaluator
from polymer_classifier.export import export_to_onnx, export_to_torchscript
from polymer_classifier.utils import set_seed


def main():
    parser = argparse.ArgumentParser(description="Train State-of-the-Art Polymer Classifier")
    parser.add_argument("--model", type=str, default="convnext_tiny", help="Backbone architecture (convnext_tiny, efficientnet_b2, resnet50d)")
    parser.add_argument("--epochs", type=int, default=12, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--base-lr", type=float, default=1e-4, help="Learning rate for backbone")
    parser.add_argument("--head-lr", type=float, default=8e-4, help="Learning rate for classification head")
    parser.add_argument("--loss", type=str, default="focal", choices=["focal", "class_balanced_ce", "ce"], help="Loss function")
    parser.add_argument("--sampler", type=str, default="sqrt_frequency", choices=["sqrt_frequency", "inverse_frequency", "none"], help="Batch sampling mode")
    parser.add_argument("--samples-per-epoch", type=int, default=15000, help="Number of balanced samples per epoch")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no-export", action="store_true", help="Skip ONNX and TorchScript export")
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()

    config = ClassifierConfig(
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        base_lr=args.base_lr,
        head_lr=args.head_lr,
        loss_type=args.loss,
        sampler=args.sampler,
        samples_per_epoch=args.samples_per_epoch,
        seed=args.seed,
    )

    print(f"Dataset root: {config.data_dir}")
    print(f"Target device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # 1. Create DataLoaders
    print("\n[Step 1/4] Preparing datasets and balanced samplers...")
    train_loader, val_loader, test_loader, class_counts, classes = create_dataloaders(config)
    print(f"Loaded {len(classes)} classes: {classes}")
    print(f"Class distribution in training set:")
    for c_idx, c_name in enumerate(classes):
        print(f"  {c_name:<8}: {class_counts[c_idx]:>5} crops ({class_counts[c_idx]/sum(class_counts)*100:>5.2f}%)")

    # 2. Build Model
    print(f"\n[Step 2/4] Initializing pretrained {config.model_name} with multi-layer classification head...")
    model = build_classifier(config)

    # 3. Train
    print("\n[Step 3/4] Starting training loop...")
    trainer = PolymerTrainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        class_counts=class_counts,
        classes=classes,
        device=device,
    )
    train_results = trainer.train()

    # 4. Final Evaluation on Held-Out Test Set
    if test_loader is not None:
        print("\n[Step 4/4] Evaluating best saved checkpoint on held-out test split...")
        evaluator = PolymerEvaluator.from_checkpoint(config.checkpoint_path, device=device)
        test_metrics = evaluator.run_benchmark(test_loader, output_dir=config.save_dir, save_chart=True)

    # Export
    if not args.no_export:
        print("\nExporting model to ONNX & TorchScript...")
        onnx_path = export_to_onnx(model, config.onnx_path, device=device)
        ts_path = os.path.join(config.save_dir, "polymer_classifier.pt")
        export_to_torchscript(model, ts_path, device=device)
        print(f"Artifacts saved in {config.save_dir}")

    print("\n✅ All training, testing, and export steps completed successfully!")


if __name__ == "__main__":
    import torch
    main()
