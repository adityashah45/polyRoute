#!/usr/bin/env python3
"""
CLI Tool for Plastic Polymer Classification.
Analyzes any image crop or photo, predicts polymer resin type,
and displays full probability distribution and recyclability recommendations.

Usage:
    python classify.py <path_to_image.jpg>
    python classify.py <path_to_image.jpg> --top 5
"""

import os
import sys
import argparse

from polymer_classifier.config import BASE_DIR, get_device
from polymer_classifier.infer import PolymerPredictor


def main():
    parser = argparse.ArgumentParser(description="Classify Plastic Polymer Resin from Image")
    parser.add_argument("image_path", type=str, help="Path to input image file (JPG/PNG)")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint (.pth or .npz). Defaults to best trained model.",
    )
    parser.add_argument("--top", type=int, default=3, help="Number of top classes to display")
    args = parser.parse_args()

    if not os.path.exists(args.image_path):
        print(f"❌ Error: Image '{args.image_path}' not found.")
        sys.exit(1)

    # Determine checkpoint
    checkpoint_candidates = [
        args.checkpoint,
        os.path.join(BASE_DIR, "models", "best_polymer_model.pth"),
        os.path.join(BASE_DIR, "models", "model_weights.npz"),
    ]
    checkpoint_path = next((p for p in checkpoint_candidates if p and os.path.exists(p)), None)

    if not checkpoint_path:
        print(f"❌ Error: No trained model checkpoint found in {os.path.join(BASE_DIR, 'models')}.")
        sys.exit(1)

    device = get_device()
    predictor = PolymerPredictor.from_checkpoint(checkpoint_path, device=device)

    result = predictor.predict(args.image_path, top_k=args.top)

    top_class = result["predicted_class"]
    conf = result["confidence"]
    resin_code = result["resin_code"]
    full_name = result["full_name"]

    print("\n" + "=" * 70)
    print("        🔬 POLYMER RESIN CLASSIFICATION REPORT")
    print("=" * 70)
    print(f"📁 Image Path       : {args.image_path}")
    print(f"🏷️ Detected Polymer : {resin_code} {top_class} ({full_name})")
    print(f"🎯 AI Confidence    : {conf * 100:.2f}%")
    print(f"⚙️ Execution Engine : {result.get('engine', 'PyTorch (ConvNeXt-Tiny)')}")
    print(f"📦 Common Uses      : {result['common_uses']}")
    print(f"♻️ Recyclability     : {result['recyclability']}")

    # Probability Distribution Bars
    print("\n📊 Probability Distribution Across Resin Classes:")
    print("-" * 70)
    all_probs = result["all_probabilities"]
    sorted_probs = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)
    for cls_name, prob in sorted_probs:
        bar_len = int(prob * 32)
        bar = "█" * bar_len + "░" * (32 - bar_len)
        marker = " ◄ [TOP]" if cls_name == top_class else ""
        print(f"  {cls_name:<8} : {prob * 100:>6.2f}% | {bar}{marker}")

    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
