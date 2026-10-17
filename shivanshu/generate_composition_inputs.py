#!/usr/bin/env python3
"""
CLI Tool: Generates inputs for the Composition Estimator.
1. Predictions JSON (image, material, polymer, confidence, probs)
2. Confusion Matrix JSON (data/confusion_matrix.json)

Usage:
    # Generate both on held-out test split
    python generate_composition_inputs.py --all

    # Generate predictions for a specific image folder
    python generate_composition_inputs.py --images dataset_crops/test/PET --output data/predictions.json

    # Generate only the confusion matrix
    python generate_composition_inputs.py --confusion-matrix
"""

import os
import sys
import glob
import json
import argparse
from typing import List

from polymer_classifier.config import BASE_DIR, get_device, ClassifierConfig
from polymer_classifier.infer import PolymerPredictor

try:
    from polymer_classifier.dataset import create_dataloaders
    from polymer_classifier.evaluator import PolymerEvaluator
    HAS_EVALUATOR = True
except ImportError:
    HAS_EVALUATOR = False
    create_dataloaders = None
    PolymerEvaluator = None

from polymer_classifier.estimator_adapter import (
    predict_for_estimator,
    export_estimator_confusion_matrix,
    ESTIMATOR_POLYMERS,
)


def get_active_checkpoint():
    candidates = [
        os.path.join(BASE_DIR, "models", "best_polymer_model.pth"),
        os.path.join(BASE_DIR, "models", "model_weights.npz"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    raise FileNotFoundError("No trained model checkpoint found in models/.")


def main():
    parser = argparse.ArgumentParser(description="Generate Composition Estimator Inputs")
    parser.add_argument("--all", action="store_true", help="Generate both predictions and confusion matrix")
    parser.add_argument("--images", type=str, help="Directory or glob pattern of images to predict on")
    parser.add_argument("--output", type=str, default="data/predictions.json", help="Predictions JSON output path")
    parser.add_argument("--confusion-matrix", action="store_true", help="Generate data/confusion_matrix.json")
    parser.add_argument("--cm-output", type=str, default="data/confusion_matrix.json", help="Confusion matrix output path")
    parser.add_argument("--checkpoint", type=str, default=None, help="Custom checkpoint path")
    args = parser.parse_args()

    checkpoint_path = args.checkpoint or get_active_checkpoint()
    device = get_device()
    print(f"Using checkpoint: {checkpoint_path}")
    print(f"Device: {device}")

    # Generate Confusion Matrix
    if args.all or args.confusion_matrix:
        print("\n[1/2] Computing 7-Class Confusion Matrix for Composition Estimator...")
        config = ClassifierConfig()
        _, _, test_loader, _, _ = create_dataloaders(config)
        if test_loader is not None:
            evaluator = PolymerEvaluator.from_checkpoint(checkpoint_path, device=device)
            cm_payload = export_estimator_confusion_matrix(evaluator, test_loader, output_path=args.cm_output)
            print(f"   Confusion Matrix Source: {cm_payload['source']}")
            print(f"   Labels ({len(cm_payload['labels'])}): {cm_payload['labels']}")
        else:
            print("⚠️ Warning: Test loader unavailable for confusion matrix.")

    # Generate Predictions
    if args.all or args.images or not args.confusion_matrix:
        print("\n[2/2] Generating Predictions for Composition Estimator...")
        predictor = PolymerPredictor.from_checkpoint(checkpoint_path, device=device)

        if args.images:
            if os.path.isdir(args.images):
                img_list = sorted(glob.glob(os.path.join(args.images, "**", "*.jpg"), recursive=True))
                if not img_list:
                    img_list = sorted(glob.glob(os.path.join(args.images, "*.png")))
            else:
                img_list = sorted(glob.glob(args.images))
        else:
            # Default to test set crops
            img_list = sorted(glob.glob(os.path.join(BASE_DIR, "dataset_crops", "test", "*", "*.jpg")))

        if not img_list:
            print("❌ No images found to predict on!")
            return

        print(f"Processing {len(img_list)} images...")
        records = predict_for_estimator(img_list, predictor, output_json_path=args.output)

        # Print first 3 sample records
        print("\nSample Output Records (first 3):")
        print(json.dumps(records[:3], indent=2))

    # Test with composition.py if present
    try:
        from composition import estimate_composition
        print("\n🧪 Running quick check with composition.estimate_composition...")
        pred_data = json.load(open(args.output))
        res = estimate_composition(pred_data, total_kg=50)
        print(f"✅ composition test passed! Result: {res.get('kg', res)}")
    except ImportError:
        pass
    except Exception as e:
        print(f"⚠️ Composition validation note: {e}")

    print("\n✨ Done! Ready for composition estimation.")


if __name__ == "__main__":
    main()
