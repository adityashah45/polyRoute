#!/usr/bin/env python3
"""
Bundle packager for Polymer Classification project.
Creates a self-contained, clean tar.gz distribution archive with:
- Models (PyTorch checkpoint, ONNX, TorchScript)
- Core framework (polymer_classifier/)
- CLI scripts (classify, evaluate, train, generate_composition_inputs, benchmark)
- Composition estimator inputs (data/confusion_matrix.json, data/predictions.json)
- Test dataset split (for immediate out-of-the-box verification)
- Documentation and configs
"""

import os
import tarfile

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUNDLE_NAME = "polymer_classification_bundle.tar.gz"
BUNDLE_PATH = os.path.join(BASE_DIR, BUNDLE_NAME)

ITEMS_TO_INCLUDE = [
    "polymer_classifier",
    "models",
    "data",
    "assets",
    "scripts",
    "tests",
    "classify.py",
    "train.py",
    "evaluate.py",
    "generate_composition_inputs.py",
    "benchmark_comparison.py",
    "requirements.txt",
    "requirements-torch.txt",
    "class_mapping.json",
    "README.md",
    ".gitignore",
]


def create_bundle():
    print(f"📦 Packaging Polymer Classification project into {BUNDLE_NAME}...")

    with tarfile.open(BUNDLE_PATH, "w:gz") as tar:
        for item in ITEMS_TO_INCLUDE:
            full_path = os.path.join(BASE_DIR, item)
            if os.path.exists(full_path):
                print(f"  + Adding {item}...")
                tar.add(full_path, arcname=item)
            else:
                print(f"  ! Warning: {item} not found, skipping.")

    size_mb = os.path.getsize(BUNDLE_PATH) / (1024 * 1024)
    print(f"\n✅ Final bundle created: {BUNDLE_PATH} ({size_mb:.2f} MB)")
    return BUNDLE_PATH


if __name__ == "__main__":
    create_bundle()
