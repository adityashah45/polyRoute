#!/usr/bin/env python3
"""
Builds pure NumPy model weights for native inference on IBM LinuxONE (s390x).
Extracts multi-scale color moments, spatial grids, texture gradients, and aspect ratios,
trains a class-balanced multinomial classifier, and saves weights into model_weights.npz.
Zero PyTorch required at inference time!
"""

import os
import sys
import glob
from typing import Tuple, List
import numpy as np
from PIL import Image

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "dataset_crops")
MODELS_DIR = os.path.join(BASE_DIR, "models")
CLASSES = ["PE-HD", "PE-LD", "PET", "PP", "PS", "PVC"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}


def extract_features_from_image(img: Image.Image) -> np.ndarray:
    """Extracts a 132-dim robust visual descriptor from a PIL image."""
    img_rgb = img.convert("RGB").resize((96, 96), Image.Resampling.BILINEAR)
    arr = np.array(img_rgb, dtype=np.float32) / 255.0

    # 1. Color channel histograms (16 bins x 3 = 48)
    hist_r, _ = np.histogram(arr[:, :, 0], bins=16, range=(0, 1), density=True)
    hist_g, _ = np.histogram(arr[:, :, 1], bins=16, range=(0, 1), density=True)
    hist_b, _ = np.histogram(arr[:, :, 2], bins=16, range=(0, 1), density=True)

    # 2. Grayscale luminance histogram (16 bins)
    gray = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    hist_gray, _ = np.histogram(gray, bins=16, range=(0, 1), density=True)

    # 3. Channel statistics: mean, std, skewness (9 features)
    means = np.mean(arr, axis=(0, 1))
    stds = np.std(arr, axis=(0, 1))
    skews = np.mean(((arr - means) / (stds + 1e-5)) ** 3, axis=(0, 1))

    # 4. Spatial 4x4 grid averages (4x4x3 = 48 features)
    grid = arr.reshape(4, 24, 4, 24, 3).mean(axis=(1, 3)).flatten()

    # 5. Texture & edge gradients (6 features)
    diff_x = np.abs(arr[:, 1:, :] - arr[:, :-1, :]).mean(axis=(0, 1))
    diff_y = np.abs(arr[1:, :, :] - arr[:-1, :, :]).mean(axis=(0, 1))
    diff_std = np.std(np.abs(arr[:, 1:, :] - arr[:, :-1, :]), axis=(0, 1))

    # 6. Aspect ratio & relative dimensions (5 features)
    w, h = img.size
    aspect_ratio = float(w) / max(1.0, float(h))
    log_aspect = np.log(max(1e-2, aspect_ratio))
    log_area = np.log(max(1.0, float(w * h)))
    specular_hi = np.mean(gray > 0.92)
    specular_lo = np.mean(gray < 0.08)

    feat = np.concatenate([
        hist_r, hist_g, hist_b, hist_gray,  # 64
        means, stds, skews,                  # 9
        grid,                                # 48
        diff_x, diff_y, diff_std,            # 9
        np.array([log_aspect, log_area, specular_hi, specular_lo], dtype=np.float32)  # 4
    ])
    return feat.astype(np.float32)


def extract_split(split_name: str, max_per_class: int = 1500) -> Tuple[np.ndarray, np.ndarray]:
    features = []
    labels = []
    split_dir = os.path.join(DATA_DIR, split_name)

    for c_idx, c_name in enumerate(CLASSES):
        cls_dir = os.path.join(split_dir, c_name)
        if not os.path.exists(cls_dir):
            continue
        imgs = glob.glob(os.path.join(cls_dir, "*.jpg"))
        if max_per_class and len(imgs) > max_per_class:
            # Sample evenly to balance training representations
            np.random.seed(42)
            imgs = list(np.random.choice(imgs, max_per_class, replace=False))

        print(f"  [{split_name.upper()}] Extracting {len(imgs)} features for {c_name}...")
        for p in imgs:
            try:
                with Image.open(p) as img:
                    feat = extract_features_from_image(img)
                    features.append(feat)
                    labels.append(c_idx)
            except Exception:
                continue

    X = np.array(features, dtype=np.float32)
    y = np.array(labels, dtype=np.int64)
    return X, y


def train_numpy_model():
    print("==================================================")
    print("Building Native s390x Pure NumPy Classifier")
    print("==================================================")

    print("\n1. Extracting training features...")
    X_train, y_train = extract_split("train", max_per_class=1200)
    print(f"Total training samples: {X_train.shape[0]}, Features dim: {X_train.shape[1]}")

    print("\n2. Computing feature normalization (mean, std)...")
    feat_mean = np.mean(X_train, axis=0)
    feat_std = np.std(X_train, axis=0) + 1e-6
    X_norm = (X_train - feat_mean) / feat_std

    # Add bias column
    N, D = X_norm.shape
    num_classes = len(CLASSES)
    X_bias = np.hstack([X_norm, np.ones((N, 1), dtype=np.float32)])

    # Compute class weights for loss balancing
    class_counts = [np.sum(y_train == c) for c in range(num_classes)]
    total_samples = len(y_train)
    weights_per_class = np.array([total_samples / (num_classes * max(1, c)) for c in class_counts], dtype=np.float32)
    sample_weights = weights_per_class[y_train]

    print("\n3. Training Softmax Logistic Regression via L-BFGS / Gradient Descent...")
    # Initialize weights
    np.random.seed(42)
    W = np.random.randn(D + 1, num_classes).astype(np.float32) * 0.01

    lr = 0.05
    reg = 1e-3
    epochs = 400

    one_hot = np.zeros((N, num_classes), dtype=np.float32)
    one_hot[np.arange(N), y_train] = 1.0

    for ep in range(epochs):
        logits = X_bias @ W
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        # Weighted gradient
        grad = (X_bias.T @ ((probs - one_hot) * sample_weights[:, None])) / N + reg * W
        W -= lr * grad

        if (ep + 1) % 100 == 0:
            preds = np.argmax(probs, axis=1)
            train_acc = np.mean(preds == y_train) * 100
            print(f"  Epoch {ep + 1}/{epochs} - Loss: {np.mean(-np.log(np.clip(probs[np.arange(N), y_train], 1e-7, 1.0))):.4f} - Acc: {train_acc:.2f}%")

    print("\n4. Evaluating on Test Split...")
    X_test, y_test = extract_split("test", max_per_class=None)
    X_test_norm = (X_test - feat_mean) / feat_std
    X_test_bias = np.hstack([X_test_norm, np.ones((len(X_test), 1), dtype=np.float32)])

    test_logits = X_test_bias @ W
    test_exp = np.exp(test_logits - np.max(test_logits, axis=1, keepdims=True))
    test_probs = test_exp / np.sum(test_exp, axis=1, keepdims=True)
    test_preds = np.argmax(test_probs, axis=1)

    overall_acc = np.mean(test_preds == y_test) * 100
    print(f"\n✅ Pure NumPy Test Accuracy: {overall_acc:.2f}% ({np.sum(test_preds == y_test)}/{len(y_test)})")
    for c_idx, c_name in enumerate(CLASSES):
        mask = (y_test == c_idx)
        if np.sum(mask) > 0:
            c_acc = np.mean(test_preds[mask] == y_test[mask]) * 100
            print(f"  {c_name:<8}: {c_acc:.1f}% ({np.sum(test_preds[mask] == y_test[mask])}/{np.sum(mask)})")

    # Save weights
    os.makedirs(MODELS_DIR, exist_ok=True)
    out_models_path = os.path.join(MODELS_DIR, "model_weights.npz")

    save_dict = {
        "W": W,
        "mean": feat_mean,
        "std": feat_std,
        "classes": np.array(CLASSES),
    }

    np.savez_compressed(out_models_path, **save_dict)
    print(f"\n✅ Saved s390x-native weights to: {out_models_path}")


if __name__ == "__main__":
    train_numpy_model()
