"""Validates the class-mix methods on simulated batches with a KNOWN true mix.

For each sample size:
  1. draw items from a known true polymer mix,
  2. pass each item through the confusion matrix (simulated classifier mistakes),
  3. estimate the mix with plain counting and with error correction,
  4. measure bias (systematic error) and average error against the truth.

Usage (from src/):
    python -m comp_analyzer.validate
"""

import numpy as np

from . import POLYMERS, load_confusion
from .config import CONFUSION_FILE
from .mix import class_mix

TRUE_MIX = np.array([0.35, 0.15, 0.03, 0.20, 0.15, 0.07, 0.05])  # PET, HDPE, PVC, LDPE, PP, PS, Other
SAMPLE_SIZES = [50, 100, 400, 1000, 2000]
TRIALS = 200


def total_variation(a, b):
    """Share of the mix that is wrong 
    0 = perfect, 1 = completely wrong"""
    return np.abs(a - b).sum() / 2


def main():
    confusion, synthetic = load_confusion(CONFUSION_FILE)
    if synthetic:
        print("NOTE: using the synthetic confusion matrix")
    rng = np.random.default_rng(1)

    print(f"{'items':>6} | {'count: bias':>12} {'error':>7} | {'adjusted: bias':>15} {'error':>7}")
    for n in SAMPLE_SIZES:
        estimates = {"count": [], "adjusted": []}
        for _ in range(TRIALS):
            true_labels = rng.choice(len(POLYMERS), size=n, p=TRUE_MIX)
            predicted = [rng.choice(len(POLYMERS), p=confusion[t]) for t in true_labels]
            items = [{"material": "plastic", "polymer": POLYMERS[p]} for p in predicted]
            for method in estimates:
                estimates[method].append(class_mix(items, method, confusion))

        cells = []
        for method, values in estimates.items():
            values = np.array(values)
            bias = total_variation(values.mean(axis=0), TRUE_MIX)
            error = np.mean([total_variation(v, TRUE_MIX) for v in values])
            width = 12 if method == "count" else 15
            cells.append(f"{bias:>{width}.3f} {error:>7.3f}")
        print(f"{n:>6} | " + " | ".join(cells))


if __name__ == "__main__":
    main()
