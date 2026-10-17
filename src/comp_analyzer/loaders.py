"""Loads the reference data: typical item masses and the classifier's confusion matrix."""

import csv
import json

import numpy as np

from .config import CONFUSION_FILE, ITEM_MASS_FILE, POLYMERS, WEIGHT_PARAMS_FILE


def load_item_mass(path=ITEM_MASS_FILE):
    """Typical item mass (grams) per class

    File: CSV with columns class, typical_item_mass_g, source
    Returns: ({class: grams}, is_placeholder)
    """
    with open(path) as f:
        rows = list(csv.DictReader(f))
    masses = {r["class"]: float(r["typical_item_mass_g"]) for r in rows}
    is_placeholder = any(r.get("source", "").upper() == "PLACEHOLDER" for r in rows)
    return masses, is_placeholder


def load_weight_params(path=WEIGHT_PARAMS_FILE):
    """Parameters for estimating the batch weight (item mass spread, bulk density).

    File: JSON with mass_cv and bulk_density_kg_m3 {low, typical, high}
    Returns: (dict, is_placeholder)
    """
    with open(path) as f:
        params = json.load(f)
    is_placeholder = "PLACEHOLDER" in str(params.get("source", "")).upper()
    return params, is_placeholder


def load_confusion(path=CONFUSION_FILE):
    """Classifier confusion matrix as row-normalised P(predicted | true).

    File: JSON with "labels" and "matrix" (rows = true class, columns = predicted, values = counts)
    Returns: (7x7 numpy array in POLYMERS order, is_synthetic)
    """
    with open(path) as f:
        data = json.load(f)

    labels = data["labels"]
    counts = np.array(data["matrix"], dtype=float)

    # Reorder rows and columns to POLYMERS order so indices always line up
    order = [labels.index(p) for p in POLYMERS]
    counts = counts[np.ix_(order, order)]

    # Turn counts into probabilities per row. A class with no test images
    # is assumed to be classified perfectly (identity row)
    row_sums = counts.sum(axis=1, keepdims=True)
    safe_sums = np.where(row_sums == 0, 1, row_sums)
    probs = np.where(row_sums > 0, counts / safe_sums, np.eye(len(POLYMERS)))

    is_synthetic = "SYNTHETIC" in str(data.get("source", "")).upper()
    return probs, is_synthetic
