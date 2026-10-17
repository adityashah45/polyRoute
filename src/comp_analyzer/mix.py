"""classes mix: the share of each polymer among the plastic items (by item count).

Three methods:
  count    : plain counts of the predicted labels
  soft     : sum of each item's class probabilities ("probs")
  adjusted : corrects classifier mistakes using its confusion matrix
             (adjusted classify-and-count, solved with non-negative least squares)
""" 

import numpy as np
from scipy.optimize import nnls

from .config import ADJUST_MIN_ITEMS, POLYMERS


def hard_counts(plastic_items):
    """Number of items predicted as each polymer."""
    counts = np.zeros(len(POLYMERS))
    for item in plastic_items:
        counts[POLYMERS.index(item["polymer"])] += 1
    return counts


def soft_counts(plastic_items):
    """Sum of class probabilities; items without 'probs' count as one full item."""
    counts = np.zeros(len(POLYMERS))
    for item in plastic_items:
        probs = item.get("probs")
        if probs:
            vec = np.array([float(probs.get(p, 0.0)) for p in POLYMERS])
            if (vec.sum() > 0):
                counts += vec/vec.sum()
        else:
            counts[POLYMERS.index(item["polymer"])] += 1
    return counts


def adjusted_mix(plastic_items, confusion):
    """Recover the true mix from biased predictions.

    observed predicted shares  q = confusion^T @ true_shares-> solve for true_shares >= 0
    then normalise to sum to 1
    """
    q = hard_counts(plastic_items)
    q = q / q.sum()
    true_shares, _ = nnls(confusion.T, q)
    if true_shares.sum() == 0:
        return q
    return true_shares / true_shares.sum()


def class_mix(plastic_items, method, confusion=None):
    """Share of each polymer among plastic items """
    if not plastic_items:
        return np.zeros(len(POLYMERS))
    if method == "adjusted":
        return adjusted_mix(plastic_items, confusion)
    counts =0
    if method == "soft":
        counts = soft_counts(plastic_items)
    else:
        counts = hard_counts(plastic_items)
    return counts/counts.sum()


def choose_method(plastic_items, method="auto", confusion=None):
    """Pick the best method for this sample."""
    if method != "auto":
        if method == "adjusted" and confusion is None:
            raise ValueError("method='adjusted' needs a confusion matrix")
        return method
    if confusion is not None and len(plastic_items) >= ADJUST_MIN_ITEMS:
        return "adjusted"
    if any(item.get("probs") for item in plastic_items):
        return "soft"
    return "count"
