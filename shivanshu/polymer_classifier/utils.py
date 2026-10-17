"""
Metrics, logging, and evaluation utilities for Polymer Classification.
Computes Macro F1, Balanced Accuracy, confusion matrices, and generates visual charts.
"""

import os
import random
from typing import Dict, List, Any, Optional
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)


def set_seed(seed: int = 42):
    """Sets random seeds for reproducibility across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_classification_metrics(
    y_true: List[int],
    y_pred: List[int],
    classes: List[str],
    y_prob: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Computes comprehensive evaluation metrics including Balanced Accuracy and Macro F1,
    which are essential for severely imbalanced datasets.
    """
    y_true_np = np.array(y_true)
    y_pred_np = np.array(y_pred)

    acc = float(accuracy_score(y_true_np, y_pred_np))
    bal_acc = float(balanced_accuracy_score(y_true_np, y_pred_np))
    macro_f1 = float(f1_score(y_true_np, y_pred_np, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true_np, y_pred_np, average="weighted", zero_division=0))
    macro_precision = float(precision_score(y_true_np, y_pred_np, average="macro", zero_division=0))
    macro_recall = float(recall_score(y_true_np, y_pred_np, average="macro", zero_division=0))

    cm = confusion_matrix(y_true_np, y_pred_np, labels=list(range(len(classes))))

    # Per-class metrics
    per_class = {}
    report_dict = classification_report(
        y_true_np,
        y_pred_np,
        target_names=classes,
        output_dict=True,
        zero_division=0,
    )
    for c in classes:
        if c in report_dict:
            per_class[c] = {
                "precision": float(report_dict[c]["precision"]),
                "recall": float(report_dict[c]["recall"]),
                "f1": float(report_dict[c]["f1-score"]),
                "support": int(report_dict[c]["support"]),
            }

    # Top-2 Accuracy if probabilities are provided
    top2_acc = None
    if y_prob is not None and y_prob.ndim == 2:
        top2_preds = np.argsort(y_prob, axis=1)[:, -2:]
        top2_correct = sum(
            target in top2 for target, top2 in zip(y_true, top2_preds)
        )
        top2_acc = float(top2_correct / len(y_true))

    return {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "top2_accuracy": top2_acc,
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "total_samples": len(y_true),
    }


def format_classification_table(metrics: Dict[str, Any], classes: List[str]) -> str:
    """Formats metrics as a clean, human-readable terminal table."""
    lines = []
    lines.append("=" * 68)
    lines.append(f"{'Class':<10}{'Precision':<14}{'Recall':<14}{'F1-Score':<14}{'Support':<10}")
    lines.append("-" * 68)
    for c in classes:
        info = metrics["per_class"].get(c, {"precision": 0, "recall": 0, "f1": 0, "support": 0})
        p = f"{info['precision']*100:.2f}%"
        r = f"{info['recall']*100:.2f}%"
        f1 = f"{info['f1']*100:.2f}%"
        sup = str(info["support"])
        lines.append(f"{c:<10}{p:<14}{r:<14}{f1:<14}{sup:<10}")
    lines.append("-" * 68)
    acc_str = f"Acc: {metrics['accuracy']*100:.2f}%"
    bal_str = f"Bal Acc: {metrics['balanced_accuracy']*100:.2f}%"
    f1_str = f"Macro F1: {metrics['macro_f1']*100:.2f}%"
    lines.append(f"{'OVERALL':<10}{acc_str:<18}{bal_str:<20}{f1_str}")
    if metrics.get("top2_accuracy") is not None:
        top2_str = f"Top-2 Acc: {metrics['top2_accuracy']*100:.2f}%"
        lines.append(f"{top2_str}")
    lines.append("=" * 68)
    return "\n".join(lines)


def format_confusion_matrix_ascii(cm: List[List[int]], classes: List[str]) -> str:
    """Formats the confusion matrix as a clear ASCII table."""
    col_width = 8
    header = f"{'True \\ Pred':<12}" + "".join(f"{c:>{col_width}}" for c in classes)
    sep = "-" * len(header)
    rows = [header, sep]
    for idx, row in enumerate(cm):
        row_str = f"{classes[idx]:<12}" + "".join(f"{val:>{col_width}}" for val in row)
        rows.append(row_str)
    return "\n".join(rows)


def save_confusion_matrix_figure(
    cm: np.ndarray,
    classes: List[str],
    output_path: str,
    title: str = "Polymer Classification Confusion Matrix",
):
    """Generates and saves a high-quality annotated confusion matrix heatmap."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        plt.figure(figsize=(8, 6), dpi=300)
        # Normalized matrix by true class (recall)
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_norm = np.divide(cm.astype(float), row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)

        # Annotations showing count and percentage
        annot = np.empty_like(cm, dtype=object)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                annot[i, j] = f"{cm[i, j]}\n({cm_norm[i, j]*100:.1f}%)"

        sns.heatmap(
            cm_norm,
            annot=annot,
            fmt="",
            cmap="Blues",
            xticklabels=classes,
            yticklabels=classes,
            cbar=True,
            vmin=0.0,
            vmax=1.0,
        )
        plt.xlabel("Predicted Class", fontsize=11, fontweight="bold")
        plt.ylabel("True Class", fontsize=11, fontweight="bold")
        plt.title(title, fontsize=12, fontweight="bold", pad=12)
        plt.tight_layout()
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        plt.savefig(output_path)
        plt.close()
    except Exception as e:
        print(f"Warning: Could not save confusion matrix figure: {e}")
