"""
Loss functions engineered for extreme class imbalance in polymer resin classification.
Includes Focal Loss with label smoothing and Cui et al. Class-Balanced Loss.
"""

from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_effective_num_weights(
    class_counts: List[int], beta: float = 0.999
) -> torch.Tensor:
    """
    Computes class weights based on the Effective Number of Samples (Cui et al., CVPR 2019).
    E_n = (1 - beta^n) / (1 - beta)
    weight = (1 - beta) / (1 - beta^n)
    """
    weights = []
    for count in class_counts:
        if count <= 0:
            weights.append(1.0)
        else:
            effective_num = (1.0 - (beta ** count)) / (1.0 - beta)
            weights.append(1.0 / effective_num)

    weights_tensor = torch.tensor(weights, dtype=torch.float32)
    # Normalize weights so their sum equals the number of classes
    weights_tensor = weights_tensor / weights_tensor.sum() * len(class_counts)
    return weights_tensor


class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss with optional class weights and label smoothing.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    Downweights well-classified examples (e.g. abundant PET) to concentrate gradients
    on hard minority classes (e.g. PE-LD and PVC).
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("alpha", alpha if alpha is not None else None)
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = logits.size(1)
        log_probs = F.log_softmax(logits, dim=1)
        probs = torch.exp(log_probs)

        # Apply label smoothing if configured
        if self.label_smoothing > 0.0:
            smoothed_targets = torch.full_like(logits, self.label_smoothing / (num_classes - 1))
            smoothed_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)
        else:
            smoothed_targets = F.one_hot(targets, num_classes=num_classes).float()

        # Gather target probabilities
        p_t = (probs * smoothed_targets).sum(dim=1)
        focal_weight = torch.pow(1.0 - p_t.clamp(min=1e-6, max=1.0 - 1e-6), self.gamma)

        # Cross entropy component with smoothed targets
        ce_loss = -(smoothed_targets * log_probs).sum(dim=1)
        loss = focal_weight * ce_loss

        # Apply per-class alpha weights if provided
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            loss = loss * alpha_t

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


def get_criterion(
    loss_type: str,
    class_counts: List[int],
    gamma: float = 2.0,
    label_smoothing: float = 0.08,
    device: torch.device = torch.device("cpu"),
) -> nn.Module:
    """
    Constructs the appropriate loss criterion with balanced class weights.
    """
    weights = compute_effective_num_weights(class_counts, beta=0.999).to(device)

    if loss_type == "focal":
        return FocalLoss(
            gamma=gamma,
            alpha=weights,
            label_smoothing=label_smoothing,
            reduction="mean",
        )
    elif loss_type == "class_balanced_ce":
        return nn.CrossEntropyLoss(
            weight=weights,
            label_smoothing=label_smoothing,
        )
    else:
        return nn.CrossEntropyLoss(label_smoothing=label_smoothing)
