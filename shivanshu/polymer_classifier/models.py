"""
Model architecture module for Polymer Classification.
Provides transfer learning backbones with customizable non-linear classifier heads.
"""

from typing import Dict, Any, List, Tuple
import torch
import torch.nn as nn
import timm

from .config import ClassifierConfig, DEFAULT_CLASSES


class PolymerClassificationHead(nn.Module):
    """
    Multi-layer non-linear classification head with LayerNorm, GELU, and Dropout.
    Provides superior feature separation for visually subtle polymer classes.
    """

    def __init__(self, in_features: int, num_classes: int, dropout_rate: float = 0.3):
        super().__init__()
        hidden_dim = min(512, max(128, in_features // 2))
        self.head = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout_rate * 0.67),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class PolymerClassifier(nn.Module):
    """
    Full polymer classifier module wrapping a pretrained backbone and custom head.
    """

    def __init__(
        self,
        model_name: str = "convnext_tiny",
        pretrained: bool = True,
        num_classes: int = len(DEFAULT_CLASSES),
        dropout_rate: float = 0.3,
    ):
        super().__init__()
        self.model_name = model_name
        self.num_classes = num_classes

        # Build backbone with timm (pooled features output)
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # Remove original head to get pooled representation
        )
        in_features = self.backbone.num_features

        # Multi-layer classification head
        self.classifier = PolymerClassificationHead(
            in_features=in_features,
            num_classes=num_classes,
            dropout_rate=dropout_rate,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            logits = self.forward(x)
            return torch.softmax(logits, dim=1)

    def get_parameter_groups(
        self, base_lr: float, head_lr: float, weight_decay: float = 1e-2
    ) -> List[Dict[str, Any]]:
        """
        Creates differential learning rate parameter groups:
        - Lower learning rate for pretrained backbone to prevent feature destruction
        - Higher learning rate for newly initialized classification head
        """
        decay_params = []
        no_decay_params = []

        # Backbone params
        for name, param in self.backbone.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim <= 1 or name.endswith(".bias"):
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        head_params = [p for p in self.classifier.parameters() if p.requires_grad]

        return [
            {"params": decay_params, "lr": base_lr, "weight_decay": weight_decay},
            {"params": no_decay_params, "lr": base_lr, "weight_decay": 0.0},
            {"params": head_params, "lr": head_lr, "weight_decay": weight_decay},
        ]


def build_classifier(config: ClassifierConfig) -> PolymerClassifier:
    """Builds and returns a PolymerClassifier instance from config."""
    model = PolymerClassifier(
        model_name=config.model_name,
        pretrained=config.pretrained,
        num_classes=config.num_classes,
        dropout_rate=config.dropout_rate,
    )
    return model


def load_classifier_checkpoint(
    checkpoint_path: str,
    device: torch.device,
    model_name: str = "convnext_tiny",
) -> Tuple[PolymerClassifier, Dict[str, Any]]:
    """Loads a saved checkpoint into a PolymerClassifier model."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    ckpt_model_name = checkpoint.get("model_name", model_name)
    classes = checkpoint.get("classes", DEFAULT_CLASSES)

    model = PolymerClassifier(
        model_name=ckpt_model_name,
        pretrained=False,
        num_classes=len(classes),
    )
    state_dict = {
        k: v.float() if v.is_floating_point() else v
        for k, v in checkpoint["model_state_dict"].items()
    }
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, checkpoint
