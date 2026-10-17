"""
Configuration module for Polymer Classification.
Defines hyperparameters, dataset paths, resin class mappings, and runtime defaults.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "dataset_crops")
MODELS_DIR = os.path.join(BASE_DIR, "models")
MAPPING_PATH = os.path.join(BASE_DIR, "class_mapping.json")

# Standard 6 polymer classes in fixed canonical order matching original index mapping
DEFAULT_CLASSES: List[str] = ["PE-HD", "PE-LD", "PET", "PP", "PS", "PVC"]
IDX_TO_CLASS: Dict[int, str] = {i: c for i, c in enumerate(DEFAULT_CLASSES)}
CLASS_TO_IDX: Dict[str, int] = {c: i for i, c in enumerate(DEFAULT_CLASSES)}

# Polymer metadata for display and reporting
POLYMER_INFO: Dict[str, Dict[str, str]] = {
    "PE-HD": {
        "full_name": "High-Density Polyethylene (HDPE)",
        "code": "♴",
        "common_uses": "Milk jugs, detergent bottles, shampoo containers, pipes",
        "recyclability": "High (Widely recyclable mechanically)",
    },
    "PE-LD": {
        "full_name": "Low-Density Polyethylene (LDPE)",
        "code": "♶",
        "common_uses": "Plastic wrap, grocery bags, squeeze bottles, flexible lids",
        "recyclability": "Moderate (Specialized soft plastic film stream)",
    },
    "PET": {
        "full_name": "Polyethylene Terephthalate (PETE)",
        "code": "♳",
        "common_uses": "Beverage bottles, food trays, polyester fibers",
        "recyclability": "Very High (Bottle-to-bottle closed loop)",
    },
    "PP": {
        "full_name": "Polypropylene (PP)",
        "code": "♷",
        "common_uses": "Yogurt containers, bottle caps, straws, microwaveware",
        "recyclability": "Moderate-High (Thermal resistance, mechanical recycling)",
    },
    "PS": {
        "full_name": "Polystyrene (PS)",
        "code": "♸",
        "common_uses": "Styrofoam cups, disposable cutlery, CD cases, meat trays",
        "recyclability": "Low (Economically challenging, chemical recycling preferred)",
    },
    "PVC": {
        "full_name": "Polyvinyl Chloride (PVC)",
        "code": "♵",
        "common_uses": "Pipes, medical tubing, credit cards, blister packaging",
        "recyclability": "Very Low (Chlorine content requires specialized non-thermal routing)",
    },
}


@dataclass
class ClassifierConfig:
    # Model architecture
    model_name: str = "convnext_tiny"  # Options: convnext_tiny, efficientnet_b2, resnet50d, mobilenet_v2
    pretrained: bool = True
    num_classes: int = len(DEFAULT_CLASSES)
    dropout_rate: float = 0.3
    image_size: int = 224

    # Optimization & Scheduler
    epochs: int = 12
    batch_size: int = 32
    base_lr: float = 1e-4             # LR for backbone
    head_lr: float = 8e-4             # LR for custom classification head
    weight_decay: float = 1e-2
    warmup_epochs: int = 1
    min_lr: float = 1e-6

    # Loss & Class Imbalance Handling
    loss_type: str = "focal"          # Options: "focal", "class_balanced_ce", "ce"
    focal_gamma: float = 2.0
    label_smoothing: float = 0.08
    sampler: str = "sqrt_frequency"   # Options: "sqrt_frequency", "inverse_frequency", "none"
    samples_per_epoch: int = 15000    # If using balanced sampler, target steps per epoch

    # Hardware & Performance
    num_workers: int = 4
    pin_memory: bool = True
    use_amp: bool = True              # Mixed precision FP16
    seed: int = 42

    # Paths
    data_dir: str = DATA_DIR
    save_dir: str = MODELS_DIR
    checkpoint_name: str = "best_polymer_model.pth"
    onnx_name: str = "polymer_classifier.onnx"
    classes: List[str] = field(default_factory=lambda: list(DEFAULT_CLASSES))

    @property
    def checkpoint_path(self) -> str:
        return os.path.join(self.save_dir, self.checkpoint_name)

    @property
    def onnx_path(self) -> str:
        return os.path.join(self.save_dir, self.onnx_name)


def get_device():
    """Returns the best available torch device (CUDA GPU or CPU), or 'cpu' if torch is unavailable."""
    if HAS_TORCH and torch is not None:
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        return torch.device("cpu")
    return "cpu"
