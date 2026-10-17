"""
PolymerClassifier - State-of-the-Art Deep Learning Framework for Plastic Resin Classification.
"""

from .config import ClassifierConfig
from .infer import PolymerPredictor, NumPyPolymerPredictor

try:
    from .models import build_classifier, PolymerClassifier
    from .dataset import create_dataloaders, get_transforms
except ImportError:
    build_classifier = None
    PolymerClassifier = None
    create_dataloaders = None
    get_transforms = None

__version__ = "2.0.0"
__all__ = [
    "ClassifierConfig",
    "PolymerPredictor",
    "NumPyPolymerPredictor",
]
