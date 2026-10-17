"""
High-performance inference engine for Polymer Resin Classification.
Supports:
1. PyTorch Deep Learning Engine (ConvNeXt / MobileNet)
2. Native NumPy Engine (Zero PyTorch, Zero ONNX required - 100% IBM LinuxONE s390x compatible)
"""

import os
from typing import Union, List, Dict, Any, Optional
from PIL import Image
import numpy as np

from .config import (
    DEFAULT_CLASSES,
    POLYMER_INFO,
    BASE_DIR,
)

# Optional PyTorch import
try:
    import torch
    from torchvision import transforms
    from .models import PolymerClassifier, load_classifier_checkpoint
    from .config import get_device
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None


class NumPyPolymerPredictor:
    """
    Pure NumPy inference engine. Zero dependencies on PyTorch, ONNX, or C extensions.
    Guaranteed 100% compatible with IBM LinuxONE (s390x) and any minimal Python environment.
    """

    def __init__(self, npz_path: str, classes: Optional[List[str]] = None):
        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"NumPy weights not found at: {npz_path}")
        npz = np.load(npz_path)
        self.W = npz["W"]
        self.mean = npz["mean"]
        self.std = npz["std"]
        self.classes = list(npz["classes"]) if "classes" in npz else (classes or list(DEFAULT_CLASSES))

    def _extract_features(self, img: Image.Image) -> np.ndarray:
        img_rgb = img.convert("RGB").resize((96, 96), Image.Resampling.BILINEAR)
        arr = np.array(img_rgb, dtype=np.float32) / 255.0

        hist_r, _ = np.histogram(arr[:, :, 0], bins=16, range=(0, 1), density=True)
        hist_g, _ = np.histogram(arr[:, :, 1], bins=16, range=(0, 1), density=True)
        hist_b, _ = np.histogram(arr[:, :, 2], bins=16, range=(0, 1), density=True)

        gray = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
        hist_gray, _ = np.histogram(gray, bins=16, range=(0, 1), density=True)

        means = np.mean(arr, axis=(0, 1))
        stds = np.std(arr, axis=(0, 1))
        skews = np.mean(((arr - means) / (stds + 1e-5)) ** 3, axis=(0, 1))

        grid = arr.reshape(4, 24, 4, 24, 3).mean(axis=(1, 3)).flatten()

        diff_x = np.abs(arr[:, 1:, :] - arr[:, :-1, :]).mean(axis=(0, 1))
        diff_y = np.abs(arr[1:, :, :] - arr[:-1, :, :]).mean(axis=(0, 1))
        diff_std = np.std(np.abs(arr[:, 1:, :] - arr[:, :-1, :]), axis=(0, 1))

        w, h = img.size
        aspect_ratio = float(w) / max(1.0, float(h))
        log_aspect = np.log(max(1e-2, aspect_ratio))
        log_area = np.log(max(1.0, float(w * h)))
        specular_hi = np.mean(gray > 0.92)
        specular_lo = np.mean(gray < 0.08)

        feat = np.concatenate([
            hist_r, hist_g, hist_b, hist_gray,
            means, stds, skews,
            grid,
            diff_x, diff_y, diff_std,
            np.array([log_aspect, log_area, specular_hi, specular_lo], dtype=np.float32)
        ])
        return feat.astype(np.float32)

    def predict(self, image_input: Union[str, Image.Image, np.ndarray], top_k: int = 3) -> Dict[str, Any]:
        if isinstance(image_input, str):
            img = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, np.ndarray):
            img = Image.fromarray(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            img = image_input.convert("RGB")
        else:
            raise ValueError(f"Unsupported image type: {type(image_input)}")

        feat = self._extract_features(img)
        feat_norm = (feat - self.mean) / self.std

        # Add bias term if W includes it
        if self.W.shape[0] == feat_norm.shape[0] + 1:
            feat_norm = np.append(feat_norm, 1.0)

        logits = feat_norm @ self.W
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)

        sorted_indices = probs.argsort()[::-1]
        top_idx = sorted_indices[0]
        top_class = self.classes[top_idx]
        top_conf = float(probs[top_idx])

        rankings = []
        for rank_idx in sorted_indices[:top_k]:
            c_name = self.classes[rank_idx]
            info = POLYMER_INFO.get(c_name, {})
            rankings.append({
                "class": c_name,
                "confidence": float(probs[rank_idx]),
                "full_name": info.get("full_name", c_name),
                "resin_code": info.get("code", ""),
            })

        all_probs = {self.classes[i]: float(probs[i]) for i in range(len(self.classes))}
        info_top = POLYMER_INFO.get(top_class, {})

        return {
            "predicted_class": top_class,
            "confidence": top_conf,
            "full_name": info_top.get("full_name", top_class),
            "resin_code": info_top.get("code", ""),
            "common_uses": info_top.get("common_uses", ""),
            "recyclability": info_top.get("recyclability", ""),
            "top_k": rankings,
            "all_probabilities": all_probs,
            "engine": "Native NumPy (s390x LinuxONE)",
        }


class PolymerPredictor:
    """
    Inference wrapper for classifying plastic polymer resins.
    Automatically uses PyTorch on GPU/CPU if available, or falls back to Native NumPy
    on IBM LinuxONE / s390x systems where PyTorch is not installed.
    """

    def __init__(
        self,
        model: Any,
        classes: Optional[List[str]] = None,
        device: Optional[Any] = None,
        image_size: int = 224,
    ):
        self.device = device
        self.model = model
        self.classes = classes or list(DEFAULT_CLASSES)
        self.image_size = image_size
        self.is_torch = HAS_TORCH and hasattr(model, "forward")

        if self.is_torch:
            self.model.to(self.device)
            self.model.eval()
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Optional[str] = None,
        device: Optional[Any] = None,
        image_size: int = 224,
    ) -> Union["PolymerPredictor", NumPyPolymerPredictor]:
        # If PyTorch is unavailable or npz file is requested, use NumPyPredictor
        npz_candidates = [
            checkpoint_path if checkpoint_path and checkpoint_path.endswith(".npz") else None,
            os.path.join(BASE_DIR, "models", "model_weights.npz"),
        ]
        valid_npz = next((p for p in npz_candidates if p and os.path.exists(p)), None)

        if not HAS_TORCH or (checkpoint_path and checkpoint_path.endswith(".npz")):
            if valid_npz:
                return NumPyPolymerPredictor(valid_npz)
            raise RuntimeError("PyTorch is not installed and model_weights.npz was not found in models/.")

        # PyTorch mode
        device = device or get_device()
        pth_candidates = [
            checkpoint_path,
            os.path.join(BASE_DIR, "models", "best_polymer_model.pth"),
        ]
        valid_pth = next((p for p in pth_candidates if p and os.path.exists(p)), None)

        if valid_pth:
            model, checkpoint = load_classifier_checkpoint(valid_pth, device)
            classes = checkpoint.get("classes", DEFAULT_CLASSES)
            return cls(model=model, classes=classes, device=device, image_size=image_size)

        if valid_npz:
            print("PyTorch model not found, falling back to Native NumPy Predictor.")
            return NumPyPolymerPredictor(valid_npz)

        raise FileNotFoundError(f"No model checkpoint (.pth or .npz) found in {os.path.join(BASE_DIR, 'models')}")

    def _prepare_image(self, image_input: Union[str, Image.Image, np.ndarray]):
        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                raise FileNotFoundError(f"Image not found at: {image_input}")
            img = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, np.ndarray):
            img = Image.fromarray(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            img = image_input.convert("RGB")
        else:
            raise ValueError(f"Unsupported image type: {type(image_input)}")

        tensor = self.transform(img).unsqueeze(0).to(self.device)
        return tensor

    def predict(
        self,
        image_input: Union[str, Image.Image, np.ndarray],
        top_k: int = 3,
    ) -> Dict[str, Any]:
        if not self.is_torch:
            return self.model.predict(image_input, top_k=top_k)

        with torch.no_grad():
            tensor = self._prepare_image(image_input)
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

        sorted_indices = probs.argsort()[::-1]
        top_idx = sorted_indices[0]
        top_class = self.classes[top_idx]
        top_conf = float(probs[top_idx])

        rankings = []
        for rank_idx in sorted_indices[:top_k]:
            c_name = self.classes[rank_idx]
            info = POLYMER_INFO.get(c_name, {})
            rankings.append({
                "class": c_name,
                "confidence": float(probs[rank_idx]),
                "full_name": info.get("full_name", c_name),
                "resin_code": info.get("code", ""),
            })

        all_probs = {self.classes[i]: float(probs[i]) for i in range(len(self.classes))}
        info_top = POLYMER_INFO.get(top_class, {})

        return {
            "predicted_class": top_class,
            "confidence": top_conf,
            "full_name": info_top.get("full_name", top_class),
            "resin_code": info_top.get("code", ""),
            "common_uses": info_top.get("common_uses", ""),
            "recyclability": info_top.get("recyclability", ""),
            "top_k": rankings,
            "all_probabilities": all_probs,
            "engine": "PyTorch (ConvNeXt-Tiny)",
        }
