"""
Estimator Adapter: Bridges the Polymer Classifier output to the Composition Estimator.
Implements the exact JSON prediction and confusion matrix schema required:
- Predictions: JSON list of dicts with keys: image, material, polymer, confidence, probs
- Confusion Matrix: data/confusion_matrix.json with 7 polymer classes: PET, HDPE, PVC, LDPE, PP, PS, Other
"""

import os
import json
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix

from .config import BASE_DIR, get_device
from .infer import PolymerPredictor
from .evaluator import PolymerEvaluator

# The exact 7 polymer class names required by the composition estimator
ESTIMATOR_POLYMERS: List[str] = ["PET", "HDPE", "PVC", "LDPE", "PP", "PS", "Other"]

# Mapping from internal dataset labels to estimator standard labels
INTERNAL_TO_ESTIMATOR_MAP: Dict[str, str] = {
    "PE-HD": "HDPE",
    "PE-LD": "LDPE",
    "PET": "PET",
    "PP": "PP",
    "PS": "PS",
    "PVC": "PVC",
    "Other": "Other",
}


def to_estimator_record(
    image_name: str,
    polymer_probs: Dict[str, float],
    plastic_prob: float = 1.0,
    threshold: float = 0.6,
) -> Dict[str, Any]:
    """
    Constructs a single prediction record according to the exact schema:
    - material: "plastic", "non_plastic", or "uncertain"
    - polymer: top polymer if plastic, else null
    - confidence: float 0-1
    - probs: dict of 7 polymers summing to 1 (for plastic items)
    """
    filter_conf = max(plastic_prob, 1.0 - plastic_prob)

    if filter_conf < threshold:
        return {
            "image": os.path.basename(image_name),
            "material": "uncertain",
            "polymer": None,
            "confidence": round(float(filter_conf), 4),
        }

    if plastic_prob < 0.5:
        return {
            "image": os.path.basename(image_name),
            "material": "non_plastic",
            "polymer": None,
            "confidence": round(float(1.0 - plastic_prob), 4),
        }

    # Map internal probabilities to 7 estimator classes
    estimator_probs = {p: 0.0 for p in ESTIMATOR_POLYMERS}
    for internal_cls, prob in polymer_probs.items():
        est_name = INTERNAL_TO_ESTIMATOR_MAP.get(internal_cls, "Other")
        if est_name in estimator_probs:
            estimator_probs[est_name] += float(prob)

    # Normalize to ensure exact sum to 1.0
    total = sum(estimator_probs.values())
    if total > 0:
        estimator_probs = {k: v / total for k, v in estimator_probs.items()}

    # Round to 4 decimal places
    rounded_probs = {k: round(v, 4) for k, v in estimator_probs.items()}
    # Adjust largest value so sum is exactly 1.0
    diff = round(1.0 - sum(rounded_probs.values()), 4)
    top_key = max(rounded_probs, key=rounded_probs.get)
    rounded_probs[top_key] = round(rounded_probs[top_key] + diff, 4)

    top_polymer = max(rounded_probs, key=rounded_probs.get)
    top_conf = rounded_probs[top_polymer]

    return {
        "image": os.path.basename(image_name),
        "material": "plastic",
        "polymer": top_polymer,
        "confidence": top_conf,
        "probs": rounded_probs,
    }


def predict_for_estimator(
    image_paths: List[str],
    predictor: PolymerPredictor,
    output_json_path: Optional[str] = None,
    threshold: float = 0.6,
) -> List[Dict[str, Any]]:
    """
    Runs predictions across images and formats records for the composition estimator.
    """
    records = []
    for img_path in image_paths:
        res = predictor.predict(img_path)
        record = to_estimator_record(
            image_name=img_path,
            polymer_probs=res["all_probabilities"],
            plastic_prob=1.0,  # Single-object plastic crop
            threshold=threshold,
        )
        records.append(record)

    if output_json_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_json_path)), exist_ok=True)
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2)
        print(f"✅ Estimator predictions saved to: {output_json_path} ({len(records)} records)")

    return records


def export_estimator_confusion_matrix(
    evaluator: PolymerEvaluator,
    test_loader: DataLoader,
    output_path: str = "data/confusion_matrix.json",
    source_tag: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Computes and exports the 7x7 confusion matrix in the exact schema:
    {
      "source": "Polymer classifier test set, 491 images, 17 Oct 2026",
      "labels": ["PET", "HDPE", "PVC", "LDPE", "PP", "PS", "Other"],
      "matrix": [[...], ...]
    }
    """
    all_targets = []
    all_preds = []

    evaluator.model.eval()
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(evaluator.device)
            logits = evaluator.model(inputs)
            preds = torch.argmax(logits, dim=1)

            all_targets.extend(targets.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())

    # Map target indices and predicted indices to estimator labels
    mapped_true = [INTERNAL_TO_ESTIMATOR_MAP[evaluator.classes[t]] for t in all_targets]
    mapped_pred = [INTERNAL_TO_ESTIMATOR_MAP[evaluator.classes[p]] for p in all_preds]

    cm = confusion_matrix(mapped_true, mapped_pred, labels=ESTIMATOR_POLYMERS)

    date_str = datetime.now().strftime("%d %b %Y")
    total_imgs = len(all_targets)
    source_text = source_tag or f"Polymer classifier test set, {total_imgs} images, {date_str}"

    payload = {
        "source": source_text,
        "labels": ESTIMATOR_POLYMERS,
        "matrix": cm.tolist(),
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"✅ Estimator confusion matrix saved to: {output_path}")
    print(f"   Matrix shape: {cm.shape} with labels: {ESTIMATOR_POLYMERS}")
    return payload
