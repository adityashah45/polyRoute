"""Generate synthetic classifier output for testing the composition analysing

Every item mimics one classifier prediction:

{"image", "material", "polymer", "confidence", "probs"}

"probs" gives a probability for every polymer
"""

import json
import os
import random

from .config import SYNTHETIC_DIR

POLYMERS = ["PET", "HDPE", "PVC", "LDPE", "PP", "PS", "Other"]

# Default distributions
DEFAULT_MAT_WEIGHTS = {"plastic": 85, "non_plastic": 10, "uncertain": 5}
DEFAULT_POLY_WEIGHTS = {
    "PET": 35, "HDPE": 15, "LDPE": 20, "PP": 15, "PS": 7, "PVC": 3, "Other": 5
}
PET_HEAVY_WEIGHTS = {"PET": 70, "HDPE": 10, "PP": 10, "LDPE": 10}


def make_probs(predicted, confidence, rng):
    others = [p for p in POLYMERS if p != predicted]
    raw = [rng.random() for _ in others]
    scale = (1 - confidence) / sum(raw)
    probs = {p: round(r * scale, 4) for p, r in zip(others, raw)}
    probs[predicted] = round(1 - sum(probs.values()), 4)
    return {p: probs[p] for p in POLYMERS}


def generate_items(count, mat_weights, poly_weights, rng):
    materials, m_weights = list(mat_weights.keys()), list(mat_weights.values())
    polymers, p_weights = list(poly_weights.keys()), list(poly_weights.values())

    items = []
    for i in range(count):
        material = rng.choices(materials, weights=m_weights, k=1)[0]
        item = {"image": f"img_{i:04d}.jpg", "material": material, "polymer": None}

        if material == "plastic":
            item["polymer"] = rng.choices(polymers, weights=p_weights, k=1)[0]
            item["confidence"] = round(rng.uniform(0.60, 0.99), 2)
            item["probs"] = make_probs(item["polymer"], item["confidence"], rng)
        elif material == "non_plastic":
            item["confidence"] = round(rng.uniform(0.60, 0.99), 2)
        else:  # uncertain
            item["confidence"] = round(rng.uniform(0.30, 0.59), 2)

        items.append(item)
    return items


def main():
    SYNTHETIC_DIR.mkdir(parents=True, exist_ok=True)

    # filename: (item_count, material_weights, polymer_weights)
    datasets = {
        "small_20.json": (20, DEFAULT_MAT_WEIGHTS, DEFAULT_POLY_WEIGHTS),
        "medium_100.json": (100, DEFAULT_MAT_WEIGHTS, DEFAULT_POLY_WEIGHTS),
        "large_500.json": (500, DEFAULT_MAT_WEIGHTS, DEFAULT_POLY_WEIGHTS),
        "xlarge_1000.json": (1000, DEFAULT_MAT_WEIGHTS, DEFAULT_POLY_WEIGHTS),
        "pet_heavy_100.json": (100, DEFAULT_MAT_WEIGHTS, PET_HEAVY_WEIGHTS),
        "all_non_plastic.json": (10, {"non_plastic": 1}, DEFAULT_POLY_WEIGHTS),
        "only_uncertain.json": (10, {"uncertain": 1}, DEFAULT_POLY_WEIGHTS),
        "empty.json": (0, DEFAULT_MAT_WEIGHTS, DEFAULT_POLY_WEIGHTS),
    }

    for filename, (count, mat_w, poly_w) in datasets.items():
        rng = random.Random(filename)
        data = generate_items(count, mat_w, poly_w, rng)
        with open(SYNTHETIC_DIR / filename, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"{SYNTHETIC_DIR / filename}: {len(data)} items")


if __name__ == "__main__":
    main()
