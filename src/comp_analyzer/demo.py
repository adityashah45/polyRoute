"""Compare the estimation methods on one classifier output file.

Usage (from src/):
    python -m comp_analyzer.demo <items.json> [weight]
    python -m comp_analyzer.demo medium_100.json 50     # a bare name is looked up in comp_analyzer/data/synthetic/

weight can be:
    50            measured batch weight in kg
    volume=1.2    volume of the batch in m3
    count=2000    number of items in the whole batch
    (nothing)     the photos show the whole batch
"""

import json
import sys

from . import (POLYMERS, estimate_composition, load_confusion,
                         load_item_mass, load_weight_params)
from pathlib import Path

from .config import CONFUSION_FILE, ITEM_MASS_FILE, SYNTHETIC_DIR, WEIGHT_PARAMS_FILE


def parse_weight(arg):
    """Turn the command-line weight into (total_kg, batch)."""
    if arg is None:
        return None, None
    if arg.startswith("volume="):
        return None, {"volume_m3": float(arg.split("=")[1])}
    if arg.startswith("count="):
        return None, {"item_count": int(arg.split("=")[1])}
    return float(arg), None


def show(title, result):
    lo, hi = result["total_kg_ci"]
    print(f"\n== {title}  (method: {result['method']}, mass-weighted: {result['mass_weighted']})")
    print(f"Batch weight: {result['total_kg']} kg ({result['total_kg_source']}, 95%: {lo} - {hi})")
    print(f"Plastic share of batch: {result['plastic_fraction']:.1%} -> {result['plastic_kg']} kg")
    print(f"{'Polymer':<8}{'Items':>6}{'kg':>9}   95% interval (kg)")
    for p in POLYMERS:
        if p in result["kg"]:
            plo, phi = result["kg_ci"].get(p, ["-", "-"])
            print(f"{p:<8}{result['item_counts'].get(p, 0):>6}{result['kg'][p]:>9.2f}   {plo} - {phi}")
    for w in result["warnings"]:
        print(f"WARNING: {w}")


def main():
    if len(sys.argv) not in (2, 3):
        print(__doc__)
        sys.exit(1)

    path = Path(sys.argv[1])
    if not path.exists() and (SYNTHETIC_DIR / path).exists():
        path = SYNTHETIC_DIR / path
    total_kg, batch = parse_weight(sys.argv[2] if len(sys.argv) == 3 else None)
    with open(path) as f:
        items = json.load(f)

    masses, masses_placeholder = load_item_mass(ITEM_MASS_FILE)
    confusion, confusion_synthetic = load_confusion(CONFUSION_FILE)
    params, params_placeholder = load_weight_params(WEIGHT_PARAMS_FILE)

    print(f"File: {path}  ({len(items)} items)")
    for flag, name in [(masses_placeholder, "item_mass.csv"), (confusion_synthetic, "confusion_matrix.json"),
                       (params_placeholder, "weight_params.json")]:
        if flag:
            print(f"NOTE: {name} holds placeholder/synthetic values")

    common = dict(total_kg=total_kg, batch=batch, weight_params=params)
    show("1. Counts only", estimate_composition(items, method="count", **common))
    show("2. + mass weighting", estimate_composition(items, method="count", masses=masses, **common))
    show("3. + error correction", estimate_composition(items, method="adjusted", masses=masses,
                                                      confusion=confusion, **common))
    show("4. Auto (what the app uses)", estimate_composition(items, masses=masses,
                                                            confusion=confusion, **common))


if __name__ == "__main__":
    main()
