"""Checks the classifier output before using it."""

from .config import MATERIALS, POLYMERS


def check_items(items):
    for i, item in enumerate(items):
        material = item.get("material")
        if material not in MATERIALS:
            raise ValueError(f"Item {i}: unknown material {material!r}")
        if material == "plastic" and item.get("polymer") not in POLYMERS:
            raise ValueError(f"Item {i}: plastic item with unknown polymer {item.get('polymer')!r}")


def split_items(items):
    plastic = [it for it in items if it["material"] == "plastic"]
    n_non_plastic = 0
    for it in items:
        if (it["material"] == "non-plastic"):
            n_non_plastic+=1
    n_uncertain=0
    for it in items:
        if (it["material"] == "uncertain"):
            n_uncertain+=1
    return plastic, n_non_plastic, n_uncertain
