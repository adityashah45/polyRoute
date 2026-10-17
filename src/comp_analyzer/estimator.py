""" The main file which runs all steps, adds warnings and builds the result dict"""

import numpy as np

from .config import N_BOOTSTRAP, POLYMERS, SMALL_SAMPLE
from .mass import kg_per_polymer
from .mix import choose_method, class_mix, hard_counts
from .uncertainty import bootstrap
from .validation import check_items, split_items
from .weight import estimate_total_kg

MIN_KG = 0.005  # values below this are treated as zero


def point_estimate(items, total_kg, method, masses, confusion):
    """One estimate without intervals
    Returns (kg array, plastic fraction)."""
    plastic, n_non_plastic, _ = split_items(items)
    mix = class_mix(plastic, method, confusion)
    return kg_per_polymer(mix, len(plastic), n_non_plastic, total_kg, masses)


def estimate_composition(items, total_kg=None, batch=None, method="auto", masses=None,
                         confusion=None, weight_params=None, n_boot=N_BOOTSTRAP, seed=0,
                         kg_override=None):
    """Estimate kg of each polymer in a batch.

    items         : classifier results (interface A)
    total_kg      : measured batch weight (kg); if None, it is estimated from `batch`
    batch         : how to estimate the weight if not measured:
                    {"volume_m3": ...} or {"item_count": ...}; None = photos show the whole batch
    method        : "auto", "count", "soft" or "adjusted"
    masses        : {class: typical item mass in g}; None = all items weigh the same
    confusion     : confusion matrix from load_confusion; None = trust the labels
    weight_params : from load_weight_params (item mass spread, bulk density)
    n_boot        : bootstrap resamples for 95% intervals (0 = skip)
    seed          : random seed, so results are reproducible
    kg_override   : optional {polymer: kg} typed by the user; replaces the estimate
    """
    if total_kg is not None and total_kg <= 0:
        raise ValueError("total_kg must be greater than 0")
    check_items(items)

    masses = masses or {}
    plastic, n_non_plastic, n_uncertain = split_items(items)
    method = choose_method(plastic, method, confusion)

    # Batch weight: measured if given, otherwise estimated
    weight_batch = {}
    if total_kg:
        weight_batch = {"total_kg": total_kg}
    else:
        weight_batch= batch or {}
    weight = estimate_total_kg(items, masses, weight_batch, weight_params)
    total = weight["kg"]

    kg_vec, plastic_fraction = point_estimate(items, total, method, masses, confusion)

    # Bootstrap: resample items AND redraw the weight, so both uncertainties count
    kg_ci, total_ci = {}, [round(total, 2), round(total, 2)]
    if n_boot and items:
        def one_draw(sample, rng):
            t = weight["sampler"](sample, rng)
            return np.append(point_estimate(sample, t, method, masses, confusion)[0], t)

        low, high = bootstrap(items, one_draw, n_boot=n_boot, seed=seed)
        kg_ci = {p: [round(float(lo), 2), round(float(hi), 2)]
                 for p, v, lo, hi in zip(POLYMERS, kg_vec, low[:-1], high[:-1]) if v > MIN_KG}
        total_ci = [round(float(low[-1]), 2), round(float(high[-1]), 2)]

    kg = {p: round(float(v), 2) for p, v in zip(POLYMERS, kg_vec) if v > MIN_KG}
    item_counts = {p: int(c) for p, c in zip(POLYMERS, hard_counts(plastic)) if c > 0}

    warnings = []
    if not plastic:
        warnings.append("No plastic items found in the sample.")
    elif len(plastic) < SMALL_SAMPLE:
        warnings.append(f"Only {len(plastic)} plastic items sampled; intervals are wide. Add more photos.")
    if weight["source"] != "measured":
        warnings.append(f"Batch weight was estimated ({weight['source']}); weighing the batch gives better results.")

    if kg_override:
        kg = {p: round(float(v), 2) for p, v in kg_override.items() if float(v) > 0}
        kg_ci = {}
        plastic_fraction = 1.0
        if total:
            plastic_fraction = min(sum(kg.values())/total, 1.0)
        method = "user_override"

    return {
        "total_kg": round(total, 2),
        "total_kg_ci": total_ci,
        "total_kg_source": weight["source"],
        "plastic_kg": round(sum(kg.values()), 2),
        "plastic_fraction": round(float(plastic_fraction), 3),
        "kg": kg,
        "kg_ci": kg_ci,
        "item_counts": item_counts,
        "excluded": {"non_plastic": n_non_plastic, "uncertain": n_uncertain},
        "method": method,
        "mass_weighted": bool(masses),
        "warnings": warnings,
    }
