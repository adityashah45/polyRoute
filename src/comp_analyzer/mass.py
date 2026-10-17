"""mass weighting and scaling to the batch weight.

Only the RATIOS between typical masses matter, because total_kg is measured.
"""

import numpy as np

from .config import POLYMERS


def mass_weighted_shares(mix, masses):
    """Convert shares by item count into shares by mass.

    mix    : share of each polymer by item count (sums to 1)
    masses : {class: typical item mass in g}; missing classes count as equal weight
    """
    m = np.array([masses.get(p, 1.0) for p in POLYMERS])
    weighted = mix * m
    return weighted / weighted.sum() if weighted.sum() > 0 else weighted


def plastic_mass_fraction(mix, n_plastic, n_non_plastic, masses):
    """Share of the batch that is plastic, by mass. Uncertain items are left out."""
    if n_plastic == 0:
        return 0.0
    m = np.array([masses.get(p, 1.0) for p in POLYMERS])
    plastic_mass = n_plastic * (mix * m).sum()
    non_plastic_mass = n_non_plastic * masses.get("non_plastic", 1.0)
    return plastic_mass / (plastic_mass + non_plastic_mass)


def kg_per_polymer(mix, n_plastic, n_non_plastic, total_kg, masses):
    """kg of each polymer in the batch. Returns (numpy array, plastic fraction)."""
    fraction = plastic_mass_fraction(mix, n_plastic, n_non_plastic, masses)
    return mass_weighted_shares(mix, masses) * fraction * total_kg, fraction
