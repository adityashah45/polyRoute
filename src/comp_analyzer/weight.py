"""Estimates the total batch weight when the user hasn't weighed it.

The user describes the batch with ONE of these (best first):
    {"total_kg": k}        measured weight            -> used as is
    {"volume_m3": v.0}     volume of the waste        -> volume x bulk density
    {"item_count": n}      number of items in batch   -> count x average item mass
    None / {}              photos show the whole batch -> sum of typical item masses

Every method returns a point value AND a sampler, so the uncertainty of the
weight can be carried into the bootstrap intervals of the composition estimate
"""

import numpy as np

DEFAULT_PARAMS = {"mass_cv": 0.5, "bulk_density_kg_m3": {"low": 25, "typical": 50, "high": 100}}


def typical_mass_g(item, masses, fallback_g):
    """Typical mass of one photographed item, in grams."""
    if item["material"] == "plastic":
        return masses.get(item["polymer"], fallback_g)
    if item["material"] == "non_plastic":
        return masses.get("non_plastic", fallback_g)
    return fallback_g


def _sample_masses_g(typical_g, cv, rng):
    """Draw actual item masses around their typical values (lognormal to keep them positive)."""
    sigma2 = np.log(1 + cv ** 2)
    mu = np.log(typical_g) - sigma2 / 2
    return rng.lognormal(mu, np.sqrt(sigma2))


def estimate_total_kg(items, masses, batch=None, params=None):
    """Return {"kg", "source", "sampler"}.

    sampler(sample_items, rng) -> one random draw of the batch weight (kg),
    used inside the bootstrap so weight uncertainty widens the intervals
    """
    batch = batch or {}
    params = params or DEFAULT_PARAMS
    cv = params["mass_cv"]

    known = [typical_mass_g(it, masses, 1.0) for it in items if it["material"] != "uncertain"]
    fallback_g = float(np.mean(known)) if known else 1.0

    def typical_array(sample):
        return np.array([typical_mass_g(it, masses, fallback_g) for it in sample])

    # 1. Measured weight
    if batch.get("total_kg"):
        kg = float(batch["total_kg"])
        return {"kg": kg, "source": "measured", "sampler": lambda sample, rng: kg}

    # 2. Volume x bulk density (density drawn from a triangular low / typical / high range)
    if batch.get("volume_m3"):
        v = float(batch["volume_m3"])
        d = params["bulk_density_kg_m3"]
        return {
            "kg": v * d["typical"],
            "source": "volume",
            "sampler": lambda sample, rng: v * rng.triangular(d["low"], d["typical"], d["high"]),
        }

    if not items:
        raise ValueError("Cannot estimate the weight: no items and no weight, volume or item count given")

    # 3. Item count x average item mass of the sampled mix
    if batch.get("item_count"):
        n = int(batch["item_count"])
        kg = n * typical_array(items).mean() / 1000

        def sampler(sample, rng):
            avg_g = _sample_masses_g(typical_array(sample), cv, rng).mean()
            return n * avg_g / 1000

        return {"kg": kg, "source": "item_count", "sampler": sampler}

    # 4. Photos show the whole batch: add up the photographed items
    kg = typical_array(items).sum() / 1000

    def sampler(sample, rng):
        return _sample_masses_g(typical_array(sample), cv, rng).sum() / 1000

    return {"kg": kg, "source": "photos_only", "sampler": sampler}
