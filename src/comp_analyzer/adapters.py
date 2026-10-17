"""Convert the composition result into the input format other modules expect."""

from .config import POLYMERS


def to_optimiser_input(result, total="batch"):
    """
    Builds the optimiser's "composition_data" from estimate_composition() output

    total : "batch"   -> total_kg = whole batch (plastic + non-plastic); sum(kg) <= total_kg
            "plastic" -> total_kg = plastic only; sum(kg) == total_kg (up to rounding)

    margin[p] = half-width of the 95% interval of polymer p, as a share of the
    plastic weight (e.g. 0.0997 = ±9.97 percentage points). 0.0 when no interval exists

    """
    if total not in ("batch", "plastic"):
        raise ValueError('total must be "batch" or "plastic"')
    plastic_kg = result["plastic_kg"]
    margin = {}
    for p in POLYMERS:
        if p not in result["kg"]:
            continue
        low, high = result["kg_ci"].get(p, [result["kg"][p], result["kg"][p]])
        margin[p] = 0.0
        if plastic_kg:
            margin[p] = round(min((high - low)/(2*plastic_kg), 1.0), 4)

    return {
        "composition_data": {
            "total_kg": float(result["total_kg"] if total == "batch" else plastic_kg),
            "kg": {p: result["kg"][p] for p in POLYMERS if p in result["kg"]},
            "margin": margin,
            "excluded": dict(result["excluded"]),
        }
    }
