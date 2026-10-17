"""Convert the composition result into the input format other modules expect."""

from .config import POLYMERS


def to_optimiser_input(result):
    """
    Builds the optimiser's "composition_data" from estimate_composition() output

    margin[p] = half-width of the 95% interval of polymer p, as a share of the
    plastic weight (e.g. 0.0997 = ±9.97 percentage points). 0.0 when no interval exists

    """
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
            "total_kg": float(result["total_kg"]),
            "kg": {p: result["kg"][p] for p in POLYMERS if p in result["kg"]},
            "margin": margin,
            "excluded": dict(result["excluded"]),
        }
    }
