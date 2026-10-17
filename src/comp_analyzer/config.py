"""Shared constants for the comp_analyzer package."""

from pathlib import Path

# Polymer classes, in the order used by every array in this package
POLYMERS = ["PET", "HDPE", "PVC", "LDPE", "PP", "PS", "Other"]

# Allowed values of "material" in the classifier output
MATERIALS = {"plastic", "non_plastic", "uncertain"}

# Below this many plastic items, show a "add more photos" warning
SMALL_SAMPLE = 30

# Error correction removes bias but adds variance. Our simulation
# (comp_analyzer/validate.py) shows it only beats plain counting on large samples.
ADJUST_MIN_ITEMS = 500

# Bootstrap settings for the 95% intervals
N_BOOTSTRAP = 500
CI_PERCENTILES = (2.5, 97.5)


DATA_DIR = Path(__file__).resolve().parent / "data"
SYNTHETIC_DIR = DATA_DIR / "synthetic"
ITEM_MASS_FILE = DATA_DIR / "item_mass.csv"
CONFUSION_FILE = DATA_DIR / "confusion_matrix.json"
WEIGHT_PARAMS_FILE = DATA_DIR / "weight_params.json"
