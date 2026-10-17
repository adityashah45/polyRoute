"""bootstraps 95% intervals.

Resamples the photographed items with replacement many times, re-run the
estimate each time (including a fresh draw of the batch weight when it was
estimated), and take the 2.5th and 97.5th percentiles.
"""

import numpy as np

from .config import CI_PERCENTILES, N_BOOTSTRAP


def bootstrap(items, estimate_fn, n_boot=N_BOOTSTRAP, seed=0):
    """Run estimate_fn on n_boot resamples.

    items       : the classifier results
    estimate_fn : function(sample_items, rng) -> numpy array (any length)
    Returns     : (low, high) numpy arrays, same length as estimate_fn's output
    """
    rng = np.random.default_rng(seed)
    n = len(items)
    samples = np.array([
        estimate_fn([items[i] for i in rng.integers(0, n, n)], rng)
        for _ in range(n_boot)
    ])
    low, high = np.percentile(samples, CI_PERCENTILES, axis=0)
    return low, high
