from __future__ import annotations

from typing import Literal

import numpy as np


def paired_permutation_test_brier(
    y_true,
    prob_a,
    prob_b,
    n_permutations: int = 10000,
    seed: int = 42,
    alternative: Literal["two-sided", "less", "greater"] = "two-sided",
) -> dict:
    """Paired permutation test on per-sample Brier loss differences.

    Returns the observed mean difference: loss(A) - loss(B).
    Negative values favor model A.
    """
    y_true = np.asarray(y_true, dtype=float)
    diff = (np.asarray(prob_a, dtype=float) - y_true) ** 2 - (np.asarray(prob_b, dtype=float) - y_true) ** 2
    obs = float(np.mean(diff))
    rng = np.random.default_rng(seed)
    null = np.empty(n_permutations, dtype=float)
    for i in range(n_permutations):
        signs = rng.choice([-1.0, 1.0], size=diff.shape[0])
        null[i] = np.mean(diff * signs)
    if alternative == "two-sided":
        p = np.mean(np.abs(null) >= abs(obs))
    elif alternative == "less":
        p = np.mean(null <= obs)
    elif alternative == "greater":
        p = np.mean(null >= obs)
    else:
        raise ValueError("alternative must be two-sided, less, or greater")
    return {"mean_brier_difference": obs, "p_value": float((p * n_permutations + 1) / (n_permutations + 1))}


def bca_ci(values, alpha: float = 0.05) -> tuple[float, float]:
    """Simple percentile CI placeholder.

    The paper reports BCa CIs. For exact BCa intervals, replace this with a bootstrap
    implementation that computes bias correction and acceleration for the statistic of interest.
    """
    arr = np.asarray(values, dtype=float)
    return (float(np.nanpercentile(arr, 100 * alpha / 2)), float(np.nanpercentile(arr, 100 * (1 - alpha / 2))))
