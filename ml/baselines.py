"""Baseline selectors for comparison.

Model A -- original fixed weights a=0.25, b=0.25, c=0.30, d=0.20.
Model B -- a single globally optimized fixed weight set, found by an exhaustive
grid search over the simplex so that model A is a member of the search space.

Both use the preserved reliability equation W = aT + bS + cSNR + dDOP.
"""
import numpy as np

from predict import FIXED_FALLBACK_WEIGHTS, select_scores

MODEL_A_WEIGHTS = FIXED_FALLBACK_WEIGHTS


def reliability(W, terms):
    """Terms must have shape (..., 2, 4) -> per-receiver score (..., 2)."""
    return np.sum(np.asarray(W)[..., None, :] * np.asarray(terms), axis=-1)


def selected_errors(decisions, errors, available):
    """Actual error of the selected receiver on epochs where one is available."""
    decisions = np.asarray(decisions)
    errors = np.asarray(errors)
    available = np.asarray(available, dtype=bool)
    ok = np.flatnonzero(available.any(axis=-1))
    sel = decisions[ok]
    keep = sel >= 0
    return errors[ok[keep], sel[keep]]


def grid_search_optimal_weights(terms, errors, available, step=0.05):
    """Exhaustively search the simplex at 'step' resolution and return the set
    minimizing mean selected error on the provided (validation) epochs.

    Returns (best_weights, best_mean_error, candidates_evaluated).
    """
    terms = np.asarray(terms, dtype=float)
    errors = np.asarray(errors, dtype=float)
    available = np.asarray(available, dtype=bool)

    use = available.any(axis=-1) & (errors.sum(axis=-1) >= 0)
    T, E = terms[use], errors[use]
    acc = available[use]
    n = int(round(1.0 / step))
    best_w, best_err = None, np.inf
    count = 0
    for a in range(n + 1):
        for b in range(n + 1 - a):
            for c in range(n + 1 - a - b):
                d = n - a - b - c
                w = np.array([a, b, c, d], dtype=float) * step
                count += 1
                select = select_scores(reliability(w, T), acc)
                err = selected_errors(select, E, acc)
                mean_err = float(err.mean())
                if mean_err < best_err:
                    best_err, best_w = mean_err, w
    return best_w, float(best_err), count