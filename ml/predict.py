"""Portable NumPy inference for the adaptive-weight reliability system.

The neural network acts as an ADAPTIVE WEIGHT LEARNER, not a selector. It
outputs the four reliability coefficients

    [a, b, c, d] = f(X)

(softmax-constrained so a, b, c, d >= 0 and a + b + c + d = 1). These plug
into the preserved original reliability equation for each receiver i:

    W_i = a*T_i + b*S_i + c*SNR_i + d*DOP_i

and the deterministic selection rule remains  argmax(W).  Everything used at
inference is available in real time; this module is dependency-free (NumPy).
"""
from pathlib import Path
import numpy as np

from features import reliability_scores, build_features, eligible

WEIGHT_NAMES = ['alpha', 'beta', 'gamma', 'delta']

# Original fixed coefficients (Model A / safe fallback).
FIXED_FALLBACK_WEIGHTS = np.array([0.25, 0.25, 0.30, 0.20], dtype=float)


class DynamicWeights:
    """Loads an exported model.npz and returns dynamic reliability weights."""

    def __init__(self, path=None):
        path = Path(path) if path else Path(__file__).parent / 'artifacts/model.npz'
        with np.load(path, allow_pickle=False) as z:
            self.mean = z['mean'].astype(float)
            self.std = np.maximum(z['std'].astype(float), 1e-6)
            self.p = {k: z[k].astype(float) for k in ('w1', 'b1', 'w2', 'b2', 'w3', 'b3')}

    def weights(self, X):
        """Softmax reliability weights a, b, c, d for conditioning vectors X."""
        X = np.asarray(X, dtype=float)
        z = (X - self.mean) / self.std
        z = np.maximum(z @ self.p['w1'] + self.p['b1'], 0)
        z = np.maximum(z @ self.p['w2'] + self.p['b2'], 0)
        z = z @ self.p['w3'] + self.p['b3']
        e = np.exp(z - z.max(axis=-1, keepdims=True))
        return np.where(np.isfinite(e), e / e.sum(axis=-1, keepdims=True), FIXED_FALLBACK_WEIGHTS)

    def reliability(self, W, terms):
        """Reliability scores W1, W2 = terms (..,2,4) weighted by W (..,4)."""
        return np.sum(np.asarray(W)[..., None, :] * np.asarray(terms), axis=-1)


def apply_hysteresis(W, available, threshold):
    """Deterministic selection with a deadband.

    If |W1 - W2| < threshold the previously selected receiver is retained;
    otherwise the highest-W available receiver wins. Unavailable receivers are
    never selected. Returns one decision (-1 = none) per timestamp.
    """
    W = np.asarray(W)
    available = np.asarray(available, dtype=bool)
    n = len(W)
    decisions = np.full(n, -1, dtype=int)
    prev = -1
    for i in range(n):
        if not available[i].any():
            decisions[i], prev = -1, -1
            continue
        w = W[i]
        candidates = np.flatnonzero(available[i])
        best = int(candidates[np.argmax(w[candidates])])
        if (prev in candidates and abs(float(W[i, 0] - W[i, 1])) < threshold):
            best = prev
        decisions[i] = best
        prev = best
    return decisions


def select_scores(W, available):
    """Hard argmax selection per timestamp (deployment rule, no hysteresis)."""
    W = np.where(np.asarray(available, dtype=bool), W, -np.inf)
    sel = np.argmax(W, axis=-1)
    return np.where(~np.asarray(available).any(axis=-1), -1, sel)