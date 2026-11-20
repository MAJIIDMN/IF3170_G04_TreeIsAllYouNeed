from __future__ import annotations

import numpy as np

ENTROPY = "entropy"
GINI = "gini"


def impurity_rows(weights: np.ndarray, measure: str = ENTROPY) -> np.ndarray:
    totals = weights.sum(axis=1, keepdims=True)
    p = np.divide(weights, totals, out=np.zeros_like(weights), where=totals > 0.0)
    if measure == GINI:
        return 1.0 - (p * p).sum(axis=1)
    logp = np.where(p > 0.0, np.log2(np.where(p > 0.0, p, 1.0)), 0.0)
    return -(p * logp).sum(axis=1)


def impurity(class_weights, measure: str = ENTROPY) -> float:
    w = np.asarray(class_weights, dtype=float)[None, :]
    return float(impurity_rows(w, measure)[0])


def entropy(class_weights) -> float:
    return impurity(class_weights, ENTROPY)


def split_info_rows(shares: np.ndarray) -> np.ndarray:
    logs = np.where(shares > 0.0, np.log2(np.where(shares > 0.0, shares, 1.0)), 0.0)
    return -(shares * logs).sum(axis=1)


def select_split(candidates, use_ratio: bool = True):
    if not candidates:
        return None
    if not use_ratio:
        return max(candidates, key=lambda c: c[0])
    mean_gain = sum(c[0] for c in candidates) / len(candidates)
    pool = [c for c in candidates if c[0] >= mean_gain] or list(candidates)
    return max(pool, key=lambda c: c[1])
