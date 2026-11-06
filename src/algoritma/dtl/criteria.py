from __future__ import annotations

import numpy as np


def entropy(class_weights) -> float:
    """H = -sum_c p_c log2 p_c over a weighted class distribution."""
    w = np.asarray(class_weights, dtype=float)
    total = float(w.sum())
    if total <= 0.0:
        return 0.0
    p = w[w > 0.0] / total
    return float(-(p * np.log2(p)).sum())


def information_gain(parent_weights, child_weights) -> float:
    parent = np.asarray(parent_weights, dtype=float)
    children = [np.asarray(c, dtype=float) for c in child_weights]
    parent_total = float(parent.sum())
    known = float(sum(c.sum() for c in children))
    if parent_total <= 0.0 or known <= 0.0:
        return 0.0
    h_after = 0.0
    for c in children:
        w = float(c.sum())
        if w > 0.0:
            h_after += (w / known) * entropy(c)
    return (known / parent_total) * (entropy(parent) - h_after)


def split_info(child_weights) -> float:
    """SplitInfo = -sum_i p_i log2 p_i over the children's weight shares."""
    children = [np.asarray(c, dtype=float) for c in child_weights]
    known = float(sum(c.sum() for c in children))
    if known <= 0.0:
        return 0.0
    shares = np.array([float(c.sum()) / known for c in children if c.sum() > 0.0])
    return float(-(shares * np.log2(shares)).sum())


def gain_ratio(gain: float, info: float) -> float:
    """Ratio of a split; 0 when the split carries no information."""
    return gain / info if info > 0.0 else 0.0


def select_split(candidates):
    """Highest gain-ratio candidate whose gain is at least the mean gain of
    all candidates (Materi 10c p. 27). Ties go to the first candidate in
    input order; returns the chosen (gain, ratio, payload) tuple or None."""
    if not candidates:
        return None
    mean_gain = sum(g for g, _, _ in candidates) / len(candidates)
    pool = [c for c in candidates if c[0] >= mean_gain] or list(candidates)
    best = pool[0]
    for cand in pool[1:]:
        if cand[1] > best[1]:
            best = cand
    return best
