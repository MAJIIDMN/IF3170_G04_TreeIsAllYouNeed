from __future__ import annotations

import numpy as np

from .criteria import impurity, impurity_rows, split_info_rows


def known_mask(values) -> np.ndarray:
    if values.dtype == object:
        return np.fromiter((v is not None for v in values), dtype=bool,
                           count=len(values))
    return ~np.isnan(values)


def _scan_boundaries(ordered, node_weight, cfg):
    cum = np.cumsum(ordered, axis=0)
    total = cum[-1]
    known_total = float(total.sum())
    if known_total <= 0.0 or node_weight <= 0.0:
        return None
    left = cum[:-1]
    right = total - left
    wl = left.sum(axis=1)
    wr = right.sum(axis=1)
    valid = (wl >= cfg.min_samples_leaf) & (wr >= cfg.min_samples_leaf)
    if not valid.any():
        return None
    parent = impurity(total, cfg.measure)
    after = (wl * impurity_rows(left, cfg.measure)
             + wr * impurity_rows(right, cfg.measure)) / known_total
    gains = (known_total / node_weight) * (parent - after)
    gains = np.where(valid, gains, -np.inf)
    best = int(np.argmax(gains))
    shares = np.array([[wl[best], wr[best]]]) / known_total
    info = float(split_info_rows(shares)[0])
    gain = float(gains[best])
    return gain, (gain / info if info > 0.0 else 0.0), best


def best_split_continuous(values, y_idx, w, cfg):
    known = known_mask(values)
    if not known.any():
        return None
    distinct, inverse = np.unique(values[known], return_inverse=True)
    if distinct.size < 2:
        return None
    ordered = np.zeros((distinct.size, cfg.n_classes))
    np.add.at(ordered, (inverse, y_idx[known]), w[known])
    found = _scan_boundaries(ordered, float(w.sum()), cfg)
    if found is None:
        return None
    gain, ratio, i = found
    return gain, ratio, float(distinct[i] + (distinct[i + 1] - distinct[i]) / 2.0)


def _category_weights(values, y_idx, w, known, n_classes):
    uniques, inverse = np.unique(values[known], return_inverse=True)
    weights = np.zeros((uniques.size, n_classes))
    np.add.at(weights, (inverse, y_idx[known]), w[known])
    return uniques, weights


# Sumber: L. Breiman et al., Classification and Regression Trees, 1984, Thm. 4.5.
def best_split_subset(values, y_idx, w, cfg):
    known = known_mask(values)
    if not known.any():
        return None
    uniques, weights = _category_weights(values, y_idx, w, known, cfg.n_classes)
    if uniques.size < 2:
        return None
    totals = weights.sum(axis=1)
    order = np.argsort(weights[:, -1] / np.maximum(totals, 1e-12), kind="stable")
    found = _scan_boundaries(weights[order], float(w.sum()), cfg)
    if found is None:
        return None
    gain, ratio, i = found
    names = uniques[order]
    return gain, ratio, (frozenset(names[:i + 1].tolist()),
                         frozenset(names[i + 1:].tolist()))


def best_split_multiway(values, y_idx, w, cfg):
    known = known_mask(values)
    if not known.any():
        return None
    uniques, weights = _category_weights(values, y_idx, w, known, cfg.n_classes)
    sizes = weights.sum(axis=1)
    if uniques.size < 2 or sizes.min() < cfg.min_samples_leaf:
        return None
    known_total = float(sizes.sum())
    after = float((sizes * impurity_rows(weights, cfg.measure)).sum()) / known_total
    parent = impurity(weights.sum(axis=0), cfg.measure)
    gain = (known_total / float(w.sum())) * (parent - after)
    info = float(split_info_rows((sizes / known_total)[None, :])[0])
    return gain, (gain / info if info > 0.0 else 0.0), uniques.tolist()
