from __future__ import annotations

import numpy as np

from .criteria import entropy, gain_ratio, information_gain, split_info


def known_mask(values) -> np.ndarray:
    """Boolean mask of rows whose attribute value is present."""
    if values.dtype == object:
        return np.fromiter((v is not None for v in values), dtype=bool,
                           count=len(values))
    return ~np.isnan(values)


def _entropy_rows(m: np.ndarray) -> np.ndarray:
    """Entropy of each row of an (m, k) matrix of class weights."""
    totals = m.sum(axis=1, keepdims=True)
    p = np.divide(m, totals, out=np.zeros_like(m), where=totals > 0.0)
    logp = np.where(p > 0.0, np.log2(np.where(p > 0.0, p, 1.0)), 0.0)
    return -(p * logp).sum(axis=1)


def best_split_continuous(values, y_idx, w, n_classes, min_samples_leaf):
    """Candidate ``(gain, ratio, threshold)`` splits for a float column.

    ``values`` is a float64 array of one node's rows with np.nan for missing.
    Candidates are midpoints between adjacent known values whose value
    differs; each side must keep at least ``min_samples_leaf`` known rows.
    """
    known = known_mask(values)
    if int(known.sum()) < 2 * min_samples_leaf:
        return []
    idx = np.flatnonzero(known)
    order = idx[np.argsort(values[idx], kind="stable")]
    v = values[order]
    ys = y_idx[order]
    ws = w[order]
    boundaries = np.flatnonzero(np.diff(v) > 0.0)
    boundaries = boundaries[(boundaries + 1 >= min_samples_leaf) &
                            (v.size - boundaries - 1 >= min_samples_leaf)]
    if boundaries.size == 0:
        return []
    onehot = np.zeros((v.size, n_classes))
    onehot[np.arange(v.size), ys] = ws
    cum = np.cumsum(onehot, axis=0)
    total = cum[-1]
    left = cum[boundaries]
    right = total - left
    known_total = float(total.sum())
    parent = np.bincount(y_idx, weights=w, minlength=n_classes)
    if parent.sum() <= 0.0 or known_total <= 0.0:
        return []
    wl = left.sum(axis=1)
    wr = right.sum(axis=1)
    h_after = (wl / known_total) * _entropy_rows(left) + \
              (wr / known_total) * _entropy_rows(right)
    gains = (known_total / float(parent.sum())) * (entropy(parent) - h_after)
    pl = wl / known_total
    pr = wr / known_total
    infos = -(pl * np.log2(pl) + pr * np.log2(pr))
    ratios = np.divide(gains, infos, out=np.zeros_like(gains), where=infos > 0.0)
    thresholds = v[boundaries] + (v[boundaries + 1] - v[boundaries]) / 2.0
    return [(float(g), float(r), float(t))
            for g, r, t in zip(gains, ratios, thresholds)]


def best_split_categorical(values, y_idx, w, n_classes, min_samples_leaf):
    """Multiway split over the values observed in an object column.

    ``values`` is an object array with None for missing rows. Returns
    ``(gain, ratio, observed_values)`` or None; every observed value needs at
    least ``min_samples_leaf`` known rows or the whole split is rejected.
    """
    known = known_mask(values)
    n_known = int(known.sum())
    if n_known < 2 * min_samples_leaf:
        return None
    uniques = np.unique(values[known])
    if uniques.size < 2:
        return None
    parent = np.bincount(y_idx, weights=w, minlength=n_classes)
    children = []
    for u in uniques:
        m = known & (values == u)
        if int(m.sum()) < min_samples_leaf:
            return None
        children.append(np.bincount(y_idx[m], weights=w[m],
                                   minlength=n_classes))
    if sum(float(c.sum()) for c in children) <= 0.0:
        return None
    gain = information_gain(parent, children)
    return gain, gain_ratio(gain, split_info(children)), uniques
