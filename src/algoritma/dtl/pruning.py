from __future__ import annotations

from statistics import NormalDist

import numpy as np

from .tree import Node, children_of, iter_internal, iter_nodes, step


# C4.5 also collapses a subtree that is at most this many errors better.
PRUNE_MARGIN = 0.1


def _node_error(node) -> float:
    """Misclassification weight if the node were a leaf."""
    w = node.class_weights
    return float(w.sum() - w.max())


def reduced_error_prune(root, columns, y_idx, val_idx):
    """Greedy reduced error pruning on a validation slice.

    Returns ``(root, report)``; ``report`` holds validation accuracy before
    and after plus the number of pruned nodes.
    """
    val_idx = np.asarray(val_idx, dtype=int)
    y_val = y_idx[val_idx]
    n_val = len(val_idx)
    if n_val == 0:
        return root, {"val_accuracy_before": None, "val_accuracy_after": None,
                      "pruned_nodes": 0}

    rows_through: dict[Node, list[int]] = {}
    cur_pred = np.empty(n_val, dtype=int)
    for i, r in enumerate(val_idx):
        row = [col[int(r)] for col in columns]
        node = root
        rows_through.setdefault(node, []).append(i)
        while not node.leaf:
            node = step(node, row)
            rows_through.setdefault(node, []).append(i)
        cur_pred[i] = node.plurality()
    for key in rows_through:
        rows_through[key] = np.asarray(rows_through[key], dtype=int)

    errors_before = int(np.count_nonzero(cur_pred != y_val))
    candidates = list(iter_internal(root))
    pruned = 0
    while candidates:
        best_t, best_gain = None, 0
        for t in candidates:
            rows = rows_through.get(t)
            if rows is None or rows.size == 0:
                continue
            pl = t.plurality()
            gain = (int(np.count_nonzero(cur_pred[rows] != y_val[rows]))
                    - int(np.count_nonzero(pl != y_val[rows])))
            if gain > best_gain:
                best_t, best_gain = t, gain
        if best_t is None:
            break
        rows = rows_through[best_t]
        cur_pred[rows] = best_t.plurality()
        dead = {n for n in iter_nodes(best_t) if n is not best_t}
        best_t.become_leaf()
        for d in dead:
            rows_through.pop(d, None)
        candidates = [c for c in candidates
                      if c is not best_t and c not in dead]
        pruned += 1

    errors_after = int(np.count_nonzero(cur_pred != y_val))
    report = {
        "val_accuracy_before": (n_val - errors_before) / n_val,
        "val_accuracy_after": (n_val - errors_after) / n_val,
        "pruned_nodes": pruned,
    }
    return root, report


def cost_complexity_prune(root, alpha: float):
    """Minimize error + alpha * leaves, with alpha in units of row weight.

    One bottom-up pass finds the optimal subtree, because the best cost of a
    subtree only depends on the best costs of its children (Breiman et al.,
    1984, chapter 10). A tie collapses the node, which keeps the smaller tree.
    """
    if alpha < 0.0:
        raise ValueError(f"alpha must be non-negative, got {alpha}")
    pruned = 0

    def best_cost(node) -> float:
        nonlocal pruned
        as_leaf = _node_error(node) + alpha
        if node.leaf:
            return as_leaf
        as_subtree = sum(best_cost(child) for child in children_of(node))
        if as_leaf <= as_subtree:
            node.become_leaf()
            pruned += 1
            return as_leaf
        return as_subtree

    best_cost(root)
    return root, {"pruned_nodes": pruned}


def _extra_errors(weight: float, errors: float, confidence: float, z: float) -> float:
    """Errors to add for the upper confidence limit, as in C4.5 release 8.

    Zero errors use the exact binomial limit 1 - CF^(1/N), under one error the
    value is interpolated, and above that a corrected normal limit is used.
    """
    if errors < 1e-6:
        return weight * (1.0 - confidence ** (1.0 / weight))
    if errors < 0.9999:
        none = weight * (1.0 - confidence ** (1.0 / weight))
        return none + errors * (_extra_errors(weight, 1.0, confidence, z) - none)
    if errors + 0.5 >= weight:
        return z * (weight - errors)
    corrected = errors + 0.5
    coefficient = z * z
    limit = (corrected + coefficient / 2.0 + z * np.sqrt(
        corrected * (1.0 - corrected / weight) + coefficient / 4.0)) / (weight + coefficient)
    return weight * limit - errors


def pessimistic_prune(root, confidence: float):
    """C4.5 error-based pruning (Quinlan, 1993, chapter 4).

    A subtree is replaced by a leaf when the leaf's estimated errors are no
    worse than the sum over the subtree, within ``PRUNE_MARGIN``. Lower
    ``confidence`` prunes more.
    Subtree raising from the original C4.5 is not implemented.
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    z = NormalDist().inv_cdf(1.0 - confidence)
    pruned = 0

    def estimate(node) -> float:
        nonlocal pruned
        weight, errors = float(node.class_weights.sum()), _node_error(node)
        as_leaf = errors + _extra_errors(weight, errors, confidence, z)
        if node.leaf:
            return as_leaf
        as_subtree = sum(estimate(child) for child in children_of(node))
        if as_leaf <= as_subtree + PRUNE_MARGIN:
            node.become_leaf()
            pruned += 1
            return as_leaf
        return as_subtree

    estimate(root)
    return root, {"pruned_nodes": pruned}
