from __future__ import annotations

import numpy as np

from .tree import Node, children_of, iter_internal, iter_nodes, step


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
    if alpha < 0.0:
        raise ValueError(f"alpha must be non-negative, got {alpha}")
    pruned = 0
    while True:
        internal = list(iter_internal(root))
        if not internal:
            break
        leaves_err: dict[Node, tuple[int, float]] = {}

        def stats(n: Node) -> tuple[int, float]:
            if n.leaf:
                leaves_err[n] = (1, _node_error(n))
                return leaves_err[n]
            leaves = 0
            err = 0.0
            for child in children_of(n):
                sub_leaves, sub_err = stats(child)
                leaves += sub_leaves
                err += sub_err
            leaves_err[n] = (leaves, err)
            return leaves, err

        stats(root)
        best_t, best_alpha = None, None
        for n in internal:  # preorder: ties collapse the earliest node
            leaves, sub_err = leaves_err[n]
            if leaves <= 1:
                continue
            a = (_node_error(n) - sub_err) / (leaves - 1)
            if best_alpha is None or a < best_alpha:
                best_t, best_alpha = n, a
        if best_t is None or best_alpha > alpha:
            break
        best_t.become_leaf()
        pruned += 1
    return root, {"pruned_nodes": pruned}
