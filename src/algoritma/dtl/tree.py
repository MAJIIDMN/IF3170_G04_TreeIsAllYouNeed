from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .criteria import select_split
from .splitting import (best_split_categorical, best_split_continuous,
                        known_mask)

CONTINUOUS = "continuous"
CATEGORICAL = "categorical"
MULTIWAY = "multiway"
BINARY = "binary"


@dataclass(frozen=True)
class BuildConfig:
    n_classes: int
    max_depth: int | None
    min_samples_leaf: int
    min_gain: float


class Node:
    __slots__ = ("leaf", "class_weights", "feature", "kind", "children",
                 "shares", "default", "threshold", "left", "right", "p_left",
                 "leaf_id")

    def __init__(self):
        self.leaf = True
        self.class_weights = None
        self.feature = None
        self.kind = None
        self.children = None    # multiway: {value: Node}
        self.shares = None      # multiway: {value: share of known weight}
        self.default = None     # multiway: value; binary: side0/1
        self.threshold = None
        self.left = None
        self.right = None
        self.p_left = None
        self.leaf_id = None

    def plurality(self) -> int:
        """Majority class of the training rows that reached this node."""
        return int(np.argmax(self.class_weights))

    def become_leaf(self) -> None:
        """Drop the split, keeping class weights (used by pruning)."""
        self.leaf = True
        self.feature = None
        self.kind = None
        self.children = None
        self.shares = None
        self.default = None
        self.threshold = None
        self.left = None
        self.right = None
        self.p_left = None
        self.leaf_id = None


def children_of(node):
    if node.kind == MULTIWAY:
        return list(node.children.values())
    return [node.left, node.right]


def iter_nodes(node):
    """All nodes in preorder."""
    yield node
    if not node.leaf:
        yield from (k for child in children_of(node) for k in iter_nodes(child))


def iter_internal(node):
    return (n for n in iter_nodes(node) if not n.leaf)


def tree_depth(node) -> int:
    if node.leaf:
        return 0
    return 1 + max(tree_depth(k) for k in children_of(node))


def assign_leaf_ids(root) -> int:
    """Number leaves in preorder; returns the leaf count."""
    counter = 0

    def walk(n):
        nonlocal counter
        if n.leaf:
            n.leaf_id = counter
            counter += 1
            return
        for child in children_of(n):
            walk(child)

    walk(root)
    return counter


def step(node, row):
    """Child reached from ``node`` for one sample ``row``."""
    v = row[node.feature]
    if node.kind == MULTIWAY:
        if v is None:
            return node.children[node.default]
        child = node.children.get(v)
        return node.children[node.default] if child is None else child
    if v is None or (isinstance(v, (float, np.floating)) and np.isnan(v)):
        return node.left if node.default == 0 else node.right
    return node.left if v <= node.threshold else node.right


def traverse(root, row) -> Node:
    """Leaf reached by one row (missing value follows the common child)."""
    node = root
    while not node.leaf:
        node = step(node, row)
    return node


def predict_row(root, row, n_classes: int) -> np.ndarray:
    leaf = traverse(root, row)
    total = float(leaf.class_weights.sum())
    if total <= 0.0:
        return np.full(n_classes, 1.0 / n_classes)
    return leaf.class_weights / total


def build_tree(columns, kinds, y_idx, w, idx, depth, cfg: BuildConfig) -> Node:
    node = Node()
    node.class_weights = np.bincount(y_idx[idx], weights=w[idx],
                                     minlength=cfg.n_classes).astype(float)
    if (idx.size < 2 * cfg.min_samples_leaf
            or int((node.class_weights > 0).sum()) <= 1
            or (cfg.max_depth is not None and depth >= cfg.max_depth)):
        return node

    y_n = y_idx[idx]
    w_n = w[idx]
    candidates = []
    for f, kind in enumerate(kinds):
        vals = columns[f][idx]
        if kind == CONTINUOUS:
            cands = best_split_continuous(vals, y_n, w_n, cfg.n_classes,
                                          cfg.min_samples_leaf)
            cands = [c for c in cands if c[0] > cfg.min_gain]
            if cands:
                # threshold by greatest gain (Materi 10c p. 19); the
                # attribute-level choice below still runs on gain ratio
                best = max(cands, key=lambda c: c[0])
                candidates.append((best[0], best[1], (BINARY, f, best[2])))
        else:
            split = best_split_categorical(vals, y_n, w_n, cfg.n_classes,
                                           cfg.min_samples_leaf)
            if split is not None and split[0] > cfg.min_gain:
                candidates.append((split[0], split[1],
                                   (MULTIWAY, f, split[2])))
    chosen = select_split(candidates)
    if chosen is None:
        return node

    _, _, (kind, f, payload) = chosen
    node.leaf = False
    node.feature = f
    node.kind = kind
    vals = columns[f][idx]
    known = known_mask(vals)
    miss = ~known
    known_w = float(w[idx][known].sum())

    if kind == MULTIWAY:
        node.children = {}
        node.shares = {}
        for u in payload:
            child_known = idx[known & (vals == u)]
            share = float(w[child_known].sum()) / known_w if known_w > 0 else 0.0
            node.children[u] = _build_child(columns, kinds, y_idx, w, idx,
                                            child_known, miss, share, depth,
                                            cfg)
            node.shares[u] = share
        node.default = max(node.shares, key=node.shares.get)
    else:
        thr = payload
        left_known = idx[known & (vals <= thr)]
        right_known = idx[known & (vals > thr)]
        wl = float(w[left_known].sum())
        wr = float(w[right_known].sum())
        node.threshold = float(thr)
        node.p_left = wl / (wl + wr) if wl + wr > 0 else 0.5
        node.left = _build_child(columns, kinds, y_idx, w, idx, left_known,
                                 miss, node.p_left, depth, cfg)
        node.right = _build_child(columns, kinds, y_idx, w, idx, right_known,
                                  miss, 1.0 - node.p_left, depth, cfg)
        node.default = 0 if node.p_left >= 0.5 else 1
    return node


def _build_child(columns, kinds, y_idx, w, idx, child_known, miss, share,
                 depth, cfg: BuildConfig) -> Node:
    """Build one child, appending the node's missing rows with weight share
    (Materi 10c p. 22: p_i added to each value for splitting)."""
    if miss.any():
        child_idx = np.concatenate([child_known, idx[miss]])
        child_w = w.copy()
        child_w[idx[miss]] = w[idx[miss]] * share
    else:
        child_idx, child_w = child_known, w
    return build_tree(columns, kinds, y_idx, child_w, child_idx, depth + 1, cfg)
