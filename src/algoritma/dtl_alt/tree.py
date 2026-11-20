from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .criteria import select_split
from .splitting import (best_split_continuous, best_split_multiway,
                        best_split_subset, known_mask)

CONTINUOUS = "continuous"
CATEGORICAL = "categorical"
MULTIWAY = "multiway"
BINARY = "binary"
SUBSET = "subset"


@dataclass(frozen=True)
class BuildConfig:
    n_classes: int
    max_depth: int | None
    min_samples_leaf: float
    min_gain: float
    measure: str = "entropy"
    use_ratio: bool = True
    categorical_split: str = SUBSET


class Node:
    __slots__ = ("leaf", "class_weights", "feature", "kind", "children",
                 "shares", "default", "threshold", "left", "right", "p_left",
                 "left_values", "right_values", "leaf_id")

    def __init__(self):
        self.leaf = True
        self.class_weights = None
        self.leaf_id = None
        self._clear_split()

    def _clear_split(self):
        self.feature = None
        self.kind = None
        self.children = None
        self.shares = None
        self.default = None
        self.threshold = None
        self.left = None
        self.right = None
        self.p_left = None
        self.left_values = None
        self.right_values = None

    def plurality(self) -> int:
        return int(np.argmax(self.class_weights))

    def become_leaf(self) -> None:
        self.leaf = True
        self.leaf_id = None
        self._clear_split()


def children_of(node):
    if node.kind == MULTIWAY:
        return list(node.children.values())
    return [node.left, node.right]


def iter_nodes(node):
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
    v = row[node.feature]
    if node.kind == MULTIWAY:
        child = None if v is None else node.children.get(v)
        return node.children[node.default] if child is None else child
    common = node.left if node.default == 0 else node.right
    if node.kind == SUBSET:
        if v in node.left_values:
            return node.left
        return node.right if v in node.right_values else common
    if v is None or (isinstance(v, (float, np.floating)) and np.isnan(v)):
        return common
    return node.left if v <= node.threshold else node.right


def traverse(root, row) -> Node:
    node = root
    while not node.leaf:
        node = step(node, row)
    return node


def _member_mask(values, allowed) -> np.ndarray:
    return np.fromiter((v in allowed for v in values), dtype=bool,
                       count=len(values))


def _routes(node, column):
    if node.kind == MULTIWAY:
        routes = [(child, _member_mask(column, {value}), node.shares[value])
                  for value, child in node.children.items()]
    else:
        if node.kind == SUBSET:
            left = _member_mask(column, node.left_values)
            right = _member_mask(column, node.right_values)
        else:
            known = ~np.isnan(column)
            left = known & (column <= node.threshold)
            right = known & (column > node.threshold)
        routes = [(node.left, left, node.p_left),
                  (node.right, right, 1.0 - node.p_left)]
    routed = np.zeros(len(column), dtype=bool)
    for _, mask, _ in routes:
        routed |= mask
    return routes, ~routed


# Sumber: J. R. Quinlan, C4.5: Programs for Machine Learning, 1993, ch. 3.
def accumulate_proba(node, columns, idx, weight, out, fractional):
    if idx.size == 0:
        return
    if node.leaf:
        out[idx] += weight[:, None] * (node.class_weights / node.class_weights.sum())
        return
    routes, unknown = _routes(node, columns[node.feature][idx])
    for child, mask, share in routes:
        rows, row_weight = idx[mask], weight[mask]
        if unknown.any():
            if fractional:
                rows = np.concatenate([rows, idx[unknown]])
                row_weight = np.concatenate([row_weight, weight[unknown] * share])
            elif child is _common_child(node):
                rows = np.concatenate([rows, idx[unknown]])
                row_weight = np.concatenate([row_weight, weight[unknown]])
        accumulate_proba(child, columns, rows, row_weight, out, fractional)


def _common_child(node):
    if node.kind == MULTIWAY:
        return node.children[node.default]
    return node.left if node.default == 0 else node.right


def _candidates(columns, kinds, y_n, w_n, idx, cfg):
    out = []
    for f, kind in enumerate(kinds):
        values = columns[f][idx]
        if kind == CONTINUOUS:
            split, split_kind = best_split_continuous(values, y_n, w_n, cfg), BINARY
        elif cfg.categorical_split == SUBSET:
            split, split_kind = best_split_subset(values, y_n, w_n, cfg), SUBSET
        else:
            split, split_kind = best_split_multiway(values, y_n, w_n, cfg), MULTIWAY
        if split is not None and split[0] > cfg.min_gain:
            out.append((split[0], split[1], (split_kind, f, split[2])))
    return out


def build_tree(columns, kinds, y_idx, idx, w, depth, cfg: BuildConfig) -> Node:
    node = Node()
    y_n = y_idx[idx]
    node.class_weights = np.bincount(y_n, weights=w, minlength=cfg.n_classes).astype(float)
    if (float(w.sum()) < 2 * cfg.min_samples_leaf
            or int((node.class_weights > 0).sum()) <= 1
            or (cfg.max_depth is not None and depth >= cfg.max_depth)):
        return node

    chosen = select_split(_candidates(columns, kinds, y_n, w, idx, cfg), cfg.use_ratio)
    if chosen is None:
        return node

    kind, f, payload = chosen[2]
    node.leaf = False
    node.feature = f
    node.kind = kind
    values = columns[f][idx]
    known = known_mask(values)
    missing = ~known

    def grow(mask, share):
        rows = np.concatenate([idx[mask], idx[missing]])
        weights = np.concatenate([w[mask], w[missing] * share])
        return build_tree(columns, kinds, y_idx, rows, weights, depth + 1, cfg)

    if kind == MULTIWAY:
        masks = {u: known & _member_mask(values, {u}) for u in payload}
        known_weight = float(w[known].sum())
        node.shares = {u: float(w[m].sum()) / known_weight for u, m in masks.items()}
        node.children = {u: grow(m, node.shares[u]) for u, m in masks.items()}
        node.default = max(node.shares, key=node.shares.get)
        return node

    if kind == SUBSET:
        node.left_values, node.right_values = payload
        left = known & _member_mask(values, node.left_values)
        right = known & _member_mask(values, node.right_values)
    else:
        node.threshold = float(payload)
        left = known & (values <= node.threshold)
        right = known & (values > node.threshold)
    weight_left, weight_right = float(w[left].sum()), float(w[right].sum())
    node.p_left = weight_left / (weight_left + weight_right)
    node.default = 0 if node.p_left >= 0.5 else 1
    node.left = grow(left, node.p_left)
    node.right = grow(right, 1.0 - node.p_left)
    return node
