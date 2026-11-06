from __future__ import annotations

import json

import numpy as np

from .tree import BINARY, MULTIWAY, Node

FORMAT = "if3170-dtl"
VERSION = 1


def _json_value(v):
    return v.item() if isinstance(v, np.generic) else v


def node_to_dict(node: Node) -> dict:
    d = {"leaf": bool(node.leaf),
         "class_weights": [float(x) for x in node.class_weights]}
    if node.leaf:
        return d
    d["feature"] = int(node.feature)
    if node.kind == MULTIWAY:
        d["kind"] = MULTIWAY
        d["children"] = [[_json_value(v), node_to_dict(ch)]
                         for v, ch in node.children.items()]
        d["shares"] = [[_json_value(v), float(s)]
                       for v, s in node.shares.items()]
        d["default"] = _json_value(node.default)
    else:
        d["kind"] = BINARY
        d["threshold"] = float(node.threshold)
        d["p_left"] = float(node.p_left)
        d["default"] = int(node.default)
        d["left"] = node_to_dict(node.left)
        d["right"] = node_to_dict(node.right)
    return d


def node_from_dict(d: dict) -> Node:
    node = Node()
    node.leaf = bool(d["leaf"])
    node.class_weights = np.asarray(d["class_weights"], dtype=float)
    if node.leaf:
        return node
    node.feature = int(d["feature"])
    node.kind = d["kind"]
    if node.kind == MULTIWAY:
        node.children = {v: node_from_dict(cd) for v, cd in d["children"]}
        node.shares = {v: float(s) for v, s in d["shares"]}
        node.default = d["default"]
    else:
        node.threshold = float(d["threshold"])
        node.p_left = float(d["p_left"])
        node.default = int(d["default"])
        node.left = node_from_dict(d["left"])
        node.right = node_from_dict(d["right"])
    return node


def save_model(path, params, classes, feature_names, kinds, pruning_report,
               root, named) -> None:
    payload = {
        "format": FORMAT,
        "version": VERSION,
        "params": params,
        "classes": [_json_value(c) for c in classes],
        "feature_names": [_json_value(n) for n in feature_names],
        "kinds": list(kinds),
        "named_fit": bool(named),
        "pruning_report": pruning_report,
        "root": node_to_dict(root),
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def load_model(path) -> dict:
    with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)
    if payload.get("format") != FORMAT:
        raise ValueError(f"{path} is not a {FORMAT} model file")
    if payload.get("version") != VERSION:
        raise ValueError(f"unsupported model version {payload.get('version')}")
    payload["root"] = node_from_dict(payload["root"])
    return payload
