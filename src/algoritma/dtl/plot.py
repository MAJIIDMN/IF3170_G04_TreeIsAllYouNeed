from __future__ import annotations

import heapq

from .tree import MULTIWAY, SUBSET, children_of


def top_branchings(root, top_n=None):
    if top_n is not None and top_n < 1:
        raise ValueError(f"top_n must be >= 1 or None, got {top_n}")
    shown, heap, order = [], [], 0
    if not root.leaf:
        heap.append((0.0, order, root))
    while heap and (top_n is None or len(shown) < top_n):
        _, _, node = heapq.heappop(heap)
        shown.append(node)
        for child in children_of(node):
            if not child.leaf:
                order += 1
                heapq.heappush(heap, (-float(child.class_weights.sum()), order, child))
    return shown


def _edge_labels(node):
    if node.kind == MULTIWAY:
        return [str(v) for v in node.children]
    return ["ya", "tidak"]


def _split_text(node, names):
    name = str(names[node.feature])
    if node.kind == MULTIWAY:
        return f"{name} ?"
    if node.kind == SUBSET:
        return f"{name} in\n{{{', '.join(sorted(map(str, node.subset)))}}}"
    return f"{name}\n<= {node.threshold:.4g}"


def plot_tree(root, names, classes, path, top_n=15, dpi=160):
    from matplotlib import colormaps
    from matplotlib.figure import Figure

    shown = set(top_branchings(root, top_n))
    position, cursor, deepest = {}, 0.0, 0

    def place(node, depth):
        nonlocal cursor, deepest
        deepest = max(deepest, depth)
        if node in shown:
            xs = [place(child, depth + 1) for child in children_of(node)]
            x = (xs[0] + xs[-1]) / 2.0
        else:
            x = cursor
            cursor += 1.0
        position[node] = (x, -float(depth))
        return x

    place(root, 0)
    fig = Figure(figsize=(max(6.0, 1.35 * cursor), max(3.0, 1.35 * (deepest + 1))))
    ax = fig.add_subplot()
    palette = colormaps["Set2"]
    for node, (x, y) in position.items():
        weights = node.class_weights
        total = float(weights.sum())
        k = node.plurality()
        text = f"n={total:.0f}\n{classes[k]} ({weights[k] / total if total else 0.0:.0%})"
        if node in shown:
            text = f"{_split_text(node, names)}\n{text}"
            for child, label in zip(children_of(node), _edge_labels(node)):
                cx, cy = position[child]
                ax.plot([x, cx], [y, cy], color="0.55", lw=1.0, zorder=1)
                ax.text((x + cx) / 2.0, (y + cy) / 2.0, label, fontsize=7, ha="center", va="center", zorder=3,
                        bbox=dict(facecolor="white", edgecolor="none", pad=0.6))
            style = dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="0.25")
        else:
            style = dict(boxstyle="round,pad=0.35", facecolor=palette(k % 8), edgecolor="0.25",
                         linestyle="-" if node.leaf else "--")
            if not node.leaf:
                text += "\n..."
        ax.text(x, y, text, fontsize=7, ha="center", va="center", zorder=2, bbox=style)
    ax.set_xlim(-0.7, cursor - 0.3)
    ax.set_ylim(-deepest - 0.5, 0.5)
    ax.axis("off")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return len(shown)
