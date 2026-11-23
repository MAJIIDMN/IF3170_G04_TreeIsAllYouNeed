"""Melatih DTL dan menyimpan modelnya.

Guru (HistGradientBoosting dan RandomForest dari scikit-learn) hanya dipakai di sini untuk
memberi label. Model akhir adalah satu pohon C4.5 yang diimplementasikan sendiri.

Lingkungan saat model dilatih: Python 3.14.6, numpy 2.4.6, pandas 3.0.3, scikit-learn 1.9.0,
numba 0.65.1. numba tidak ada di requirements.txt karena notebook tidak membutuhkannya.

Referensi:
- L. Breiman dan N. Shang, "Born Again Trees," UC Berkeley, 1996.
- T. Vidal dan M. Schiffer, "Born-Again Tree Ensembles," ICML 2020, arXiv:2003.11132.
- C. Bucila, R. Caruana, A. Niculescu-Mizil, "Model Compression," KDD 2006 (data sintetis).
- J. R. Quinlan, C4.5: Programs for Machine Learning, 1993 (gain ratio).
- L. Breiman dkk., Classification and Regression Trees, 1984 (cost-complexity pruning).
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "algoritma"))
sys.path.insert(0, str(ROOT / "src" / "kaggle"))

from dtl import DTL  # noqa: E402
from dtl.tree import BINARY, CONTINUOUS, Node, assign_leaf_ids, tree_depth  # noqa: E402
from dtl_new_features import TARGET, build_features, clean, load_data  # noqa: E402

DATA_DIR = ROOT / "src" / "kaggle" / "dataset"
OUTPUT_DIR = ROOT / "output" / "dtl_new"
THRESHOLD = 0.4
COPIES, SYNTHETIC_WEIGHT = 12, 0.4
MAX_DEPTH, MIN_LEAF, CCP_ALPHA = 30, 5, 2.5


class Teacher:
    def fit(self, X, y):
        self.boosters = [
            HistGradientBoostingClassifier(max_iter=300, learning_rate=0.04, max_leaf_nodes=24,
                                           l2_regularization=3, random_state=seed,
                                           max_features=0.5).fit(X, y) for seed in range(3)]
        self.forest = RandomForestClassifier(80, min_samples_leaf=2, max_features=0.4,
                                             random_state=0).fit(X, y)
        return self

    def proba(self, X):
        boosted = np.mean([m.predict_proba(X)[:, 1] for m in self.boosters], 0)
        return 0.6 * boosted + 0.4 * self.forest.predict_proba(X)[:, 1]


def out_of_fold_teacher(X, y):
    proba = np.zeros(len(X))
    for held in np.array_split(np.random.default_rng(2).permutation(len(X)), 5):
        rest = np.setdiff1d(np.arange(len(X)), held)
        proba[held] = Teacher().fit(X[rest], y[rest]).proba(X[held])
    return proba


def synthesize(A, columns, copies, rng, swap=0.2, lead_sd=15, rate_sd=0.1):
    """Salin baris, geser lead time dan harga, lalu tukar sebagian kolom antarbaris."""
    at = {name: k for k, name in enumerate(columns)}
    lead, book, rate = at["days_before_arrival"], at["book_ord"], at["nightly_rate_idr"]
    out = []
    for _ in range(copies):
        Z, m = A.copy(), len(A)
        shift = np.where(rng.random(m) < 0.7, rng.normal(0, lead_sd, m), 0)
        new_lead = np.maximum(Z[:, lead] + shift, 0)
        Z[:, book] -= new_lead - Z[:, lead]
        Z[:, lead] = new_lead
        Z[:, rate] *= np.where(rng.random(m) < 0.7, np.exp(rng.normal(0, rate_sd, m)), 1)
        for name in ("special_requests", "num_adults", "num_children", "needs_parking"):
            z, other = rng.random(m) < swap, rng.permutation(m)
            Z[z, at[name]] = A[other[z], at[name]]
        for prefix in ("room_category_", "meal_package_", "booking_channel_"):
            cols = [k for name, k in at.items() if name.startswith(prefix)]
            share = swap * (0.3 if prefix == "booking_channel_" else 1)
            z, other = rng.random(m) < share, rng.permutation(m)
            Z[np.ix_(z, cols)] = A[np.ix_(other[z], cols)]
        for names, share in ((("sib_te", "sib_n"), swap),
                             (("weekend_nights", "weekday_nights", "nights", "weekend_share"),
                              swap * 0.5)):
            cols = [at[name] for name in names]
            z, other = rng.random(m) < share, rng.permutation(m)
            Z[np.ix_(z, cols)] = A[np.ix_(other[z], cols)]
        out.append(Z)
    return np.vstack(out)


@njit(cache=True)
def _entropy(a, b):
    t, h = a + b, 0.0
    if t <= 0:
        return 0.0
    if a > 0:
        h -= a / t * np.log2(a / t)
    if b > 0:
        h -= b / t * np.log2(b / t)
    return h


@njit(cache=True)
def grow(X, w0, w1, order, max_depth, min_leaf):
    """C4.5: gain ratio di antara fitur yang gain-nya di atas rata-rata. Semua fitur numerik."""
    p, n = X.shape[1], order.shape[1]
    size = 2 * n + 5
    feature = np.full(size, -1, np.int32)
    threshold = np.zeros(size)
    left, right = np.zeros(size, np.int32), np.zeros(size, np.int32)
    counts = np.zeros((size, 2))
    stack = np.zeros((size, 4), np.int64)
    stack[0] = np.array([0, 0, n, 0])
    top, used = 1, 1
    gains, ratios, cuts = np.zeros(p), np.zeros(p), np.zeros(p)
    buf = np.zeros(n, order.dtype)
    while top > 0:
        top -= 1
        node, lo, hi, depth = stack[top]
        a = b = 0.0
        for q in range(lo, hi):
            a += w0[order[0, q]]
            b += w1[order[0, q]]
        counts[node, 0], counts[node, 1] = a, b
        if hi - lo < 2 * min_leaf or a <= 1e-12 or b <= 1e-12 or depth >= max_depth:
            continue
        T, H, s, k = a + b, _entropy(a, b), 0.0, 0
        for f in range(p):
            gains[f], ratios[f] = -1.0, 0.0
            l0 = l1 = 0.0
            for q in range(lo, hi - 1):
                i = order[f, q]
                l0 += w0[i]
                l1 += w1[i]
                x, nx = X[i, f], X[order[f, q + 1], f]
                if nx <= x or q - lo + 1 < min_leaf or hi - q - 1 < min_leaf:
                    continue
                L = l0 + l1
                R = T - L
                g = H - (L / T * _entropy(l0, l1) + R / T * _entropy(a - l0, b - l1))
                if g > gains[f]:
                    gains[f], cuts[f] = g, x + (nx - x) / 2
                    ratios[f] = g / _entropy(L, R) if L > 0 and R > 0 else 0.0
            if gains[f] <= 1e-7:
                gains[f] = -1.0
            else:
                s += gains[f]
                k += 1
        if k == 0:
            continue
        best, score = -1, -1.0
        for f in range(p):
            if gains[f] > 0 and gains[f] >= s / k and ratios[f] > score:
                best, score = f, ratios[f]
        if best < 0:
            continue
        mid = lo
        for f in range(p):
            u = v = 0
            u = lo
            for q in range(lo, hi):
                i = order[f, q]
                if X[i, best] <= cuts[best]:
                    order[f, u] = i
                    u += 1
                else:
                    buf[v] = i
                    v += 1
            for q in range(v):
                order[f, u + q] = buf[q]
            mid = u
        feature[node], threshold[node] = best, cuts[best]
        left[node], right[node] = used, used + 1
        stack[top] = np.array([used + 1, mid, hi, depth + 1])
        stack[top + 1] = np.array([used, lo, mid, depth + 1])
        top += 2
        used += 2
    return feature[:used], threshold[:used], left[:used], right[:used], counts[:used]


def prune(feature, left, right, counts, alpha):
    """Cost-complexity pruning satu lintasan; indeks anak selalu lebih besar dari induknya."""
    leaves, errors = np.ones(len(feature)), counts.min(axis=1)
    for node in range(len(feature) - 1, -1, -1):
        if feature[node] < 0:
            continue
        sub_leaves = leaves[left[node]] + leaves[right[node]]
        sub_errors = errors[left[node]] + errors[right[node]]
        if errors[node] - sub_errors <= alpha * (sub_leaves - 1) + 1e-12:
            feature[node] = -1
        else:
            errors[node], leaves[node] = sub_errors, sub_leaves


def to_dtl(feature, threshold, left, right, counts, names):
    def convert(i):
        node = Node()
        node.class_weights = counts[i].copy()
        if feature[i] < 0:
            return node
        node.leaf, node.kind = False, BINARY
        node.feature, node.threshold = int(feature[i]), float(threshold[i])
        node.left, node.right = convert(left[i]), convert(right[i])
        node.p_left = float(counts[left[i]].sum() / counts[i].sum())
        node.default = 0 if node.p_left >= 0.5 else 1
        return node

    model = DTL(max_depth=MAX_DEPTH, min_samples_leaf=MIN_LEAF, pruning="cost_complexity",
                ccp_alpha=CCP_ALPHA)
    model.root_, model.classes_ = convert(0), np.array([0, 1])
    model.feature_names_, model.kinds_ = list(names), [CONTINUOUS] * len(names)
    model._named_fit_, model.pruning_report_ = True, None
    model.n_leaves_, model.depth_ = assign_leaf_ids(model.root_), tree_depth(model.root_)
    return model


def fit_tree(X, y):
    columns, A, y = list(X.columns), np.ascontiguousarray(X.to_numpy(float)), np.asarray(y, float)
    teacher = Teacher().fit(A, y)
    soft = 0.5 * out_of_fold_teacher(A, y) + 0.5 * y
    Z = synthesize(A, columns, COPIES, np.random.default_rng(3))
    hard = (teacher.proba(Z) >= 0.5).astype(float)
    rows = np.vstack([A, Z])
    target = np.r_[soft, hard]
    weight = np.r_[np.ones(len(A)), np.full(len(Z), SYNTHETIC_WEIGHT)]
    order = np.argsort(rows, axis=0, kind="stable").T.astype(np.int32).copy()
    tree = grow(rows, weight * (1 - target), weight * target, order, MAX_DEPTH, MIN_LEAF)
    prune(tree[0], tree[2], tree[3], tree[4], CCP_ALPHA)
    return to_dtl(*tree, columns)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_raw, test = load_data(DATA_DIR)
    train = clean(train_raw)
    X, X_test = build_features(train, train[TARGET], test)
    model = fit_tree(X, train[TARGET])
    model.save(OUTPUT_DIR / "dtl_model.json")
    labels = (model.predict_proba(X_test)[:, 1] >= THRESHOLD).astype(int)
    pd.DataFrame({"id": test["id"], TARGET: labels}).to_csv(OUTPUT_DIR / "submission.csv", index=False)
    print(f"daun={model.n_leaves_} kedalaman={model.depth_} batal={labels.mean():.4f}")


if __name__ == "__main__":
    main()
