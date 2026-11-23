import pickle
import warnings
from collections import OrderedDict

import numpy as np

KERNELS = ("linear", "poly", "rbf", "laplacian")

# Sumber: C.-C. Chang & C.-J. Lin, "LIBSVM: A Library for Support Vector Machines", ACM TIST 2(3), 2011.
TAU = 1e-12

PREDICT_BATCH_ROWS = 4096
LAPLACIAN_BATCH_ROWS = 256


def _l1_distances(A, B):
    if len(A) == 1:
        return np.abs(B - A[0]).sum(axis=1)[None]
    distances = np.zeros((len(A), len(B)))
    for column in range(A.shape[1]):
        distances += np.abs(A[:, column, None] - B[None, :, column])
    return distances


class _KernelRowCache:
    def __init__(self, compute_row, n_samples, cache_mb):
        self._compute_row = compute_row
        self._capacity = max(2, int(cache_mb * 1e6 // (8 * n_samples)))
        self._rows = OrderedDict()

    def __getitem__(self, index):
        row = self._rows.get(index)
        if row is not None:
            self._rows.move_to_end(index)
            return row
        row = self._compute_row(index)
        self._rows[index] = row
        if len(self._rows) > self._capacity:
            self._rows.popitem(last=False)
        return row


class SVM:
    def __init__(self, C=1.0, kernel="rbf", degree=3, gamma="scale", coef0=0.0, tol=1e-3,
                 max_iter=10_000_000, class_weight=None, cache_mb=1024):
        if kernel not in KERNELS:
            raise ValueError(f"kernel must be one of {KERNELS}, got {kernel!r}")
        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.tol = tol
        self.max_iter = max_iter
        self.class_weight = class_weight
        self.cache_mb = cache_mb

    def fit(self, X, y):
        X = self._as_matrix(X)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError(f"SVM needs exactly two classes, got {len(self.classes_)}")

        sign = np.where(y == self.classes_[1], 1.0, -1.0)
        upper = self._box_limits(y)
        self.gamma_ = self._resolve_gamma(X)

        alpha, bias = self._solve(X, sign, upper)

        support = alpha > 0
        self.support_ = np.flatnonzero(support)
        self.support_vectors_ = X[support]
        self.dual_coef_ = (alpha * sign)[support]
        self.intercept_ = bias
        self.coef_ = self.dual_coef_ @ self.support_vectors_ if self.kernel == "linear" else None
        return self

    def decision_function(self, X):
        self._check_fitted()
        X = self._as_matrix(X)
        support_sq = np.einsum("ij,ij->i", self.support_vectors_, self.support_vectors_)
        scores = np.empty(len(X))
        batch_rows = LAPLACIAN_BATCH_ROWS if self.kernel == "laplacian" else PREDICT_BATCH_ROWS
        for start in range(0, len(X), batch_rows):
            batch = slice(start, start + batch_rows)
            scores[batch] = self._gram(X[batch], self.support_vectors_, support_sq) @ self.dual_coef_
        return scores + self.intercept_

    def predict(self, X, threshold=0.0):
        scores = self.decision_function(X)
        return self.classes_[(scores > threshold).astype(int)]

    def save(self, path):
        with open(path, "wb") as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as file:
            return pickle.load(file)

    # Sumber: J. Platt, "Sequential Minimal Optimization", MSR-TR-98-14, 1998;
    # R.-E. Fan, P.-H. Chen & C.-J. Lin, "Working Set Selection Using Second Order Information", JMLR 6, 2005.
    def _solve(self, X, sign, upper):
        n_samples = len(X)
        squared_norms = np.einsum("ij,ij->i", X, X)
        rows = _KernelRowCache(
            lambda index: self._gram(X[index:index + 1], X, squared_norms)[0],
            n_samples,
            self.cache_mb,
        )
        diagonal = self._gram_diagonal(squared_norms)

        alpha = np.zeros(n_samples)
        score = sign.copy()

        can_rise = sign > 0
        can_fall = sign < 0

        self.n_iter_ = 0
        converged = False
        while self.n_iter_ < self.max_iter:
            i = int(np.argmax(np.where(can_rise, score, -np.inf)))
            largest = score[i]
            smallest = np.where(can_fall, score, np.inf).min()
            if largest - smallest < self.tol:
                converged = True
                break

            row_i = rows[i]

            gap = largest - score
            curvature = diagonal[i] + diagonal - 2.0 * row_i
            curvature = np.where(curvature > 0, curvature, TAU)
            gain = np.where(can_fall & (gap > 0), gap * gap / curvature, -np.inf)
            j = int(np.argmax(gain))

            row_j = rows[j]

            step = gap[j] / curvature[j]
            room_i = upper[i] - alpha[i] if sign[i] > 0 else alpha[i]
            room_j = alpha[j] if sign[j] > 0 else upper[j] - alpha[j]
            step = min(step, room_i, room_j)

            alpha[i] = self._move(alpha[i], sign[i] * step, step == room_i, upper[i])
            alpha[j] = self._move(alpha[j], -sign[j] * step, step == room_j, upper[j])
            score -= step * (row_i - row_j)

            for index in (i, j):
                is_lower = alpha[index] > 0
                is_below_upper = alpha[index] < upper[index]
                positive = sign[index] > 0
                can_rise[index] = is_below_upper if positive else is_lower
                can_fall[index] = is_lower if positive else is_below_upper

            self.n_iter_ += 1

        if not converged:
            warnings.warn(f"SMO stopped at max_iter={self.max_iter} before reaching tol={self.tol}")
            largest = score[can_rise].max()
            smallest = score[can_fall].min()

        free = (alpha > 0) & (alpha < upper)
        bias = score[free].mean() if free.any() else (largest + smallest) / 2.0
        return alpha, bias

    @staticmethod
    def _move(value, delta, hits_bound, upper):
        if hits_bound:
            return upper if delta > 0 else 0.0
        return value + delta

    def _gram(self, A, B, B_squared_norms):
        # Sumber: B. Scholkopf & A. Smola, "Learning with Kernels", MIT Press, 2002 (kernel Laplacian).
        if self.kernel == "laplacian":
            return np.exp(-self.gamma_ * _l1_distances(A, B))
        dot = A @ B.T
        if self.kernel == "linear":
            return dot
        if self.kernel == "poly":
            return (self.gamma_ * dot + self.coef0) ** self.degree
        A_squared_norms = np.einsum("ij,ij->i", A, A)
        distances = A_squared_norms[:, None] + B_squared_norms - 2.0 * dot
        return np.exp(-self.gamma_ * np.maximum(distances, 0.0))

    def _gram_diagonal(self, squared_norms):
        if self.kernel == "linear":
            return squared_norms
        if self.kernel == "poly":
            return (self.gamma_ * squared_norms + self.coef0) ** self.degree
        return np.ones_like(squared_norms)

    def _resolve_gamma(self, X):
        if self.gamma == "scale":
            return 1.0 / (X.shape[1] * X.var())
        if self.gamma == "auto":
            return 1.0 / X.shape[1]
        return float(self.gamma)

    def _box_limits(self, y):
        if self.class_weight is None:
            return np.full(len(y), float(self.C))
        if self.class_weight == "balanced":
            counts = {label: (y == label).sum() for label in self.classes_}
            weights = {label: len(y) / (2.0 * count) for label, count in counts.items()}
        else:
            weights = self.class_weight
        upper = np.empty(len(y))
        for label in self.classes_:
            upper[y == label] = self.C * weights.get(label.item(), 1.0)
        return upper

    def _check_fitted(self):
        if not hasattr(self, "support_vectors_"):
            raise RuntimeError("Call fit before predicting")

    @staticmethod
    def _as_matrix(X):
        X = np.ascontiguousarray(X, dtype=np.float64)
        if not np.isfinite(X).all():
            raise ValueError("X contains NaN or infinity, impute before using SVM")
        return X
