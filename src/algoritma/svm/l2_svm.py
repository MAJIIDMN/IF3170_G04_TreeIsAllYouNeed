import pickle

import numpy as np

BLOCK_ROWS = 1500
CHANGE_TOL = 2e-4


def hamming_gram(A, B, weights, mixture):
    gram = np.zeros((len(A), len(B)), dtype=np.float32)
    for w, share in zip(weights, mixture):
        scaled = A * w
        exponent = scaled @ B.T
        exponent *= 2.0
        exponent -= scaled.sum(axis=1)[:, None]
        exponent -= (B @ w)[None]
        np.minimum(exponent, 0.0, out=exponent)
        np.exp(exponent, out=exponent)
        exponent *= np.float32(share)
        gram += exponent
    return gram


def blocked_gram(A, B, weights, mixture):
    gram = np.empty((len(A), len(B)), dtype=np.float32)
    for start in range(0, len(A), BLOCK_ROWS):
        gram[start:start + BLOCK_ROWS] = hamming_gram(A[start:start + BLOCK_ROWS], B, weights, mixture)
    return gram


# Sumber: M. Hestenes & E. Stiefel, J. Res. NBS 49(6), 1952 (conjugate gradient, tiap kolom rhs satu sistem).
def conjugate_gradient(multiply, rhs, x, tol, max_iter):
    residual = rhs - multiply(x)
    direction = residual.copy()
    norm = (residual * residual).sum(axis=0)
    target = tol ** 2 * np.maximum((rhs * rhs).sum(axis=0), 1e-30)
    for _ in range(max_iter):
        if (norm < target).all():
            break
        product = multiply(direction)
        step = norm / np.maximum((direction * product).sum(axis=0), 1e-30)
        x = x + step * direction
        residual -= step * product
        new_norm = (residual * residual).sum(axis=0)
        direction = residual + (new_norm / np.maximum(norm, 1e-30)) * direction
        norm = new_norm
    return x


# Sumber: O. Chapelle, "Training a Support Vector Machine in the Primal", Neural Computation 19(5), 2007
# (SVM dengan slack kuadrat, dilatih dengan Newton pada himpunan support vector);
# J. Suykens & J. Vandewalle, Neural Processing Letters 9, 1999 (loss="least_squares", LS-SVM).
class L2SVM:
    def __init__(self, weights, mixture=(1.0,), reg=1.0, loss="squared_hinge", tol=1e-5, cg_iter=500,
                 max_iter=30):
        if loss not in ("squared_hinge", "least_squares"):
            raise ValueError(f"loss must be squared_hinge or least_squares, got {loss!r}")
        self.weights = [np.asarray(w, dtype=np.float32) for w in weights]
        self.mixture = list(mixture)
        self.reg = reg
        self.loss = loss
        self.tol = tol
        self.cg_iter = cg_iter
        self.max_iter = max_iter

    def fit(self, X, y, gram=None):
        X = np.ascontiguousarray(X, dtype=np.float32)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError(f"L2SVM needs exactly two classes, got {len(self.classes_)}")
        sign = np.where(y == self.classes_[1], 1.0, -1.0)
        K = blocked_gram(X, X, self.weights, self.mixture) if gram is None else gram

        active = np.ones(len(y))
        solution = np.zeros((len(y), 2))
        for self.n_iter_ in range(1, self.max_iter + 1):
            mask = active[:, None]

            def multiply(v):
                return mask * (K @ (mask * v).astype(np.float32)) + self.reg * v

            rhs = mask * np.column_stack([sign, np.ones(len(y))])
            solution = conjugate_gradient(multiply, rhs, mask * solution, self.tol, self.cg_iter)
            bias = float((active @ solution[:, 0]) / (active @ solution[:, 1]))
            alpha = active * (solution[:, 0] - bias * solution[:, 1])
            if self.loss == "least_squares":
                break
            scores = K @ alpha.astype(np.float32) + bias
            new_active = (sign * scores < 1.0).astype(float)
            changed = int((new_active != active).sum())
            if changed <= CHANGE_TOL * len(y):
                break
            active = new_active

        support = alpha != 0
        self.support_ = np.flatnonzero(support)
        self.support_vectors_ = X[support].astype(bool)
        self.dual_coef_ = alpha[support].astype(np.float32)
        self.intercept_ = bias
        return self

    def decision_function(self, X):
        if not hasattr(self, "support_vectors_"):
            raise RuntimeError("Call fit before predicting")
        X = np.ascontiguousarray(X, dtype=np.float32)
        support = self.support_vectors_.astype(np.float32)
        return np.concatenate([
            hamming_gram(X[start:start + BLOCK_ROWS], support, self.weights, self.mixture) @ self.dual_coef_
            for start in range(0, len(X), BLOCK_ROWS)]).astype(float) + self.intercept_

    def predict(self, X, threshold=0.0):
        return self.classes_[(self.decision_function(X) > threshold).astype(int)]

    def __getstate__(self):
        state = dict(self.__dict__)
        if "support_vectors_" in state:
            state["n_bits_"] = state["support_vectors_"].shape[1]
            state["support_vectors_"] = np.packbits(state["support_vectors_"], axis=1)
        return state

    def __setstate__(self, state):
        bits = state.pop("n_bits_", None)
        self.__dict__.update(state)
        if bits is not None:
            self.support_vectors_ = np.unpackbits(state["support_vectors_"], axis=1, count=bits).astype(bool)

    def save(self, path):
        with open(path, "wb") as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as file:
            return pickle.load(file)
