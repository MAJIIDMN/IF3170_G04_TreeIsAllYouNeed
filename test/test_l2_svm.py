import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "algoritma" / "svm"))

from l2_svm import L2SVM, hamming_gram  # noqa: E402


def make_data(n=400, bits=20, seed=0):
    rng = np.random.default_rng(seed)
    X = (rng.random((n, bits)) > 0.5).astype(np.float32)
    y = (X[:, 0] + X[:, 1] * X[:, 2] + rng.normal(0, 0.5, n) > 0.9).astype(int)
    return X, y, [np.full(bits, 0.1), np.full(bits, 0.4)], [0.6, 0.4]


class L2SVMTest(unittest.TestCase):
    def test_gram_matches_weighted_hamming_distance(self):
        X, _, weights, mixture = make_data(30)
        K = hamming_gram(X, X, [w.astype(np.float32) for w in weights], mixture)
        distances = np.abs(X[:, None] - X[None]).astype(float)
        expected = sum(m * np.exp(-(distances * w).sum(axis=2)) for w, m in zip(weights, mixture))
        np.testing.assert_allclose(K, expected, atol=1e-5)

    def test_squared_hinge_optimality_conditions(self):
        X, y, weights, mixture = make_data()
        model = L2SVM(weights, mixture, reg=0.5, tol=1e-7).fit(X, y)
        sign = np.where(y == 1, 1.0, -1.0)
        alpha = np.zeros(len(y))
        alpha[model.support_] = model.dual_coef_
        scores = model.decision_function(X)
        self.assertLess(abs(alpha.sum()), 1e-3)
        np.testing.assert_allclose(0.5 * alpha, sign * np.maximum(0, 1 - sign * scores), atol=1e-3)

    def test_least_squares_residual_is_alpha(self):
        X, y, weights, mixture = make_data()
        model = L2SVM(weights, mixture, reg=1.0, loss="least_squares", tol=1e-7).fit(X, y)
        residual = np.where(y == 1, 1.0, -1.0) - model.decision_function(X)
        np.testing.assert_allclose(model.dual_coef_, residual[model.support_], atol=1e-3)

    def test_save_and_load(self):
        X, y, weights, mixture = make_data()
        model = L2SVM(weights, mixture).fit(X, y)
        with tempfile.TemporaryDirectory() as folder:
            model.save(Path(folder) / "model.pkl")
            loaded = L2SVM.load(Path(folder) / "model.pkl")
        np.testing.assert_array_equal(loaded.decision_function(X), model.decision_function(X))

    def test_unknown_loss_is_rejected(self):
        with self.assertRaises(ValueError):
            L2SVM([np.ones(3)], loss="hinge")


if __name__ == "__main__":
    unittest.main()
