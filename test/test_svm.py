import sys
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np
from sklearn.datasets import make_blobs, make_classification, make_moons
from sklearn.metrics.pairwise import laplacian_kernel
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "algoritma" / "svm"))

from svm import SVM  # noqa: E402


def noisy_data(n=400, seed=0):
    X, y = make_classification(
        n_samples=n, n_features=8, n_informative=5, weights=[0.67], flip_y=0.1, random_state=seed
    )
    return StandardScaler().fit_transform(X), y


class SeparableDataTest(unittest.TestCase):
    def test_linear_kernel_classifies_separable_blobs_perfectly(self):
        X, y = make_blobs(n_samples=100, centers=[[-3, -3], [3, 3]], cluster_std=0.5, random_state=0)
        model = SVM(kernel="linear").fit(X, y)
        self.assertTrue(np.array_equal(model.predict(X), y))

    def test_gaussian_kernel_solves_moons_that_linear_cannot(self):
        X, y = make_moons(n_samples=300, noise=0.05, random_state=0)
        gaussian = (SVM(kernel="rbf", C=10).fit(X, y).predict(X) == y).mean()
        linear = (SVM(kernel="linear", C=10).fit(X, y).predict(X) == y).mean()
        self.assertGreater(gaussian, 0.99)
        self.assertLess(linear, 0.95)


class MatchesLibsvmTest(unittest.TestCase):
    CONFIGS = [
        {"kernel": "linear", "C": 1.0},
        {"kernel": "linear", "C": 0.1},
        {"kernel": "poly", "C": 1.0, "degree": 3, "coef0": 1.0},
        {"kernel": "poly", "C": 5.0, "degree": 2, "coef0": 0.5, "gamma": 0.2},
        {"kernel": "rbf", "C": 1.0},
        {"kernel": "rbf", "C": 10.0, "gamma": 0.05},
        {"kernel": "rbf", "C": 1.0, "class_weight": "balanced"},
        {"kernel": "rbf", "C": 1.0, "class_weight": {0: 1.0, 1: 3.0}},
    ]

    def test_decision_values_match_sklearn(self):
        X, y = noisy_data()
        X_new, _ = noisy_data(n=200, seed=1)
        for config in self.CONFIGS:
            with self.subTest(config=config):
                ours = SVM(tol=1e-5, **config).fit(X, y).decision_function(X_new)
                reference = SVC(tol=1e-5, **config).fit(X, y).decision_function(X_new)
                np.testing.assert_allclose(ours, reference, atol=5e-3)

    def test_support_vector_count_matches_sklearn(self):
        X, y = noisy_data()
        for config in self.CONFIGS:
            with self.subTest(config=config):
                ours = SVM(tol=1e-5, **config).fit(X, y)
                reference = SVC(tol=1e-5, **config).fit(X, y)
                self.assertLessEqual(abs(len(ours.support_vectors_) - reference.n_support_.sum()), 2)

    def test_laplacian_kernel_matches_sklearn(self):
        X, y = noisy_data()
        X_new, _ = noisy_data(n=200, seed=1)
        ours = SVM(kernel="laplacian", gamma=0.1, C=2.0, tol=1e-5).fit(X, y).decision_function(X_new)
        reference = SVC(kernel=lambda A, B: laplacian_kernel(A, B, gamma=0.1), C=2.0, tol=1e-5).fit(X, y)
        np.testing.assert_allclose(ours, reference.decision_function(X_new), atol=5e-3)

    def test_gamma_scale_uses_feature_variance(self):
        X, y = noisy_data()
        X = X * 7.0
        self.assertAlmostEqual(SVM().fit(X, y).gamma_, 1.0 / (X.shape[1] * X.var()))

    def test_gamma_auto_uses_feature_count(self):
        X, y = noisy_data()
        self.assertAlmostEqual(SVM(gamma="auto").fit(X, y).gamma_, 1.0 / X.shape[1])


class DualConstraintTest(unittest.TestCase):
    def test_dual_coefficients_sum_to_zero(self):
        X, y = noisy_data()
        model = SVM(kernel="rbf").fit(X, y)
        self.assertAlmostEqual(model.dual_coef_.sum(), 0.0, places=8)

    def test_dual_coefficients_respect_box_with_class_weights(self):
        X, y = noisy_data()
        model = SVM(C=2.0, class_weight={0: 1.0, 1: 3.0}).fit(X, y)
        positive = model.dual_coef_[model.dual_coef_ > 0]
        negative = model.dual_coef_[model.dual_coef_ < 0]
        self.assertLessEqual(positive.max(), 6.0 + 1e-12)
        self.assertGreaterEqual(negative.min(), -2.0 - 1e-12)

    def test_linear_coef_reproduces_decision_function(self):
        X, y = noisy_data()
        model = SVM(kernel="linear").fit(X, y)
        np.testing.assert_allclose(X @ model.coef_ + model.intercept_, model.decision_function(X))


class CacheTest(unittest.TestCase):
    def test_tiny_cache_gives_same_model_as_large_cache(self):
        X, y = noisy_data()
        tiny = SVM(cache_mb=0.01).fit(X, y)
        large = SVM(cache_mb=100).fit(X, y)
        self.assertEqual(tiny.n_iter_, large.n_iter_)
        np.testing.assert_array_equal(tiny.decision_function(X), large.decision_function(X))


class LabelTest(unittest.TestCase):
    def test_predict_returns_original_labels(self):
        X, y = noisy_data()
        labels = np.where(y == 1, "batal", "tidak")
        self.assertEqual(set(SVM().fit(X, labels).predict(X)), {"batal", "tidak"})

    def test_decision_is_positive_for_larger_label(self):
        X, y = noisy_data()
        model = SVM().fit(X, y)
        np.testing.assert_array_equal(model.predict(X), (model.decision_function(X) > 0).astype(int))

    def test_threshold_shifts_prediction(self):
        X, y = noisy_data()
        model = SVM().fit(X, y)
        self.assertGreater(model.predict(X, threshold=-0.5).sum(), model.predict(X).sum())


class PersistenceTest(unittest.TestCase):
    def test_loaded_model_predicts_identically(self):
        X, y = noisy_data()
        model = SVM(kernel="poly").fit(X, y)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "svm.pkl"
            model.save(path)
            loaded = SVM.load(path)
        np.testing.assert_array_equal(model.decision_function(X), loaded.decision_function(X))


class ValidationTest(unittest.TestCase):
    def test_unknown_kernel_is_rejected(self):
        with self.assertRaises(ValueError):
            SVM(kernel="sigmoid")

    def test_single_class_is_rejected(self):
        with self.assertRaises(ValueError):
            SVM().fit(np.zeros((4, 2)), np.ones(4))

    def test_more_than_two_classes_is_rejected(self):
        with self.assertRaises(ValueError):
            SVM().fit(np.arange(6.0).reshape(3, 2), np.array([0, 1, 2]))

    def test_missing_values_are_rejected(self):
        X, y = noisy_data()
        X[0, 0] = np.nan
        with self.assertRaises(ValueError):
            SVM().fit(X, y)

    def test_predict_before_fit_is_rejected(self):
        with self.assertRaises(RuntimeError):
            SVM().predict(np.zeros((1, 2)))

    def test_iteration_limit_warns_and_keeps_a_usable_model(self):
        X, y = noisy_data()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = SVM(max_iter=3).fit(X, y)
        self.assertTrue(any("max_iter" in str(w.message) for w in caught))
        self.assertEqual(model.n_iter_, 3)
        self.assertEqual(len(model.predict(X)), len(X))

    def test_converged_fit_does_not_warn(self):
        X, y = noisy_data()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            SVM().fit(X, y)


if __name__ == "__main__":
    unittest.main()
