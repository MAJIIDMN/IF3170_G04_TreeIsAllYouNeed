import pickle
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from algoritma.knn.knn import KNN
from kaggle.knn_submission import BookingPreprocessor, ORIGINAL, add_features, clean_training_data, restore_preprocessor


class TestKNN(unittest.TestCase):
    def test_distances_and_neighbor_order(self):
        X = np.array([[0, 0], [3, 4], [5, 0]], dtype=np.float32)
        y = np.array([0, 1, 1])
        manhattan = KNN(n_neighbors=3, metric="manhattan").fit(X, y)
        distances, neighbors = manhattan.kneighbors([[0, 0]])
        np.testing.assert_allclose(distances, [[0, 5, 7]])
        np.testing.assert_array_equal(neighbors, [[0, 2, 1]])
        euclidean = KNN(n_neighbors=3, metric="euclidean").fit(X, y)
        distances, _ = euclidean.kneighbors([[0, 0]])
        np.testing.assert_allclose(distances, [[0, 5, 5]])
        minkowski = KNN(n_neighbors=3, metric="minkowski", p=3).fit(X, y)
        distances, _ = minkowski.kneighbors([[0, 0]])
        self.assertAlmostEqual(distances[0, 1], (3**3 + 4**3)**(1/3))

    def test_majority_vote_and_probabilities(self):
        X = np.array([[0], [1], [2], [9], [10]], dtype=np.float32)
        y = np.array([0, 0, 1, 1, 1])
        model = KNN(n_neighbors=3, metric="manhattan").fit(X, y)
        self.assertEqual(model.predict([[1.1]])[0], 0)
        np.testing.assert_allclose(model.predict_proba([[1.1]]), [[2 / 3, 1 / 3]])

    def test_tie_uses_smallest_label(self):
        X = np.array([[0], [2]], dtype=np.float32)
        y = np.array([1, 0])
        model = KNN(n_neighbors=2).fit(X, y)
        self.assertEqual(model.predict([[1]])[0], 0)

    def test_matches_sklearn(self):
        rng = np.random.default_rng(42)
        X = rng.random((40, 5), dtype=np.float32)
        y = rng.integers(0, 2, size=40)
        query = rng.random((12, 5), dtype=np.float32)
        for metric in ("manhattan", "euclidean", "minkowski"):
            model = KNN(n_neighbors=5, metric=metric, p=3).fit(X, y)
            reference = KNeighborsClassifier(n_neighbors=5, metric=metric, p=3).fit(X, y)
            np.testing.assert_array_equal(model.predict(query), reference.predict(query))

    def test_predict_many_k_matches_individual_models(self):
        rng = np.random.default_rng(17)
        features = rng.normal(size=(52, 4)).astype(np.float32)
        labels = rng.integers(0, 2, size=52)
        query = rng.normal(size=(11, 4)).astype(np.float32)
        for metric in ["manhattan", "euclidean", "minkowski"]:
            multi = KNN(n_neighbors=7, metric=metric, p=3).fit(features, labels)
            predictions = multi.predict_many_k(query, [1, 3, 5, 7])
            for k in [1, 3, 5, 7]:
                individual = KNN(n_neighbors=k, metric=metric, p=3).fit(features, labels)
                np.testing.assert_array_equal(predictions[k], individual.predict(query))

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            KNN(n_neighbors=0)
        with self.assertRaises(ValueError):
            KNN(metric="unknown")
        with self.assertRaises(RuntimeError):
            KNN().predict([[1]])
        model = KNN(n_neighbors=2).fit([[0], [1]], [0, 1])
        with self.assertRaises(ValueError):
            model.predict([[0, 1]])
        with self.assertRaises(ValueError):
            model.predict([[np.nan]])

    def test_preprocessing_uses_training_only(self):
        data = pd.DataFrame({name: [1, 2, 3] for name in ORIGINAL})
        data["meal_package"] = ["Breakfast", None, "Room Only"]
        data["room_category"] = ["Type B", "Type B", "Type A"]
        data["booking_channel"] = ["Online", "Online", "Offline"]
        training = add_features(data.iloc[:2])
        validation = add_features(data.iloc[2:])
        prep = BookingPreprocessor().fit(training)
        X = prep.transform(training, ["num_adults", "meal_package"])
        V = prep.transform(validation, ["num_adults", "meal_package"])
        self.assertEqual(X.shape[1], V.shape[1])
        self.assertGreater(float(V[0, 0]), 1.0)
        self.assertEqual(prep.categorical_stats["meal_package"][0], "Breakfast")

    def test_model_pickle_roundtrip(self):
        model = KNN(n_neighbors=1).fit([[0, 0], [2, 2]], [0, 1])
        reloaded = pickle.loads(pickle.dumps(model))
        np.testing.assert_array_equal(model.predict([[1.8, 2]]), reloaded.predict([[1.8, 2]]))


    def test_preprocessor_state_survives_pickle(self):
        data = pd.DataFrame({name: [1, 2] for name in ORIGINAL})
        data["meal_package"] = ["Breakfast", "Room Only"]
        data["room_category"] = ["Type A", "Type B"]
        data["booking_channel"] = ["Online", "Offline"]
        prepared = add_features(data)
        preprocessor = BookingPreprocessor().fit(prepared)
        state = {
            "numeric_stats": preprocessor.numeric_stats,
            "categorical_stats": preprocessor.categorical_stats
        }
        restored = restore_preprocessor(pickle.loads(pickle.dumps(state)))
        np.testing.assert_array_equal(
            preprocessor.transform(prepared, ORIGINAL),
            restored.transform(prepared, ORIGINAL)
        )

    def test_duplicate_cleaning_excludes_identifier(self):
        data = pd.DataFrame({name: [1, 1] for name in ORIGINAL})
        data["id"] = ["one", "two"]
        data["is_canceled"] = [0, 0]
        self.assertEqual(len(clean_training_data(data)), 1)


if __name__ == "__main__":
    unittest.main()
