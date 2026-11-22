import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src" / "algoritma"))

from dtl import DTL
from dtl.criteria import (entropy, information_gain, select_split)  # noqa: E402
from dtl.splitting import (best_split_categorical,  # noqa: E402
                           best_split_continuous)


def playtennis():
    outlook = ["Sunny", "Sunny", "Overcast", "Rain", "Rain", "Rain",
               "Overcast", "Sunny", "Sunny", "Rain", "Sunny", "Overcast",
               "Overcast", "Rain"]
    temp = ["Hot", "Hot", "Hot", "Mild", "Cool", "Cool", "Cool", "Mild",
            "Cool", "Mild", "Mild", "Mild", "Hot", "Mild"]
    humidity = ["High", "High", "High", "High", "Normal", "Normal", "Normal",
                "High", "Normal", "Normal", "Normal", "High", "Normal",
                "High"]
    wind = ["Weak", "Strong", "Weak", "Weak", "Weak", "Strong", "Strong",
            "Weak", "Weak", "Weak", "Strong", "Strong", "Weak", "Strong"]
    play = ["No", "No", "Yes", "Yes", "Yes", "No", "Yes", "No", "Yes", "Yes",
            "Yes", "Yes", "Yes", "No"]
    X = np.array([outlook, temp, humidity, wind]).T
    y = np.array([1 if p == "Yes" else 0 for p in play])
    return X, y


def mixed_data(seed=0, n=240):
    rng = np.random.default_rng(seed)
    clean = rng.normal(size=n)
    cat = rng.choice(["u", "v", "w"], size=n)
    y = ((clean > 0) ^ (cat == "v")).astype(int)
    X = np.empty((n, 2), dtype=object)
    X[:, 0] = clean
    X[:, 1] = cat
    X[rng.choice(n, 15, replace=False), 0] = np.nan
    X[rng.choice(n, 12, replace=False), 1] = None
    return X, y


def clean_mixed(seed=1, n=200):
    rng = np.random.default_rng(seed)
    clean = rng.normal(size=n)
    cat = rng.choice(["u", "v", "w"], size=n)
    y = ((clean > 0) ^ (cat == "v")).astype(int)
    X = np.empty((n, 2), dtype=object)
    X[:, 0] = clean
    X[:, 1] = cat
    return X, y


class TestCriteria(unittest.TestCase):
    def test_entropy_golden_values(self):
        self.assertAlmostEqual(entropy([5, 5]), 1.0, places=10)
        self.assertAlmostEqual(entropy([3, 1]), 0.81, places=2)
        self.assertAlmostEqual(entropy([2, 4]), 0.92, places=2)
        self.assertEqual(entropy([5, 0]), 0.0)
        self.assertEqual(entropy([0, 5]), 0.0)
        self.assertAlmostEqual(entropy([1, 1, 1]), np.log2(3), places=10)

    def test_information_gain_outlook(self):
        parent = [9, 5]
        children = [[2, 3], [4, 0], [3, 2]]
        self.assertAlmostEqual(information_gain(parent, children), 0.246,
                               places=2)
        humidity = information_gain([9, 5], [[3, 4], [6, 1]])
        wind = information_gain([9, 5], [[6, 2], [3, 3]])
        temp = information_gain([9, 5], [[2, 2], [4, 2], [3, 1]])
        self.assertAlmostEqual(humidity, 0.151, places=2)
        self.assertAlmostEqual(wind, 0.048, places=2)
        self.assertAlmostEqual(temp, 0.029, places=2)

    def test_information_gain_missing_fraction(self):
        parent = [6, 5]
        children = [[4, 2], [1, 3]]
        expected = (10 / 11) * (entropy([6, 5])
                                - (6 / 10) * entropy([4, 2])
                                - (4 / 10) * entropy([1, 3]))
        self.assertAlmostEqual(information_gain(parent, children), expected,
                               places=12)

    def test_select_split_applies_above_average_gain(self):
        chosen = select_split([(0.10, 0.50, "a"), (0.06, 0.90, "b")])
        self.assertEqual(chosen[2], "a")

    def test_select_split_ties_and_empty(self):
        self.assertEqual(
            select_split([(0.1, 0.5, "x"), (0.1, 0.5, "y")])[2], "x")
        self.assertIsNone(select_split([]))


class TestSplitting(unittest.TestCase):
    def test_continuous_midpoints(self):
        values = np.array([72, 40, 90, 60, 48, 80], dtype=float)
        y = np.array([1, 0, 0, 1, 0, 1])
        w = np.ones(6)
        cands = best_split_continuous(values, y, w, 2, min_samples_leaf=1)
        self.assertEqual({c[2] for c in cands},
                         {44.0, 54.0, 66.0, 76.0, 85.0})
        gain54 = next(c[0] for c in cands if c[2] == 54.0)
        self.assertEqual(max(c[0] for c in cands), gain54)
        self.assertAlmostEqual(gain54, 0.459, places=3)
        self.assertAlmostEqual(gain54,
                               information_gain([3, 3], [[0, 2], [3, 1]]),
                               places=10)

    def test_continuous_respects_min_samples_leaf(self):
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        y = np.array([0, 0, 0, 1, 1, 1])
        w = np.ones(6)
        self.assertEqual(
            best_split_continuous(values, y, w, 2, min_samples_leaf=4), [])
        self.assertTrue(
            best_split_continuous(values, y, w, 2, min_samples_leaf=2))

    def test_categorical_rejects_rare_value(self):
        values = np.array(["a", "a", "b", "b", "rare", None], dtype=object)
        y = np.array([0, 1, 0, 1, 0, 1])
        w = np.ones(6)
        self.assertIsNone(
            best_split_categorical(values, y, w, 2, min_samples_leaf=2))
        split = best_split_categorical(values, y, w, 2, min_samples_leaf=1)
        self.assertIsNotNone(split)
        gain, _, uniques = split
        self.assertEqual(list(uniques), ["a", "b", "rare"])
        self.assertGreater(gain, 0.0)

    def test_categorical_missing_stays_out_of_children(self):
        values = np.array(["a", "a", "b", None, "b", None], dtype=object)
        y = np.array([0, 0, 1, 1, 1, 0])
        w = np.ones(6)
        split = best_split_categorical(values, y, w, 2, min_samples_leaf=1)
        gain, _, uniques = split
        expected = information_gain([3, 3], [[2, 0], [0, 2]])
        self.assertAlmostEqual(gain, expected, places=10)
        self.assertAlmostEqual(gain, 4 / 6, places=10)
        self.assertEqual(list(uniques), ["a", "b"])


class TestDTL(unittest.TestCase):
    def test_playtennis_root_and_fit(self):
        X, y = playtennis()
        model = DTL(max_depth=None, min_samples_leaf=1).fit(X, y)
        self.assertEqual(model.root_.feature, 0)
        self.assertEqual(int((model.predict(X) == y).sum()), 14)
        self.assertLessEqual(model.depth_, 4)

    def test_missing_values_and_unseen_category(self):
        X, y = mixed_data()
        model = DTL(max_depth=6, min_samples_leaf=3).fit(X, y)
        proba = model.predict_proba(X)
        self.assertEqual(proba.shape, (len(y), 2))
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-9)
        self.assertGreater((model.predict(X) == y).mean(), 0.8)
        unseen = np.array([[0.5, "zzz"]], dtype=object)
        self.assertEqual(model.predict_proba(unseen).shape, (1, 2))

    def test_determinism(self):
        X, y = mixed_data(seed=3)
        m1 = DTL(max_depth=8, min_samples_leaf=3).fit(X, y)
        m2 = DTL(max_depth=8, min_samples_leaf=3).fit(X, y)
        self.assertEqual(m1.root_.feature, m2.root_.feature)
        np.testing.assert_array_equal(m1.predict(X), m2.predict(X))

    def test_tie_breaks_to_lowest_feature(self):
        col = np.array(["a", "a", "b", "b", "a", "b"], dtype=object)
        X = np.array([col, col]).T
        y = np.array([0, 1, 0, 1, 0, 1])
        model = DTL(max_depth=3, min_samples_leaf=1).fit(X, y)
        self.assertEqual(model.root_.feature, 0)

    def test_depth_and_min_leaf(self):
        X, y = clean_mixed()
        model = DTL(max_depth=4, min_samples_leaf=5).fit(X, y)
        self.assertLessEqual(model.depth_, 4)
        counts = np.bincount(model.apply(X), minlength=model.n_leaves_)
        self.assertEqual(counts.sum(), len(y))
        self.assertGreaterEqual(int(counts.min()), 5)

    def test_pandas_input_and_column_order(self):
        pd = __import__("pandas")
        df = pd.DataFrame({
            "num": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0,
                    9.0, 10.0, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5],
            "cat": ["a", "b", "a", "b", "a", "b", "a", "b",
                    "a", "b", "a", "b", "a", "b", "a", "b"],
        })
        y = pd.Series([0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0])
        model = DTL(max_depth=3, min_samples_leaf=1).fit(df, y)
        self.assertEqual(model.feature_names_, ["num", "cat"])
        pred = model.predict(df)
        pred_shuffled = model.predict(df[["cat", "num"]])
        np.testing.assert_array_equal(pred, pred_shuffled)

    def test_save_load_roundtrip(self):
        X, y = mixed_data(seed=5)
        model = DTL(max_depth=6, min_samples_leaf=3).fit(X, y)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.json"
            model.save(path)
            payload = json.loads(path.read_text())
            self.assertEqual(payload["format"], "if3170-dtl")
            loaded = DTL.load(path)
        np.testing.assert_array_equal(model.predict(X), loaded.predict(X))
        np.testing.assert_allclose(model.predict_proba(X),
                                   loaded.predict_proba(X))
        self.assertEqual(model.feature_names_, loaded.feature_names_)
        self.assertEqual(model.n_leaves_, loaded.n_leaves_)
        self.assertEqual(model.depth_, loaded.depth_)

    def test_reduced_error_pruning(self):
        rng = np.random.default_rng(7)
        n = 400
        x = rng.uniform(size=(n, 2))
        y = ((x[:, 0] > 0.5) != (x[:, 1] > 0.5)).astype(int)
        flip = rng.choice(n, 150, replace=False)
        y[flip] = 1 - y[flip]
        model = DTL(max_depth=None, min_samples_leaf=1,
                    pruning="reduced_error", val_fraction=0.3,
                    random_state=0).fit(x, y)
        report = model.pruning_report_
        self.assertGreaterEqual(report["val_accuracy_after"],
                                report["val_accuracy_before"] - 1e-12)
        self.assertGreaterEqual(report["pruned_nodes"], 1)
        unpruned = DTL(max_depth=None, min_samples_leaf=1).fit(x, y)
        self.assertLessEqual(model.n_leaves_, unpruned.n_leaves_)

    def test_cost_complexity_pruning(self):
        X, y = playtennis()
        unpruned = DTL(max_depth=None, min_samples_leaf=1).fit(X, y)
        collapsed = DTL(max_depth=None, min_samples_leaf=1,
                        pruning="cost_complexity",
                        ccp_alpha=1e9).fit(X, y)
        self.assertEqual(collapsed.n_leaves_, 1)
        gentle = DTL(max_depth=None, min_samples_leaf=1,
                     pruning="cost_complexity",
                     ccp_alpha=1e-6).fit(X, y)
        self.assertLessEqual(gentle.n_leaves_, unpruned.n_leaves_)

    def test_parameter_validation(self):
        with self.assertRaises(ValueError):
            DTL(pruning="bogus")
        with self.assertRaises(ValueError):
            DTL(min_samples_leaf=0)
        with self.assertRaises(ValueError):
            DTL(pruning="reduced_error", val_fraction=0.0)
        with self.assertRaises(ValueError):
            DTL(ccp_alpha=-1.0)
        with self.assertRaises(RuntimeError):
            DTL().predict(np.zeros((2, 2)))


if __name__ == "__main__":
    unittest.main()
