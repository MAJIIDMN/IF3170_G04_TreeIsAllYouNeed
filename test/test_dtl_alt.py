import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "algoritma"))

from dtl_alt import DTL  # noqa: E402


def numeric_data(n=600, seed=0):
    return make_classification(n_samples=n, n_features=6, n_informative=4, flip_y=0.05, random_state=seed)


def mixed_frame(n=800, seed=0):
    rng = np.random.default_rng(seed)
    amount = rng.normal(size=n)
    color = rng.choice(["red", "green", "blue", "rare"], size=n, p=[0.45, 0.3, 0.24, 0.01])
    y = ((amount > 0.2) ^ (color == "red")).astype(int)
    frame = pd.DataFrame({"amount": amount, "color": color})
    frame.loc[rng.random(n) < 0.15, "amount"] = np.nan
    frame.loc[rng.random(n) < 0.15, "color"] = None
    return frame, y


class SklearnAgreementTest(unittest.TestCase):
    def test_gain_matches_entropy_tree(self):
        X, y = numeric_data()
        ours = DTL(max_depth=4, min_samples_leaf=5, criterion="gain").fit(X, y)
        reference = DecisionTreeClassifier(max_depth=4, min_samples_leaf=5, criterion="entropy",
                                           random_state=0).fit(X, y)
        self.assertGreater((ours.predict(X) == reference.predict(X)).mean(), 0.97)

    def test_gini_matches_gini_tree(self):
        X, y = numeric_data()
        ours = DTL(max_depth=4, min_samples_leaf=5, criterion="gini").fit(X, y)
        reference = DecisionTreeClassifier(max_depth=4, min_samples_leaf=5, criterion="gini",
                                           random_state=0).fit(X, y)
        self.assertGreater((ours.predict(X) == reference.predict(X)).mean(), 0.97)


class MissingValueTest(unittest.TestCase):
    def test_fractional_prediction_mixes_both_branches(self):
        X = np.array([[0.0], [0.0], [0.0], [1.0]])
        y = np.array([0, 0, 0, 1])
        model = DTL(min_samples_leaf=1, criterion="gain").fit(X, y)
        np.testing.assert_allclose(model.predict_proba(np.array([[np.nan]])), [[0.75, 0.25]])

    def test_common_branch_prediction_uses_one_leaf(self):
        X = np.array([[0.0], [0.0], [0.0], [1.0]])
        y = np.array([0, 0, 0, 1])
        model = DTL(min_samples_leaf=1, criterion="gain", missing="common").fit(X, y)
        np.testing.assert_allclose(model.predict_proba(np.array([[np.nan]])), [[1.0, 0.0]])

    def test_probabilities_sum_to_one_with_missing_values(self):
        frame, y = mixed_frame()
        for split in ("subset", "multiway"):
            with self.subTest(split=split):
                model = DTL(max_depth=6, categorical_split=split).fit(frame, y)
                np.testing.assert_allclose(model.predict_proba(frame).sum(axis=1), 1.0)

    def test_child_weights_add_up_to_parent(self):
        frame, y = mixed_frame()
        root = DTL(max_depth=4).fit(frame, y).root_
        np.testing.assert_allclose(root.left.class_weights + root.right.class_weights,
                                   root.class_weights)

    def test_leaf_size_counts_fractional_weight(self):
        frame, y = mixed_frame()
        model = DTL(max_depth=None, min_samples_leaf=10).fit(frame, y)
        stack = [model.root_]
        while stack:
            node = stack.pop()
            self.assertGreaterEqual(node.class_weights.sum(), 10 - 1e-9)
            if not node.leaf:
                stack += [node.left, node.right]

    def test_unseen_category_is_treated_as_missing(self):
        frame, y = mixed_frame()
        model = DTL(max_depth=4).fit(frame, y)
        unseen = frame.iloc[:5].assign(color="purple")
        missing = frame.iloc[:5].assign(color=None)
        np.testing.assert_allclose(model.predict_proba(unseen), model.predict_proba(missing))


class CategoricalSplitTest(unittest.TestCase):
    def test_subset_split_survives_a_rare_category(self):
        frame, y = mixed_frame()
        subset = DTL(max_depth=3, min_samples_leaf=20).fit(frame[["color"]], y)
        multiway = DTL(max_depth=3, min_samples_leaf=20, categorical_split="multiway").fit(frame[["color"]], y)
        self.assertFalse(subset.root_.leaf)
        self.assertTrue(multiway.root_.leaf)


class PruningTest(unittest.TestCase):
    def test_every_pruning_shrinks_a_deep_tree(self):
        X, y = numeric_data()
        deep = DTL(max_depth=None, min_samples_leaf=1).fit(X, y).n_leaves_
        for params in ({"pruning": "cost_complexity", "ccp_alpha": 1.0},
                       {"pruning": "pessimistic"},
                       {"pruning": "reduced_error"}):
            with self.subTest(params=params):
                model = DTL(max_depth=None, min_samples_leaf=1, **params).fit(X, y)
                self.assertLess(model.n_leaves_, deep)

    def test_large_alpha_collapses_to_root(self):
        X, y = numeric_data()
        model = DTL(pruning="cost_complexity", ccp_alpha=1e6).fit(X, y)
        self.assertEqual(model.n_leaves_, 1)

    def test_lower_confidence_prunes_more(self):
        X, y = numeric_data()
        strict = DTL(max_depth=None, min_samples_leaf=1, pruning="pessimistic", confidence=0.05).fit(X, y)
        loose = DTL(max_depth=None, min_samples_leaf=1, pruning="pessimistic", confidence=0.5).fit(X, y)
        self.assertLess(strict.n_leaves_, loose.n_leaves_)


class ApiTest(unittest.TestCase):
    def test_saved_model_predicts_identically(self):
        frame, y = mixed_frame()
        for split in ("subset", "multiway"):
            with self.subTest(split=split):
                model = DTL(max_depth=6, categorical_split=split).fit(frame, y)
                with tempfile.TemporaryDirectory() as folder:
                    path = Path(folder) / "dtl.json"
                    model.save(path)
                    loaded = DTL.load(path)
                np.testing.assert_array_equal(model.predict_proba(frame), loaded.predict_proba(frame))

    def test_apply_returns_a_valid_leaf_for_each_row(self):
        frame, y = mixed_frame()
        model = DTL(max_depth=4).fit(frame, y)
        leaves = model.apply(frame)
        self.assertEqual(len(leaves), len(frame))
        self.assertLess(leaves.max(), model.n_leaves_)

    def test_threshold_shifts_prediction(self):
        X, y = numeric_data()
        model = DTL(max_depth=4).fit(X, y)
        self.assertGreater(model.predict(X, threshold=0.2).sum(), model.predict(X, threshold=0.8).sum())

    def test_invalid_parameters_are_rejected(self):
        for params in ({"criterion": "x"}, {"missing": "x"}, {"categorical_split": "x"}, {"pruning": "x"},
                       {"max_depth": 0}, {"min_samples_leaf": 0}):
            with self.subTest(params=params), self.assertRaises(ValueError):
                DTL(**params)

    def test_predict_before_fit_is_rejected(self):
        with self.assertRaises(RuntimeError):
            DTL().predict(np.zeros((1, 2)))


if __name__ == "__main__":
    unittest.main()
