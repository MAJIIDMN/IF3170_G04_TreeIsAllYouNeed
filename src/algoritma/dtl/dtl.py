from __future__ import annotations

import numpy as np

from .pruning import (cost_complexity_prune, pessimistic_prune,
                      reduced_error_prune)
from .serialization import load_model, save_model
from .tree import (CATEGORICAL, CONTINUOUS, MULTIWAY, SUBSET, BuildConfig,
                   accumulate_proba, assign_leaf_ids, build_tree, tree_depth,
                   traverse)

__all__ = ["DTL"]

_PRUNING = ("none", "reduced_error", "cost_complexity", "pessimistic")
_CRITERIA = ("gain_ratio", "gain", "gini")
_CATEGORICAL_SPLITS = (SUBSET, MULTIWAY)
_MISSING = ("fractional", "common")


def _is_missing(v) -> bool:
    if v is None:
        return True
    if isinstance(v, (float, np.floating)):
        return bool(np.isnan(v))
    return False


def _array_kind(col) -> str:
    if col.dtype.kind in "US":
        return CATEGORICAL
    if col.dtype.kind == "O":
        for v in col:
            if _is_missing(v):
                continue
            return CATEGORICAL if isinstance(v, str) else CONTINUOUS
        return CONTINUOUS
    return CONTINUOUS


def _array_column(col, kind) -> np.ndarray:
    if kind == CONTINUOUS:
        if col.dtype.kind == "O":
            out = np.empty(col.shape[0], dtype=float)
            for i, v in enumerate(col):
                out[i] = np.nan if _is_missing(v) else float(v)
            return out
        return np.asarray(col, dtype=float)
    out = np.array(col, dtype=object, copy=True)
    for i, v in enumerate(out):
        if _is_missing(v):
            out[i] = None
    return out


def _pandas_column(series, kind) -> np.ndarray:
    if kind == CONTINUOUS:
        return series.to_numpy(dtype=float)
    na = np.asarray(series.isna())
    vals = np.array(series.to_numpy(dtype=object), dtype=object, copy=True)
    vals[na] = None
    return vals


def _pandas_fit_series(series):
    """pandas 3 labels strings as dtype \"str\", so kind is decided by
    probing float conversion instead of matching the dtype name."""
    if str(series.dtype) == "category":
        return _pandas_column(series, CATEGORICAL), CATEGORICAL
    try:
        return series.to_numpy(dtype=float), CONTINUOUS
    except (TypeError, ValueError):
        return _pandas_column(series, CATEGORICAL), CATEGORICAL


def _is_dataframe(X) -> bool:
    return hasattr(X, "columns") and hasattr(X, "iloc")


def _stack(columns) -> np.ndarray:
    n = len(columns[0])
    out = np.empty((n, len(columns)), dtype=object)
    for f, col in enumerate(columns):
        out[:, f] = col
    return out


class DTL:
    """C4.5-style decision tree with native categorical and missing values.

    Args:
        max_depth: Depth limit, or None for no limit.
        min_samples_leaf: Least row weight on each side of a split. A row with
            a missing split value counts fractionally.
        min_gain: A split needs a gain above this value.
        pruning: 'none', 'reduced_error', 'cost_complexity', or 'pessimistic'.
        ccp_alpha: Cost per leaf for 'cost_complexity', in units of row weight.
        val_fraction: Share of rows held out for 'reduced_error'.
        random_state: Seed of the 'reduced_error' holdout.
        criterion: 'gain_ratio' (C4.5), 'gain' (ID3), or 'gini' (CART).
        categorical_split: 'subset' for a binary partition of categories, or
            'multiway' for one child per category.
        missing: 'fractional' spreads a row over all branches at prediction
            time, 'common' follows the most common branch.
        confidence: Confidence factor of 'pessimistic' pruning.
    """

    def __init__(self, max_depth=12, min_samples_leaf=5, min_gain=1e-7,
                 pruning="none", ccp_alpha=0.0, val_fraction=0.3,
                 random_state=42, criterion="gain_ratio",
                 categorical_split=SUBSET, missing="fractional",
                 confidence=0.25):
        if pruning not in _PRUNING:
            raise ValueError(f"pruning must be one of {_PRUNING}, "
                             f"got {pruning!r}")
        if criterion not in _CRITERIA:
            raise ValueError(f"criterion must be one of {_CRITERIA}, "
                             f"got {criterion!r}")
        if categorical_split not in _CATEGORICAL_SPLITS:
            raise ValueError(f"categorical_split must be one of "
                             f"{_CATEGORICAL_SPLITS}, got {categorical_split!r}")
        if missing not in _MISSING:
            raise ValueError(f"missing must be one of {_MISSING}, "
                             f"got {missing!r}")
        if max_depth is not None and max_depth < 1:
            raise ValueError(f"max_depth must be >= 1, got {max_depth}")
        if min_samples_leaf <= 0:
            raise ValueError(f"min_samples_leaf must be > 0, "
                             f"got {min_samples_leaf}")
        if min_gain < 0.0:
            raise ValueError(f"min_gain must be >= 0, got {min_gain}")
        if ccp_alpha < 0.0:
            raise ValueError(f"ccp_alpha must be >= 0, got {ccp_alpha}")
        if pruning == "reduced_error" and not 0.0 < val_fraction < 1.0:
            raise ValueError("val_fraction must be in (0, 1) for "
                             "pruning='reduced_error'")
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_gain = min_gain
        self.pruning = pruning
        self.ccp_alpha = ccp_alpha
        self.val_fraction = val_fraction
        self.random_state = random_state
        self.criterion = criterion
        self.categorical_split = categorical_split
        self.missing = missing
        self.confidence = confidence

    def fit(self, X, y):
        columns, names, kinds, named = self._fit_columns(X)
        y_arr = np.asarray(y)
        if y_arr.ndim != 1:
            raise ValueError(f"y must be one-dimensional, got shape "
                             f"{y_arr.shape}")
        if len(y_arr) != len(columns[0]):
            raise ValueError(f"X has {len(columns[0])} rows but y has "
                             f"{len(y_arr)}")
        self.classes_ = np.unique(y_arr)
        y_idx = np.searchsorted(self.classes_, y_arr)
        cfg = BuildConfig(n_classes=len(self.classes_),
                          max_depth=self.max_depth,
                          min_samples_leaf=self.min_samples_leaf,
                          min_gain=self.min_gain,
                          measure="gini" if self.criterion == "gini" else "entropy",
                          use_ratio=self.criterion == "gain_ratio",
                          categorical_split=self.categorical_split)
        n = len(y_arr)
        train_idx, val_idx = np.arange(n), None
        if self.pruning == "reduced_error":
            perm = np.random.default_rng(self.random_state).permutation(n)
            n_val = min(max(int(round(n * self.val_fraction)), 1), n - 1)
            val_idx, train_idx = perm[:n_val], perm[n_val:]
        root = build_tree(columns, kinds, y_idx, train_idx,
                          np.ones(len(train_idx)), 0, cfg)
        self.pruning_report_ = None
        if self.pruning == "reduced_error":
            root, self.pruning_report_ = reduced_error_prune(root, columns, y_idx, val_idx)
        elif self.pruning == "cost_complexity":
            root, self.pruning_report_ = cost_complexity_prune(root, self.ccp_alpha)
        elif self.pruning == "pessimistic":
            root, self.pruning_report_ = pessimistic_prune(root, self.confidence)
        self.root_ = root
        self.n_leaves_ = assign_leaf_ids(root)
        self.depth_ = tree_depth(root)
        self.feature_names_ = names
        self.kinds_ = kinds
        self._named_fit_ = named
        return self

    def _fit_columns(self, X):
        if _is_dataframe(X):
            names = list(X.columns)
            kinds = []
            cols = []
            for name in names:
                series = X[name]
                col, kind = _pandas_fit_series(series)
                kinds.append(kind)
                cols.append(col)
            return cols, names, kinds, True
        arr = np.asarray(X)
        if arr.ndim != 2:
            raise ValueError(f"X must be 2-dimensional, got shape {arr.shape}")
        names = [f"feature_{i}" for i in range(arr.shape[1])]
        kinds = [_array_kind(arr[:, i]) for i in range(arr.shape[1])]
        cols = [_array_column(arr[:, i], kinds[i])
                for i in range(arr.shape[1])]
        return cols, names, kinds, False

    def _predict_columns(self, X):
        if _is_dataframe(X):
            names = list(X.columns)
            if self._named_fit_:
                if names == self.feature_names_:
                    series_list = [X[c] for c in names]
                elif (len(names) == len(self.feature_names_)
                      and set(names) == set(self.feature_names_)):
                    series_list = [X[c] for c in self.feature_names_]
                else:
                    raise ValueError(
                        f"expected features {self.feature_names_}, "
                        f"got {names}")
            else:
                if len(names) != len(self.kinds_):
                    raise ValueError(
                        f"expected {len(self.kinds_)} features, "
                        f"got {len(names)}")
                series_list = [X[c] for c in names]
            return [_pandas_column(s, k)
                    for s, k in zip(series_list, self.kinds_)]
        arr = np.asarray(X)
        if arr.ndim != 2:
            raise ValueError(f"X must be 2-dimensional, got shape {arr.shape}")
        if arr.shape[1] != len(self.kinds_):
            raise ValueError(f"expected {len(self.kinds_)} features, "
                             f"got {arr.shape[1]}")
        return [_array_column(arr[:, i], k)
                for i, k in enumerate(self.kinds_)]

    def _check_fitted(self):
        if not hasattr(self, "root_"):
            raise RuntimeError("this DTL instance is not fitted yet")

    def predict_proba(self, X) -> np.ndarray:
        self._check_fitted()
        columns = self._predict_columns(X)
        n = len(columns[0])
        out = np.zeros((n, len(self.classes_)))
        accumulate_proba(self.root_, columns, np.arange(n), np.ones(n), out,
                         self.missing == "fractional")
        return out

    def predict(self, X, threshold=None) -> np.ndarray:
        """Predict labels; ``threshold`` applies to the larger of two classes."""
        proba = self.predict_proba(X)
        if threshold is None:
            return self.classes_[np.argmax(proba, axis=1)]
        if len(self.classes_) != 2:
            raise ValueError("threshold needs exactly two classes")
        return self.classes_[(proba[:, 1] > threshold).astype(int)]

    def apply(self, X) -> np.ndarray:
        """Index of the leaf each row lands in (leaf ids are preorder)."""
        self._check_fitted()
        rows = _stack(self._predict_columns(X))
        return np.array([traverse(self.root_, row).leaf_id for row in rows])

    def save(self, path) -> None:
        self._check_fitted()
        params = {
            "max_depth": self.max_depth,
            "min_samples_leaf": self.min_samples_leaf,
            "min_gain": self.min_gain,
            "pruning": self.pruning,
            "ccp_alpha": self.ccp_alpha,
            "val_fraction": self.val_fraction,
            "random_state": self.random_state,
            "criterion": self.criterion,
            "categorical_split": self.categorical_split,
            "missing": self.missing,
            "confidence": self.confidence,
        }
        save_model(path, params, self.classes_, self.feature_names_,
                   self.kinds_, self.pruning_report_, self.root_,
                   self._named_fit_)

    @classmethod
    def load(cls, path) -> "DTL":
        payload = load_model(path)
        model = cls(**payload["params"])
        model.classes_ = np.asarray(payload["classes"])
        model.feature_names_ = list(payload["feature_names"])
        model.kinds_ = list(payload["kinds"])
        model.pruning_report_ = payload["pruning_report"]
        model._named_fit_ = bool(payload.get("named_fit", True))
        model.root_ = payload["root"]
        model.n_leaves_ = assign_leaf_ids(model.root_)
        model.depth_ = tree_depth(model.root_)
        return model
