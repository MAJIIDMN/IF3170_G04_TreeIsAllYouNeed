from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "src" / "kaggle" / "dataset"
TARGET = "is_canceled"
ID_COL = "id"

RAW_NUMERIC = [
    "num_adults", "num_children", "weekend_nights", "weekday_nights", "needs_parking",
    "days_before_arrival", "is_returning_guest", "prior_cancellations",
    "prior_completed_bookings", "nightly_rate_idr", "special_requests",
]
RAW_CATEGORICAL = ["meal_package", "room_category", "booking_channel"]
RAW_DATE = ["arrival_year", "arrival_month", "arrival_day"]
RAW_FEATURES = RAW_NUMERIC + RAW_CATEGORICAL + RAW_DATE

COUNT_CAP = {
    "num_adults": 3, "num_children": 2, "weekend_nights": 4, "weekday_nights": 7,
    "needs_parking": 1, "is_returning_guest": 1, "prior_cancellations": 2,
    "prior_completed_bookings": 4, "special_requests": 4, "arrival_year": 1,
    "arrival_month": 12, "arrival_day": 31,
}
STRING_FEATURES = RAW_CATEGORICAL
RARE_MIN_COUNT = 30
WINSOR = (0.5, 99.5)

def load_train(drop_duplicates=True):
    train = pd.read_csv(DATA_DIR / "train.csv")
    if drop_duplicates:
        features = [c for c in train.columns if c not in (ID_COL, TARGET)]
        train = train.drop_duplicates(subset=features + [TARGET]).reset_index(drop=True)
    return train

def load_test():
    return pd.read_csv(DATA_DIR / "test.csv")

def macro_f1(y_true, y_pred):
    scores = []
    for label in (0, 1):
        tp = np.sum((y_true == label) & (y_pred == label))
        denominator = np.sum(y_true == label) + np.sum(y_pred == label)
        scores.append(2 * tp / denominator if denominator else 0.0)
    return float(np.mean(scores))

def roc_auc(y_true, score):
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score))
    sorted_scores = score[order]
    boundaries = np.flatnonzero(np.diff(sorted_scores)) + 1
    starts = np.concatenate([[0], boundaries])
    ends = np.concatenate([boundaries, [len(score)]])
    for s, e in zip(starts, ends):
        ranks[order[s:e]] = (s + e + 1) / 2.0
    positive = y_true == 1
    n_pos, n_neg = positive.sum(), (~positive).sum()
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))

def stratified_folds(y, n_folds, seed):
    rng = np.random.default_rng(seed)
    fold_of = np.empty(len(y), dtype=int)
    for label in np.unique(y):
        members = rng.permutation(np.flatnonzero(y == label))
        fold_of[members] = np.arange(len(members)) % n_folds
    return [(np.flatnonzero(fold_of != k), np.flatnonzero(fold_of == k)) for k in range(n_folds)]

def _entropy(counts):
    total = counts.sum()
    if total <= 0:
        return 0.0
    p = counts[counts > 0] / total
    return float(-(p * np.log2(p)).sum())

# Algoritma di bawah merupakan diskretisasi berbasis entropi dengan kriteria berhenti MDL.
# Referensi: "Multi-Interval Discretization of Continuous-Valued Attributes for Classification Learning" (Fayyad, Irani, 1993)
def mdl_cuts(values, labels, max_cuts=15):
    order = np.argsort(values, kind="mergesort")
    v, y = values[order], labels[order].astype(int)
    k_all = int(y.max()) + 1
    distinct, first = np.unique(v, return_index=True)
    if distinct.size < 2:
        return []
    counts = np.zeros((distinct.size, k_all))
    np.add.at(counts, (np.searchsorted(distinct, v), y), 1.0)
    cum = np.vstack([np.zeros(k_all), np.cumsum(counts, axis=0)])
    cuts = []

    def split(lo, hi, budget):
        if hi - lo < 2 or budget <= 0:
            return
        total = cum[hi] - cum[lo]
        n = total.sum()
        parent = _entropy(total)
        best, best_gain = None, 0.0
        for i in range(lo + 1, hi):
            left, right = cum[i] - cum[lo], cum[hi] - cum[i]
            gain = parent - (left.sum() * _entropy(left) + right.sum() * _entropy(right)) / n
            if gain > best_gain:
                best, best_gain = i, gain
        if best is None:
            return
        left, right = cum[best] - cum[lo], cum[hi] - cum[best]
        k, k1, k2 = (total > 0).sum(), (left > 0).sum(), (right > 0).sum()
        delta = np.log2(3.0 ** k - 2.0) - (k * parent - k1 * _entropy(left) - k2 * _entropy(right))
        # kriteria berhenti MDL (referensi di atas)
        if best_gain <= (np.log2(n - 1.0) + delta) / n:
            return
        cuts.append(float((distinct[best - 1] + distinct[best]) / 2.0))
        split(lo, best, budget - 1)
        split(best, hi, budget - 1)

    split(0, distinct.size, max_cuts)
    return sorted(cuts)

class Preprocessor:
    def __init__(self, spec, winsorize=True):
        self.spec = dict(spec)
        self.winsorize = winsorize
        self.feature_names_ = list(self.spec)
        self.feature_types_ = [
            "gaussian" if rep in ("gauss", "log") else "categorical" for rep in self.spec.values()
        ]
        self.parts_ = {}

    def fit(self, df, y=None):
        self.categories_, self.cuts_, self.bounds_, self.offsets_ = {}, {}, {}, {}
        for name, rep in self.spec.items():
            if isinstance(rep, tuple):
                self._fit_cross(name, rep[1], df, y)
                continue
            column = df[name]
            if name in STRING_FEATURES and rep == "cat":
                counts = column.value_counts()
                self.categories_[name] = sorted(counts[counts >= RARE_MIN_COUNT].index) + ["lainnya"]
            elif rep in ("gauss", "log"):
                values = self._base(column, rep).dropna().to_numpy()
                self.bounds_[name] = tuple(np.percentile(values, WINSOR)) if self.winsorize else (-np.inf, np.inf)
            elif rep == "mdl":
                known = column.notna().to_numpy()
                self.cuts_[name] = mdl_cuts(column.to_numpy(float)[known], np.asarray(y)[known])
            elif rep == "cat":
                self.offsets_[name] = float(column.min()) if name == "arrival_year" else 0.0
            elif rep[0] in "qw" and rep[1:].isdigit():
                self.cuts_[name] = self._bin_edges(column.dropna().to_numpy(float), rep)
        return self

    def transform(self, df):
        columns = []
        for name, rep in self.spec.items():
            if isinstance(rep, tuple):
                columns.append(self._cross_codes(name, df))
                continue
            column = df[name]
            if rep in ("gauss", "log"):
                low, high = self.bounds_[name]
                columns.append(self._base(column, rep).clip(low, high).to_numpy(float))
            elif name in STRING_FEATURES and rep == "cat":
                columns.append(self._string_codes(column, name))
            elif rep == "cat":
                values = column.to_numpy(float) - self.offsets_[name]
                columns.append(np.where(np.isnan(values), np.nan, np.clip(values, 0, COUNT_CAP[name])))
            else:
                values = column.to_numpy(float)
                codes = np.searchsorted(np.asarray(self.cuts_[name]), values, side="right").astype(float)
                columns.append(np.where(np.isnan(values), np.nan, codes))
        return np.column_stack(columns)

    def fit_transform(self, df, y=None):
        return self.fit(df, y).transform(df)

    def to_dict(self):
        spec = {name: (["cross", [list(part) for part in rep[1]]] if isinstance(rep, tuple) else rep)
                for name, rep in self.spec.items()}
        return {
            "spec": spec,
            "winsorize": self.winsorize,
            "categories": {name: list(values) for name, values in self.categories_.items()},
            "cuts": {name: [float(c) for c in cuts] for name, cuts in self.cuts_.items()},
            "bounds": {name: [float(low), float(high)] for name, (low, high) in self.bounds_.items()},
            "offsets": {name: float(value) for name, value in self.offsets_.items()},
            "parts": {name: {"subs": [sub.to_dict() for sub in info["subs"]],
                             "sizes": [int(s) for s in info["sizes"]],
                             "ids": [[float(key), int(value)] for key, value in info["ids"].items()]}
                      for name, info in self.parts_.items()},
        }

    @classmethod
    def from_dict(cls, payload):
        spec = {name: (("cross", [tuple(part) for part in rep[1]]) if isinstance(rep, list) else rep)
                for name, rep in payload["spec"].items()}
        prep = cls(spec, winsorize=payload["winsorize"])
        prep.categories_ = {name: list(values) for name, values in payload["categories"].items()}
        prep.cuts_ = {name: list(cuts) for name, cuts in payload["cuts"].items()}
        prep.bounds_ = {name: tuple(bound) for name, bound in payload["bounds"].items()}
        prep.offsets_ = dict(payload["offsets"])
        prep.parts_ = {name: {"subs": [cls.from_dict(sub) for sub in info["subs"]],
                              "sizes": list(info["sizes"]),
                              "ids": {float(key): int(value) for key, value in info["ids"]}}
                       for name, info in payload["parts"].items()}
        return prep

    def _fit_cross(self, name, parts, df, y):
        subs = [Preprocessor({feature: rep}, winsorize=self.winsorize).fit(df, y)
                for feature, rep in parts]
        codes = np.column_stack([sub.transform(df)[:, 0] for sub in subs])
        sizes = [int(np.nanmax(codes[:, k])) + 1 for k in range(len(subs))]
        joint = self._ravel(codes, sizes)
        known = ~np.isnan(joint)
        values, counts = np.unique(joint[known], return_counts=True)
        keep = values[counts >= RARE_MIN_COUNT]
        self.parts_[name] = {"subs": subs, "sizes": sizes, "ids": {float(v): i for i, v in enumerate(keep)}}

    def _cross_codes(self, name, df):
        info = self.parts_[name]
        codes = np.column_stack([sub.transform(df)[:, 0] for sub in info["subs"]])
        joint = self._ravel(codes, info["sizes"])
        other = float(len(info["ids"]))
        return np.array([np.nan if np.isnan(v) else info["ids"].get(float(v), other) for v in joint])

    @staticmethod
    def _ravel(codes, sizes):
        joint = np.zeros(len(codes))
        for k, size in enumerate(sizes):
            joint = joint * size + np.minimum(codes[:, k], size - 1)
        return np.where(np.isnan(codes).any(axis=1), np.nan, joint)

    @staticmethod
    def _base(column, rep):
        column = column.astype(float)
        return np.log1p(column.clip(lower=0)) if rep == "log" else column

    def _string_codes(self, column, name):
        categories = self.categories_[name]
        known = column.notna().to_numpy()
        values = column.where(column.isin(categories) | ~column.notna(), "lainnya")
        lookup = {c: i for i, c in enumerate(categories)}
        return np.array([lookup.get(v, np.nan) if k else np.nan for v, k in zip(values, known)], dtype=float)

    @staticmethod
    def _bin_edges(values, rep):
        k = int(rep[1:])
        if rep[0] == "q":
            edges = np.unique(np.quantile(values, np.linspace(0, 1, k + 1)[1:-1]))
        else:
            edges = np.linspace(values.min(), values.max(), k + 1)[1:-1]
        return edges.tolist()


def cross_validate(spec, make_model, df, y, n_folds=5, seed=42, winsorize=True, thresholds=None, model_kwargs=None):
    folds = stratified_folds(y, n_folds, seed)
    odds = np.zeros(len(y))
    f1s = []
    for fit_idx, val_idx in folds:
        prep = Preprocessor(spec, winsorize=winsorize)
        X_fit = prep.fit_transform(df.iloc[fit_idx], y[fit_idx])
        X_val = prep.transform(df.iloc[val_idx])
        model = make_model(**(model_kwargs or {})).fit(X_fit, y[fit_idx], prep.feature_types_)
        odds[val_idx] = model.decision_function(X_val)
        f1s.append(macro_f1(y[val_idx], (odds[val_idx] > 0).astype(int)))
    if thresholds is None:
        thresholds = np.round(np.arange(-2.0, 2.0001, 0.1), 2)
    curves = np.array([[macro_f1(y[v], (odds[v] > t).astype(int)) for t in thresholds] for _, v in folds])
    held = [curves[k, int(np.argmax(np.delete(curves, k, axis=0).mean(axis=0)))] for k in range(len(folds))]
    return {
        "macro_f1": float(np.mean(f1s)), "macro_f1_std": float(np.std(f1s)),
        "macro_f1_held_out_threshold": float(np.mean(held)),
        "best_threshold": float(thresholds[int(np.argmax(curves.mean(axis=0)))]),
        "roc_auc": roc_auc(y, odds), "odds": odds,
    }
