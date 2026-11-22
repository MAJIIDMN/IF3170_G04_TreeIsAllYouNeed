import os

os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "algoritma" / "svm"))

from svm import SVM  # noqa: E402

DATA_DIR = ROOT / "src" / "kaggle" / "dataset"
OUTPUT_DIR = ROOT / "output" / "svm"

SEED = 42
FOLDS = 3
WORKERS = 5
TARGET = "is_canceled"
RARE_MIN_COUNT = 30

NUMERIC = [
    "num_adults", "num_children", "weekend_nights", "weekday_nights", "needs_parking",
    "is_returning_guest", "prior_cancellations", "prior_completed_bookings", "special_requests",
    "days_before_arrival", "nightly_rate_idr", "total_nights", "lead_log", "rate_is_zero",
    "special_requests_missing",
]
TIME_NUMERIC = ["arrival_idx", "booking_idx"]
CATEGORICAL = ["meal_package", "room_category", "booking_channel", "arrival_ym", "booking_ym"]
CLIP_UPPER = {"num_children": 3, "nightly_rate_idr": 6_000_000}
LOG_COLUMNS = ["prior_cancellations", "prior_completed_bookings"]

GAMMA_MULTIPLIERS = [0.5, 1.0, 2.0, 4.0]
C_VALUES = [1.0, 3.0, 10.0, 30.0]
THRESHOLDS = np.round(np.arange(-1.0, 1.0001, 0.05), 2)

ROUND3_C_VALUES = [1.5, 2.0, 3.0, 4.0, 6.0, 8.0]
ROUND3_GAMMA_MULTIPLIERS = [1.0, 1.5, 2.0, 2.5, 3.0]
ROUND3_LEADERS = 4
ROUND3_REPEAT_SEEDS = [7, 2026]

BEST = {"kernel": "rbf", "C": 2.0, "gamma_multiplier": 2.5, "class_weight": None, "use_time": True,
        "threshold": -0.2}


def load_train():
    train = pd.read_csv(DATA_DIR / "train.csv")
    features = [column for column in train.columns if column not in ("id", TARGET)]
    return train.drop_duplicates(subset=features + [TARGET]).reset_index(drop=True)


def add_features(df):
    out = df.copy()
    arrival = pd.to_datetime(
        {"year": out["arrival_year"], "month": out["arrival_month"], "day": out["arrival_day"]}
    )
    booking = arrival - pd.to_timedelta(out["days_before_arrival"], unit="D")
    out["arrival_idx"] = (arrival - pd.Timestamp("2023-07-01")).dt.days.astype(float)
    out["booking_idx"] = out["arrival_idx"] - out["days_before_arrival"]
    out["arrival_ym"] = arrival.dt.strftime("%Y-%m")
    out["booking_ym"] = booking.dt.strftime("%Y-%m")
    out["total_nights"] = out["weekend_nights"] + out["weekday_nights"]
    out["lead_log"] = np.log1p(out["days_before_arrival"])
    out["rate_is_zero"] = out["nightly_rate_idr"].eq(0).astype(float)
    out["special_requests_missing"] = out["special_requests"].isna().astype(float)
    return out


class Preprocessor:
    def __init__(self, use_time=False, onehot_scale=1.0):
        self.numeric_ = NUMERIC + TIME_NUMERIC if use_time else NUMERIC
        self.onehot_scale = onehot_scale

    def fit(self, df):
        self.channel_rate_ = df.groupby("booking_channel")["nightly_rate_idr"].median()
        self.medians_ = df[self.numeric_].median()
        self.categories_ = {}
        for column in CATEGORICAL:
            counts = df[column].fillna("kosong").value_counts()
            self.categories_[column] = sorted(counts[counts >= RARE_MIN_COUNT].index) + ["lainnya"]
        matrix = self._numeric(df)
        self.mean_ = matrix.mean(axis=0)
        self.std_ = matrix.std(axis=0)
        self.std_[self.std_ == 0] = 1.0
        return self

    def transform(self, df):
        numeric = (self._numeric(df) - self.mean_) / self.std_
        one_hot = [self.onehot_scale * self._one_hot(df, column) for column in CATEGORICAL]
        return np.hstack([numeric] + one_hot)

    def _numeric(self, df):
        out = df[self.numeric_].copy()
        out["nightly_rate_idr"] = out["nightly_rate_idr"].fillna(df["booking_channel"].map(self.channel_rate_))
        out = out.fillna(self.medians_)
        for column, upper in CLIP_UPPER.items():
            out[column] = out[column].clip(upper=upper)
        out[LOG_COLUMNS] = np.log1p(out[LOG_COLUMNS])
        return out.to_numpy(dtype=np.float64)

    def _one_hot(self, df, column):
        categories = self.categories_[column]
        values = df[column].fillna("kosong")
        values = values.where(values.isin(categories), "lainnya").to_numpy()
        return (values[:, None] == np.array(categories, dtype=object)[None, :]).astype(np.float64)


def stratified_folds(y, n_folds, seed):
    rng = np.random.default_rng(seed)
    fold_of = np.empty(len(y), dtype=int)
    for label in np.unique(y):
        members = rng.permutation(np.flatnonzero(y == label))
        fold_of[members] = np.arange(len(members)) % n_folds
    return [(np.flatnonzero(fold_of != k), np.flatnonzero(fold_of == k)) for k in range(n_folds)]


def macro_f1(y_true, y_pred):
    scores = []
    for label in (0, 1):
        true_positive = np.sum((y_true == label) & (y_pred == label))
        denominator = np.sum(y_true == label) + np.sum(y_pred == label)
        scores.append(2 * true_positive / denominator if denominator else 0.0)
    return float(np.mean(scores))


def scale_gamma(X):
    return 1.0 / (X.shape[1] * X.var())


def make_preprocessor(config):
    return Preprocessor(config.get("use_time", False), config.get("onehot_scale", 1.0))


def make_svm(config, X):
    return SVM(
        kernel=config["kernel"],
        C=config["C"],
        gamma=config["gamma_multiplier"] * scale_gamma(X),
        degree=config.get("degree", 3),
        coef0=config.get("coef0", 0.0),
        class_weight=config["class_weight"],
        cache_mb=512,
    )


def fit_fold(task):
    config, fit_idx, val_idx = task
    frame = add_features(load_train())
    y = frame[TARGET].to_numpy()
    prep = make_preprocessor(config).fit(frame.iloc[fit_idx])
    X_fit, X_val = prep.transform(frame.iloc[fit_idx]), prep.transform(frame.iloc[val_idx])
    start = time.perf_counter()
    model = make_svm(config, X_fit).fit(X_fit, y[fit_idx])
    return config, val_idx, model.decision_function(X_val), time.perf_counter() - start


def held_out_threshold_score(fold_curves):
    scores = []
    for k in range(len(fold_curves)):
        others = np.delete(fold_curves, k, axis=0).mean(axis=0)
        scores.append(fold_curves[k, int(np.argmax(others))])
    return float(np.mean(scores))


def evaluate(configs, seed=SEED):
    y = load_train()[TARGET].to_numpy()
    folds = stratified_folds(y, FOLDS, seed)
    tasks = [(config, fit_idx, val_idx) for config in configs for fit_idx, val_idx in folds]

    scores = {}
    seconds = {}
    with ProcessPoolExecutor(WORKERS) as pool:
        for config, val_idx, fold_scores, elapsed in pool.map(fit_fold, tasks):
            key = str(config)
            scores.setdefault(key, np.zeros(len(y)))[val_idx] = fold_scores
            seconds[key] = seconds.get(key, 0.0) + elapsed

    rows = []
    for config in configs:
        key = str(config)
        per_fold = [macro_f1(y[v], (scores[key][v] > 0).astype(int)) for _, v in folds]
        fold_curves = np.array([
            [macro_f1(y[v], (scores[key][v] > t).astype(int)) for t in THRESHOLDS]
            for _, v in folds
        ])
        curve = fold_curves.mean(axis=0)
        best = int(np.argmax(curve))
        rows.append({
            **config,
            "seed": seed,
            "macro_f1": float(np.mean(per_fold)),
            "macro_f1_std": float(np.std(per_fold)),
            "best_threshold": float(THRESHOLDS[best]),
            "macro_f1_at_threshold": float(curve[best]),
            "macro_f1_held_out_threshold": held_out_threshold_score(fold_curves),
            "fit_seconds": seconds[key],
        })
        print(rows[-1], flush=True)
    return pd.DataFrame(rows)


def tune():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    grid = [
        {"kernel": "rbf", "C": c, "gamma_multiplier": g, "class_weight": None}
        for c in C_VALUES for g in GAMMA_MULTIPLIERS
    ]
    results = evaluate(grid)

    top = results.sort_values("macro_f1", ascending=False).iloc[0]
    weighted = [
        {"kernel": "rbf", "C": float(top["C"]), "gamma_multiplier": float(top["gamma_multiplier"]),
         "class_weight": "balanced"},
    ]
    results = pd.concat([results, evaluate(weighted)], ignore_index=True)
    results.to_csv(OUTPUT_DIR / "tuning.csv", index=False)
    print(results.sort_values("macro_f1_at_threshold", ascending=False).head(8).to_string())


def tune_round2():
    def rbf(c, g, **extra):
        return {"kernel": "rbf", "C": c, "gamma_multiplier": g, "class_weight": None, **extra}

    def poly(c, degree, g=1.0):
        return {"kernel": "poly", "C": c, "gamma_multiplier": g, "class_weight": None,
                "degree": degree, "coef0": 1.0}

    grid = [
        rbf(10.0, 1.0),
        rbf(3.0, 2.0, use_time=True), rbf(10.0, 1.0, use_time=True),
        rbf(30.0, 0.5, use_time=True), rbf(10.0, 2.0, use_time=True),
        rbf(10.0, 1.0, onehot_scale=0.5), rbf(10.0, 1.0, onehot_scale=1.5),
        rbf(10.0, 1.0, onehot_scale=2.0), rbf(3.0, 2.0, onehot_scale=2.0),
        rbf(10.0, 1.0, use_time=True, onehot_scale=0.5),
        rbf(10.0, 1.0, use_time=True, onehot_scale=2.0),
        poly(1.0, 3), poly(3.0, 3), poly(1.0, 4), poly(3.0, 4), poly(1.0, 5),
    ]
    results = evaluate(grid)
    results.to_csv(OUTPUT_DIR / "tuning_round2.csv", index=False)
    columns = ["kernel", "C", "gamma_multiplier", "degree", "use_time", "onehot_scale", "macro_f1",
               "macro_f1_std", "best_threshold", "macro_f1_at_threshold", "fit_seconds"]
    print(results.sort_values("macro_f1", ascending=False)[columns].round(4).to_string())


def tune_round3():
    def rbf(c, g, **extra):
        return {"kernel": "rbf", "C": c, "gamma_multiplier": g, "class_weight": None, **extra}

    baseline = rbf(BEST["C"], BEST["gamma_multiplier"])
    grid = [rbf(c, g, use_time=True) for c in ROUND3_C_VALUES for g in ROUND3_GAMMA_MULTIPLIERS]
    path = OUTPUT_DIR / "tuning_round3.csv"

    results = evaluate([baseline] + grid)
    results.to_csv(path, index=False)

    ranked = results.iloc[1:].sort_values("macro_f1_held_out_threshold", ascending=False)
    leaders = [
        rbf(float(row["C"]), float(row["gamma_multiplier"]), use_time=True)
        for _, row in ranked.head(ROUND3_LEADERS).iterrows()
    ]
    for seed in ROUND3_REPEAT_SEEDS:
        results = pd.concat([results, evaluate([baseline] + leaders, seed)], ignore_index=True)
        results.to_csv(path, index=False)

    repeated = results.groupby(["C", "gamma_multiplier", "use_time"], dropna=False).filter(
        lambda group: len(group) > 1
    )
    summary = repeated.groupby(["C", "gamma_multiplier", "use_time"], dropna=False).agg(
        runs=("seed", "size"),
        macro_f1=("macro_f1", "mean"),
        held_out=("macro_f1_held_out_threshold", "mean"),
        held_out_spread=("macro_f1_held_out_threshold", "std"),
        threshold=("best_threshold", "mean"),
    )
    print(summary.sort_values("held_out", ascending=False).round(4).to_string())


def submit():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train = add_features(load_train())
    test_raw = pd.read_csv(DATA_DIR / "test.csv")
    test = add_features(test_raw)

    prep = make_preprocessor(BEST).fit(train)
    X_train, X_test = prep.transform(train), prep.transform(test)
    model = make_svm(BEST, X_train).fit(X_train, train[TARGET].to_numpy())
    model.save(OUTPUT_DIR / "svm_model.pkl")

    labels = model.predict(X_test, threshold=BEST["threshold"]).astype(int)
    submission = pd.DataFrame({"id": test_raw["id"], TARGET: labels})
    submission.to_csv(OUTPUT_DIR / "submission.csv", index=False)
    print(f"iterations={model.n_iter_} support_vectors={len(model.support_vectors_)} "
          f"canceled_share={labels.mean():.4f} rows={len(submission)}")


if __name__ == "__main__":
    {"tune": tune, "tune2": tune_round2, "tune3": tune_round3, "submit": submit}[sys.argv[1]]()
