import os

os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import copy
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "algoritma"))

from dtl_alt import DTL  # noqa: E402
from dtl_alt.pruning import pessimistic_prune  # noqa: E402

DATA_DIR = ROOT / "src" / "kaggle" / "dataset"
OUTPUT_DIR = ROOT / "output" / "dtl"

TARGET = "is_canceled"
WORKERS = 8
FOLDS = 5
TUNE_SEEDS = [42, 7]
CV_PLAN = [(5, 2026), (5, 99), (5, 123), (3, 42), (3, 7), (3, 2026)]
THRESHOLDS = np.round(np.arange(0.2, 0.8001, 0.02), 2)

RAW = [
    "num_adults", "num_children", "weekend_nights", "weekday_nights", "meal_package",
    "needs_parking", "room_category", "days_before_arrival", "arrival_year", "arrival_month",
    "arrival_day", "booking_channel", "is_returning_guest", "prior_cancellations",
    "prior_completed_bookings", "nightly_rate_idr", "special_requests",
]
TIME_FEATURES = ["arrival_idx", "booking_idx", "total_nights", "arrival_dow", "arrival_month_idx"]
FEATURE_SETS = {
    "time": RAW + TIME_FEATURES,
    "time_missing_flag": RAW + TIME_FEATURES + ["special_requests_missing"],
    "full": RAW + TIME_FEATURES + ["special_requests_missing", "booking_month_idx"],
}

LEAF_SIZES = [2, 3, 4, 6]
CONFIDENCES = [0.03, 0.05, 0.07, 0.1, 0.15, 0.2]

BEST = {"features": "full", "max_depth": 30, "min_samples_leaf": 2, "confidence": 0.05}


def load_train():
    train = pd.read_csv(DATA_DIR / "train.csv")
    return train.drop_duplicates(subset=RAW + [TARGET]).reset_index(drop=True)


def add_features(df):
    out = df.copy()
    arrival = pd.to_datetime(
        {"year": out["arrival_year"], "month": out["arrival_month"], "day": out["arrival_day"]}
    )
    out["arrival_idx"] = (arrival - pd.Timestamp("2023-07-01")).dt.days.astype(float)
    out["booking_idx"] = out["arrival_idx"] - out["days_before_arrival"]
    out["total_nights"] = out["weekend_nights"] + out["weekday_nights"]
    out["arrival_dow"] = arrival.dt.day_of_week.astype(float)
    out["arrival_month_idx"] = (out["arrival_year"] - 2023) * 12 + out["arrival_month"]
    out["special_requests_missing"] = out["special_requests"].isna().astype(float)
    out["booking_month_idx"] = np.floor(out["booking_idx"] / 30.4375)
    return out


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


def score(y, proba, folds):
    curves = np.array([
        [macro_f1(y[v], (proba[v] > t).astype(int)) for t in THRESHOLDS] for _, v in folds
    ])
    held_out = [
        curves[k, int(np.argmax(np.delete(curves, k, axis=0).mean(axis=0)))]
        for k in range(len(folds))
    ]
    per_fold = [macro_f1(y[v], (proba[v] > 0.5).astype(int)) for _, v in folds]
    return {
        "macro_f1": float(np.mean(per_fold)),
        "macro_f1_std": float(np.std(per_fold)),
        "macro_f1_held_out_threshold": float(np.mean(held_out)),
        "best_threshold": float(THRESHOLDS[int(np.argmax(curves.mean(axis=0)))]),
    }


def make_tree(config):
    return DTL(
        max_depth=config["max_depth"],
        min_samples_leaf=config["min_samples_leaf"],
        pruning="pessimistic",
        confidence=config["confidence"],
    )


def fit_fold(task):
    config, fit_idx, val_idx = task
    frame = add_features(load_train())
    columns = FEATURE_SETS[config["features"]]
    model = make_tree(config).fit(frame.iloc[fit_idx][columns], frame[TARGET].to_numpy()[fit_idx])
    return val_idx, model.predict_proba(frame.iloc[val_idx][columns])[:, 1]


def fit_fold_all_confidences(task):
    features, leaf_size, seed, fit_idx, val_idx = task
    frame = add_features(load_train())
    columns = FEATURE_SETS[features]
    model = DTL(max_depth=BEST["max_depth"], min_samples_leaf=leaf_size)
    model.fit(frame.iloc[fit_idx][columns], frame[TARGET].to_numpy()[fit_idx])
    unpruned = model.root_
    probabilities = {}
    for confidence in CONFIDENCES:
        model.root_ = copy.deepcopy(unpruned)
        pessimistic_prune(model.root_, confidence)
        probabilities[confidence] = model.predict_proba(frame.iloc[val_idx][columns])[:, 1]
    return features, leaf_size, seed, val_idx, probabilities


def tune():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    y = load_train()[TARGET].to_numpy()
    splits = {seed: stratified_folds(y, FOLDS, seed) for seed in TUNE_SEEDS}
    tasks = [
        (features, leaf_size, seed, fit_idx, val_idx)
        for features in FEATURE_SETS for leaf_size in LEAF_SIZES
        for seed in TUNE_SEEDS for fit_idx, val_idx in splits[seed]
    ]
    collected = {}
    with ProcessPoolExecutor(WORKERS) as pool:
        for features, leaf_size, seed, val_idx, probabilities in pool.map(fit_fold_all_confidences, tasks):
            for confidence, proba in probabilities.items():
                key = (features, leaf_size, confidence, seed)
                collected.setdefault(key, np.zeros(len(y)))[val_idx] = proba
    rows = [
        {"features": f, "min_samples_leaf": l, "confidence": c, "seed": s, **score(y, p, splits[s])}
        for (f, l, c, s), p in collected.items()
    ]
    results = pd.DataFrame(rows)
    results.to_csv(OUTPUT_DIR / "tuning.csv", index=False)
    summary = results.groupby(["features", "min_samples_leaf", "confidence"])[
        ["macro_f1", "macro_f1_held_out_threshold"]
    ].mean()
    print(summary.sort_values("macro_f1", ascending=False).head(10).round(4).to_string())


def cv():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    y = load_train()[TARGET].to_numpy()
    rows = []
    with ProcessPoolExecutor(WORKERS) as pool:
        for n_folds, seed in CV_PLAN:
            folds = stratified_folds(y, n_folds, seed)
            proba = np.zeros(len(y))
            tasks = [(BEST, fit_idx, val_idx) for fit_idx, val_idx in folds]
            for val_idx, fold_proba in pool.map(fit_fold, tasks):
                proba[val_idx] = fold_proba
            rows.append({"folds": n_folds, "seed": seed, **score(y, proba, folds)})
            print(rows[-1], flush=True)
    results = pd.DataFrame(rows)
    results.to_csv(OUTPUT_DIR / "cv.csv", index=False)
    print(results.groupby("folds")["macro_f1"].mean().round(4).to_string())


def submit():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train = add_features(load_train())
    test_raw = pd.read_csv(DATA_DIR / "test.csv")
    test = add_features(test_raw)

    columns = FEATURE_SETS[BEST["features"]]
    model = make_tree(BEST).fit(train[columns], train[TARGET].to_numpy())
    model.save(OUTPUT_DIR / "dtl_model.json")

    labels = model.predict(test[columns], threshold=0.5).astype(int)
    submission = pd.DataFrame({"id": test_raw["id"], TARGET: labels})
    submission.to_csv(OUTPUT_DIR / "submission.csv", index=False)
    print(f"leaves={model.n_leaves_} depth={model.depth_} canceled_share={labels.mean():.4f} "
          f"rows={len(submission)}")


if __name__ == "__main__":
    {"tune": tune, "cv": cv, "submit": submit}[sys.argv[1]]()
