from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from algoritma.knn.knn import KNN

DATA_DIR = ROOT / "src" / "kaggle" / "dataset"
OUTPUT_DIR = ROOT / "output" / "knn"
SEED = 42
TARGET = "is_canceled"
CATEGORICAL = ["meal_package", "room_category", "booking_channel"]
ORIGINAL = [
    "num_adults", "num_children", "weekend_nights", "weekday_nights",
    "meal_package", "needs_parking", "room_category", "days_before_arrival",
    "arrival_year", "arrival_month", "arrival_day", "booking_channel",
    "is_returning_guest", "prior_cancellations", "prior_completed_bookings",
    "nightly_rate_idr", "special_requests"
]
ENGINEERED = ["total_nights", "lead_log", "arrival_index", "booking_index"]
ALL_FEATURES = ORIGINAL + ENGINEERED
ENGINEERED_17 = [
    "num_adults", "num_children", "needs_parking", "days_before_arrival",
    "arrival_month", "booking_channel", "is_returning_guest", "prior_cancellations",
    "prior_completed_bookings", "nightly_rate_idr", "special_requests",
    "meal_package", "room_category", "total_nights", "lead_log",
    "arrival_index", "booking_index"
]


def canonical(features):
    available = set(features)
    return [feature for feature in ALL_FEATURES if feature in available]


def load_data():
    train = pd.read_csv(DATA_DIR / "train.csv", dtype={"id": str})
    test = pd.read_csv(DATA_DIR / "test.csv", dtype={"id": str})
    sample = pd.read_csv(DATA_DIR / "sample_submission.csv", dtype={"id": str})
    return train, test, sample


def clean_training_data(frame):
    return frame.drop_duplicates(subset=ORIGINAL + [TARGET]).reset_index(drop=True)


def add_features(frame):
    data = frame.copy()
    for feature in ORIGINAL:
        if feature not in CATEGORICAL:
            data[feature] = pd.to_numeric(data[feature], errors="coerce")
    data["total_nights"] = data["weekend_nights"] + data["weekday_nights"]
    data["lead_log"] = np.log1p(data["days_before_arrival"].clip(lower=0))
    dates = pd.to_datetime(
        {"year": data["arrival_year"], "month": data["arrival_month"], "day": data["arrival_day"]},
        errors="coerce"
    )
    data["arrival_index"] = (dates - pd.Timestamp("2023-01-01")).dt.days.astype(float)
    data["booking_index"] = data["arrival_index"] - data["days_before_arrival"]
    return data


class BookingPreprocessor:
    def __init__(self):
        self.numeric_stats = {}
        self.categorical_stats = {}
        self.fitted_ = False

    def fit(self, frame):
        self.numeric_stats = {}
        self.categorical_stats = {}
        for feature in ALL_FEATURES:
            if feature in CATEGORICAL:
                series = frame[feature]
                modes = series.dropna().mode()
                fill = str(modes.iloc[0]) if len(modes) else "__MISSING__"
                values = series.fillna(fill).astype(str)
                self.categorical_stats[feature] = (fill, sorted(set(values)))
            else:
                series = pd.to_numeric(frame[feature], errors="coerce")
                median = float(series.median()) if series.notna().any() else 0.0
                values = series.fillna(median).to_numpy(np.float64)
                self.numeric_stats[feature] = (median, float(values.min()), float(values.max()))
        self.fitted_ = True
        return self

    def transform_columns(self, frame):
        if not self.fitted_:
            raise RuntimeError("Preprocessor has not been fitted")
        result = {}
        for feature in ALL_FEATURES:
            if feature in CATEGORICAL:
                fill, categories = self.categorical_stats[feature]
                values = frame[feature].fillna(fill).astype(str).to_numpy()
                result[feature] = (
                    np.column_stack([values == category for category in categories]).astype(np.float32)
                    / np.sqrt(2.0)
                )
            else:
                median, minimum, maximum = self.numeric_stats[feature]
                series = pd.to_numeric(frame[feature], errors="coerce")
                values = series.fillna(median).to_numpy(np.float64)
                span = maximum - minimum
                scaled = (values - minimum) / span if span > 0 else np.zeros(len(values))
                result[feature] = scaled.astype(np.float32)[:, None]
        return result

    def transform(self, frame, features):
        return assemble(self.transform_columns(frame), features)


def assemble(columns, features):
    return np.hstack([columns[feature] for feature in canonical(features)]).astype(np.float32)


def candidate_features():
    variants = {
        "original_17": ORIGINAL,
        "original_21": ALL_FEATURES,
        "engineered_17": ENGINEERED_17,
        "engineered_without_meal": [x for x in ENGINEERED_17 if x != "meal_package"],
        "engineered_without_room": [x for x in ENGINEERED_17 if x != "room_category"],
        "engineered_without_both_categories": [x for x in ENGINEERED_17 if x not in {"meal_package", "room_category"}],
        "engineered_without_booking_index": [x for x in ENGINEERED_17 if x != "booking_index"],
        "engineered_without_history": [x for x in ENGINEERED_17 if x not in {"prior_cancellations", "prior_completed_bookings"}],
        "engineered_without_lead_log": [x for x in ENGINEERED_17 if x != "lead_log"],
        "engineered_without_total_nights": [x for x in ENGINEERED_17 if x != "total_nights"]
    }
    return {name: canonical(features) for name, features in variants.items()}


def make_cv_frame(stage, evaluations, features_by_name, folds, k, metric, p=None):
    rows = []
    for name, scores in evaluations.items():
        result = {
            "stage": stage,
            "candidate": name,
            "k": k,
            "metric": metric,
            "p": p if p is not None else (1 if metric == "manhattan" else 2),
            "folds": folds,
            "n_features": len(features_by_name[name]),
            "features": ";".join(features_by_name[name]),
            "cv_macro_f1": float(np.mean([score[0] for score in scores])),
            "cv_f1_sd": float(np.std([score[0] for score in scores], ddof=1)),
            "cv_accuracy": float(np.mean([score[1] for score in scores]))
        }
        for index, score in enumerate(scores, 1):
            result[f"fold{index}_f1"] = score[0]
        rows.append(result)
    return pd.DataFrame(rows).sort_values(
        ["cv_macro_f1", "cv_f1_sd", "candidate"], ascending=[False, True, True]
    ).reset_index(drop=True)


def evaluate_feature_sets(data, labels, indices, configurations, k=5, metric="manhattan", folds=3):
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
    evaluations = {name: [] for name in configurations}
    for fold, (train_positions, validation_positions) in enumerate(cv.split(indices, labels[indices]), 1):
        training_indices = indices[train_positions]
        validation_indices = indices[validation_positions]
        preprocessor = BookingPreprocessor().fit(data.iloc[training_indices])
        training_columns = preprocessor.transform_columns(data.iloc[training_indices])
        validation_columns = preprocessor.transform_columns(data.iloc[validation_indices])
        for name, features in configurations.items():
            train_matrix = assemble(training_columns, features)
            validation_matrix = assemble(validation_columns, features)
            model = KNN(n_neighbors=k, metric=metric).fit(train_matrix, labels[training_indices])
            predicted = model.predict(validation_matrix)
            evaluations[name].append((
                float(f1_score(labels[validation_indices], predicted, average="macro")),
                float(accuracy_score(labels[validation_indices], predicted))
            ))
        print(f"  Fold {fold}/{folds} complete", flush=True)
    return make_cv_frame("feature_cv", evaluations, configurations, folds, k, metric)


def tune_knn_after_features(data, labels, indices, feature_sets, folds=5):
    k_values = [3, 5, 7, 9, 11, 15, 21]
    metrics = ["manhattan", "euclidean", "minkowski"]
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
    evaluations = {
        (name, metric, k): []
        for name in feature_sets for metric in metrics for k in k_values
    }
    for fold, (train_positions, validation_positions) in enumerate(cv.split(indices, labels[indices]), 1):
        training_indices = indices[train_positions]
        validation_indices = indices[validation_positions]
        preprocessor = BookingPreprocessor().fit(data.iloc[training_indices])
        training_columns = preprocessor.transform_columns(data.iloc[training_indices])
        validation_columns = preprocessor.transform_columns(data.iloc[validation_indices])
        for name, features in feature_sets.items():
            train_matrix = assemble(training_columns, features)
            validation_matrix = assemble(validation_columns, features)
            for metric in metrics:
                model = KNN(n_neighbors=max(k_values), metric=metric, p=3).fit(
                    train_matrix, labels[training_indices]
                )
                predictions = model.predict_many_k(validation_matrix, k_values)
                for k in k_values:
                    predicted = predictions[k]
                    evaluations[(name, metric, k)].append((
                        float(f1_score(labels[validation_indices], predicted, average="macro")),
                        float(accuracy_score(labels[validation_indices], predicted))
                    ))
        print(f"  Fold {fold}/{folds} complete", flush=True)
    rows = []
    for (name, metric, k), scores in evaluations.items():
        row = {
            "stage": "joint_tuning_after_features",
            "candidate": name,
            "k": k,
            "metric": metric,
            "p": 3 if metric == "minkowski" else 1 if metric == "manhattan" else 2,
            "folds": folds,
            "n_features": len(feature_sets[name]),
            "features": ";".join(feature_sets[name]),
            "cv_macro_f1": float(np.mean([score[0] for score in scores])),
            "cv_f1_sd": float(np.std([score[0] for score in scores], ddof=1)),
            "cv_accuracy": float(np.mean([score[1] for score in scores]))
        }
        for index, score in enumerate(scores, 1):
            row[f"fold{index}_f1"] = score[0]
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["cv_macro_f1", "cv_f1_sd", "candidate"], ascending=[False, True, True]
    ).reset_index(drop=True)


def evaluate_holdout(data, labels, development, holdout, features, k, metric):
    preprocessor = BookingPreprocessor().fit(data.iloc[development])
    train_matrix = preprocessor.transform(data.iloc[development], features)
    holdout_matrix = preprocessor.transform(data.iloc[holdout], features)
    model = KNN(n_neighbors=k, metric=metric, p=3).fit(train_matrix, labels[development])
    scratch = model.predict(holdout_matrix)
    sklearn = KNeighborsClassifier(
        n_neighbors=k, metric=metric, p=3, weights="uniform", algorithm="brute", n_jobs=1
    ).fit(train_matrix, labels[development])
    sklearn_predicted = sklearn.predict(holdout_matrix)
    evaluation = {
        "macro_f1": float(f1_score(labels[holdout], scratch, average="macro")),
        "accuracy": float(accuracy_score(labels[holdout], scratch)),
        "confusion_matrix": confusion_matrix(labels[holdout], scratch),
        "sklearn_agreement": float(np.mean(scratch == sklearn_predicted)),
        "sklearn_macro_f1": float(f1_score(labels[holdout], sklearn_predicted, average="macro")),
        "classification_report": classification_report(labels[holdout], scratch, output_dict=True)
    }
    return evaluation


def restore_preprocessor(state):
    preprocessor = BookingPreprocessor()
    preprocessor.numeric_stats = state["numeric_stats"]
    preprocessor.categorical_stats = state["categorical_stats"]
    preprocessor.fitted_ = True
    return preprocessor


def make_final_model(train, test, features, k, metric):
    prepared_train = add_features(train)
    prepared_test = add_features(test)
    preprocessor = BookingPreprocessor().fit(prepared_train)
    training_matrix = preprocessor.transform(prepared_train, features)
    test_matrix = preprocessor.transform(prepared_test, features)
    model = KNN(n_neighbors=k, metric=metric, p=3).fit(
        training_matrix, train[TARGET].to_numpy(np.int8)
    )
    payload = {
        "model": model,
        "preprocessor_state": {
            "numeric_stats": preprocessor.numeric_stats,
            "categorical_stats": preprocessor.categorical_stats
        },
        "features": list(features),
        "seed": SEED,
        "metric": metric,
        "k": k
    }
    with (OUTPUT_DIR / "model.pkl").open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with (OUTPUT_DIR / "model.pkl").open("rb") as handle:
        restored = pickle.load(handle)
    fresh_predicted = model.predict(test_matrix)
    restored_matrix = restore_preprocessor(restored["preprocessor_state"]).transform(prepared_test, restored["features"])
    loaded_predicted = restored["model"].predict(restored_matrix)
    if not np.array_equal(fresh_predicted, loaded_predicted):
        raise RuntimeError("Predictions changed after loading the model")
    submission = pd.DataFrame({"id": test["id"], TARGET: loaded_predicted})
    submission.to_csv(OUTPUT_DIR / "submission.csv", index=False)
    return submission, restored


def run_pipeline():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    raw_train, raw_test, sample = load_data()
    if not sample["id"].equals(raw_test["id"]):
        raise ValueError("Sample submission IDs do not match test IDs")
    train = clean_training_data(raw_train)
    data = add_features(train)
    labels = train[TARGET].to_numpy(np.int8)
    development, holdout = train_test_split(
        np.arange(len(train)), test_size=0.2, random_state=SEED, stratify=labels
    )
    print("\nFeature selection = 3-fold CV", flush=True)
    variants = candidate_features()
    features_cv3 = evaluate_feature_sets(
        data, labels, development, variants, k=5, metric="euclidean", folds=3
    )
    finalists = {
        name: variants[name] for name in features_cv3["candidate"].head(2).tolist()
    }
    print("\nHyperparameter tuning = 5-fold CV", flush=True)
    hyper = tune_knn_after_features(
        data, labels, development, finalists, folds=5
    )
    chosen_name = str(hyper.iloc[0]["candidate"])
    chosen_features = finalists[chosen_name]
    best_k = int(hyper.iloc[0]["k"])
    best_metric = str(hyper.iloc[0]["metric"])
    print("\nBest model", flush=True)
    print(f"Features = {len(chosen_features)}", flush=True)
    print(f"k = {best_k}", flush=True)
    print(f"Distance = {best_metric.capitalize()}", flush=True)
    print(f"CV macro F1 = {hyper.iloc[0]['cv_macro_f1']:.6f}", flush=True)
    combined = pd.concat([
        features_cv3.assign(stage="feature_selection_cv3"),
        hyper.assign(stage="joint_tuning_cv5")
    ], ignore_index=True)
    combined.to_csv(OUTPUT_DIR / "tuning.csv", index=False)
    evaluation = evaluate_holdout(data, labels, development, holdout, chosen_features, best_k, best_metric)
    submission, saved_model = make_final_model(train, raw_test, chosen_features, best_k, best_metric)
    if len(submission) != len(raw_test) or submission["id"].duplicated().any():
        raise RuntimeError("Invalid submission format")
    print(f"Holdout macro F1 = {evaluation['macro_f1']:.6f}", flush=True)
    print(f"Holdout accuracy = {evaluation['accuracy'] * 100:.2f}%", flush=True)
    print(f"Scratch vs sklearn = {evaluation['sklearn_agreement'] * 100:.2f}% match", flush=True)
    print(f"\nModel saved = {OUTPUT_DIR / 'model.pkl'}", flush=True)
    print(f"Submission saved = {OUTPUT_DIR / 'submission.csv'}", flush=True)
    return {
        "raw_train": raw_train,
        "train": train,
        "test": raw_test,
        "development": development,
        "holdout": holdout,
        "features_cv3": features_cv3,
        "finalists": finalists,
        "hyper": hyper,
        "features": chosen_features,
        "selected_name": chosen_name,
        "k": best_k,
        "metric": best_metric,
        "evaluation": evaluation,
        "submission": submission,
        "saved_model": saved_model
    }


if __name__ == "__main__":
    run_pipeline()
