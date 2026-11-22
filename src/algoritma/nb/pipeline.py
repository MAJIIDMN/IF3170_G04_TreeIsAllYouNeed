import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from nb import NaiveBayes 
from preprocessing import (ID_COL, TARGET, Preprocessor, cross_validate, load_test, load_train)
from tan import TANNaiveBayes

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
ARTIFACT_VERSION = 1
SEED = 42

def cross(*parts):
    return ("cross", list(parts))

def build_pool():
    pool = {}
    for name in ["num_adults", "num_children", "weekend_nights", "weekday_nights", "needs_parking",
                 "is_returning_guest", "prior_cancellations", "prior_completed_bookings",
                 "special_requests", "arrival_year", "arrival_month", "arrival_day",
                 "meal_package", "room_category", "booking_channel"]:
        pool[name] = "cat"
    pool["lead_q5"] = cross(("days_before_arrival", "q5"))
    pool["lead_q10"] = cross(("days_before_arrival", "q10"))
    pool["lead_mdl"] = cross(("days_before_arrival", "mdl"))
    pool["rate_q5"] = cross(("nightly_rate_idr", "q5"))
    pool["rate_q10"] = cross(("nightly_rate_idr", "q10"))
    pool["rate_mdl"] = cross(("nightly_rate_idr", "mdl"))
    pool["days_before_arrival"] = "gauss"
    pool["nightly_rate_idr"] = "gauss"
    period = (("arrival_year", "cat"), ("arrival_month", "cat"))
    pool["x_period"] = cross(*period)
    pool["x_lead5_period"] = cross(("days_before_arrival", "q5"), *period)
    pool["x_lead4_period"] = cross(("days_before_arrival", "q4"), *period)
    pool["x_period_channel"] = cross(*period, ("booking_channel", "cat"))
    pool["x_sr_channel"] = cross(("special_requests", "cat"), ("booking_channel", "cat"))
    pool["x_history"] = cross(("is_returning_guest", "cat"), ("prior_completed_bookings", "cat"),
                              ("prior_cancellations", "cat"))
    pool["x_channel_lead5"] = cross(("booking_channel", "cat"), ("days_before_arrival", "q5"))
    pool["x_channel_rate5"] = cross(("booking_channel", "cat"), ("nightly_rate_idr", "q5"))
    pool["x_meal_channel"] = cross(("meal_package", "cat"), ("booking_channel", "cat"))
    pool["x_sr_lead5"] = cross(("special_requests", "cat"), ("days_before_arrival", "q5"))
    pool["x_adults_children"] = cross(("num_adults", "cat"), ("num_children", "cat"))
    pool["x_nights"] = cross(("weekend_nights", "cat"), ("weekday_nights", "cat"))
    pool["x_lead5_rate5"] = cross(("days_before_arrival", "q5"), ("nightly_rate_idr", "q5"))
    pool["x_room_meal"] = cross(("room_category", "cat"), ("meal_package", "cat"))
    pool["x_sr_rate5"] = cross(("special_requests", "cat"), ("nightly_rate_idr", "q5"))
    pool["x_lead5_parking"] = cross(("days_before_arrival", "q5"), ("needs_parking", "cat"))
    pool["x_channel_lead3"] = cross(("booking_channel", "cat"), ("days_before_arrival", "q3"))
    return pool

FINAL_FEATURES = ["x_lead5_period", "x_sr_channel", "rate_mdl", "weekend_nights", "needs_parking", "room_category", "arrival_day", "prior_completed_bookings"]
MODELS = {
    "nb": {"cls": NaiveBayes, "alpha": 1.0, "threshold": -0.2},
    "tan": {"cls": TANNaiveBayes, "alpha": 0.5, "threshold": -0.2},
}
ALPHA_GRID = (0.5, 1.0, 2.0)
THRESHOLD_GRID = np.round(np.arange(-4.0, 4.0001, 0.1), 2)
FINAL_ALPHA = MODELS["nb"]["alpha"]
FINAL_THRESHOLD = MODELS["nb"]["threshold"]

COST_FP = 2.0
COST_FN = 1.0
COST_THRESHOLD = float(np.log(COST_FP / COST_FN))

def expected_cost(y_true, y_pred, c_fp=COST_FP, c_fn=COST_FN):
    false_positive = np.sum((y_pred == 1) & (y_true == 0))
    false_negative = np.sum((y_pred == 0) & (y_true == 1))
    return float((c_fp * false_positive + c_fn * false_negative) / len(y_true))

def final_spec():
    pool = build_pool()
    return {name: pool[name] for name in FINAL_FEATURES}

def pca_fit(X, n_components=None):
    mean = X.mean(axis=0)
    scale = X.std(axis=0)
    scale[scale == 0] = 1.0
    Z = (X - mean) / scale
    _, singular, vt = np.linalg.svd(Z, full_matrices=False)
    variance = singular ** 2 / (len(Z) - 1)
    ratio = variance / variance.sum()
    k = n_components or len(ratio)
    return mean, scale, vt[:k], ratio

def pca_transform(X, mean, scale, components):
    return ((X - mean) / scale) @ components.T

def artifact_path(model_name):
    return OUTPUT_DIR / f"{model_name}_artifact.json"

def train(model_name, train_df=None):
    config = MODELS[model_name]
    train_df = load_train() if train_df is None else train_df
    y = train_df[TARGET].to_numpy()
    prep = Preprocessor(final_spec()).fit(train_df, y)
    model = config["cls"](alpha=config["alpha"]).fit(prep.transform(train_df), y, prep.feature_types_)
    return prep, model

def save_artifact(model_name, prep, model, meta):
    meta = {"format_version": ARTIFACT_VERSION, "model": model_name,
            "alpha": MODELS[model_name]["alpha"], "threshold": MODELS[model_name]["threshold"],
            "features": FINAL_FEATURES, "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "python_version": platform.python_version(), "numpy_version": np.__version__, **meta}
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = artifact_path(model_name)
    with open(path, "w", encoding="utf-8") as file:
        json.dump({"meta": meta, "preprocessor": prep.to_dict(), "model": model.to_dict()}, file)
    return path

def load_artifact(model_name):
    with open(artifact_path(model_name), encoding="utf-8") as file:
        payload = json.load(file)
    if payload["meta"]["format_version"] != ARTIFACT_VERSION:
        raise ValueError(f"artifact format {payload['meta']['format_version']} != {ARTIFACT_VERSION}")
    prep = Preprocessor.from_dict(payload["preprocessor"])
    model = MODELS[model_name]["cls"].from_dict(payload["model"])
    return prep, model, payload["meta"]

def predict(model_name, test_df, threshold=None):
    prep, model, meta = load_artifact(model_name)
    threshold = meta["threshold"] if threshold is None else threshold
    return model.predict(prep.transform(test_df), threshold=threshold).astype(int)

def write_submission(model_name, cost_sensitive=False):
    test_df = load_test()
    threshold = COST_THRESHOLD if cost_sensitive else None
    labels = predict(model_name, test_df, threshold)
    filename = f"submission_{model_name}{'_cost_sensitive' if cost_sensitive else ''}.csv"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({ID_COL: test_df[ID_COL], TARGET: labels}).to_csv(OUTPUT_DIR / filename, index=False)
    print(f"{filename} canceled_share={labels.mean():.4f} rows={len(labels)}")
    return OUTPUT_DIR / filename

def tune(model_name):
    train_df = load_train()
    y = train_df[TARGET].to_numpy()
    results = []
    for alpha in ALPHA_GRID:
        r = cross_validate(final_spec(), MODELS[model_name]["cls"], train_df, y, seed=SEED,
                           thresholds=THRESHOLD_GRID, model_kwargs={"alpha": alpha})
        results.append((r["macro_f1_held_out_threshold"], r["roc_auc"], r["best_threshold"], alpha))
        print(f"alpha={alpha} cv_f1={results[-1][0]:.4f} cv_auc={r['roc_auc']:.4f} "
              f"threshold={r['best_threshold']:+.1f}", flush=True)
    score, auc, threshold, alpha = max(results, key=lambda item: (item[0], item[1]))
    print(f"best: alpha={alpha} threshold={threshold:+.1f} cv_f1={score:.4f} "
          f"(MODELS['{model_name}'] currently alpha={MODELS[model_name]['alpha']} "
          f"threshold={MODELS[model_name]['threshold']:+.1f})")
    return alpha, threshold

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["tune", "train", "predict"])
    parser.add_argument("--model", choices=sorted(MODELS), required=True)
    parser.add_argument("--cost-sensitive", action="store_true", help="predict with the FP=2xFN threshold")
    args = parser.parse_args(argv)
    if args.command == "tune":
        tune(args.model)
    elif args.command == "train":
        train_df = load_train()
        y = train_df[TARGET].to_numpy()
        config = MODELS[args.model]
        cv = cross_validate(final_spec(), config["cls"], train_df, y, seed=SEED,
                            thresholds=THRESHOLD_GRID, model_kwargs={"alpha": config["alpha"]})
        prep, model = train(args.model, train_df)
        path = save_artifact(args.model, prep, model, {
            "seed": SEED, "n_train_rows": len(train_df), "cv_macro_f1": cv["macro_f1_held_out_threshold"],
            "cv_roc_auc": cv["roc_auc"], "cv_best_threshold": cv["best_threshold"]})
        print(f"saved {path}  cv_f1={cv['macro_f1_held_out_threshold']:.4f} cv_auc={cv['roc_auc']:.4f}")
    else:
        write_submission(args.model, cost_sensitive=args.cost_sensitive)


if __name__ == "__main__":
    main()
