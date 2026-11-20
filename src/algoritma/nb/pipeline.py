import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from nb import NaiveBayes  # noqa: E402
from preprocessing import ID_COL, TARGET, Preprocessor, load_test, load_train

OUTPUT_DIR = Path(__file__).resolve().parent / "output"

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
FINAL_ALPHA = 1.0
FINAL_THRESHOLD = -0.2

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

def write_submission(threshold, filename, save_model=False):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train = load_train()
    test_raw = load_test()
    y = train[TARGET].to_numpy()

    prep = Preprocessor(final_spec()).fit(train, y)
    model = NaiveBayes(alpha=FINAL_ALPHA).fit(prep.transform(train), y, prep.feature_types_)
    if save_model:
        model.save(OUTPUT_DIR / "nb_model.json")

    labels = model.predict(prep.transform(test_raw), threshold=threshold).astype(int)
    submission = pd.DataFrame({ID_COL: test_raw[ID_COL], TARGET: labels})
    submission.to_csv(OUTPUT_DIR / filename, index=False)
    print(f"{filename} threshold={threshold:.3f} canceled_share={labels.mean():.4f} rows={len(submission)}")

def submit():
    write_submission(FINAL_THRESHOLD, "submission.csv", save_model=True)

def submit_cost():
    write_submission(COST_THRESHOLD, "submission_cost_sensitive.csv")


if __name__ == "__main__":
    {"submit": submit, "submit-cost": submit_cost}[sys.argv[1]]()
