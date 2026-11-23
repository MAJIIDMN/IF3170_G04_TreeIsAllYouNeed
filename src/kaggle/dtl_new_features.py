"""Fitur untuk DTL. Semua statistik dihitung dari train saja.

Referensi:
- D. Micci-Barreca, SIGKDD Explorations 3(1), 2001 (smoothed target encoding).
- Ide fitur saudara dan tanggal booking berasal dari src/kaggle/notebook.ipynb.
"""
import numpy as np
import pandas as pd

ID_COL = "id"
TARGET = "is_canceled"
CAT_COLS = ["meal_package", "room_category", "booking_channel"]
KEY = ["arrival_year", "arrival_month", "arrival_day", "weekend_nights", "weekday_nights",
       "booking_channel"]
SOFT = ["room_category", "meal_package", "num_adults", "num_children", "special_requests",
        "needs_parking"]
LEAD_TOLERANCE = 10
RATE_TOLERANCE = 0.04
INNER_FOLDS = 5


def load_data(data_dir):
    train = pd.read_csv(f"{data_dir}/train.csv", dtype={ID_COL: str})
    test = pd.read_csv(f"{data_dir}/test.csv", dtype={ID_COL: str})
    return train, test


def clean(train):
    return train.drop(columns=[ID_COL]).drop_duplicates().reset_index(drop=True)


def add_booking_dates(df):
    out = df.drop(columns=[c for c in (ID_COL, TARGET) if c in df]).reset_index(drop=True)
    arrival = pd.to_datetime({"year": out["arrival_year"], "month": out["arrival_month"],
                              "day": out["arrival_day"]})
    out["arr_ord"] = (arrival - pd.Timestamp("2023-01-01")).dt.days.astype(float)
    out["book_ord"] = out["arr_ord"] - out["days_before_arrival"]
    out["arr_dow"] = arrival.dt.day_of_week.astype(float)
    return out


def impute_and_flag(train, other):
    fill, flags = {}, []
    for col in train.columns:
        fill[col] = train[col].mode().iloc[0] if col in CAT_COLS else train[col].median()
        if train[col].isna().any():
            flags.append(col)
    results = []
    for df in (train, other):
        out = df.copy()
        for col, value in fill.items():
            out[col] = out[col].fillna(value)
        for col in flags:
            out[f"missing_{col}"] = df[col].isna().astype(float)
        results.append(out)
    return results


def encode_siblings(reference, labels, query, prior, smoothing=1.0):
    """Rata-rata label saudara, dibobot kedekatan lead time, dan jumlah bobotnya."""
    vocab = {c: {v: float(i) for i, v in enumerate(sorted(reference[c].dropna().unique()))}
             for c in KEY + SOFT if c in CAT_COLS}

    def codes(df, cols):
        return np.column_stack([df[c].map(vocab[c]).to_numpy(float) if c in vocab
                                else df[c].to_numpy(float) for c in cols])

    groups = {}
    for j, key in enumerate(map(tuple, codes(reference, KEY))):
        groups.setdefault(key, []).append(j)
    groups = {k: np.array(v) for k, v in groups.items()}
    ref_lead, lead = (d["days_before_arrival"].to_numpy(float) for d in (reference, query))
    ref_rate, rate = (d["nightly_rate_idr"].to_numpy(float) for d in (reference, query))
    ref_soft, soft = codes(reference, SOFT), codes(query, SOFT)
    total, weight_sum = np.zeros(len(query)), np.zeros(len(query))
    for i, key in enumerate(map(tuple, codes(query, KEY))):
        idx = groups.get(key)
        if idx is None:
            continue
        gap = np.abs(ref_lead[idx] - lead[i])
        ok = np.isnan(gap) | (gap <= LEAD_TOLERANCE)
        rate_gap = np.abs(ref_rate[idx] - rate[i])
        ok &= np.isnan(rate_gap) | (rate_gap <= RATE_TOLERANCE * np.fmax(rate[i], 1.0))
        ok &= np.all(np.isnan(ref_soft[idx]) | np.isnan(soft[i]) | (ref_soft[idx] == soft[i]), axis=1)
        gap = gap[ok]
        w = np.where(np.isnan(gap), 0.1, np.clip(1 - gap / (LEAD_TOLERANCE + 1), 0, 1))
        weight_sum[i] = w.sum()
        total[i] = (w * labels[idx[ok]]).sum()
    return (total + smoothing * prior) / (weight_sum + smoothing), weight_sum


def build_features(train, labels, other):
    """Fitur train (saudara out-of-fold) dan fitur data lain (hanya memakai label train)."""
    labels = np.asarray(labels, dtype=float)
    raw_train = train.drop(columns=[TARGET]).reset_index(drop=True)
    raw_other = other.drop(columns=[c for c in (ID_COL, TARGET) if c in other]).reset_index(drop=True)
    A, B = impute_and_flag(add_booking_dates(raw_train), add_booking_dates(raw_other))

    prior = labels.mean()
    te, count = np.zeros(len(labels)), np.zeros(len(labels))
    order = np.random.default_rng(0).permutation(len(labels))
    for held in np.array_split(order, INNER_FOLDS):
        rest = np.setdiff1d(np.arange(len(labels)), held)
        te[held], count[held] = encode_siblings(raw_train.iloc[rest], labels[rest],
                                                raw_train.iloc[held], prior)
    te_other, count_other = encode_siblings(raw_train, labels, raw_other, prior)
    count_other = count_other * (INNER_FOLDS - 1) / INNER_FOLDS

    results = []
    for frame, t, c in ((A, te, count), (B, te_other, count_other)):
        frame["sib_te"], frame["sib_n"] = t, c
        frame["nights"] = frame["weekend_nights"] + frame["weekday_nights"]
        frame["weekend_share"] = frame["weekend_nights"] / np.maximum(frame["nights"], 1)
        results.append(pd.get_dummies(frame, columns=CAT_COLS, dtype=float).astype(float))
    return results[0], results[1].reindex(columns=results[0].columns, fill_value=0.0)
