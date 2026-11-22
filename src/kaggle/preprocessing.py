"""
Referensi:
- E. Ferdian, "Data Preparation," IF3170, Edunex ITB, 2026 (Materi09): imputasi, encoding,
    standardisasi, preprocessing dipelajari dari data latih saja.
"""
import numpy as np
import pandas as pd

ID_COL = "id"
TARGET = "is_canceled"
CAT_COLS = ["meal_package", "room_category", "booking_channel"]
BIN_COLS = ["needs_parking", "is_returning_guest"]
COUNT_COLS = ["num_adults", "num_children", "weekend_nights", "weekday_nights",
              "prior_cancellations", "prior_completed_bookings", "special_requests"]
CONT_COLS = ["days_before_arrival", "nightly_rate_idr"]
DATE_COLS = ["arrival_year", "arrival_month", "arrival_day"]
FEATURES = COUNT_COLS + CONT_COLS + BIN_COLS + DATE_COLS + CAT_COLS

MISSING = "kosong"  
RARE = "lainnya"  


def load_data(data_dir):
    train = pd.read_csv(f"{data_dir}/train.csv", dtype={ID_COL: str})
    test = pd.read_csv(f"{data_dir}/test.csv", dtype={ID_COL: str})
    return train, test


def drop_duplicates(train):
    """Buang baris duplikat penuh (di luar id) pada TRAIN saja. Test tidak disentuh."""
    return train.drop_duplicates(subset=FEATURES + [TARGET]).reset_index(drop=True)


class Preprocessor:
    def __init__(self, min_freq=30, clip_upper=None, scale=True):
        self.min_freq = min_freq
        self.clip_upper = clip_upper if clip_upper is not None else {
            "num_children": 8, "nightly_rate_idr": 6_000_000}
        self.scale = scale

    def _base(self, df):
        out = df[FEATURES].copy()
        for col, upper in self.clip_upper.items():
            out[col] = out[col].clip(upper=upper)
        out["special_requests_missing"] = out["special_requests"].isna().astype(float)
        out["rate_is_zero"] = out["nightly_rate_idr"].eq(0).astype(float)

        arrival = pd.to_datetime({"year": out["arrival_year"], "month": out["arrival_month"],
                                  "day": out["arrival_day"]}, errors="coerce")
        booking = arrival - pd.to_timedelta(out["days_before_arrival"], unit="D")
        out["arrival_ym"] = arrival.dt.strftime("%Y-%m")
        out["booking_ym"] = booking.dt.strftime("%Y-%m")  
        return out

    @property
    def num_cols(self):
        return (COUNT_COLS + CONT_COLS + BIN_COLS + DATE_COLS
                + ["special_requests_missing", "rate_is_zero", "lead_log"])

    @property
    def cat_cols(self):
        return CAT_COLS + ["arrival_ym", "booking_ym"]

    def _impute_and_derive(self, out):
        out["special_requests"] = out["special_requests"].fillna(0)
        rate_fill = out["booking_channel"].map(self.rate_median_by_channel_)
        out["nightly_rate_idr"] = out["nightly_rate_idr"].fillna(rate_fill).fillna(
            self.medians_["nightly_rate_idr"])
        for col in COUNT_COLS + CONT_COLS + BIN_COLS + DATE_COLS:
            out[col] = out[col].fillna(self.medians_[col])
        out["lead_log"] = np.log1p(out["days_before_arrival"])
        for col in ["prior_cancellations", "prior_completed_bookings"]:
            out[col] = np.log1p(out[col])
        return out

    def _clean_cat(self, out):
        for col in self.cat_cols:
            s = out[col].astype(object).where(out[col].notna(), MISSING)
            out[col] = s.where(s.isin(self.categories_[col]), RARE)
        return out

    def fit(self, df):
        out = self._base(df)
        self.medians_ = out[COUNT_COLS + CONT_COLS + BIN_COLS + DATE_COLS].median()
        self.rate_median_by_channel_ = out.groupby("booking_channel")["nightly_rate_idr"].median()
        out = self._impute_and_derive(out)

        self.categories_ = {}
        for col in self.cat_cols:
            counts = out[col].astype(object).where(out[col].notna(), MISSING).value_counts()
            self.categories_[col] = sorted(counts[counts >= self.min_freq].index)
        out = self._clean_cat(out)

        X = self._to_matrix(out, fitting=True)
        if self.scale:
            n_num = len(self.num_cols)
            self.mean_ = X[:, :n_num].mean(axis=0)
            std = X[:, :n_num].std(axis=0)
            self.std_ = np.where(std == 0, 1.0, std)
        return self

    def _to_matrix(self, out, fitting=False):
        num = out[self.num_cols].to_numpy(float)
        blocks, names = [num], list(self.num_cols)
        if fitting:
            self.levels_ = {}
        for col in self.cat_cols:
            if fitting:
                self.levels_[col] = sorted(set(out[col]))
            levels = self.levels_[col]
            blocks.append((out[col].to_numpy()[:, None] == np.array(levels)[None, :]).astype(float))
            names += [f"{col}={lv}" for lv in levels]
        self.feature_names_ = names
        return np.hstack(blocks)

    def transform(self, df):
        out = self._clean_cat(self._impute_and_derive(self._base(df)))
        X = self._to_matrix(out)
        if self.scale:
            n_num = len(self.num_cols)
            X[:, :n_num] = (X[:, :n_num] - self.mean_) / self.std_
        return X

    def fit_transform(self, df):
        return self.fit(df).transform(df)
