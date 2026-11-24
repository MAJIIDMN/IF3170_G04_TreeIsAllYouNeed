from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src" / "algoritma"))

from dtl import DTL  # noqa: E402

DATA = REPO / "src" / "kaggle" / "dataset" / "train.csv"
N_SPLITS = 5
SEED = 42
MIN_ACCEPTABLE_F1 = 0.82


def macro_f1(y_true, y_pred, classes=(0, 1)) -> float:
    scores = []
    for c in classes:
        tp = int(np.count_nonzero((y_pred == c) & (y_true == c)))
        fp = int(np.count_nonzero((y_pred == c) & (y_true != c)))
        fn = int(np.count_nonzero((y_pred != c) & (y_true == c)))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        if precision + recall == 0.0:
            scores.append(0.0)
        else:
            scores.append(2 * precision * recall / (precision + recall))
    return float(np.mean(scores))


def stratified_folds(y, n_splits, seed):
    rng = np.random.default_rng(seed)
    buckets: list[list[int]] = [[] for _ in range(n_splits)]
    for cls in np.unique(y):
        idx = np.flatnonzero(y == cls)
        rng.shuffle(idx)
        for i, chunk in enumerate(np.array_split(idx, n_splits)):
            buckets[i].extend(chunk.tolist())
    all_idx = np.arange(len(y))
    folds = []
    for bucket in buckets:
        val = np.array(sorted(bucket))
        is_val = np.zeros(len(y), dtype=bool)
        is_val[val] = True
        folds.append((all_idx[~is_val], val))
    return folds


def impute(train_df, other_df):
    train = train_df.copy()
    other = other_df.copy()
    for col in train.columns:
        if pd.api.types.is_numeric_dtype(train[col]):
            fill = train[col].median()
        else:
            mode = train[col].mode()
            fill = mode.iloc[0] if not mode.empty else ""
        train[col] = train[col].fillna(fill)
        other[col] = other[col].fillna(fill)
    return train, other


def run_config(name, extra, X_df, y, folds):
    scores, leaves, depths = [], [], []
    t0 = time.time()
    for tr, va in folds:
        X_tr, X_va = impute(X_df.iloc[tr], X_df.iloc[va])
        model = DTL(max_depth=12, min_samples_leaf=5, **extra)
        model.fit(X_tr, y[tr])
        pred = model.predict(X_va)
        scores.append(macro_f1(y[va], pred))
        leaves.append(model.n_leaves_)
        depths.append(model.depth_)
    elapsed = time.time() - t0
    print(f"{name:26s} MacroF1={np.mean(scores):.4f}±{np.std(scores):.4f} "
          f"leaves={np.mean(leaves):6.1f} depth={np.mean(depths):4.1f} "
          f"{elapsed:6.1f}s")
    return name, float(np.mean(scores)), float(np.std(scores))


def main() -> int:
    df = pd.read_csv(DATA, dtype={"id": str})
    df = df.drop(columns=["id"]).drop_duplicates().reset_index(drop=True)
    y = df.pop("is_canceled").to_numpy()
    print(f"data: {len(df)} rows x {df.shape[1]} features, "
          f"positive rate {y.mean():.4f}")

    folds = stratified_folds(y, N_SPLITS, SEED)
    configs = [
        ("none", {}),
        ("reduced_error rf=0.3",
         {"pruning": "reduced_error", "val_fraction": 0.3,
          "random_state": SEED}),
        ("cost_complexity a=1e-3",
         {"pruning": "cost_complexity", "ccp_alpha": 1e-3}),
        ("cost_complexity a=1e-2",
         {"pruning": "cost_complexity", "ccp_alpha": 1e-2}),
    ]
    results = [run_config(name, extra, df, y, folds)
               for name, extra in configs]
    best = max(results, key=lambda r: r[1])
    print(f"\nbest: {best[0]} (Macro F1 {best[1]:.4f} ± {best[2]:.4f})")
    if best[1] < MIN_ACCEPTABLE_F1:
        print(f"FAIL: best Macro F1 {best[1]:.4f} < {MIN_ACCEPTABLE_F1}")
        return 1
    print(f"OK: best Macro F1 >= {MIN_ACCEPTABLE_F1}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
