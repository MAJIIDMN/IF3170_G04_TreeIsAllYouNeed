import argparse
import inspect
import sys
import time
import warnings
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent         
sys.path.insert(0, str(HERE.parent))       

from algoritma.lr.lr import LogisticRegression
from algoritma.lr.preprocessing import ID_COL, TARGET, Preprocessor, drop_duplicates, load_data

DATA_DIR = HERE / "dataset"
OUTPUT_DIR = HERE.parent.parent / "output" / "lr"  
SEED = 42
N_FOLDS = 5
HOLDOUT = 0.2

TRAIN_KW = dict(lr=0.1, n_epochs=100, batch_size=128, tol=1e-7, seed=SEED)

PENALTIES = [
    dict(penalty=None, lam=0.0, rho=0.5),
    dict(penalty="l2", lam=1e-4, rho=0.5),
    dict(penalty="l2", lam=1e-3, rho=0.5),
    dict(penalty="l2", lam=1e-2, rho=0.5),
    dict(penalty="l1", lam=1e-4, rho=0.5),
    dict(penalty="l1", lam=1e-3, rho=0.5),
    dict(penalty="elasticnet", lam=1e-4, rho=0.5),
    dict(penalty="elasticnet", lam=1e-3, rho=0.5),
]
SOLVERS = ["sgd", "irls_cd"]
GRID = [dict(cfg, solver=s) for s in SOLVERS for cfg in PENALTIES]
THRESHOLDS = np.round(np.arange(0.20, 0.81, 0.01), 2)

COST_FP = 2.0
COST_FN = 1.0
COST_THRESHOLDS = np.round(np.arange(0.01, 1.00, 0.01), 2)

def confusion(y, pred):
    tp = int(np.sum((y == 1) & (pred == 1)))
    tn = int(np.sum((y == 0) & (pred == 0)))
    fp = int(np.sum((y == 0) & (pred == 1)))
    fn = int(np.sum((y == 1) & (pred == 0)))
    return tn, fp, fn, tp


def f1_per_class(y, pred, cls):
    tp = np.sum((y == cls) & (pred == cls))
    fp = np.sum((y != cls) & (pred == cls))
    fn = np.sum((y == cls) & (pred != cls))
    return 0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn)


def macro_f1(y, pred):
    return (f1_per_class(y, pred, 0) + f1_per_class(y, pred, 1)) / 2

def roc_auc(y, score):
    ranks = pd.Series(score).rank().to_numpy()
    n_pos, n_neg = np.sum(y == 1), np.sum(y == 0)
    return (ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def report(y, proba, threshold):
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion(y, pred)
    return {"macro_f1": macro_f1(y, pred), "accuracy": float(np.mean(pred == y)),
            "f1_cancel": f1_per_class(y, pred, 1), "roc_auc": roc_auc(y, proba),
            "tn": tn, "fp": fp, "fn": fn, "tp": tp}

def best_threshold(y, proba):
    scores = [macro_f1(y, (proba >= t).astype(int)) for t in THRESHOLDS]
    i = int(np.argmax(scores))
    return float(THRESHOLDS[i]), float(scores[i])

def total_cost(y, pred, c_fp=COST_FP, c_fn=COST_FN):
    # total biaya salah klasifikasi = c_fp * FP + c_fn * FN
    _, fp, fn, _ = confusion(y, pred)
    return c_fp * fp + c_fn * fn


def cost_threshold(c_fp=COST_FP, c_fn=COST_FN):
    # threshold optimal Bayes (Elkan 2001): prediksi batal jika biaya harapannya lebih kecil,
    # (1 - p) * c_fp < p * c_fn  <=>  p > c_fp / (c_fp + c_fn)
    return c_fp / (c_fp + c_fn)


def best_cost_threshold(y, proba, c_fp=COST_FP, c_fn=COST_FN):
    # threshold empiris dengan total biaya terkecil
    costs = [total_cost(y, (proba >= t).astype(int), c_fp, c_fn) for t in COST_THRESHOLDS]
    i = int(np.argmin(costs))
    return float(COST_THRESHOLDS[i]), float(costs[i])


def stratified_folds(y, k, seed):
    rng = np.random.default_rng(seed)
    fold_of = np.empty(len(y), int)
    for cls in np.unique(y):
        idx = rng.permutation(np.flatnonzero(y == cls))
        fold_of[idx] = np.arange(len(idx)) % k
    return [(np.flatnonzero(fold_of != f), np.flatnonzero(fold_of == f)) for f in range(k)]

def stratified_holdout(y, frac, seed):
    rng = np.random.default_rng(seed)
    val = []
    for cls in np.unique(y):
        idx = rng.permutation(np.flatnonzero(y == cls))
        val.append(idx[:int(round(len(idx) * frac))])
    val = np.sort(np.concatenate(val))
    return np.setdiff1d(np.arange(len(y)), val), val

def cfg_from_row(row):
    pen = row["penalty"]
    return dict(penalty=None if pen is None or pd.isna(pen) else pen,
                lam=float(row["lam"]), rho=float(row["rho"]), solver=row.get("solver", "sgd"))


def name_of(cfg):
    if cfg["penalty"] is None:
        name = "none"
    else:
        extra = f",rho={cfg['rho']}" if cfg["penalty"] == "elasticnet" else ""
        name = f"{cfg['penalty']}(lam={cfg['lam']:g}{extra})"
    return f"{name} [{cfg['solver']}]" if "solver" in cfg else name


def cross_validate(train, y):
    folds = stratified_folds(y, N_FOLDS, SEED)
    prepared = []
    for fit_idx, val_idx in folds:
        prep = Preprocessor().fit(train.iloc[fit_idx])
        prepared.append((fit_idx, val_idx, prep.transform(train.iloc[fit_idx]),
                         prep.transform(train.iloc[val_idx])))

    rows = []
    for cfg in GRID:
        start = time.perf_counter()
        oof = np.zeros(len(y))
        epochs = []
        for fit_idx, val_idx, X_fit, X_val in prepared:
            model = LogisticRegression(**cfg, **TRAIN_KW).fit(X_fit, y[fit_idx])
            oof[val_idx] = model.predict_proba(X_val)
            epochs.append(len(model.loss_history))
        thr, _ = best_threshold(y, oof)
        r05, rbest = report(y, oof, 0.5), report(y, oof, thr)
        rows.append({"config": name_of(cfg), **cfg, "macro_f1@0.5": r05["macro_f1"],
                     "threshold": thr, "macro_f1@thr": rbest["macro_f1"],
                     "accuracy@thr": rbest["accuracy"], "roc_auc": r05["roc_auc"],
                     "epochs_mean": float(np.mean(epochs)), "sec": time.perf_counter() - start})
        print(f"  {rows[-1]['config']:<40} macroF1@0.5={r05['macro_f1']:.4f}  "
              f"macroF1@{thr:.2f}={rbest['macro_f1']:.4f}  AUC={r05['roc_auc']:.4f}  "
              f"epochs~{np.mean(epochs):.0f}  ({rows[-1]['sec']:.1f}s)")
    return pd.DataFrame(rows)


def oof_proba(df, y, cfg, class_weight=None):
    # probabilitas out-of-fold (stratified k-fold) untuk satu konfigurasi, dipakai memilih threshold
    oof = np.zeros(len(y))
    for fit_idx, val_idx in stratified_folds(y, N_FOLDS, SEED):
        prep = Preprocessor().fit(df.iloc[fit_idx])
        model = LogisticRegression(**cfg, **TRAIN_KW, class_weight=class_weight)
        model.fit(prep.transform(df.iloc[fit_idx]), y[fit_idx])
        oof[val_idx] = model.predict_proba(prep.transform(df.iloc[val_idx]))
    return oof


def sklearn_model(cfg, n):
    from sklearn.linear_model import LogisticRegression as SkLogisticRegression
    C = np.inf if cfg["penalty"] is None else 1.0 / (n * cfg["lam"])
    l1_ratio = {None: 0.0, "l2": 0.0, "l1": 1.0, "elasticnet": cfg["rho"]}[cfg["penalty"]]
    solver = "lbfgs" if cfg["penalty"] in (None, "l2") else "saga"
    kw = dict(solver=solver, max_iter=5000, tol=1e-6, random_state=SEED)

    if inspect.signature(SkLogisticRegression).parameters["penalty"].default == "deprecated":
        return SkLogisticRegression(C=C, l1_ratio=l1_ratio, **kw)
    if cfg["penalty"] is None:
        return SkLogisticRegression(penalty=None, **kw)
    return SkLogisticRegression(penalty=cfg["penalty"], C=C,
                                l1_ratio=l1_ratio if cfg["penalty"] == "elasticnet" else None, **kw)

def compare_sklearn(train, y, configs):
    fit_idx, val_idx = stratified_holdout(y, HOLDOUT, SEED)
    prep = Preprocessor().fit(train.iloc[fit_idx])
    X_fit, X_val = prep.transform(train.iloc[fit_idx]), prep.transform(train.iloc[val_idx])
    y_fit, y_val = y[fit_idx], y[val_idx]

    rows = []
    for cfg in configs:
        start = time.perf_counter()
        ours = LogisticRegression(**cfg, **TRAIN_KW).fit(X_fit, y_fit)
        sec_ours = time.perf_counter() - start
        p_ours = ours.predict_proba(X_val)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            start = time.perf_counter()
            sk = sklearn_model(cfg, len(y_fit)).fit(X_fit, y_fit)
            sec_sk = time.perf_counter() - start
        p_sk = sk.predict_proba(X_val)[:, 1]
        for label, p, sec, w in [("scratch", p_ours, sec_ours, ours.w), ("sklearn", p_sk, sec_sk, sk.coef_.ravel())]:
            r = report(y_val, p, 0.5)
            rows.append({"config": name_of(cfg), "impl": label, **r, "waktu_s": sec, "bobot_nol": int(np.sum(w == 0))})
        agree = np.mean((p_ours >= 0.5) == (p_sk >= 0.5))
        w_diff = np.max(np.abs(ours.w - sk.coef_.ravel()))
        print(f"  {name_of(cfg):<40} label sama {agree:.2%}  max|w_scratch - w_sklearn|={w_diff:.4f}  "
              f"nonzero w: scratch={np.sum(np.abs(ours.w) > 1e-6)} sklearn={np.sum(np.abs(sk.coef_) > 1e-6)}")
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-sklearn", action="store_true", help="lewati perbandingan scikit-learn")
    args = parser.parse_args()
    sys.stdout.reconfigure(line_buffering=True) 
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 160)

    train, test = load_data(DATA_DIR)
    train = drop_duplicates(train)
    y = train[TARGET].to_numpy(int)
    print(f"Train {train.shape} (setelah buang duplikat), test {test.shape}, batal {y.mean():.1%}")

    print(f"\n[1] Stratified {N_FOLDS}-fold CV (from scratch)")
    cv = cross_validate(train, y)
    cv.to_csv(OUTPUT_DIR / "lr_cv_results.csv", index=False)
    best = cv.sort_values("macro_f1@thr", ascending=False).iloc[0]
    best_cfg = cfg_from_row(best)
    threshold = float(best["threshold"])
    print(f"  -> terbaik: {best['config']}  threshold={threshold:.2f}  "
          f"Macro F1 OOF={best['macro_f1@thr']:.4f}")

    if not args.no_sklearn:
        print(f"\n[2] From scratch vs scikit-learn (holdout {HOLDOUT:.0%}, threshold 0.5)")
        try:
            per_penalty = (cv.assign(p=cv["penalty"].fillna("none"))
                           .sort_values("macro_f1@thr", ascending=False).groupby("p").head(1))
            configs = [cfg_from_row(r) for _, r in per_penalty.iterrows()]
            cmp = compare_sklearn(train, y, configs)
            cmp.to_csv(OUTPUT_DIR / "lr_sklearn_comparison.csv", index=False)
            print(cmp.round(4).to_string(index=False))
        except ImportError:
            print("  scikit-learn belum terinstall, tahap ini dilewati.")

    print("\n[3] Latih ulang pada seluruh train, save/load, submission")
    prep = Preprocessor().fit(train)
    X_train, X_test = prep.transform(train), prep.transform(test)
    model = LogisticRegression(**best_cfg, **TRAIN_KW).fit(X_train, y)
    model_path = OUTPUT_DIR / "lr_model.npz"
    model.save(model_path)
    loaded = LogisticRegression.load(model_path)
    p_test = loaded.predict_proba(X_test)
    assert np.array_equal(p_test, model.predict_proba(X_test)), "hasil load berbeda!"
    print(f"  model disimpan & dimuat ulang: {model_path} (prediksi identik)")
    print(f"  Macro F1 train (threshold {threshold:.2f}): "
          f"{report(y, model.predict_proba(X_train), threshold)['macro_f1']:.4f}")

    sub = pd.DataFrame({ID_COL: test[ID_COL], TARGET: (p_test >= threshold).astype(int)})
    sub_path = OUTPUT_DIR / "submission.csv"
    sub.to_csv(sub_path, index=False)
    print(f"  submission: {sub_path}  ({len(sub)} baris, prediksi batal {sub[TARGET].mean():.1%})")

    top = np.argsort(-np.abs(model.w))[:10]
    print("\n  10 bobot terbesar (|w|):")
    for i in top:
        print(f"    {prep.feature_names_[i]:<34} {model.w[i]:+.4f}")


if __name__ == "__main__":
    main()
