import numpy as np
from nb import NaiveBayes
from preprocessing import Preprocessor, macro_f1, roc_auc, stratified_folds

class FoldCache:
    def __init__(self, spec, df, y, n_folds=5, seed=42):
        self.names = list(spec)
        self.y = np.asarray(y)
        self.folds = stratified_folds(self.y, n_folds, seed)
        self.data = []
        self.types = None
        for fit_idx, val_idx in self.folds:
            prep = Preprocessor(spec)
            X_fit = prep.fit_transform(df.iloc[fit_idx], self.y[fit_idx])
            X_val = prep.transform(df.iloc[val_idx])
            self.data.append((X_fit, X_val))
            self.types = prep.feature_types_

    def score(self, names, alpha=1.0, thresholds=None):
        cols = [self.names.index(n) for n in names]
        types = [self.types[c] for c in cols]
        odds = np.zeros(len(self.y))
        for (fit_idx, val_idx), (X_fit, X_val) in zip(self.folds, self.data):
            model = NaiveBayes(alpha=alpha).fit(X_fit[:, cols], self.y[fit_idx], types)
            odds[val_idx] = model.decision_function(X_val[:, cols])
        if thresholds is None:
            thresholds = np.round(np.arange(-1.5, 1.5001, 0.1), 2)
        curves = np.array([[macro_f1(self.y[v], (odds[v] > t).astype(int)) for t in thresholds]
                           for _, v in self.folds])
        held = [curves[k, int(np.argmax(np.delete(curves, k, axis=0).mean(axis=0)))]
                for k in range(len(self.folds))]
        per_fold = [macro_f1(self.y[v], (odds[v] > 0).astype(int)) for _, v in self.folds]
        return {
            "macro_f1": float(np.mean(per_fold)), "macro_f1_std": float(np.std(per_fold)),
            "macro_f1_held_out_threshold": float(np.mean(held)),
            "best_threshold": float(thresholds[int(np.argmax(curves.mean(axis=0)))]),
            "roc_auc": roc_auc(self.y, odds),
        }

# forward selection untuk Naive Bayes
def forward_selection(cache, candidates=None, start=(), max_features=None, min_gain=1e-4, alpha=1.0):
    pool = [n for n in (candidates or cache.names) if n not in start]
    chosen = list(start)
    steps = []
    best = cache.score(chosen, alpha)["roc_auc"] if chosen else -np.inf
    while pool and (max_features is None or len(chosen) < max_features):
        auc, name = max((cache.score(chosen + [n], alpha)["roc_auc"], n) for n in pool)
        if auc - best < min_gain:
            break
        chosen.append(name)
        pool.remove(name)
        best = auc
        steps.append({"step": len(chosen), "added": name, **cache.score(chosen, alpha)})
    return steps
