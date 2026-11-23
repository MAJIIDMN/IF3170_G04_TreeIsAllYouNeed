import numpy as np
from nb import NaiveBayes

def class_index(model, y):
    return np.searchsorted(model.classes_, np.asarray(y))

def feature_contributions(model, X):
    X = model._as_matrix(X)
    out = np.zeros((len(X), len(model.classes_), X.shape[1]))
    for j, kind in enumerate(model.feature_types_):
        column = X[:, j]
        known = ~np.isnan(column)
        if not known.any():
            continue
        if kind == "gaussian":
            diff = column[known][:, None] - model.mean_[:, j][None, :]
            var = model.var_[:, j][None, :]
            out[known, :, j] = -0.5 * (np.log(2.0 * np.pi * var) + diff * diff / var)
        else:
            codes = column[known].astype(int)
            seen = (codes >= 0) & (codes < model.n_categories_[j])
            part = np.tile(model.unseen_log_[j], (len(codes), 1))
            part[seen] = model.log_theta_[j][:, codes[seen]].T
            out[known, :, j] = part
    return out

def pair_counts(a, b, index, n_classes, n_a, n_b):
    both = ~np.isnan(a) & ~np.isnan(b)
    counts = np.zeros((n_classes, n_a, n_b))
    np.add.at(counts, (index[both], a[both].astype(int), b[both].astype(int)), 1.0)
    return counts

# Sumber: N. Friedman, D. Geiger, M. Goldszmidt, "Bayesian Network Classifiers," Machine Learning 29, 1997.
def conditional_mutual_information(counts, alpha):
    smoothed = counts + alpha / (counts.shape[1] * counts.shape[2])
    joint = smoothed / smoothed.sum()
    p_c = joint.sum(axis=(1, 2), keepdims=True)
    p_ac = joint.sum(axis=2, keepdims=True)
    p_bc = joint.sum(axis=1, keepdims=True)
    return float((joint * np.log(joint * p_c / (p_ac * p_bc))).sum())

# Sumber: C. Chow dan C. Liu, "Approximating Discrete Probability Distributions with Dependence Trees," IEEE Trans. Inf. Theory 14, 1968.
def maximum_spanning_tree(weights, root):
    d = len(weights)
    parent = np.full(d, -1)
    inside = np.zeros(d, dtype=bool)
    inside[root] = True
    best = weights[root].copy()
    best_parent = np.full(d, root)
    for _ in range(d - 1):
        candidates = np.where(inside, -np.inf, best)
        j = int(np.argmax(candidates))
        parent[j] = best_parent[j]
        inside[j] = True
        better = weights[j] > best
        best = np.where(better, weights[j], best)
        best_parent = np.where(better, j, best_parent)
    return parent

class TANNaiveBayes(NaiveBayes):
    def __init__(self, alpha=1.0, root=0, class_prior=None):
        super().__init__(alpha=alpha, var_smoothing=1e-9, class_prior=class_prior)
        self.root = root

    def fit(self, X, y, feature_types):
        if any(kind != "categorical" for kind in feature_types):
            raise ValueError("TANNaiveBayes needs categorical features only")
        super().fit(X, y, feature_types)
        X = self._as_matrix(X)
        index = class_index(self, y)
        n_classes, d = len(self.classes_), X.shape[1]
        sizes = [self.n_categories_[j] for j in range(d)]
        weights = np.full((d, d), -np.inf)
        for i in range(d):
            for j in range(i + 1, d):
                counts = pair_counts(X[:, i], X[:, j], index, n_classes, sizes[i], sizes[j])
                weights[i, j] = weights[j, i] = conditional_mutual_information(counts, self.alpha)
        self.weights_ = weights
        self.parent_ = maximum_spanning_tree(weights, self.root)
        self.cpt_ = {}
        for j in range(d):
            p = self.parent_[j]
            if p < 0:
                continue
            counts = pair_counts(X[:, p], X[:, j], index, n_classes, sizes[p], sizes[j])
            self.cpt_[j] = np.log((counts + self.alpha) / (counts.sum(axis=2, keepdims=True) + self.alpha * sizes[j]))
        return self

    def joint_log_likelihood(self, X):
        X = self._as_matrix(X)
        marginal = feature_contributions(self, X)
        total = np.tile(self.class_log_prior_, (len(X), 1))
        for j in range(X.shape[1]):
            if j not in self.cpt_:
                total += marginal[:, :, j]
                continue
            table = self.cpt_[j]
            child, parent = X[:, j], X[:, self.parent_[j]]
            has_child = ~np.isnan(child)
            has_parent = ~np.isnan(parent)
            valid = has_child & has_parent
            valid[valid] = ((parent[valid] >= 0) & (parent[valid] < table.shape[1])
                            & (child[valid] >= 0) & (child[valid] < table.shape[2]))
            total[valid] += table[:, parent[valid].astype(int), child[valid].astype(int)].T
            total[has_child & ~valid] += marginal[has_child & ~valid, :, j]
        return total

    def to_dict(self):
        payload = super().to_dict()
        payload["params"] = {"alpha": self.alpha, "root": self.root,
                             "class_prior": payload["params"]["class_prior"]}
        payload["tan"] = {
            "parent": self.parent_.tolist(),
            "weights": self.weights_.tolist(),
            "cpt": {str(j): table.tolist() for j, table in self.cpt_.items()},
        }
        return payload

    @classmethod
    def from_dict(cls, payload):
        model = super().from_dict(payload)
        model.parent_ = np.asarray(payload["tan"]["parent"])
        model.weights_ = np.asarray(payload["tan"]["weights"])
        model.cpt_ = {int(j): np.asarray(table) for j, table in payload["tan"]["cpt"].items()}
        return model
