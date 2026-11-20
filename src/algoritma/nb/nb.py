import jso
import numpy as np

GAUSSIAN = "gaussian"
CATEGORICAL = "categorical"
TYPES = (GAUSSIAN, CATEGORICAL)

class NaiveBayes:
    def __init__(self, alpha=1.0, var_smoothing=1e-9, class_prior=None):
        if alpha < 0:
            raise ValueError(f"alpha must be >= 0, got {alpha}")
        if var_smoothing < 0:
            raise ValueError(f"var_smoothing must be >= 0, got {var_smoothing}")
        self.alpha = alpha
        self.var_smoothing = var_smoothing
        self.class_prior = class_prior

    def fit(self, X, y, feature_types):
        X = self._as_matrix(X)
        y = np.asarray(y)
        feature_types = list(feature_types)
        if X.shape[1] != len(feature_types):
            raise ValueError(f"X has {X.shape[1]} columns but {len(feature_types)} types")
        for kind in feature_types:
            if kind not in TYPES:
                raise ValueError(f"feature type must be one of {TYPES}, got {kind!r}")
        if len(y) != len(X):
            raise ValueError(f"X has {len(X)} rows but y has {len(y)}")

        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        index = np.searchsorted(self.classes_, y)
        self.feature_types_ = feature_types
        self.n_features_ = X.shape[1]
        self.class_count_ = np.bincount(index, minlength=n_classes).astype(float)

        if self.class_prior is None:
            prior = self.class_count_ / self.class_count_.sum()
        else:
            prior = np.asarray(self.class_prior, dtype=float)
            if prior.shape != (n_classes,) or not np.isclose(prior.sum(), 1.0) or (prior <= 0).any():
                raise ValueError("class_prior must be positive and sum to 1, one per class")
        self.class_log_prior_ = np.log(prior)

        self.mean_ = np.zeros((n_classes, self.n_features_))
        self.var_ = np.ones((n_classes, self.n_features_))
        self.log_theta_ = {}
        self.n_categories_ = {}
        self.unseen_log_ = {}
        epsilon = 0.0
        gaussian_columns = [j for j, kind in enumerate(feature_types) if kind == GAUSSIAN]
        if gaussian_columns:
            variances = np.nanvar(X[:, gaussian_columns], axis=0)
            epsilon = self.var_smoothing * float(np.max(variances))
        self.epsilon_ = epsilon

        for j, kind in enumerate(feature_types):
            column = X[:, j]
            known = ~np.isnan(column)
            if kind == GAUSSIAN:
                for c in range(n_classes):
                    values = column[known & (index == c)]
                    if values.size:
                        self.mean_[c, j] = values.mean()
                        self.var_[c, j] = values.var() + epsilon
                    else:
                        self.var_[c, j] = max(epsilon, 1.0)
            else:
                codes = column[known]
                if codes.size and (codes < 0).any() or (codes != np.floor(codes)).any():
                    raise ValueError(f"categorical feature {j} needs non-negative integer codes")
                n_cat = int(codes.max()) + 1 if codes.size else 1
                counts = np.zeros((n_classes, n_cat))
                np.add.at(counts, (index[known], codes.astype(int)), 1.0)
                totals = counts.sum(axis=1, keepdims=True)
                denominator = totals + self.alpha * n_cat
                denominator = np.where(denominator > 0, denominator, 1.0)
                theta = (counts + self.alpha) / denominator
                with np.errstate(divide="ignore"):
                    self.log_theta_[j] = np.log(theta)
                self.n_categories_[j] = n_cat
                unseen = self.alpha / denominator[:, 0]
                with np.errstate(divide="ignore"):
                    self.unseen_log_[j] = np.log(unseen)
        return self

    def joint_log_likelihood(self, X):
        self._check_fitted()
        X = self._as_matrix(X)
        if X.shape[1] != self.n_features_:
            raise ValueError(f"expected {self.n_features_} features, got {X.shape[1]}")
        total = np.tile(self.class_log_prior_, (len(X), 1))
        for j, kind in enumerate(self.feature_types_):
            column = X[:, j]
            known = ~np.isnan(column)
            if not known.any():
                continue
            if kind == GAUSSIAN:
                diff = column[known][:, None] - self.mean_[:, j][None, :]
                var = self.var_[:, j][None, :]
                total[known] += -0.5 * (np.log(2.0 * np.pi * var) + diff * diff / var)
            else:
                codes = column[known].astype(int)
                table = self.log_theta_[j]
                seen = (codes >= 0) & (codes < self.n_categories_[j])
                part = np.tile(self.unseen_log_[j], (len(codes), 1))
                part[seen] = table[:, codes[seen]].T
                total[known] += part
        return total

    def predict_log_proba(self, X):
        joint = self.joint_log_likelihood(X)
        top = joint.max(axis=1, keepdims=True)
        log_norm = top + np.log(np.exp(joint - top).sum(axis=1, keepdims=True))
        return joint - log_norm

    def predict_proba(self, X):
        return np.exp(self.predict_log_proba(X))

    def decision_function(self, X):
        if len(self.classes_) != 2:
            raise ValueError("decision_function needs exactly two classes")
        joint = self.joint_log_likelihood(X)
        return joint[:, 1] - joint[:, 0]

    def predict(self, X, threshold=0.0):
        self._check_fitted()
        if threshold == 0.0 or len(self.classes_) != 2:
            return self.classes_[np.argmax(self.joint_log_likelihood(X), axis=1)]
        return self.classes_[(self.decision_function(X) > threshold).astype(int)]

    def save(self, path):
        self._check_fitted()
        payload = {
            "params": {"alpha": self.alpha, "var_smoothing": self.var_smoothing,
                       "class_prior": None if self.class_prior is None else list(self.class_prior)},
            "classes": self.classes_.tolist(),
            "feature_types": self.feature_types_,
            "class_count": self.class_count_.tolist(),
            "class_log_prior": self.class_log_prior_.tolist(),
            "mean": self.mean_.tolist(),
            "var": self.var_.tolist(),
            "epsilon": self.epsilon_,
            "log_theta": {str(j): t.tolist() for j, t in self.log_theta_.items()},
            "unseen_log": {str(j): t.tolist() for j, t in self.unseen_log_.items()},
        }
        with open(path, "w", encoding="utf-8") as file:
            json.dump(payload, file)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as file:
            payload = json.load(file)
        model = cls(**payload["params"])
        model.classes_ = np.asarray(payload["classes"])
        model.feature_types_ = list(payload["feature_types"])
        model.n_features_ = len(model.feature_types_)
        model.class_count_ = np.asarray(payload["class_count"])
        model.class_log_prior_ = np.asarray(payload["class_log_prior"])
        model.mean_ = np.asarray(payload["mean"])
        model.var_ = np.asarray(payload["var"])
        model.epsilon_ = payload["epsilon"]
        model.log_theta_ = {int(j): np.asarray(t) for j, t in payload["log_theta"].items()}
        model.unseen_log_ = {int(j): np.asarray(t) for j, t in payload["unseen_log"].items()}
        model.n_categories_ = {j: t.shape[1] for j, t in model.log_theta_.items()}
        return model

    def _check_fitted(self):
        if not hasattr(self, "class_log_prior_"):
            raise RuntimeError("Call fit before predicting")

    @staticmethod
    def _as_matrix(X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-dimensional, got shape {X.shape}")
        if np.isinf(X).any():
            raise ValueError("X contains infinity")
        return X
