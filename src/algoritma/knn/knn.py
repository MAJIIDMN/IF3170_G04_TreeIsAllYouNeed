import numpy as np
from scipy.spatial.distance import cdist


class KNN:
    def __init__(self, n_neighbors=5, metric="manhattan", p=3, batch_size=128):
        if not isinstance(n_neighbors, (int, np.integer)) or n_neighbors < 1:
            raise ValueError("n_neighbors harus bilangan bulat positif")
        if metric not in {"manhattan", "euclidean", "minkowski"}:
            raise ValueError("metric harus manhattan, euclidean, atau minkowski")
        if metric == "minkowski" and (not np.isfinite(p) or p < 1):
            raise ValueError("p harus bernilai minimal 1")
        if not isinstance(batch_size, (int, np.integer)) or batch_size < 1:
            raise ValueError("batch_size harus bilangan bulat positif")
        self.n_neighbors = int(n_neighbors)
        self.metric = metric
        self.p = float(p)
        self.batch_size = int(batch_size)
        self.X_train_ = None
        self.y_train_ = None
        self.classes_ = None

    def fit(self, X, y):
        features = np.asarray(X, dtype=np.float32)
        labels = np.asarray(y)
        if features.ndim != 2 or len(features) == 0:
            raise ValueError("X harus matriks dua dimensi yang tidak kosong")
        if labels.ndim != 1 or len(labels) != len(features):
            raise ValueError("Jumlah label harus sama dengan jumlah baris X")
        if self.n_neighbors > len(features):
            raise ValueError("n_neighbors tidak boleh melebihi jumlah training")
        if not np.isfinite(features).all():
            raise ValueError("X tidak boleh mengandung NaN atau infinity")
        self.X_train_ = np.ascontiguousarray(features)
        self.y_train_ = labels.copy()
        self.classes_ = np.unique(labels)
        return self

    def _validate_query(self, X):
        if self.X_train_ is None:
            raise RuntimeError("Model belum di-fit")
        features = np.asarray(X, dtype=np.float32)
        if features.ndim != 2 or features.shape[1] != self.X_train_.shape[1]:
            raise ValueError("Jumlah kolom X tidak sesuai data training")
        if not np.isfinite(features).all():
            raise ValueError("X tidak boleh mengandung NaN atau infinity")
        return features

    def _distance_matrix(self, X):
        if self.metric == "manhattan":
            return cdist(X, self.X_train_, metric="cityblock")
        if self.metric == "euclidean":
            return cdist(X, self.X_train_, metric="euclidean")
        return cdist(X, self.X_train_, metric="minkowski", p=self.p)

    def _neighbors(self, X, k, ordered):
        if not 1 <= k <= len(self.X_train_):
            raise ValueError("Jumlah tetangga tidak valid")
        indices = np.empty((len(X), k), dtype=np.int32)
        distances = np.empty((len(X), k), dtype=np.float64)
        for start in range(0, len(X), self.batch_size):
            stop = min(start + self.batch_size, len(X))
            dist = self._distance_matrix(X[start:stop])
            chosen = np.argpartition(dist, kth=k - 1, axis=1)[:, :k]
            picked = np.take_along_axis(dist, chosen, axis=1)
            if ordered:
                order = np.argsort(picked, axis=1, kind="stable")
                chosen = np.take_along_axis(chosen, order, axis=1)
                picked = np.take_along_axis(picked, order, axis=1)
            indices[start:stop] = chosen
            distances[start:stop] = picked
        return distances, indices

    def kneighbors(self, X, n_neighbors=None):
        query = self._validate_query(X)
        k = self.n_neighbors if n_neighbors is None else int(n_neighbors)
        return self._neighbors(query, k, ordered=True)


    def predict_many_k(self, X, k_values):
        query = self._validate_query(X)
        values = sorted(set(k_values))
        if not values or any(not isinstance(k, (int, np.integer)) or k < 1 or k > len(self.X_train_) for k in values):
            raise ValueError("Nilai k tidak valid")
        predictions = {int(k): np.empty(len(query), dtype=self.classes_.dtype) for k in values}
        for start in range(0, len(query), self.batch_size):
            stop = min(start + self.batch_size, len(query))
            distances = self._distance_matrix(query[start:stop])
            for k in values:
                indices = np.argpartition(distances, kth=k - 1, axis=1)[:, :k]
                labels = self.y_train_[indices]
                counts = np.stack([(labels == c).sum(axis=1) for c in self.classes_], axis=1)
                predictions[int(k)][start:stop] = self.classes_[counts.argmax(axis=1)]
        return predictions

    def predict_proba(self, X):
        query = self._validate_query(X)
        _, indices = self._neighbors(query, self.n_neighbors, ordered=False)
        labels = self.y_train_[indices]
        counts = np.stack([(labels == c).sum(axis=1) for c in self.classes_], axis=1)
        return counts / self.n_neighbors

    def predict(self, X):
        query = self._validate_query(X)
        _, indices = self._neighbors(query, self.n_neighbors, ordered=False)
        labels = self.y_train_[indices]
        counts = np.stack([(labels == c).sum(axis=1) for c in self.classes_], axis=1)
        return self.classes_[counts.argmax(axis=1)]
