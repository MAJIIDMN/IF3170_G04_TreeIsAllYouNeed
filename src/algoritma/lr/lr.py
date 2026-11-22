"""
Referensi Logistic Regression:
- M. L. Khodra, "Logistic Regression: What & Why," YouTube, 2021.
     https://www.youtube.com/watch?v=QAKZ_pymEzc
- M. L. Khodra, "Logistic Regression: Stochastic Gradient Ascent," YouTube, 2021.
     https://www.youtube.com/watch?v=O5NSNMQO1yU
- GeeksforGeeks, "Logistic Regression in Machine Learning."
     https://www.geeksforgeeks.org/machine-learning/understanding-logistic-regression/

Referensi Regularisasi:
-  scikit-learn User Guide, "1.1.11. Logistic regression" (formulasi matematis penalti
     L1, L2, dan Elastic Net). https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
- R. Tibshirani, "Regression Shrinkage and Selection via the Lasso," J. R. Stat. Soc. B,
     vol. 58, no. 1, pp. 267-288, 1996.  -> penalti L1.
- H. Zou and T. Hastie, "Regularization and Variable Selection via the Elastic Net,"
     J. R. Stat. Soc. B, vol. 67, no. 2, pp. 301-320, 2005.  -> penalti Elastic Net.

"""
import numpy as np


def sigmoid(z):
    # fungsi logistik / sigmoid, z di clip biar np.exp ga overflow
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


class LogisticRegression:
    """Logistic regression biner, dilatih dengan (mini-batch) gradient descent pada log-loss.

    penalty    : None, "l1", "l2", atau "elasticnet"
    lam        : kekuatan regularisasi (lambda)
    rho        : porsi L1 pada elastic net (0 = L2 murni, 1 = L1 murni)
    batch_size : None = full batch, 1 = stochastic, k = mini-batch
    tol        : berhenti jika perubahan loss antar-epoch < tol
    """

    def __init__(self, lr=0.1, n_epochs=1000, penalty=None, lam=0.0, rho=0.5,
                 batch_size=None, tol=1e-7, seed=42):
        if penalty not in (None, "l1", "l2", "elasticnet"):
            raise ValueError("penalty harus None, 'l1', 'l2', atau 'elasticnet'")
        self.lr = lr
        self.n_epochs = n_epochs
        self.penalty = penalty
        self.lam = lam
        self.rho = rho
        self.batch_size = batch_size
        self.tol = tol
        self.seed = seed
        self.w = None          # bobot fitur
        self.b = 0.0           # bias (tidak diregularisasi)
        self.loss_history = []  # loss tiap epoch
        self.path = []          # (w, b) tiap epoch, untuk visualisasi

    # ---------- regularisasi ----------
    def _penalty_value(self, w):
        # nilai penalti yg ditambahkan ke log-loss
        if self.penalty == "l2":
            return 0.5 * self.lam * np.sum(w ** 2)
        if self.penalty == "l1":
            return self.lam * np.sum(np.abs(w))
        if self.penalty == "elasticnet":
            return self.lam * (self.rho * np.sum(np.abs(w)) + (1 - self.rho) * 0.5 * np.sum(w ** 2))
        return 0.0

    def _penalty_grad(self, w):
        # gradien penalti thdp w, |w| tidak terdiferensiasi di 0, jadi pake subgradien sign(w) (sign(0) = 0)
        if self.penalty == "l2":
            return self.lam * w
        if self.penalty == "l1":
            return self.lam * np.sign(w)
        if self.penalty == "elasticnet":
            return self.lam * (self.rho * np.sign(w) + (1 - self.rho) * w)
        return 0.0

    def loss(self, X, y, w=None, b=None):
        # log loss rata-rata = -LCL / n, ditambah penalti regularisasi, p di clip so log(0) ga terjadi
        w = self.w if w is None else w
        b = self.b if b is None else b
        p = np.clip(sigmoid(X @ w + b), 1e-15, 1 - 1e-15)
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)) + self._penalty_value(w)

    # ---------- training ----------
    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y, float).ravel()
        n, d = X.shape
        rng = np.random.default_rng(self.seed)
        self.w, self.b = np.zeros(d), 0.0
        self.loss_history, self.path = [], []
        size = self.batch_size or n

        for _ in range(self.n_epochs):
            # urutan contoh diacak tiap epoch (stochastic / mini-batch)
            order = rng.permutation(n) if self.batch_size else np.arange(n)
            for i in range(0, n, size):
                idx = order[i:i + size]
                err = sigmoid(X[idx] @ self.w + self.b) - y[idx]
                self.w = self.w - self.lr * (X[idx].T @ err / len(idx) + self._penalty_grad(self.w))
                self.b = self.b - self.lr * err.mean()

            self.loss_history.append(self.loss(X, y))
            self.path.append((self.w.copy(), self.b))
            if len(self.loss_history) > 1 and abs(self.loss_history[-2] - self.loss_history[-1]) < self.tol:
                break
        return self

    # ---------- prediksi ----------
    def predict_proba(self, X):
        # P(y = 1 | x)
        return sigmoid(np.asarray(X, float) @ self.w + self.b)

    def predict(self, X, threshold=0.5):
        # decision rule: P >= threshold -> kelas 1 (default 0,5)
        return (self.predict_proba(X) >= threshold).astype(int)

    # ---------- save / load ----------
    def save(self, path):
        np.savez(path, w=self.w, b=self.b, lr=self.lr, n_epochs=self.n_epochs,
                 penalty=str(self.penalty), lam=self.lam, rho=self.rho)

    @classmethod
    def load(cls, path):
        d = np.load(path, allow_pickle=False)
        pen = str(d["penalty"])
        model = cls(lr=float(d["lr"]), n_epochs=int(d["n_epochs"]),
                    penalty=None if pen == "None" else pen,
                    lam=float(d["lam"]), rho=float(d["rho"]))
        model.w, model.b = d["w"], float(d["b"])
        return model
