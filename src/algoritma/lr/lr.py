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

Referensi Cost-Sensitive (parameter class_weight):
- C. Elkan, "The Foundations of Cost-Sensitive Learning," in Proc. IJCAI, 2001, pp. 973-978.
     -> Teorema 2: contoh negatif diberi bobot (biaya) lebih besar agar model belajar sesuai biaya.
- scikit-learn User Guide, "class_weight" pada LogisticRegression (log-loss tiap contoh dikali bobot kelasnya).
     https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

Referensi Algoritma Optimasi (bonus, solver="irls_cd"):
- J. Friedman, T. Hastie, and R. Tibshirani, "Regularization Paths for Generalized Linear Models via
     Coordinate Descent," J. Stat. Softw., vol. 33, no. 1, pp. 1-22, 2010.
     -> Sec. 2.1-2.2: update coordinate descent + soft-thresholding untuk elastic net;
        Sec. 3: logistic regression lewat aproksimasi kuadratik (IRLS) + coordinate descent.
- T. Hastie, R. Tibshirani, and J. Friedman, The Elements of Statistical Learning, 2nd ed.
     Springer, 2009, Sec. 4.4.1 -> Newton-Raphson / IRLS untuk logistic regression.
"""
import numpy as np


def sigmoid(z):
    # fungsi logistik / sigmoid, z di clip biar np.exp ga overflow
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def soft_threshold(z, t):
    # operator soft-thresholding S(z, t) = sign(z) * max(|z| - t, 0)  [Friedman dkk. 2010, pers. (6)]
    return np.sign(z) * max(abs(z) - t, 0.0)


class LogisticRegression:

    def __init__(self, lr=0.1, n_epochs=1000, penalty=None, lam=0.0, rho=0.5,
                 batch_size=None, tol=1e-7, seed=42, class_weight=None, solver="sgd"):
        if penalty not in (None, "l1", "l2", "elasticnet"):
            raise ValueError("penalty harus None, 'l1', 'l2', atau 'elasticnet'")
        if solver not in ("sgd", "irls_cd"):
            raise ValueError("solver harus 'sgd' atau 'irls_cd'")
        self.solver = solver
        self.lr = lr
        self.n_epochs = n_epochs
        self.penalty = penalty
        self.lam = lam
        self.rho = rho
        self.batch_size = batch_size
        self.tol = tol
        self.seed = seed
        self.class_weight = class_weight
        self.w = None       
        self.b = 0.0       
        self.loss_history = [] 
        self.path = []  

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

    def _l1_l2(self):
        # pecah penalti menjadi L1 dan L2: penalti = l1 * |w|_1 + l2 / 2 * |w|^2
        if self.penalty == "l2":
            return 0.0, self.lam
        if self.penalty == "l1":
            return self.lam, 0.0
        if self.penalty == "elasticnet":
            return self.lam * self.rho, self.lam * (1 - self.rho)
        return 0.0, 0.0

    def _sample_weight(self, y):
        # bobot tiap contoh sesuai kelasnya (cost-sensitive), semua 1 kalo class_weight=None
        if self.class_weight is None:
            return np.ones(len(y))
        return np.where(y == 1, self.class_weight[1], self.class_weight[0])

    def loss(self, X, y, w=None, b=None):
        # log loss rata-rata = -LCL / n (tiap contoh dikali bobot kelasnya), ditambah penalti regularisasi,
        # p di clip so log(0) ga terjadi
        w = self.w if w is None else w
        b = self.b if b is None else b
        p = np.clip(sigmoid(X @ w + b), 1e-15, 1 - 1e-15)
        s = self._sample_weight(y)
        return -np.mean(s * (y * np.log(p) + (1 - y) * np.log(1 - p))) + self._penalty_value(w)

    def fit(self, X, y, init=None):
        # init = (w, b) titik awal, default nol seperti di [Khodra, video 2] ("boleh nilai lain")
        X, y = np.asarray(X, float), np.asarray(y, float).ravel()
        if init is None:
            self.w, self.b = np.zeros(X.shape[1]), 0.0
        else:
            self.w, self.b = np.array(init[0], float).reshape(X.shape[1]), float(init[1])
        self.loss_history, self.path = [], []
        if self.solver == "irls_cd":
            return self._fit_irls_cd(X, y)
        return self._fit_sgd(X, y)

    def _record(self, X, y):
        # simpan loss dan parameter tiap iterasi, return True kalau konvergen
        self.loss_history.append(self.loss(X, y))
        self.path.append((self.w.copy(), self.b))
        return len(self.loss_history) > 1 and abs(self.loss_history[-2] - self.loss_history[-1]) < self.tol

    def _fit_sgd(self, X, y):
        n = X.shape[0]
        rng = np.random.default_rng(self.seed)
        size = self.batch_size or n

        for _ in range(self.n_epochs):
            # urutan contoh diacak tiap epoch (stochastic / mini-batch)
            order = rng.permutation(n) if self.batch_size else np.arange(n)
            for i in range(0, n, size):
                idx = order[i:i + size]
                # error dikali bobot kelas: kesalahan pada kelas yg lebih mahal mendorong update lebih besar
                err = (sigmoid(X[idx] @ self.w + self.b) - y[idx]) * self._sample_weight(y[idx])
                self.w = self.w - self.lr * (X[idx].T @ err / len(idx) + self._penalty_grad(self.w))
                self.b = self.b - self.lr * err.mean()
            if self._record(X, y):
                break
        return self

    def _fit_irls_cd(self, X, y, max_sweeps=1000, cd_tol=1e-8):
        # Friedman dkk. (2010) Sec. 3. Tiap iterasi IRLS (= satu langkah Newton)
        n, d = X.shape
        A = np.hstack([X, np.ones((n, 1))])
        beta = np.r_[self.w, self.b]
        l1, l2 = self._l1_l2()
        pen_l1 = np.r_[np.full(d, l1), 0.0]   
        pen_l2 = np.r_[np.full(d, l2), 0.0]
        c = self._sample_weight(y)

        for _ in range(self.n_epochs):
            eta = A @ beta
            p = np.clip(sigmoid(eta), 1e-5, 1 - 1e-5) 
            s = c * p * (1 - p) # bobot IRLS = turunan kedua log-loss
            z = eta + (y - p) / (p * (1 - p))  
            G = (A.T * s) @ A / n # matriks Gram berbobot (d+1) x (d+1)
            q = (A.T * s) @ z / n
            Gb = G @ beta

            for _ in range(max_sweeps):
                max_change = 0.0
                for j in range(d + 1):
                    # minimum fungsi kuadratik thdp beta_j saja (bobot lain tetap)
                    g = q[j] - Gb[j] + G[j, j] * beta[j]
                    denom = G[j, j] + pen_l2[j]
                    new = soft_threshold(g, pen_l1[j]) / denom if denom > 0 else 0.0
                    if new != beta[j]:
                        Gb += G[:, j] * (new - beta[j])
                        max_change = max(max_change, abs(new - beta[j]))
                        beta[j] = new
                if max_change < cd_tol:
                    break

            self.w, self.b = beta[:d].copy(), float(beta[d])
            if self._record(X, y):
                break
        return self

    def predict_proba(self, X):
        # P(y = 1 | x)
        return sigmoid(np.asarray(X, float) @ self.w + self.b)

    def predict(self, X, threshold=0.5):
        # decision rule: P >= threshold -> kelas 1 (default 0,5)
        return (self.predict_proba(X) >= threshold).astype(int)

    def save(self, path):
        np.savez(path, w=self.w, b=self.b, lr=self.lr, n_epochs=self.n_epochs,
                 penalty=str(self.penalty), lam=self.lam, rho=self.rho, solver=self.solver)

    @classmethod
    def load(cls, path):
        d = np.load(path, allow_pickle=False)
        pen = str(d["penalty"])
        model = cls(lr=float(d["lr"]), n_epochs=int(d["n_epochs"]),
                    penalty=None if pen == "None" else pen,
                    lam=float(d["lam"]), rho=float(d["rho"]),
                    solver=str(d["solver"]) if "solver" in d.files else "sgd")
        model.w, model.b = d["w"], float(d["b"])
        return model
