import numpy as np
from scipy.linalg import solve_triangular

from l2_svm import hamming_gram

BINS = 128
SCALE = 20.0
MIN_WEIGHT = 0.05
FIT_ROWS = 4000
HOLD_ROWS = 3000
STEPS = 100
LEARNING_RATE = 0.07
REG_RATE = 0.02
MIX_RATE = 0.03
START_SCALES = (0.9, 2.5)
START_NOISE = 0.5
START_REG = 0.6

TRANSFORMS = {
    "days_before_arrival": lambda v: np.sqrt(np.maximum(v, 0)),
    "sib_te": lambda v: np.log(np.clip(v, 0.005, 0.995) / (1 - np.clip(v, 0.005, 0.995))),
    "sib_n": np.log1p,
}


def _thresholds(values, bins):
    unique = np.unique(values)
    if len(unique) <= bins:
        return (unique[1:] + unique[:-1]) / 2
    quantiles = np.unique(np.quantile(values, np.linspace(0, 1, bins + 1)[1:-1]))
    following = unique[np.minimum(np.searchsorted(unique, quantiles, "right"), len(unique) - 1)]
    cuts = np.unique((quantiles + following) / 2)
    return cuts[cuts < unique[-1]]


def _bin_centres(values, cuts):
    index = np.searchsorted(cuts, values)
    known = np.flatnonzero(np.bincount(index, minlength=len(cuts) + 1))
    means = np.array([values[index == i].mean() for i in known])
    return np.interp(np.arange(len(cuts) + 1), known, means)


# Sumber: Y. Gorishniy, I. Rubachev & A. Babenko, NeurIPS 2022 (kode termometer / piecewise-linear per fitur).
def thermometer(train, other, feature_weight, bins=BINS):
    codes_train, codes_other, start, names = [], [], [], []
    k = 0
    for col in train.columns:
        a, b = train[col].to_numpy(float), other[col].to_numpy(float)
        if col in TRANSFORMS:
            a, b = TRANSFORMS[col](a), TRANSFORMS[col](b)
        spread = a.std()
        if spread < 1e-9:
            continue
        unit = max(feature_weight[k], MIN_WEIGHT) / SCALE / spread
        k += 1
        cuts = _thresholds(a, bins)
        gaps = np.diff(_bin_centres(a, cuts))
        codes_train.append(a[:, None] > cuts)
        codes_other.append(b[:, None] > cuts)
        start.append(np.maximum(gaps, 1e-6) * unit)
        names += [f"{col} > {cut:.6g}" for cut in cuts]
    return (np.hstack(codes_train).astype(np.float32), np.hstack(codes_other).astype(np.float32),
            np.concatenate(start).astype(np.float32), names)


def _cholesky_solve(factor, rhs):
    half = solve_triangular(factor, rhs, lower=True, check_finite=False)
    return solve_triangular(factor, half, lower=True, trans="T", check_finite=False)


class _Adam:
    def __init__(self, shape):
        self.mean, self.square = np.zeros(shape, np.float32), np.zeros(shape, np.float32)

    def direction(self, gradient, step, eps):
        self.mean += 0.1 * (gradient - self.mean)
        self.square += 0.01 * (gradient * gradient - self.square)
        return (self.mean / (1 - 0.9 ** step)) / (np.sqrt(self.square / (1 - 0.99 ** step)) + eps)


# Sumber: O. Chapelle, V. Vapnik, O. Bousquet & S. Mukherjee, "Choosing Multiple Parameters for Support Vector
# Machines", Machine Learning 46, 2002 (parameter kernel dipelajari dengan gradien galat validasi);
# H. Owhadi & G. Yoo, "Kernel Flows", J. Comput. Phys. 389, 2019 (gradien dari sampel acak tiap langkah);
# G. Lanckriet dkk., JMLR 5, 2004 (kernel sebagai jumlah berbobot beberapa kernel);
# D. Kingma & J. Ba, ICLR 2015 (Adam).
def learn_kernel(X, y, start, seed=0, steps=STEPS, scales=START_SCALES, verbose=20):
    n = len(X)
    sign = np.where(np.asarray(y) > 0, 1.0, -1.0).astype(np.float32)
    noise = np.random.default_rng(0)
    logs = [(np.log(start * c) + noise.normal(0, START_NOISE, len(start))).astype(np.float32) for c in scales]
    M = len(logs)
    log_mix, log_reg = np.zeros(M, np.float32), np.float32(np.log(START_REG))
    adam = [_Adam(len(start)) for _ in logs]
    adam_mix, adam_reg = _Adam(M), _Adam(())
    sums, sum_mix, sum_reg, kept = [np.zeros_like(v) for v in logs], np.zeros(M), 0.0, 0
    rng = np.random.default_rng(seed)
    smooth = None
    for step in range(1, steps + 1):
        pick = rng.permutation(n)[:FIT_ROWS + HOLD_ROWS]
        A, B = X[pick[:FIT_ROWS]], X[pick[FIT_ROWS:]]
        target_A, target_B = sign[pick[:FIT_ROWS]], sign[pick[FIT_ROWS:]]
        weights = [np.exp(v) for v in logs]
        mix = np.exp(log_mix)
        mix /= mix.sum()
        reg = float(np.exp(log_reg))
        parts_A = [hamming_gram(A, A, [w], [m]) for w, m in zip(weights, mix)]
        parts_B = [hamming_gram(B, A, [w], [m]) for w, m in zip(weights, mix)]
        system = sum(parts_A)
        system[np.diag_indices(FIT_ROWS)] += reg
        factor = np.linalg.cholesky(system)
        K_B = sum(parts_B)
        mean = target_A.mean()
        alpha = _cholesky_solve(factor, target_A - mean)
        error = K_B @ alpha + mean - target_B
        loss = float(np.mean(error ** 2))
        pull = 2 * error / HOLD_ROWS
        adjoint = _cholesky_solve(factor, K_B.T @ pull)

        rate = LEARNING_RATE * min(1.0, step / 5) * (1.0 if step < steps * 0.6 else 0.4)
        average = step > steps * 0.6
        grad_mix = np.zeros(M, np.float32)
        for i in range(M):
            part_A, part_B = parts_A[i], parts_B[i]
            grad_mix[i] = pull @ (part_B @ alpha) - adjoint @ (part_A @ alpha)
            part_B *= pull[:, None]
            part_B *= -alpha[None]
            part_A *= adjoint[:, None]
            part_A *= alpha[None]
            gradient = (B.T @ part_B.sum(1) + A.T @ part_B.sum(0) - 2 * np.einsum("ij,ij->j", B, part_B @ A)
                        + A.T @ part_A.sum(1) + A.T @ part_A.sum(0) - 2 * np.einsum("ij,ij->j", A, part_A @ A))
            logs[i] -= rate * adam[i].direction(gradient * weights[i], step, 1e-7)
            if average:
                sums[i] += logs[i]
        grad_mix -= grad_mix.sum() * mix
        log_mix -= MIX_RATE * adam_mix.direction(grad_mix, step, 1e-9)
        log_reg -= REG_RATE * adam_reg.direction(np.float32(-(adjoint @ alpha) * reg), step, 1e-9)
        if average:
            kept += 1
            sum_mix += log_mix
            sum_reg += log_reg
        smooth = loss if smooth is None else 0.95 * smooth + 0.05 * loss
        if verbose and step % verbose == 0:
            print(f"langkah {step}: galat kuadrat holdout {smooth:.4f}, reg {reg:.3f}, campuran {mix.round(3)}",
                  flush=True)
    mix = np.exp(sum_mix / kept)
    return [np.exp(s / kept) for s in sums], mix / mix.sum(), float(np.exp(sum_reg / kept))
