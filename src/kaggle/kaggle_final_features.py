"""Fitur dan bobot fitur untuk notebook kaggle_final. Semua statistik dihitung dari train saja.

Sumber:
- Target encoding dihaluskan: D. Micci-Barreca, SIGKDD Explorations 3(1), 2001.
- Kesamaan tetangga bersama: R. Jarvis & E. Patrick, IEEE Transactions on Computers C-22, 1973.
- Bobot fitur dari gradien (AGOP): A. Radhakrishnan dkk., Science 383, 2024.
- Pendekatan Nystrom: C. Williams & M. Seeger, NIPS 13, 2001.
"""
import numpy as np
import pandas as pd

from dtl_new_features import (CAT_COLS, ID_COL, KEY, SOFT, TARGET, add_booking_dates, clean,
                              impute_and_flag, load_data)

LEAD_TOLERANCE = 10
RATE_TOLERANCE = 0.05
MISSING_LEAD_WEIGHT = 0.1
INNER_FOLDS = 5

CENTRES = 1500
GRADIENT_ROWS = 4000
RIDGE = 0.3
ROUNDS = 2


def sibling_pairs(reference, query, same=False):
    """Untuk tiap baris query: indeks saudara di reference dan bobot kedekatan lead time."""
    vocab = {c: {v: float(i) for i, v in enumerate(sorted(reference[c].dropna().unique()))}
             for c in KEY + SOFT if c in CAT_COLS}

    def codes(df, cols):
        return np.column_stack([df[c].map(vocab[c]).to_numpy(float) if c in vocab
                                else df[c].to_numpy(float) for c in cols])

    groups = {}
    for j, key in enumerate(map(tuple, codes(reference, KEY))):
        groups.setdefault(key, []).append(j)
    groups = {k: np.array(v) for k, v in groups.items()}
    ref_lead, lead = (d["days_before_arrival"].to_numpy(float) for d in (reference, query))
    ref_rate, rate = (d["nightly_rate_idr"].to_numpy(float) for d in (reference, query))
    ref_soft, soft = codes(reference, SOFT), codes(query, SOFT)
    pairs = []
    for i, key in enumerate(map(tuple, codes(query, KEY))):
        idx = groups.get(key)
        if idx is None:
            pairs.append((np.empty(0, int), np.empty(0)))
            continue
        gap = np.abs(ref_lead[idx] - lead[i])
        ok = np.isnan(gap) | (gap <= LEAD_TOLERANCE)
        rate_gap = np.abs(ref_rate[idx] - rate[i])
        ok &= np.isnan(rate_gap) | (rate_gap <= RATE_TOLERANCE * np.fmax(np.fmax(ref_rate[idx], rate[i]), 1.0))
        ok &= np.all(np.isnan(ref_soft[idx]) | np.isnan(soft[i]) | (ref_soft[idx] == soft[i]), axis=1)
        if same:
            ok &= idx != i
        gap = gap[ok]
        weight = np.where(np.isnan(gap), MISSING_LEAD_WEIGHT, np.clip(1 - gap / (LEAD_TOLERANCE + 1), 0, 1))
        pairs.append((idx[ok], weight))
    return pairs


def shared_neighbour_weights(pairs, reference_sets, include_self):
    """Bobot saudara dikali akar indeks Jaccard antara dua daftar saudara."""
    weights = []
    for i, (idx, w) in enumerate(pairs):
        mine = set(idx.tolist()) | ({i} if include_self else set())
        jaccard = [len(mine & reference_sets[j]) / len(mine | reference_sets[j]) for j in idx]
        weights.append(w * np.sqrt(jaccard) if len(idx) else w)
    return weights


def build_features(train, labels, other, shared_neighbours):
    """Fitur train (saudara out-of-fold) dan fitur data lain (hanya memakai label train)."""
    labels = np.asarray(labels, dtype=float)
    prior = labels.mean()
    raw_train = train.drop(columns=[TARGET]).reset_index(drop=True)
    raw_other = other.drop(columns=[c for c in (ID_COL, TARGET) if c in other]).reset_index(drop=True)
    A, B = impute_and_flag(add_booking_dates(raw_train), add_booking_dates(raw_other))

    train_pairs = sibling_pairs(raw_train, raw_train, same=True)
    other_pairs = sibling_pairs(raw_train, raw_other)
    train_weights, other_weights = ([w for _, w in pairs] for pairs in (train_pairs, other_pairs))
    if shared_neighbours:
        sets = [set(idx.tolist()) | {i} for i, (idx, _) in enumerate(train_pairs)]
        train_weights = shared_neighbour_weights(train_pairs, sets, include_self=True)
        other_weights = shared_neighbour_weights(other_pairs, sets, include_self=False)

    fold = np.empty(len(labels), int)
    for k, held in enumerate(np.array_split(np.random.default_rng(0).permutation(len(labels)), INNER_FOLDS)):
        fold[held] = k

    def encode(pairs, weights, out_of_fold):
        total, weight_sum = np.zeros(len(pairs)), np.zeros(len(pairs))
        for i, ((idx, _), w) in enumerate(zip(pairs, weights)):
            if len(idx):
                keep = fold[idx] != fold[i] if out_of_fold else slice(None)
                weight_sum[i], total[i] = w[keep].sum(), (w[keep] * labels[idx[keep]]).sum()
        return (total + prior) / (weight_sum + 1), weight_sum

    te, count = encode(train_pairs, train_weights, out_of_fold=True)
    te_other, count_other = encode(other_pairs, other_weights, out_of_fold=False)
    count_other = count_other * (INNER_FOLDS - 1) / INNER_FOLDS

    results = []
    for frame, t, c in ((A, te, count), (B, te_other, count_other)):
        frame["sib_te"], frame["sib_n"] = t, c
        frame["nights"] = frame["weekend_nights"] + frame["weekday_nights"]
        frame["weekend_share"] = frame["weekend_nights"] / np.maximum(frame["nights"], 1)
        results.append(pd.get_dummies(frame, columns=CAT_COLS, dtype=float).astype(float))
    return results[0], results[1].reindex(columns=results[0].columns, fill_value=0.0)


def standardize(train, other):
    """Standardisasi dengan statistik train. Lead time diakar, sib_te jadi logit, sib_n jadi log."""
    def logit(v):
        v = np.clip(v, 0.005, 0.995)
        return np.log(v / (1 - v))

    transforms = {"days_before_arrival": lambda v: np.sqrt(np.maximum(v, 0)), "sib_te": logit, "sib_n": np.log1p}
    names, A, B = [], [], []
    for col in train.columns:
        a, b = train[col].to_numpy(float), other[col].to_numpy(float)
        if col in transforms:
            a, b = transforms[col](a), transforms[col](b)
        if a.std() > 1e-9:
            names.append(col)
            A.append((a - a.mean()) / a.std())
            B.append((b - a.mean()) / a.std())
    return np.column_stack(A), np.column_stack(B), names


def laplacian_gram(A, B, scale):
    distances = np.zeros((len(A), len(B)))
    for col in range(A.shape[1]):
        distances += np.abs(A[:, col, None] - B[None, :, col])
    return np.exp(-distances / scale)


def feature_weights(X, y, scale):
    """Bobot fitur: akar dari RMS gradien regresi kernel ridge terhadap tiap fitur (AGOP diagonal)."""
    n, d = X.shape
    target, weights = y - 0.5, np.ones(d)
    centres = np.sort(np.random.default_rng(0).choice(n, CENTRES, replace=False))
    rows = np.random.default_rng(1).choice(n, GRADIENT_ROWS, replace=False)
    for _ in range(ROUNDS):
        Xw = X * weights
        Z = Xw[centres]
        H, g = np.zeros((CENTRES, CENTRES)), np.zeros(CENTRES)
        for start in range(0, n, 2000):
            K = laplacian_gram(Xw[start:start + 2000], Z, scale)
            H += K.T @ K
            g += K.T @ target[start:start + 2000]
        H += RIDGE * laplacian_gram(Z, Z, scale)
        H[np.diag_indices(CENTRES)] += 1e-6 * H.diagonal().mean()
        alpha = np.linalg.solve(H, g)
        Q = Xw[rows]
        gradient = np.empty((GRADIENT_ROWS, d))
        for start in range(0, GRADIENT_ROWS, 1000):
            part = Q[start:start + 1000]
            K = laplacian_gram(part, Z, scale) * alpha
            for col in range(d):
                gradient[start:start + 1000, col] = -(np.sign(part[:, col, None] - Z[None, :, col]) * K).sum(1) / scale
        weights = np.sqrt(np.sqrt(((gradient * weights) ** 2).mean(0)))
        weights *= np.sqrt(d / (weights ** 2).sum())
    return weights.round(3)
