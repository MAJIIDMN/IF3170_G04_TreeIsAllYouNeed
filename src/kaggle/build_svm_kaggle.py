import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OUT = Path(__file__).parent / "svm_kaggle_notebook.ipynb"
HR = "---"
cells = []


def md(source):
    cells.append({"cell_type": "markdown", "id": f"c{len(cells):02d}", "metadata": {}, "source": source})


def code(source):
    cells.append({"cell_type": "code", "execution_count": None, "id": f"c{len(cells):02d}",
                  "metadata": {}, "outputs": [], "source": source})


def section(title, anchor):
    md(f"{HR}\n\n# {title} <a name=\"{anchor}\"></a>\n\n{HR}")


md(f"""{HR}

# Submission Kaggle: SVM dengan Kernel yang Dipelajari

*Satu SVM from scratch (slack kuadrat) dengan kernel Hamming berbobot. Bobot kernel dipelajari dari data latih.*""")

md(f"""{HR}

## Kelompok G04: TreeIsAllYouNeed""")

md("""Alur notebook ini:

1. Fitur dibuat dari data latih saja, lalu diterapkan ke data uji.
2. Tiap fitur diubah menjadi kode termometer, yaitu bit "x > ambang".
3. Bobot tiap bit dipelajari dengan gradien galat holdout (`src/kaggle/kernel_learning.py`).
4. SVM dilatih pada seluruh data latih (`src/algoritma/svm/l2_svm.py`), disimpan, dimuat ulang, lalu dipakai untuk
   membuat submission.

Hanya NumPy, SciPy (faktorisasi Cholesky), dan pandas yang dipakai. Tidak ada ensemble dan tidak ada data luar.""")

section("Inisialisasi", "1")

code("""import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = next(p for p in [Path.cwd(), *Path.cwd().parents]
            if (p / "src" / "kaggle" / "dataset" / "train.csv").exists())
for folder in ("algoritma/svm", "kaggle"):
    sys.path.insert(0, str(REPO / "src" / folder))

from booking_features import build_features, clean, feature_weights, load_data, standardize
from kernel_learning import SCALE, learn_kernel, thermometer
from l2_svm import L2SVM

DATA = REPO / "src" / "kaggle" / "dataset"
OUTPUT = REPO / "output" / "svm"
OUTPUT.mkdir(parents=True, exist_ok=True)
WEIGHTS = OUTPUT / "kernel_weights.npz"
THRESHOLD = -0.10""")

section("Fitur dan Kode Termometer", "2")

md("""Fitur sama dengan `svm_notebook.ipynb` (46 kolom). Bobot AGOP dari kernel Laplacian dipakai sebagai titik awal.
Ambang termometer adalah titik tengah antarnilai unik, atau titik kuantil bila nilai uniknya lebih dari 128.""")

code("""train_raw, test = load_data(DATA)
train = clean(train_raw)
y = train["is_canceled"].to_numpy()
X, X_test = build_features(train, y, test, shared_neighbours=True)
Z, _, names = standardize(X, X_test)
agop = feature_weights(Z, y, SCALE)
codes, codes_test, start, bit_names = thermometer(X, X_test, agop)
print("fitur:", X.shape[1], " bit termometer:", codes.shape[1], " baris latih:", len(codes))""")

section("Belajar Kernel", "3")

md("""Kernel berbentuk K(x, z) = jumlah a_m exp(-jumlah w_mj |b_j(x) - b_j(z)|) dengan dua komponen. Tiap langkah
mengambil 4.000 baris untuk melatih dan 3.000 baris lain untuk menghitung galat, lalu bobot digeser dengan Adam.

Hasil belajar disimpan di `output/svm/kernel_weights.npz`. Pembulatan floating point bisa sedikit berbeda antar
mesin, sehingga model akhir memakai bobot tersimpan agar submission selalu sama.""")

code("""started = time.time()
weights, mixture, reg = learn_kernel(codes, y, start)
print(f"{time.time() - started:.0f} detik")
if not WEIGHTS.exists():
    np.savez(WEIGHTS, weights=np.array(weights), mixture=mixture, reg=reg)
saved = np.load(WEIGHTS)
gap = max(np.abs(np.log(a) - np.log(b)).max() for a, b in zip(weights, saved["weights"]))
print("selisih log bobot terbesar terhadap bobot tersimpan:", round(float(gap), 4))
weights, mixture, reg = list(saved["weights"]), saved["mixture"], float(saved["reg"])
print("campuran:", mixture.round(3), " regularisasi:", round(reg, 3))""")

code("""source = [name.split(" > ")[0] for name in bit_names]
pd.DataFrame({f"komponen {m + 1}": pd.Series(w).groupby(source).sum() for m, w in enumerate(weights)}
             ).sort_values("komponen 1", ascending=False).head(12).round(2)""")

section("Pelatihan dan Submission", "4")

md("""SVM memakai slack kuadrat dan dilatih dengan metode Newton pada himpunan support vector. Ambang keputusan
-0,10 dipilih dari prediksi out-of-fold data latih.""")

code("""started = time.time()
svm = L2SVM(weights, mixture, reg=reg).fit(codes, y)
print(f"iterasi Newton: {svm.n_iter_}, support vector: {len(svm.support_)}, {time.time() - started:.0f} detik")

svm.save(OUTPUT / "svm_kernel_model.pkl")
svm = L2SVM.load(OUTPUT / "svm_kernel_model.pkl")

scores = svm.decision_function(codes_test)
submission = pd.DataFrame({"id": test["id"], "is_canceled": (scores > THRESHOLD).astype(int)})
sample = pd.read_csv(DATA / "sample_submission.csv", dtype={"id": str})
assert list(submission.columns) == list(sample.columns)
assert submission["id"].tolist() == sample["id"].tolist()
submission.to_csv(OUTPUT / "submission.csv", index=False)
print("baris:", len(submission), " rasio batal:", round(submission["is_canceled"].mean(), 4),
      " jarak skor terdekat ke ambang:", float(np.abs(scores - THRESHOLD).min()))""")

section("Referensi", "5")

md("""1. O. Chapelle, "Training a Support Vector Machine in the Primal", Neural Computation 19(5), 2007.
2. O. Chapelle, V. Vapnik, O. Bousquet, dan S. Mukherjee, "Choosing Multiple Parameters for Support Vector
   Machines", Machine Learning 46, 2002.
3. H. Owhadi dan G. R. Yoo, "Kernel Flows", Journal of Computational Physics 389, 2019.
4. G. Lanckriet dkk., "Learning the Kernel Matrix with Semidefinite Programming", JMLR 5, 2004.
5. D. Kingma dan J. Ba, "Adam: A Method for Stochastic Optimization", ICLR 2015.
6. M. Hestenes dan E. Stiefel, "Methods of Conjugate Gradients for Solving Linear Systems", J. Res. NBS 49(6), 1952.
7. Y. Gorishniy, I. Rubachev, dan A. Babenko, "On Embeddings for Numerical Features in Tabular Deep Learning",
   NeurIPS 2022.
8. A. Radhakrishnan dkk., "Mechanism for feature learning in neural networks and backpropagation-free machine
   learning models", Science, 2024 (bobot awal).""")

nb = {"nbformat": 4, "nbformat_minor": 5,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}},
      "cells": cells}
OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"Wrote {len(cells)} cells -> {OUT}")
