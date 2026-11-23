import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OUT = Path(__file__).parent / "kaggle_final.ipynb"
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

# Submission Kaggle: SVM Kernel Laplacian dan DTL

*Prediksi pembatalan pesanan hotel dengan model from scratch, tanpa ensemble.*""")

md(f"""{HR}

## Kelompok G04: TreeIsAllYouNeed""")

md(f"""{HR}

## Daftar Isi

1. [**Pendahuluan**](#1)
2. [**Inisialisasi**](#2)
3. [**Data Cleaning dan Transformation**](#3)
4. [**Model 1: SVM Kernel Laplacian**](#4)
5. [**Model 2: DTL C4.5**](#5)
6. [**Referensi**](#6)""")

section("Pendahuluan", "1")

md("""Notebook ini melatih dua model dari data mentah lalu menulis dua berkas submission. Tiap model berdiri
sendiri. Prediksinya tidak digabung atau dirata-rata.

| Model | Berkas | Skor publik Kaggle | Macro F1, CV 5-fold |
|---|---|---|---|
| SVM kernel Laplacian | `output/kaggle_final/svm/submission.csv` | 0,87464 | 0,8703 (7 seed), 0,8699 (4 seed konfirmasi) |
| DTL C4.5 | `output/kaggle_final/dtl/submission.csv` | 0,87393 | 0,8672 (10 seed) |

Pada CV, seluruh fitur dan bobot fitur dihitung ulang di dalam setiap fold. CV untuk SVM memakai
implementasi SMO yang setara dengan toleransi 0,01 agar cepat, sedangkan notebook ini memakai class `SVM`
dengan toleransi 0,001. Kedua toleransi itu menghasilkan label test yang sama.

Seluruh notebook selesai dalam sekitar 1 sampai 10 menit, tergantung memori yang tersedia.""")

section("Inisialisasi", "2")

md("""Pustaka yang dipakai hanya NumPy dan pandas. Algoritma SVM dan DTL diimpor dari `src/algoritma`.""")

code("""import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = next(p for p in [Path.cwd(), *Path.cwd().parents]
            if (p / "src" / "kaggle" / "dataset" / "train.csv").exists())
for folder in ("algoritma", "algoritma/svm", "kaggle"):
    sys.path.insert(0, str(REPO / "src" / folder))

from dtl import DTL
from kaggle_final_features import build_features, clean, feature_weights, load_data, standardize
from svm import SVM

DATA = REPO / "src" / "kaggle" / "dataset"
OUTPUT = REPO / "output" / "kaggle_final"
print("numpy", np.__version__, " pandas", pd.__version__)""")

code("""train_raw, test = load_data(DATA)
sample = pd.read_csv(DATA / "sample_submission.csv", dtype={"id": str})
print("train:", train_raw.shape, " test:", test.shape)


def write_submission(labels, folder):
    submission = pd.DataFrame({"id": test["id"], "is_canceled": labels})
    assert list(submission.columns) == list(sample.columns)
    assert submission["id"].tolist() == sample["id"].tolist()
    folder.mkdir(parents=True, exist_ok=True)
    submission.to_csv(folder / "submission.csv", index=False)
    print("baris:", len(submission), " rasio batal:", round(labels.mean(), 4))""")

section("Data Cleaning dan Transformation", "3")

md("""Duplikat persis pada train dibuang. Nilai kosong diisi median atau modus train, disertai penanda kosong.
Fitur tambahan terdiri atas tanggal booking, lama menginap, dan fitur saudara, yaitu rata-rata label booking
lain yang kemungkinan satu rombongan. Saudara harus memiliki tanggal datang, jumlah malam, dan kanal yang
sama persis, selisih lead time paling banyak 10 hari, selisih harga paling banyak 5%, serta kolom lunak yang
sama atau kosong.

Untuk baris train, fitur saudara dihitung out-of-fold sehingga label baris itu sendiri tidak terlihat.
Semua statistik berasal dari train. Train dan test tidak digabung, dan label test tidak dipakai.""")

code("""train = clean(train_raw)
y = train["is_canceled"].to_numpy()
X_snn, X_snn_test = build_features(train, y, test, shared_neighbours=True)
X_dtl, X_dtl_test = build_features(train, y, test, shared_neighbours=False)
print("train:", len(train_raw), "->", len(train), "baris, fitur:", X_snn.shape[1])
print("sisa nilai kosong di test:", int(X_snn_test.isna().sum().sum()))""")

section("Model 1: SVM Kernel Laplacian", "4")

md("""Kernel Laplacian memakai jarak L1, yaitu K(a, b) = exp(-jumlah(|a - b|) / 20). Jarak L1 menjumlahkan
selisih per fitur, sehingga lebih cocok untuk data tabular daripada jarak L2 pada kernel Gaussian.""")

md("""## Bobot Fitur

Setiap fitur diberi bobot sebelum masuk ke kernel. Bobot dihitung dari train dengan dua putaran: regresi
kernel ridge dilatih, lalu bobot fitur diambil dari besar gradien fungsinya terhadap fitur tersebut. Fitur
yang tidak memengaruhi prediksi mendapat bobot kecil.""")

code("""SCALE = 20.0
A, B, names = standardize(X_snn, X_snn_test)
started = time.time()
weights = feature_weights(A, y, SCALE)
print(f"selesai dalam {time.time() - started:.0f} detik")
pd.Series(weights, index=names).sort_values(ascending=False).head(12)""")

md("""## Pelatihan, Save, dan Load

SVM dilatih dengan SMO dan C = 2. Model disimpan, dimuat kembali, lalu model hasil muat itulah yang dipakai
untuk prediksi. Booking diprediksi batal jika nilai keputusannya positif.""")

code("""started = time.time()
svm = SVM(kernel="laplacian", gamma=1 / SCALE, C=2.0, tol=1e-3, cache_mb=400).fit(A * weights, y)
print(f"iterasi SMO: {svm.n_iter_}, support vector: {len(svm.support_)}, {time.time() - started:.0f} detik")

(OUTPUT / "svm").mkdir(parents=True, exist_ok=True)
svm.save(OUTPUT / "svm" / "svm_model.pkl")
svm = SVM.load(OUTPUT / "svm" / "svm_model.pkl")""")

code("""labels = svm.predict(B * weights)
write_submission(labels, OUTPUT / "svm")""")

section("Model 2: DTL C4.5", "5")

md("""Pohon memakai gain ratio (materi kuliah) dan cost-complexity pruning (di luar materi kuliah). Fitur saudaranya memakai bobot kedekatan lead time
saja. Booking diprediksi batal jika P(batal) minimal 0,4.""")

code("""tree = DTL(max_depth=20, min_samples_leaf=2, pruning="cost_complexity", ccp_alpha=2.5).fit(X_dtl, y)
print("daun:", tree.n_leaves_, " kedalaman:", tree.depth_)

(OUTPUT / "dtl").mkdir(parents=True, exist_ok=True)
tree.save(OUTPUT / "dtl" / "dtl_model.json")
tree = DTL.load(OUTPUT / "dtl" / "dtl_model.json")

# Sumber ambang: C. Elkan, "The Foundations of Cost-Sensitive Learning", IJCAI 2001
labels = (tree.predict_proba(X_dtl_test)[:, 1] >= 0.4).astype(int)
write_submission(labels, OUTPUT / "dtl")""")

section("Referensi", "6")

md("""Metode berikut tidak ada di materi kuliah (PPT Decision Tree, Issues in DTL, dan SVM).

1. J. Platt, "Sequential Minimal Optimization", MSR-TR-98-14, 1998.
2. R.-E. Fan, P.-H. Chen, dan C.-J. Lin, "Working Set Selection Using Second Order Information for Training
   Support Vector Machines", JMLR 6, 2005.
3. C.-C. Chang dan C.-J. Lin, "LIBSVM: A Library for Support Vector Machines", ACM TIST 2(3), 2011.
4. O. Chapelle, P. Haffner, dan V. Vapnik, "Support Vector Machines for Histogram-Based Image Classification",
   IEEE Transactions on Neural Networks 10, 1999 (kernel Laplacian).
5. A. Radhakrishnan dkk., "Mechanism for feature learning in neural networks and backpropagation-free machine
   learning models", Science, 2024, doi:10.1126/science.adi5639 (bobot fitur dari gradien).
6. C. Saunders, A. Gammerman, dan V. Vovk, "Ridge Regression Learning Algorithm in Dual Variables", ICML 1998.
7. C. Williams dan M. Seeger, "Using the Nystrom Method to Speed Up Kernel Machines", NIPS 13, 2001.
8. L. Breiman dkk., *Classification and Regression Trees*, 1984 (cost-complexity pruning).
9. C. Elkan, "The Foundations of Cost-Sensitive Learning", IJCAI 2001 (ambang keputusan 0,4).
10. D. Micci-Barreca, "A preprocessing scheme for high-cardinality categorical attributes", SIGKDD
    Explorations 3(1), 2001 (target encoding).
11. F. Pargent dkk., "Regularized target encoding outperforms traditional methods", Computational Statistics 37,
    2022.
12. R. Jarvis dan E. Patrick, "Clustering Using a Similarity Measure Based on Shared Near Neighbors", IEEE
    Transactions on Computers C-22(11), 1973.""")

nb = {"nbformat": 4, "nbformat_minor": 5,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}},
      "cells": cells}
OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"Wrote {len(cells)} cells -> {OUT}")
