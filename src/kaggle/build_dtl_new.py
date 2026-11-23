import json
from pathlib import Path

OUT = Path(__file__).parent / "dtl_new.ipynb"
cells = []


def md(source):
    cells.append({"cell_type": "markdown", "id": f"c{len(cells):02d}", "metadata": {}, "source": source})


def code(source):
    cells.append({"cell_type": "code", "execution_count": None, "id": f"c{len(cells):02d}",
                  "metadata": {}, "outputs": [], "source": source})


md("""# Decision Tree Learning (DTL): Submission Kaggle

Notebook ini menghasilkan `submission.csv` dari data mentah dengan satu pohon keputusan (kelas `DTL`).

Pohon C4.5 (gain ratio, cost-complexity pruning) sudah dilatih oleh `src/kaggle/dtl_new_train.py` dan
dimuat dari `output/dtl_new/dtl_model.json`. Cara pelatihan dan referensinya ada di skrip tersebut.

| Model | Macro F1, CV 5-fold (seed 11, 22, 33, 2026) |
|---|---|
| DTL pada `notebook.ipynb` | 0,8658 |
| DTL pada notebook ini | 0,8677 |

Skor CV dihitung dengan mengulang seluruh proses pelatihan di dalam setiap fold, dengan ambang 0,4.""")

code("""import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = next(p for p in [Path.cwd(), *Path.cwd().parents]
            if (p / "src" / "kaggle" / "dataset" / "train.csv").exists())
sys.path.insert(0, str(REPO / "src" / "algoritma"))
sys.path.insert(0, str(REPO / "src" / "kaggle"))

from dtl import DTL
from dtl_new_features import build_features, clean, load_data

DATA = REPO / "src" / "kaggle" / "dataset"
OUTPUT = REPO / "output" / "dtl_new"
THRESHOLD = 0.4

train_raw, test = load_data(DATA)
print("train:", train_raw.shape, " test:", test.shape)""")

md("""## Data Cleaning dan Transformation

Duplikat persis pada train dibuang. Nilai kosong diisi median atau modus train, disertai penanda kosong.
Fitur tambahan: tanggal booking, lama menginap, dan rata-rata label booking "saudara" (satu rombongan).
Untuk baris train, rata-rata itu dihitung out-of-fold. Semua statistik berasal dari train, dan label test tidak dipakai.""")

code("""train = clean(train_raw)
X_train, X_test = build_features(train, train["is_canceled"], test)
print("train:", len(train_raw), "->", len(train), "baris")
print("fitur:", X_train.shape[1], " sisa nilai kosong di test:", int(X_test.isna().sum().sum()))""")

md("## Memuat Pohon")

code("""model = DTL.load(OUTPUT / "dtl_model.json")
assert list(X_test.columns) == model.feature_names_
print("daun:", model.n_leaves_, " kedalaman:", model.depth_)


def show(node, depth, prefix=""):
    p = node.class_weights[1] / node.class_weights.sum()
    if node.leaf or depth == 0:
        print(f"{prefix}{'daun' if node.leaf else '...'}  P(batal)={p:.2f}")
        return
    print(f"{prefix}{model.feature_names_[node.feature]} <= {node.threshold:.3f}")
    show(node.left, depth - 1, prefix + "|  ya: ")
    show(node.right, depth - 1, prefix + "|  tidak: ")


show(model.root_, 3)""")

md("""## Prediksi dan Submission

Booking diprediksi batal jika P(batal) minimal 0,4.""")

code("""labels = (model.predict_proba(X_test)[:, 1] >= THRESHOLD).astype(int)
submission = pd.DataFrame({"id": test["id"], "is_canceled": labels})

sample = pd.read_csv(DATA / "sample_submission.csv", dtype={"id": str})
assert list(submission.columns) == list(sample.columns)
assert submission["id"].tolist() == sample["id"].tolist()

submission.to_csv(OUTPUT / "submission.csv", index=False)
print("baris:", len(submission), " rasio batal:", round(labels.mean(), 4))
submission.head()""")

nb = {"nbformat": 4, "nbformat_minor": 5,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}},
      "cells": cells}
OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"Wrote {len(cells)} cells -> {OUT}")
