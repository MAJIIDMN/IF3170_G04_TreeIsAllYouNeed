import json
import sys
from pathlib import Path
from uuid import uuid4

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OUT = Path(__file__).parent / "eda.ipynb"
HR  = "---"

def md(source: str) -> dict:
    return {"cell_type": "markdown", "id": uuid4().hex[:8], "metadata": {}, "source": source}

def code(source: str) -> dict:
    return {
        "cell_type": "code", "execution_count": None, "id": uuid4().hex[:8],
        "metadata": {}, "outputs": [], "source": source,
    }

def section(title: str, anchor: str) -> dict:
    return md(
        f"{HR}\n\n"
        f"# {title} <a name=\"{anchor}\"></a>\n\n"
        f"{HR}"
    )

# Insights are written only after reading the executed notebook output.
INSIGHTS = {
    "schema": [
        "Data memuat 17 fitur, terdiri atas 14 numerik dan 3 kategorikal. Kolom hitungan bertipe float "
        "karena mengandung nilai kosong. Rentang fitur sangat timpang, dari 0-1 pada fitur biner hingga "
        "9,165 juta pada `nightly_rate_idr`.",
    ],
    "missing": [
        "Delapan kolom memiliki nilai kosong sebesar 2,96% sampai 11,09%, dengan pola yang hampir sama "
        "pada test. Hanya 60,8% baris yang lengkap, sehingga menghapus baris akan membuang 39,2% data "
        "dan tidak dapat dilakukan pada test. Imputasi wajib dilakukan.",
        "Hanya `special_requests` yang kekosongannya berkaitan dengan target (batal 40,8% saat kosong "
        "dan 32,0% saat terisi). Tujuh kolom lain tidak signifikan setelah koreksi Bonferroni, sehingga "
        "imputasi sederhana memadai untuk kolom tersebut.",
        "Kekosongan `special_requests` condong ke nilai nol. Pada Online Travel Agent, nilai nol hanya "
        "36,6% dari baris terisi, tetapi tingkat batal baris kosong menyiratkan sekitar 82,9% nol. Pola "
        "yang sama muncul pada Offline Agent (87,8% terhadap 79,9%). Estimasi ini mengasumsikan tingkat "
        "batal per nilai sama pada baris kosong dan terisi. Imputasi dengan median, yaitu 0, sesuai "
        "dengan nilai yang paling mungkin, dan penanda kosong dapat ditambahkan.",
    ],
    "duplicates": [
        "Terdapat 62 grup duplikat dengan 66 baris berlebih, dan 2 grup memiliki label berbeda. Membuang "
        "duplikat identik termasuk labelnya menghapus 64 baris (0,2%), sehingga dampaknya kecil dan aman. "
        "Sebanyak 42 baris test memiliki fitur yang sama dengan train, tetapi baris test tetap harus "
        "diprediksi dan tidak boleh dihapus.",
    ],
    "anomalies": [
        "Sebagian besar nilai yang tampak aneh ternyata informatif dan muncul juga pada test, sehingga "
        "tidak dihapus. Tarif nol memiliki tingkat batal 0,7%, total malam nol 3,1%, dan total malam "
        "minimal 15 mencapai 80,0%.",
        "`num_adults = 0` bukan kesalahan data, karena 105 dari 106 baris memiliki anak dan satu sisanya "
        "kosong. `is_returning_guest` juga sepenuhnya konsisten dengan riwayat pesanan.",
        "Median tarif kanal Complimentary adalah 0, sedangkan median keseluruhan 1,644 juta. Imputasi "
        "tarif dengan median global akan keliru pada kanal ini, sehingga imputasi per kanal lebih tepat.",
        "Outlier yang layak dipangkas hanya sedikit, yaitu `num_children` >= 9 dan tarif di atas 6 juta "
        "(masing-masing 3 baris, tidak ada pada test).",
    ],
    "target": [
        "Kelas tidak seimbang secara moderat dengan rasio 2,05 banding 1. Memprediksi seluruh pesanan "
        "tidak batal menghasilkan akurasi 0,6723 tetapi Macro F1 hanya 0,4020. Baseline inilah yang "
        "harus dilampaui, dan akurasi tidak dapat dijadikan acuan.",
    ],
    "continuous": [
        "`days_before_arrival` adalah fitur terkuat. Tingkat batal naik dari 8,5% pada desil terendah ke "
        "75,7% pada desil tertinggi, dengan lompatan tajam dari 36,3% ke 68,0% di sekitar 151 hari. "
        "Hubungan ini monotonik tetapi tidak linear.",
        "`nightly_rate_idr` tidak monotonik. Tingkat batal naik sampai puncak 47,5% pada tarif sekitar "
        "1,8 sampai 1,9 juta, lalu turun lagi. Kernel linear tidak dapat menangkap bentuk ini tanpa "
        "transformasi fitur. Tarif nol tampak sebagai kelompok terpisah yang hampir selalu tidak batal.",
    ],
    "counts": [
        "`special_requests` adalah sinyal negatif terkuat. Tingkat batal turun dari 43,2% pada nilai 0 "
        "ke 23,8% dan 14,3%, lalu 0% untuk seluruh 563 baris dengan nilai minimal 3.",
        "Tamu lama (1,5%), kebutuhan parkir (10,1%), dan riwayat pembatalan cenderung menurunkan risiko. "
        "Masa inap panjang cenderung menaikkannya. `num_children` dan `num_adults` lemah, dan nilai "
        "pucat pada grafik berasal dari terlalu sedikit baris untuk ditafsirkan.",
    ],
    "categorical": [
        "`booking_channel` paling kuat di antara ketiganya. Complimentary tidak pernah batal, Corporate "
        "11,5%, dan Online Travel Agent 36,4%. Pada `meal_package`, Half Board menonjol dengan 45,9%, "
        "sedangkan `room_category` paling lemah.",
        "Kategori langka perlu digabung agar tidak menghasilkan kolom one-hot yang hampir selalu nol. "
        "Full Board dan Type E adalah kasus terparah. Baris kosong pada kedua kolom memiliki tingkat "
        "batal mendekati rata-rata, sehingga cukup diberi kategori sendiri.",
    ],
    "monthly": [
        "Train dan test mencakup rentang tanggal yang sama, dan porsi test per bulan stabil pada 19,0% "
        "sampai 22,8%. Pembagian data bersifat acak, bukan temporal. Stratified K-Fold karena itu "
        "mewakili kondisi test Kaggle.",
        "Tingkat batal bergerak dari 2,5% pada Desember 2023 hingga 64,2% pada Juli 2023, walaupun Juli "
        "2023 hanya memuat 274 baris. Periode kedatangan membawa sinyal yang besar.",
    ],
    "calendar": [
        "Pola ini bukan musiman murni. Oktober bernilai 16,2% pada 2023 dan 46,5% pada 2024, sedangkan "
        "Juli bernilai 64,2% dan 42,0%. `arrival_month` saja tidak cukup, sehingga kombinasi tahun dan "
        "bulan perlu diperlakukan sebagai kategori.",
        "Hari kedatangan (30,5% sampai 37,2%) dan tanggal (26,8% sampai 40,7%) jauh lebih lemah. "
        "Kombinasi tahun-bulan efektif untuk Kaggle karena test berada pada periode yang sama, tetapi "
        "tidak akan berlaku untuk bulan baru pada penerapan nyata.",
    ],
    "interaction": [
        "Efek lead time bergantung pada periode. Lead time 151 sampai 250 hari memiliki tingkat batal "
        "0,72 sampai 0,82 pada Mei hingga Oktober 2024, tetapi hanya 0,38 pada Desember 2024. Periode "
        "akhir 2023 hingga Januari 2024 rendah untuk lead time di bawah 90 hari. Interaksi ini "
        "mendukung penggunaan kernel nonlinear.",
    ],
    "booking": [
        "Bulan pemesanan membawa sinyal yang tidak identik dengan bulan kedatangan, misalnya lompatan "
        "dari 17,3% pada September 2023 ke 45,1% pada Oktober 2023.",
        "Penurunan hingga 0,3% pada Desember 2024 perlu dibaca hati-hati. Seluruh kedatangan berakhir "
        "pada Desember 2024, sehingga pesanan pada bulan itu pasti memiliki lead time pendek, dan lead "
        "time pendek memang jarang batal.",
    ],
    "correlation": [
        "Korelasi terhadap target terkuat dimiliki `days_before_arrival` (0,418) dan `special_requests` "
        "(-0,261). Redundansi utama adalah `prior_completed_bookings` dengan `is_returning_guest` "
        "(0,932). Selain kelompok riwayat tamu, tidak ada pasangan fitur dengan korelasi tinggi.",
    ],
    "relevance": [
        "Mutual information menempatkan `days_before_arrival` di puncak (0,105), diikuti `arrival_ym` "
        "(0,052), `special_requests` (0,037), dan `booking_ym` (0,028). `arrival_ym` jauh melampaui "
        "`arrival_month` (0,025) dan memiliki Cramer's V tertinggi (0,298), sehingga mendukung kombinasi "
        "tahun dan bulan.",
        "`room_category` dan `num_children` memiliki nilai terendah. Ukuran ini univariat dan tidak "
        "melihat interaksi, sehingga fitur lemah tidak langsung dibuang. Pemilihan fitur diputuskan "
        "melalui validasi silang.",
    ],
    "drift": [
        "Tidak ada pergeseran distribusi. PSI tertinggi 0,0021 dan KS tertinggi 0,0149, jauh di bawah "
        "batas 0,1, dan tidak ada kategori baru pada test. Skor validasi silang dapat dijadikan "
        "perkiraan skor test.",
    ],
    "scale": [
        "Rentang `nightly_rate_idr` sekitar 870 ribu kali median rentang fitur lain, sehingga tanpa "
        "scaling jarak antarbaris praktis hanya ditentukan tarif. `prior_cancellations` (skew 25,95) dan "
        "`prior_completed_bookings` (skew 20,08) sangat miring, sehingga log atau pemangkasan perlu "
        "diterapkan sebelum standardisasi.",
    ],
    "pca": [
        "Dua komponen utama hanya menyimpan 25,7% variansi, dan model linear pada proyeksi ini mencapai "
        "ROC-AUC 0,693, jauh di bawah model pada fitur penuh. Kedua kelas juga banyak bertumpuk. Proyeksi "
        "2D sebaiknya dipakai hanya untuk gambar bonus, dan gambar itu tidak mewakili performa model.",
        "Dibutuhkan 20 dari 78 komponen untuk 90% variansi. PCA tidak mendesak bagi SVM di sini, karena "
        "78 dimensi tergolong kecil dan biaya utama SVM berasal dari jumlah baris.",
    ],
    "groups": [
        "Skema validasi tidak mengubah kesimpulan. Macro F1 SVM adalah 0,8158 pada fold acak dan 0,8148 "
        "pada fold kelompok, dan ROC-AUC boosting 0,9304 terhadap 0,9296. Stratified K-Fold dapat "
        "digunakan.",
        "Efek rombongan nyata tetapi moderat. Pada grup minimal 10 baris, 61,1% berlabel seragam, "
        "dibandingkan harapan 45,1% dari model tanpa informasi kelompok. Pembanding tingkat batal global "
        "(1,1%) menyesatkan, karena sebagian besar keseragaman sudah dijelaskan oleh lead time dan "
        "periode.",
    ],
    "kernel": [
        "Scaling bersifat wajib. Tanpa scaling, kernel Gaussian jatuh ke Macro F1 0,4021, sama dengan "
        "memprediksi seluruh baris tidak batal.",
        "Set `engineered` mengungguli `raw` pada ketiga kernel, dengan kenaikan Macro F1 sekitar 0,01. "
        "Kernel nonlinear mengungguli linear sekitar 0,02 (0,7889 dan 0,7861 terhadap 0,7672). Selisih "
        "polynomial dan Gaussian berada dalam simpangan baku pada hyperparameter bawaan, sehingga "
        "pemenangnya belum dapat ditentukan sebelum tuning.",
        "`class_weight='balanced'` menaikkan Macro F1 dari 0,7861 ke 0,7905, tetapi selisih ini masih "
        "dalam simpangan baku. Seluruh angka di sini berasal dari sampel 6.000 baris, sehingga hanya "
        "sahih untuk perbandingan antarkonfigurasi.",
    ],
    "sample_size": [
        "Data tambahan terus membantu. Macro F1 naik dari 0,7507 pada 1.000 baris ke 0,8154 pada 21.847 "
        "baris. Sampel 4.000 baris kehilangan 0,040 Macro F1, sehingga subsampel kecil bukan pilihan "
        "yang baik. Kenaikan mulai melambat setelah 16.000 baris.",
        "Waktu latih tumbuh sekitar n^1,8 sampai n^1,9 (bervariasi antarrun) dan support vector mencakup "
        "sekitar 43% data. Matriks kernel penuh seluruh train membutuhkan 5,97 GB pada float64. "
        "Implementasi from scratch karena itu memerlukan SMO yang menghitung baris kernel sesuai "
        "kebutuhan dengan cache, bukan matriks penuh.",
        "Pada data yang sama, boosting mencapai Macro F1 0,8401, atau selisih sekitar 0,025 dari SVM "
        "tanpa tuning. Angka ini hanya pembanding batas sinyal, bukan target yang pasti tercapai, dan "
        "berasal dari satu holdout.",
    ],
    "threshold": [
        "Pergeseran threshold hanya menambah sekitar 0,005 Macro F1 (0,7909 ke 0,7961 tanpa bobot dan "
        "0,7951 ke 0,7997 dengan balanced). Angka terbaik dipilih pada data yang sama, sehingga sedikit "
        "optimistis. Tanpa bobot, threshold terbaik negatif (-0,35), yang berarti model lebih diuntungkan "
        "bila lebih mudah memprediksi batal, konsisten dengan arah pembobotan kelas.",
        "Bobot kelas dan threshold mengoreksi masalah yang sama, sehingga keuntungannya tidak bertambah "
        "penuh. Keduanya dipilih melalui validasi silang saat tuning, dan kenaikan terbesar diperkirakan "
        "datang dari `C`, `gamma`, serta jumlah data.",
    ],
    "conclusion": (
        "## Ringkasan Temuan\n\n"
        "Data bersih secara struktur, tetapi 39,2% baris memiliki nilai kosong. Sinyal terkuat berasal "
        "dari lead time, periode kedatangan dan pemesanan, permintaan khusus, serta kanal pemesanan. "
        "Banyak hubungan bersifat nonlinear dan saling berinteraksi. Train dan test berasal dari "
        "distribusi yang sama, sehingga validasi silang berstrata dapat dipercaya untuk Kaggle.\n\n"
        "## Rekomendasi Preprocessing\n\n"
        "Seluruh langkah dipelajari dari data latih, lalu diterapkan pada validasi dan test tanpa "
        "menggabungkan keduanya.\n\n"
        "| Langkah | Keputusan | Bukti |\n"
        "|---|---|---|\n"
        "| Duplikat | Buang 64 baris identik pada train, test tidak disentuh | Bagian 3 |\n"
        "| Nilai kosong numerik | Imputasi median | 7 dari 8 kolom tidak terkait target |\n"
        "| `special_requests` | Imputasi 0 dan tambah penanda kosong | Kekosongan condong ke nol |\n"
        "| `nightly_rate_idr` | Imputasi median per kanal, tambah penanda tarif nol | Median Complimentary 0 |\n"
        "| Kategorikal | One-hot, kategori `kosong`, gabungkan kategori langka | Full Board dan Type E |\n"
        "| Anomali | Pertahankan, pangkas hanya outlier ekstrem | Informatif dan muncul pada test |\n"
        "| Fitur waktu | `arrival_ym` dan `booking_ym` sebagai kategori | Bagian 6 dan 7 |\n"
        "| Fitur turunan | `total_nights`, log lead time | Set `engineered` lebih baik |\n"
        "| Fitur miring | Log atau pangkas riwayat tamu | Skew di atas 20 |\n"
        "| Scaling | Standardisasi wajib | ROC-AUC 0,5548 tanpa scaling |\n"
        "| PCA | Hanya untuk visualisasi bonus | ROC-AUC 2D 0,693 |\n\n"
        "## Rekomendasi SVM\n\n"
        "1. Solver memakai SMO dengan cache baris kernel, dilatih pada seluruh data latih.\n"
        "2. Ketiga kernel diimplementasikan, dan polynomial serta Gaussian menjadi kandidat utama.\n"
        "3. Parameter `C`, `gamma`, dan `degree` dituning untuk Macro F1 dengan Stratified K-Fold.\n"
        "4. Bobot kelas dan threshold dipilih untuk Macro F1, dan `class_weight` juga melayani bonus "
        "cost-sensitive.\n"
        "5. `sklearn.svm.SVC` pada bagian ini menjadi acuan kebenaran implementasi from scratch.\n\n"
        "## Keterbatasan\n\n"
        "1. Deskripsi resmi fitur belum dicocokkan dengan tab Data pada Kaggle.\n"
        "2. Hyperparameter pada uji SVM masih bawaan, sehingga skor bersifat arahan.\n"
        "3. Fitur periode efektif untuk test Kaggle, tetapi tidak berlaku untuk bulan baru pada "
        "penerapan nyata. Evaluasi berbasis waktu diperlukan bila sistem benar-benar digunakan."
    ),
}

def insight(key: str) -> dict:
    paragraphs = INSIGHTS.get(key, ["Belum diisi."])
    return md("#### Insights\n\n" + "\n\n".join(f"> {p}" for p in paragraphs))

cells = []

# ---------------------------------------------------------------------------
# Banner (3 cells) + TOC
# ---------------------------------------------------------------------------
cells.append(md(
    f"{HR}\n\n"
    "# Exploratory Data Analysis: Pembatalan Pesanan Ronaldo Hotel\n\n"
    "*Analisis data untuk Tugas Besar 2 IF3170 Inteligensi Artifisial, "
    "sebagai dasar perancangan preprocessing dan model SVM.*"
))

cells.append(md(
    f"{HR}\n\n"
    "## Kelompok G04: TreeIsAllYouNeed\n\n"
    "- IF3170 Inteligensi Artifisial 2026/2027"
))

cells.append(md(
    f"{HR}\n\n"
    "## Daftar Isi\n\n"
    "1. [**Introduction**](#1)\n"
    "2. [**Initialization**](#2)\n"
    "3. [**Struktur dan Kualitas Data**](#3)\n"
    "4. [**Distribusi Target**](#4)\n"
    "5. [**Fitur terhadap Target**](#5)\n"
    "6. [**Pola Waktu**](#6)\n"
    "7. [**Relevansi dan Redundansi Fitur**](#7)\n"
    "8. [**Perbandingan Train dan Test**](#8)\n"
    "9. [**Implikasi untuk SVM**](#9)\n"
    "10. [**Kesimpulan dan Rekomendasi**](#10)\n"
))

# ---------------------------------------------------------------------------
# Section 1: Introduction
# ---------------------------------------------------------------------------
cells.append(section("Introduction", "1"))

cells.append(md(
    "## Overview\n\n"
    "*Ronaldo Hotel membutuhkan sistem yang mengestimasi pengunjung yang akan membatalkan "
    "pesanan. Estimasi ini membantu pengelolaan kapasitas kamar tanpa sumber daya komputasi "
    "sebesar deep learning.*"
))

cells.append(md(
    "## Aim\n\n"
    "*Notebook ini memahami struktur, kualitas, dan pola data sebelum SVM dibangun. "
    "Keluarannya adalah daftar keputusan preprocessing yang didukung bukti, serta arahan "
    "pemilihan kernel dan ukuran data latih untuk SVM from scratch.*"
))

cells.append(md(
    "## Metric\n\n"
    "***Macro F1-score** adalah metrik resmi kompetisi, yaitu rata-rata F1 kelas batal dan kelas "
    "tidak batal. Kedua kelas berbobot sama, sehingga model yang hanya memprediksi satu kelas "
    "mendapat skor rendah walaupun akurasinya tinggi.*\n\n"
    "$$\\text{Macro F1} = \\frac{F_1^{(0)} + F_1^{(1)}}{2}, \\qquad "
    "F_1^{(c)} = \\frac{2 \\cdot \\text{Precision}^{(c)} \\cdot \\text{Recall}^{(c)}}"
    "{\\text{Precision}^{(c)} + \\text{Recall}^{(c)}}$$\n\n"
    "***ROC-AUC** ikut dilaporkan untuk menilai kualitas peringkat skor tanpa bergantung pada "
    "threshold. Leaderboard publik memakai sekitar 30% data test dan peringkat akhir memakai 70% "
    "sisanya, sehingga keputusan model didasarkan pada validasi silang, bukan skor publik.*"
))

cells.append(md(
    "## Dataset\n\n"
    "*Data berasal dari kompetisi Kaggle Tugas Besar 2 IF3170 pada tautan: "
    "https://www.kaggle.com/competitions/T2-IF3170-2026-2027*\n\n"
    "*Dataset terdiri atas 27.373 baris latih berlabel dan 6.844 baris uji tanpa label. "
    "Deskripsi di bawah disimpulkan dari nama kolom dan perlu dicocokkan dengan deskripsi "
    "resmi di halaman kompetisi.*\n\n"
    "*Dataset digunakan untuk memprediksi status pembatalan setiap pesanan.*\n\n"
    "**Attributes**\n\n"
    "1.  `id`                        : Identitas unik pesanan, bukan fitur\n"
    "2.  `num_adults`                : Jumlah tamu dewasa\n"
    "3.  `num_children`              : Jumlah tamu anak\n"
    "4.  `weekend_nights`            : Jumlah malam akhir pekan\n"
    "5.  `weekday_nights`            : Jumlah malam hari kerja\n"
    "6.  `meal_package`              : Paket makan\n"
    "7.  `needs_parking`             : Kebutuhan parkir (0 atau 1)\n"
    "8.  `room_category`             : Kategori kamar (Type A sampai Type G)\n"
    "9.  `days_before_arrival`       : Jarak hari antara pemesanan dan kedatangan\n"
    "10. `arrival_year`              : Tahun kedatangan\n"
    "11. `arrival_month`             : Bulan kedatangan\n"
    "12. `arrival_day`               : Tanggal kedatangan\n"
    "13. `booking_channel`           : Kanal pemesanan\n"
    "14. `is_returning_guest`        : Status tamu lama (0 atau 1)\n"
    "15. `prior_cancellations`       : Jumlah pembatalan sebelumnya\n"
    "16. `prior_completed_bookings`  : Jumlah pesanan selesai sebelumnya\n"
    "17. `nightly_rate_idr`          : Tarif per malam dalam rupiah\n"
    "18. `special_requests`          : Jumlah permintaan khusus\n\n"
    "**Target**\n"
    " `is_canceled`  : Status pembatalan (0: Tidak batal, 1: Batal)"
))

cells.append(md(
    "## Approach: EDA Terarah untuk SVM\n\n"
    "*EDA ini tidak berhenti pada deskripsi statistik. Setiap temuan diterjemahkan menjadi "
    "keputusan bagi SVM, yang sensitif terhadap skala fitur, bergantung pada encoding numerik, "
    "dan memiliki biaya latih yang tumbuh cepat terhadap jumlah data.*\n\n"
    "```\n"
    "Kualitas data (missing, duplikat, anomali)\n"
    "    |\n"
    "    +-- Pola fitur dan waktu terhadap target\n"
    "         |\n"
    "    Perbandingan train dan test (tanpa penggabungan)\n"
    "         |\n"
    "    Uji cepat SVM -> kernel, skala, ukuran data latih\n"
    "```\n\n"
    "*Train dan test tidak pernah digabung. Pada uji cepat, seluruh imputasi, encoding, dan "
    "scaling berada di dalam pipeline sehingga hanya dipelajari dari fold latih.*"
))

# ---------------------------------------------------------------------------
# Section 2: Initialization
# ---------------------------------------------------------------------------
cells.append(section("Initialization", "2"))

cells.append(md(
    "## Environment Setup\n\n"
    "Notebook ini berjalan pada CPU dan tidak memerlukan GPU. Dependensi tercatat pada "
    "`requirements.txt` di root repository dan dipasang dengan "
    "`uv pip install -r requirements.txt` di dalam `.venv`."
))

cells.append(md("## Import Libraries"))

cells.append(code(
    "import os\n"
    "import random\n"
    "import time\n"
    "from pathlib import Path\n"
    "\n"
    "import matplotlib.pyplot as plt\n"
    "import numpy as np\n"
    "import pandas as pd\n"
    "import seaborn as sns\n"
    "from IPython.display import display\n"
    "from matplotlib.colors import LinearSegmentedColormap\n"
    "from scipy import stats\n"
    "from sklearn.compose import ColumnTransformer\n"
    "from sklearn.decomposition import PCA\n"
    "from sklearn.ensemble import HistGradientBoostingClassifier\n"
    "from sklearn.feature_selection import mutual_info_classif\n"
    "from sklearn.impute import SimpleImputer\n"
    "from sklearn.linear_model import LogisticRegression\n"
    "from sklearn.metrics import accuracy_score, f1_score, roc_auc_score\n"
    "from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_validate, train_test_split\n"
    "from sklearn.pipeline import Pipeline\n"
    "from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler\n"
    "from sklearn.svm import SVC"
))

cells.append(md("## Seed Everything"))

cells.append(code(
    "def seed_everything(seed: int = 42):\n"
    "    random.seed(seed)\n"
    "    os.environ['PYTHONHASHSEED'] = str(seed)\n"
    "    np.random.seed(seed)\n"
    "\n"
    "seed_everything(42)"
))

cells.append(md(
    "## Settings\n\n"
    "Seluruh path, kelompok kolom, dan parameter uji dipusatkan di sini. "
    "Perpindahan antara Kaggle dan lokal tidak memerlukan perubahan di luar kelas ini."
))

cells.append(code(
    "class Settings:\n"
    "    SEED         = 42\n"
    "    _KAGGLE_HITS = sorted(Path('/kaggle/input').glob('**/train.csv'))\n"
    "\n"
    "    DATA_DIR   = _KAGGLE_HITS[0].parent if _KAGGLE_HITS else Path('dataset')\n"
    "    OUTPUT_DIR = Path('/kaggle/working') if _KAGGLE_HITS else Path('../../output')\n"
    "    FIG_DIR    = OUTPUT_DIR / 'eda'\n"
    "\n"
    "    ID_COL     = 'id'\n"
    "    TARGET     = 'is_canceled'\n"
    "    CAT_COLS   = ['meal_package', 'room_category', 'booking_channel']\n"
    "    BIN_COLS   = ['needs_parking', 'is_returning_guest']\n"
    "    COUNT_COLS = [\n"
    "        'num_adults', 'num_children', 'weekend_nights', 'weekday_nights',\n"
    "        'prior_cancellations', 'prior_completed_bookings', 'special_requests',\n"
    "    ]\n"
    "    CONT_COLS  = ['days_before_arrival', 'nightly_rate_idr']\n"
    "    DATE_COLS  = ['arrival_year', 'arrival_month', 'arrival_day']\n"
    "    NUM_COLS   = COUNT_COLS + CONT_COLS + BIN_COLS + DATE_COLS\n"
    "    FEATURES   = NUM_COLS + CAT_COLS\n"
    "\n"
    "    PROBE_SAMPLE = 6000\n"
    "    PROBE_FOLDS  = 3\n"
    "    SAMPLE_SIZES = [1000, 2000, 4000, 8000, 16000]\n"
    "    HOLDOUT_SIZE = 0.2\n"
    "\n"
    "    NEG_COLOR = '#2a78d6'\n"
    "    POS_COLOR = '#eb6834'\n"
    "    MID_COLOR = '#e8e7e3'\n"
    "    SURFACE   = '#fcfcfb'\n"
    "    INK       = '#0b0b0b'\n"
    "    MUTED     = '#52514e'\n"
    "\n"
    "CFG = Settings()\n"
    "CFG.FIG_DIR.mkdir(parents=True, exist_ok=True)\n"
    "\n"
    "plt.rcParams.update({\n"
    "    'figure.facecolor': CFG.SURFACE, 'axes.facecolor': CFG.SURFACE,\n"
    "    'savefig.facecolor': CFG.SURFACE, 'figure.dpi': 110,\n"
    "    'axes.edgecolor': CFG.MUTED, 'axes.labelcolor': CFG.MUTED,\n"
    "    'axes.titlecolor': CFG.INK, 'axes.titlesize': 11, 'axes.labelsize': 9,\n"
    "    'axes.spines.top': False, 'axes.spines.right': False,\n"
    "    'axes.grid': True, 'grid.color': CFG.MID_COLOR, 'grid.linewidth': 0.6,\n"
    "    'axes.axisbelow': True, 'xtick.color': CFG.MUTED, 'ytick.color': CFG.MUTED,\n"
    "    'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.frameon': False,\n"
    "    'legend.fontsize': 8, 'font.size': 9,\n"
    "})\n"
    "pd.set_option('display.max_columns', 50, 'display.width', 160)\n"
    "\n"
    "DIVERGING = LinearSegmentedColormap.from_list(\n"
    "    'blue_gray_orange', [CFG.NEG_COLOR, CFG.MID_COLOR, CFG.POS_COLOR]\n"
    ")\n"
    "SEQUENTIAL = LinearSegmentedColormap.from_list(\n"
    "    'orange_ramp', [CFG.SURFACE, CFG.POS_COLOR, '#8a3312']\n"
    ")\n"
    "\n"
    "def save_fig(name: str):\n"
    "    plt.tight_layout()\n"
    "    plt.savefig(CFG.FIG_DIR / f'{name}.png', dpi=150, bbox_inches='tight')\n"
    "    plt.show()"
))

cells.append(md(
    "## Load Dataset\n\n"
    "Berkas dibaca sebagai dua objek terpisah. Keduanya tidak digabung di bagian mana pun."
))

cells.append(code(
    "train = pd.read_csv(CFG.DATA_DIR / 'train.csv')\n"
    "test  = pd.read_csv(CFG.DATA_DIR / 'test.csv')\n"
    "\n"
    "assert not set(train[CFG.ID_COL]) & set(test[CFG.ID_COL])\n"
    "assert train[CFG.ID_COL].is_unique and test[CFG.ID_COL].is_unique\n"
    "assert set(train.columns) - set(test.columns) == {CFG.TARGET}\n"
    "\n"
    "print(f'Train: {train.shape}  Test: {test.shape}')"
))

# ---------------------------------------------------------------------------
# Section 3: Data quality
# ---------------------------------------------------------------------------
cells.append(section("Struktur dan Kualitas Data", "3"))

cells.append(md(
    "Bagian ini memeriksa tipe kolom, nilai hilang, duplikat, dan nilai yang tidak wajar. "
    "Hasilnya menentukan langkah cleaning, yang pada SVM wajib tuntas karena model tidak "
    "dapat menerima nilai kosong."
))

cells.append(md(
    "## Skema Kolom\n\n"
    "Ringkasan tipe, kardinalitas, dan rentang setiap fitur memperlihatkan kolom yang "
    "perlu encoding serta perbedaan skala antarfitur."
))

cells.append(code(
    "schema = pd.DataFrame({\n"
    "    'dtype': train[CFG.FEATURES].dtypes.astype(str),\n"
    "    'n_unique': train[CFG.FEATURES].nunique(),\n"
    "    'missing_pct': (train[CFG.FEATURES].isna().mean() * 100).round(2),\n"
    "    'min': train[CFG.NUM_COLS].min(),\n"
    "    'median': train[CFG.NUM_COLS].median(),\n"
    "    'max': train[CFG.NUM_COLS].max(),\n"
    "}).loc[CFG.FEATURES]\n"
    "schema"
))

cells.append(insight("schema"))

cells.append(md(
    "## Nilai Hilang\n\n"
    "Persentase nilai hilang dibandingkan antara train dan test untuk melihat apakah pola "
    "kekosongan sama pada kedua berkas."
))

cells.append(code(
    "missing = pd.DataFrame({\n"
    "    'train': train[CFG.FEATURES].isna().mean() * 100,\n"
    "    'test': test[CFG.FEATURES].isna().mean() * 100,\n"
    "})\n"
    "missing = missing[missing['train'] > 0].sort_values('train')\n"
    "\n"
    "ax = missing.plot.barh(\n"
    "    figsize=(7, 3.6), width=0.75, color=[CFG.NEG_COLOR, CFG.POS_COLOR], edgecolor=CFG.SURFACE\n"
    ")\n"
    "ax.set_title('Persentase nilai hilang per kolom')\n"
    "ax.set_xlabel('Persentase baris (%)')\n"
    "ax.grid(axis='y', visible=False)\n"
    "save_fig('missing_pct')\n"
    "\n"
    "row_missing = train[CFG.FEATURES].isna().sum(axis=1).value_counts().sort_index()\n"
    "print('Jumlah kolom kosong per baris:')\n"
    "print(row_missing.to_string())\n"
    "print(f'Baris lengkap: {row_missing[0] / len(train):.1%}')"
))

cells.append(md(
    "Mekanisme kekosongan diuji dengan chi-square antara indikator kosong dan target. "
    "Kekosongan yang berkaitan dengan target berarti indikator kosong itu sendiri membawa "
    "sinyal, sehingga imputasi saja akan membuang informasi."
))

cells.append(code(
    "rows = []\n"
    "for col in missing.index:\n"
    "    is_missing = train[col].isna()\n"
    "    table = pd.crosstab(is_missing, train[CFG.TARGET])\n"
    "    rows.append({\n"
    "        'kolom': col,\n"
    "        'n_kosong': int(is_missing.sum()),\n"
    "        'batal_saat_kosong': train.loc[is_missing, CFG.TARGET].mean(),\n"
    "        'batal_saat_terisi': train.loc[~is_missing, CFG.TARGET].mean(),\n"
    "        'p_value': stats.chi2_contingency(table)[1],\n"
    "    })\n"
    "\n"
    "missing_test = pd.DataFrame(rows).set_index('kolom').sort_values('p_value')\n"
    "missing_test['signifikan_bonferroni'] = missing_test['p_value'] < 0.05 / len(missing_test)\n"
    "missing_test.round(4)"
))

cells.append(md(
    "Kolom yang signifikan ditelusuri lebih jauh per kanal. Bila kekosongan acak, tingkat batal "
    "baris kosong setara campuran baris bernilai nol dan positif sesuai proporsinya. Proporsi "
    "nol yang tersirat pada baris kosong dapat dihitung dari ketiga tingkat batal tersebut."
))

cells.append(code(
    "sr_missing = train['special_requests'].isna()\n"
    "by_channel = train.assign(sr_missing=sr_missing).groupby('booking_channel').agg(\n"
    "    n=('sr_missing', 'size'),\n"
    "    pct_kosong=('sr_missing', 'mean'),\n"
    "    pct_nol=('special_requests', lambda s: s.eq(0).sum() / s.notna().sum()),\n"
    "    batal=(CFG.TARGET, 'mean'),\n"
    ").round(3)\n"
    "display(by_channel)\n"
    "\n"
    "compare = pd.DataFrame({\n"
    "    'batal_kosong': train[sr_missing].groupby('booking_channel')[CFG.TARGET].mean(),\n"
    "    'batal_nol': train[train['special_requests'].eq(0)].groupby('booking_channel')[CFG.TARGET].mean(),\n"
    "    'batal_positif': train[train['special_requests'].gt(0)].groupby('booking_channel')[CFG.TARGET].mean(),\n"
    "})\n"
    "compare['pct_nol_terisi'] = by_channel['pct_nol']\n"
    "compare['pct_nol_tersirat_pada_kosong'] = (\n"
    "    (compare['batal_kosong'] - compare['batal_positif'])\n"
    "    / (compare['batal_nol'] - compare['batal_positif'])\n"
    ")\n"
    "display(compare.round(3))\n"
    "\n"
    "lead_bins = pd.qcut(train['days_before_arrival'], 5, duplicates='drop')\n"
    "print('Persentase special_requests kosong per kuintil lead time:')\n"
    "print(sr_missing.groupby(lead_bins, observed=True).mean().round(3).to_string())"
))

cells.append(insight("missing"))

cells.append(md(
    "## Duplikat\n\n"
    "Kolom `id` selalu unik, sehingga duplikat diperiksa pada seluruh fitur. Duplikat "
    "yang tersebar ke fold berbeda dapat membuat skor validasi terlalu optimistis."
))

cells.append(code(
    "dup_mask   = train.duplicated(subset=CFG.FEATURES, keep=False)\n"
    "dup_groups = train[dup_mask].groupby(CFG.FEATURES, dropna=False)[CFG.TARGET]\n"
    "conflicts  = int((dup_groups.nunique() > 1).sum())\n"
    "\n"
    "train_keys = train[CFG.FEATURES].drop_duplicates()\n"
    "overlap    = len(test.merge(train_keys, on=CFG.FEATURES, how='inner'))\n"
    "\n"
    "print(f'Baris dalam grup duplikat : {dup_mask.sum()}')\n"
    "print(f'Jumlah grup duplikat      : {dup_groups.ngroups}')\n"
    "print(f'Baris berlebih            : {train.duplicated(subset=CFG.FEATURES).sum()}')\n"
    "print(f'Grup dengan label berbeda : {conflicts}')\n"
    "print(f'Duplikat di dalam test    : {test.duplicated(subset=CFG.FEATURES).sum()}')\n"
    "print(f'Baris test sama dgn train : {overlap}')"
))

cells.append(insight("duplicates"))

cells.append(md(
    "## Nilai Tidak Wajar\n\n"
    "Beberapa kombinasi nilai tampak tidak masuk akal secara bisnis. Setiap aturan dihitung "
    "pada train dan test, bersama tingkat batalnya, untuk memutuskan apakah nilai tersebut "
    "derau atau justru informatif."
))

cells.append(code(
    "def anomaly_rules(df: pd.DataFrame) -> dict:\n"
    "    nights = df['weekend_nights'] + df['weekday_nights']\n"
    "    return {\n"
    "        'num_adults = 0': df['num_adults'].eq(0),\n"
    "        'total malam = 0': nights.eq(0),\n"
    "        'nightly_rate_idr = 0': df['nightly_rate_idr'].eq(0),\n"
    "        'num_children >= 9': df['num_children'].ge(9),\n"
    "        'prior_cancellations >= 10': df['prior_cancellations'].ge(10),\n"
    "        'total malam >= 15': nights.ge(15),\n"
    "        'tarif > 6 juta': df['nightly_rate_idr'].gt(6_000_000),\n"
    "    }\n"
    "\n"
    "train_rules, test_rules = anomaly_rules(train), anomaly_rules(test)\n"
    "anomalies = pd.DataFrame({\n"
    "    'n_train': {k: int(m.sum()) for k, m in train_rules.items()},\n"
    "    'batal_train': {k: train.loc[m, CFG.TARGET].mean() for k, m in train_rules.items()},\n"
    "    'n_test': {k: int(m.sum()) for k, m in test_rules.items()},\n"
    "}).round(3)\n"
    "display(anomalies)\n"
    "\n"
    "zero_adults = train[train['num_adults'].eq(0)]\n"
    "print('num_children saat num_adults = 0:')\n"
    "print(zero_adults['num_children'].value_counts(dropna=False).sort_index().to_string())\n"
    "\n"
    "zero_rate = train[train['nightly_rate_idr'].eq(0)]\n"
    "print('\\nTarif 0 per kanal:')\n"
    "print(zero_rate.groupby('booking_channel')[CFG.TARGET].agg(['size', 'mean']).round(3).to_string())\n"
    "\n"
    "comp = train[train['booking_channel'].eq('Complimentary')]\n"
    "print('\\nDistribusi tarif kanal Complimentary:')\n"
    "print(comp['nightly_rate_idr'].describe().round(0).to_string())\n"
    "\n"
    "print('\\nMedian tarif per kanal:')\n"
    "print(train.groupby('booking_channel')['nightly_rate_idr'].median().to_string())\n"
    "\n"
    "print('\\nKonsistensi tamu lama:')\n"
    "prior = train['prior_cancellations'] + train['prior_completed_bookings']\n"
    "print(pd.crosstab(train['is_returning_guest'], prior.gt(0)).to_string())"
))

cells.append(insight("anomalies"))

# ---------------------------------------------------------------------------
# Section 4: Target
# ---------------------------------------------------------------------------
cells.append(section("Distribusi Target", "4"))

cells.append(md(
    "Distribusi kelas menentukan baseline dan perlu tidaknya pembobotan kelas pada SVM."
))

cells.append(code(
    "counts = train[CFG.TARGET].value_counts().sort_index()\n"
    "rates  = counts / counts.sum()\n"
    "\n"
    "fig, ax = plt.subplots(figsize=(5, 2.2))\n"
    "labels = ['Tidak batal (0)', 'Batal (1)']\n"
    "bars = ax.barh(labels, counts.values, color=[CFG.NEG_COLOR, CFG.POS_COLOR], height=0.6)\n"
    "for bar, n, r in zip(bars, counts.values, rates.values):\n"
    "    ax.text(bar.get_width() + 250, bar.get_y() + bar.get_height() / 2,\n"
    "            f'{n:,} ({r:.1%})', va='center', color=CFG.INK)\n"
    "ax.set_xlim(0, counts.max() * 1.3)\n"
    "ax.set_title('Distribusi is_canceled pada train')\n"
    "ax.set_xlabel('Jumlah pesanan')\n"
    "ax.grid(axis='y', visible=False)\n"
    "save_fig('target_distribution')\n"
    "\n"
    "print(f'Akurasi baseline kelas mayoritas: {rates.max():.4f}')\n"
    "print(f'Rasio kelas negatif terhadap positif: {counts[0] / counts[1]:.2f}')\n"
    "print(f'Macro F1 jika semua diprediksi tidak batal: {f1_score(train[CFG.TARGET], np.zeros(len(train)), average=\"macro\"):.4f}')\n"
    "print(f'Macro F1 jika semua diprediksi batal: {f1_score(train[CFG.TARGET], np.ones(len(train)), average=\"macro\"):.4f}')"
))

cells.append(insight("target"))

# ---------------------------------------------------------------------------
# Section 5: Features vs target
# ---------------------------------------------------------------------------
cells.append(section("Fitur terhadap Target", "5"))

cells.append(md(
    "Bagian ini melihat bentuk distribusi fitur dan hubungannya dengan target. Bentuk "
    "hubungan penting bagi SVM karena hubungan nonlinear tidak tertangkap kernel linear "
    "tanpa transformasi fitur."
))

cells.append(md(
    "## Fitur Kontinu\n\n"
    "Dua fitur kontinu dibandingkan per kelas. Tingkat batal per desil memperlihatkan "
    "bentuk hubungan secara langsung tanpa asumsi linear."
))

cells.append(code(
    "fig, axes = plt.subplots(2, 2, figsize=(10, 6))\n"
    "for row, col in zip(axes, CFG.CONT_COLS):\n"
    "    for value, color, label in [(0, CFG.NEG_COLOR, 'Tidak batal'), (1, CFG.POS_COLOR, 'Batal')]:\n"
    "        row[0].hist(train.loc[train[CFG.TARGET].eq(value), col].dropna(), bins=50,\n"
    "                    density=True, histtype='step', linewidth=2, color=color, label=label)\n"
    "    row[0].set_title(f'Distribusi {col}')\n"
    "    row[0].set_ylabel('Densitas')\n"
    "    row[0].legend()\n"
    "\n"
    "    deciles = pd.qcut(train[col], 10, duplicates='drop')\n"
    "    curve = train.groupby(deciles, observed=True).agg(\n"
    "        x=(col, 'median'), rate=(CFG.TARGET, 'mean')\n"
    "    )\n"
    "    row[1].plot(curve['x'], curve['rate'], color=CFG.POS_COLOR, linewidth=2, marker='o', markersize=5)\n"
    "    row[1].axhline(train[CFG.TARGET].mean(), color=CFG.MUTED, linewidth=1, linestyle='--')\n"
    "    row[1].set_ylim(0, 1)\n"
    "    row[1].set_title(f'Tingkat batal per desil {col}')\n"
    "    row[1].set_ylabel('Tingkat batal')\n"
    "    row[1].set_xlabel(f'Median {col} pada desil')\n"
    "save_fig('continuous_features')\n"
    "\n"
    "lead_deciles = pd.qcut(train['days_before_arrival'], 10, duplicates='drop')\n"
    "print(train.groupby(lead_deciles, observed=True)[CFG.TARGET].agg(['size', 'mean']).round(3).to_string())"
))

cells.append(insight("continuous"))

cells.append(md(
    "## Fitur Hitungan dan Biner\n\n"
    "Fitur hitungan memiliki sedikit nilai unik, sehingga tingkat batal dapat dibaca per "
    "nilai. Nilai dengan kurang dari 30 baris ditandai pucat karena estimasinya tidak stabil. "
    "Untuk keterbacaan, nilai teratas digabung, yaitu `weekday_nights` >= 10 serta "
    "`prior_cancellations` dan `prior_completed_bookings` >= 5."
))

cells.append(code(
    "def plot_rate_grid(source, cols, ncols, name, figsize):\n"
    "    nrows = int(np.ceil(len(cols) / ncols))\n"
    "    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)\n"
    "    axes = np.atleast_1d(axes).ravel()\n"
    "    overall = source[CFG.TARGET].mean()\n"
    "    for ax, col in zip(axes, cols):\n"
    "        stat = source.groupby(col, dropna=False)[CFG.TARGET].agg(['size', 'mean'])\n"
    "        labels = [\n"
    "            'kosong' if pd.isna(v) else (str(int(v)) if isinstance(v, (int, float)) else str(v))\n"
    "            for v in stat.index\n"
    "        ]\n"
    "        alphas = np.where(stat['size'] < 30, 0.3, 1.0)\n"
    "        colors = [(*plt.matplotlib.colors.to_rgb(CFG.POS_COLOR), a) for a in alphas]\n"
    "        ax.barh(labels, stat['mean'], color=colors, height=0.7)\n"
    "        ax.axvline(overall, color=CFG.MUTED, linewidth=1, linestyle='--')\n"
    "        ax.set_xlim(0, 1)\n"
    "        ax.set_title(col)\n"
    "        ax.invert_yaxis()\n"
    "        ax.grid(axis='y', visible=False)\n"
    "        for y, n in enumerate(stat['size']):\n"
    "            ax.text(0.99, y, f'n={n:,}', va='center', ha='right', fontsize=7, color=CFG.MUTED)\n"
    "    for ax in axes[len(cols):]:\n"
    "        ax.set_visible(False)\n"
    "    fig.supxlabel('Tingkat batal (garis putus-putus: rata-rata keseluruhan)', fontsize=9, color=CFG.MUTED)\n"
    "    save_fig(name)\n"
    "\n"
    "train_view = train.assign(\n"
    "    weekday_nights=train['weekday_nights'].clip(upper=10),\n"
    "    prior_cancellations=train['prior_cancellations'].clip(upper=5),\n"
    "    prior_completed_bookings=train['prior_completed_bookings'].clip(upper=5),\n"
    ")\n"
    "plot_rate_grid(train_view, CFG.COUNT_COLS + CFG.BIN_COLS, 3, 'count_binary_rates', (11, 9))"
))

cells.append(insight("counts"))

cells.append(md(
    "## Fitur Kategorikal\n\n"
    "Tiga fitur kategorikal dibandingkan dengan cara yang sama. Kategori langka perlu "
    "perhatian karena one-hot encoding menghasilkan kolom yang hampir selalu nol."
))

cells.append(code(
    "plot_rate_grid(train, CFG.CAT_COLS, 3, 'categorical_rates', (11, 3.4))\n"
    "\n"
    "for col in CFG.CAT_COLS:\n"
    "    table = pd.crosstab(train[col].fillna('kosong'), train[CFG.TARGET])\n"
    "    chi2, p, dof, _ = stats.chi2_contingency(table)\n"
    "    print(f'{col:18} chi2={chi2:9.1f}  dof={dof}  p={p:.2e}')\n"
    "\n"
    "print('\\nKategori langka:')\n"
    "for col, value in [('meal_package', 'Full Board'), ('room_category', 'Type E'), ('booking_channel', 'Airline')]:\n"
    "    print(f'{value:12} train={train[col].eq(value).sum():3}  test={test[col].eq(value).sum():3}')"
))

cells.append(insight("categorical"))

# ---------------------------------------------------------------------------
# Section 6: Time
# ---------------------------------------------------------------------------
cells.append(section("Pola Waktu", "6"))

cells.append(md(
    "Tanggal kedatangan tersedia sebagai tiga kolom terpisah. Bagian ini menyusunnya kembali "
    "untuk melihat apakah target berubah terhadap waktu, dan apakah pembagian train dan test "
    "bersifat acak atau temporal."
))

cells.append(md(
    "## Tren Bulanan\n\n"
    "Volume dan tingkat batal ditampilkan pada dua panel terpisah menurut bulan kedatangan. "
    "Jumlah baris test ikut ditampilkan untuk memeriksa cara pembagian data."
))

cells.append(code(
    "def add_period(df: pd.DataFrame) -> pd.DataFrame:\n"
    "    out = df.copy()\n"
    "    out['arrival_ym'] = (\n"
    "        out['arrival_year'].astype(str) + '-' + out['arrival_month'].astype(str).str.zfill(2)\n"
    "    )\n"
    "    out['arrival_date'] = pd.to_datetime(\n"
    "        {'year': out['arrival_year'], 'month': out['arrival_month'], 'day': out['arrival_day']},\n"
    "        errors='coerce',\n"
    "    )\n"
    "    return out\n"
    "\n"
    "train_t, test_t = add_period(train), add_period(test)\n"
    "print(f'Tanggal tidak valid: train={train_t.arrival_date.isna().sum()}, test={test_t.arrival_date.isna().sum()}')\n"
    "print(f'Rentang train: {train_t.arrival_date.min().date()} s.d. {train_t.arrival_date.max().date()}')\n"
    "print(f'Rentang test : {test_t.arrival_date.min().date()} s.d. {test_t.arrival_date.max().date()}')\n"
    "\n"
    "monthly = train_t.groupby('arrival_ym')[CFG.TARGET].agg(['size', 'mean'])\n"
    "monthly['test_size'] = test_t.groupby('arrival_ym').size()\n"
    "monthly['test_share'] = monthly['test_size'] / (monthly['test_size'] + monthly['size'])\n"
    "\n"
    "fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True)\n"
    "ax1.plot(monthly.index, monthly['mean'], color=CFG.POS_COLOR, linewidth=2, marker='o', markersize=5)\n"
    "ax1.axhline(train[CFG.TARGET].mean(), color=CFG.MUTED, linewidth=1, linestyle='--')\n"
    "ax1.set_ylim(0, 1)\n"
    "ax1.set_ylabel('Tingkat batal')\n"
    "ax1.set_title('Tingkat batal per bulan kedatangan')\n"
    "\n"
    "ax2.bar(monthly.index, monthly['size'], color=CFG.NEG_COLOR, width=0.7, label='Train')\n"
    "ax2.bar(monthly.index, monthly['test_size'], bottom=monthly['size'], color=CFG.POS_COLOR,\n"
    "        width=0.7, label='Test')\n"
    "ax2.set_ylabel('Jumlah pesanan')\n"
    "ax2.set_title('Volume pesanan per bulan kedatangan')\n"
    "ax2.legend(ncols=2, loc='upper left')\n"
    "ax2.tick_params(axis='x', rotation=45)\n"
    "ax2.grid(axis='x', visible=False)\n"
    "save_fig('monthly_trend')\n"
    "\n"
    "print(monthly.round(3).to_string())"
))

cells.append(insight("monthly"))

cells.append(md(
    "## Bulan Kalender dan Hari\n\n"
    "Periode yang sama pada dua tahun dibandingkan untuk membedakan musim berulang dari "
    "perubahan antartahun. Hari dalam pekan dan tanggal diperiksa sebagai kandidat fitur."
))

cells.append(code(
    "fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))\n"
    "\n"
    "for year, color in [(2023, CFG.NEG_COLOR), (2024, CFG.POS_COLOR)]:\n"
    "    rate = train[train['arrival_year'].eq(year)].groupby('arrival_month')[CFG.TARGET].mean()\n"
    "    axes[0].plot(rate.index, rate.values, color=color, linewidth=2, marker='o', markersize=5, label=str(year))\n"
    "axes[0].set_xticks(range(1, 13))\n"
    "axes[0].set_ylim(0, 1)\n"
    "axes[0].set_title('Tingkat batal per bulan, per tahun')\n"
    "axes[0].set_xlabel('Bulan')\n"
    "axes[0].legend()\n"
    "\n"
    "dow = train_t.groupby(train_t['arrival_date'].dt.day_of_week)[CFG.TARGET].mean()\n"
    "axes[1].bar(['Sen', 'Sel', 'Rab', 'Kam', 'Jum', 'Sab', 'Min'], dow.values, color=CFG.POS_COLOR, width=0.7)\n"
    "axes[1].set_ylim(0, 1)\n"
    "axes[1].set_title('Tingkat batal per hari kedatangan')\n"
    "axes[1].grid(axis='x', visible=False)\n"
    "\n"
    "day = train.groupby('arrival_day')[CFG.TARGET].mean()\n"
    "axes[2].plot(day.index, day.values, color=CFG.POS_COLOR, linewidth=2)\n"
    "axes[2].set_ylim(0, 1)\n"
    "axes[2].set_title('Tingkat batal per tanggal')\n"
    "axes[2].set_xlabel('Tanggal')\n"
    "save_fig('calendar_patterns')\n"
    "\n"
    "print('Rentang tingkat batal per hari  :', round(dow.min(), 3), '-', round(dow.max(), 3))\n"
    "print('Rentang tingkat batal per tanggal:', round(day.min(), 3), '-', round(day.max(), 3))"
))

cells.append(insight("calendar"))

cells.append(md(
    "## Interaksi Lead Time dan Periode\n\n"
    "Bila efek lead time berbeda antarperiode, kedua fitur berinteraksi dan batas keputusan "
    "yang lurus tidak cukup. Sel dengan kurang dari 30 baris dikosongkan."
))

cells.append(code(
    "lead_group = pd.cut(\n"
    "    train_t['days_before_arrival'], [-1, 7, 30, 90, 150, 250, 500],\n"
    "    labels=['0-7', '8-30', '31-90', '91-150', '151-250', '251+'],\n"
    ")\n"
    "pivot_rate = train_t.pivot_table(\n"
    "    index=lead_group, columns='arrival_ym', values=CFG.TARGET, aggfunc='mean', observed=True\n"
    ")\n"
    "pivot_size = train_t.pivot_table(\n"
    "    index=lead_group, columns='arrival_ym', values=CFG.TARGET, aggfunc='size', observed=True\n"
    ")\n"
    "\n"
    "fig, ax = plt.subplots(figsize=(11, 3.6))\n"
    "sns.heatmap(\n"
    "    pivot_rate.where(pivot_size >= 30), cmap=SEQUENTIAL, vmin=0, vmax=1, annot=True, fmt='.2f',\n"
    "    annot_kws={'size': 7}, linewidths=1, linecolor=CFG.SURFACE,\n"
    "    cbar_kws={'label': 'Tingkat batal'}, ax=ax,\n"
    ")\n"
    "ax.set_title('Tingkat batal menurut lead time dan bulan kedatangan')\n"
    "ax.set_xlabel('Bulan kedatangan')\n"
    "ax.set_ylabel('Lead time (hari)')\n"
    "ax.tick_params(axis='y', rotation=0)\n"
    "ax.grid(False)\n"
    "save_fig('lead_period_interaction')"
))

cells.append(insight("interaction"))

cells.append(md(
    "## Bulan Pemesanan\n\n"
    "Tanggal pemesanan tidak tersedia langsung, tetapi dapat diturunkan dari tanggal kedatangan "
    "dikurangi `days_before_arrival`. Fitur ini diketahui saat pesanan dibuat, sehingga tidak "
    "membocorkan target."
))

cells.append(code(
    "def add_booking_month(df: pd.DataFrame) -> pd.DataFrame:\n"
    "    out = df.copy()\n"
    "    booking_date = out['arrival_date'] - pd.to_timedelta(out['days_before_arrival'], unit='D')\n"
    "    out['booking_ym'] = booking_date.dt.strftime('%Y-%m')\n"
    "    return out\n"
    "\n"
    "train_t, test_t = add_booking_month(train_t), add_booking_month(test_t)\n"
    "\n"
    "booked = train_t.groupby('booking_ym')[CFG.TARGET].agg(['size', 'mean'])\n"
    "booked = booked[booked['size'] >= 100]\n"
    "\n"
    "fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5), sharex=True)\n"
    "ax1.plot(booked.index, booked['mean'], color=CFG.POS_COLOR, linewidth=2, marker='o', markersize=5)\n"
    "ax1.axhline(train[CFG.TARGET].mean(), color=CFG.MUTED, linewidth=1, linestyle='--')\n"
    "ax1.set_ylim(0, 1)\n"
    "ax1.set_ylabel('Tingkat batal')\n"
    "ax1.set_title('Tingkat batal per bulan pemesanan (minimal 100 baris)')\n"
    "\n"
    "ax2.bar(booked.index, booked['size'], color=CFG.NEG_COLOR, width=0.7)\n"
    "ax2.set_ylabel('Jumlah pesanan')\n"
    "ax2.set_title('Volume per bulan pemesanan')\n"
    "ax2.tick_params(axis='x', rotation=45)\n"
    "ax2.grid(axis='x', visible=False)\n"
    "save_fig('booking_month')\n"
    "\n"
    "print(f'Jumlah bulan pemesanan unik: {train_t.booking_ym.nunique()}')\n"
    "print(booked.round(3).to_string())"
))

cells.append(insight("booking"))

# ---------------------------------------------------------------------------
# Section 7: Relevance
# ---------------------------------------------------------------------------
cells.append(section("Relevansi dan Redundansi Fitur", "7"))

cells.append(md(
    "Dua pertanyaan dijawab di sini, yaitu fitur mana yang membawa sinyal dan fitur mana "
    "yang saling tumpang tindih. Korelasi hanya menangkap hubungan monotonik, sehingga "
    "mutual information ikut digunakan untuk hubungan nonlinear."
))

cells.append(md(
    "## Korelasi Spearman\n\n"
    "Spearman dipilih karena banyak fitur miring dan berupa hitungan. Target disertakan "
    "sebagai baris terakhir."
))

cells.append(code(
    "corr = train[CFG.NUM_COLS + [CFG.TARGET]].corr(method='spearman')\n"
    "\n"
    "fig, ax = plt.subplots(figsize=(9.5, 7.5))\n"
    "sns.heatmap(\n"
    "    corr, mask=np.triu(np.ones_like(corr, dtype=bool)), cmap=DIVERGING, vmin=-1, vmax=1,\n"
    "    annot=True, fmt='.2f', annot_kws={'size': 7}, linewidths=1, linecolor=CFG.SURFACE,\n"
    "    cbar_kws={'label': 'Korelasi Spearman', 'shrink': 0.7}, ax=ax,\n"
    ")\n"
    "ax.set_title('Korelasi Spearman antarfitur numerik dan target')\n"
    "ax.grid(False)\n"
    "save_fig('spearman_correlation')\n"
    "\n"
    "pairs = corr.drop(index=CFG.TARGET, columns=CFG.TARGET).where(\n"
    "    np.triu(np.ones((len(CFG.NUM_COLS), len(CFG.NUM_COLS)), dtype=bool), k=1)\n"
    ").stack()\n"
    "print('Pasangan fitur dengan |rho| terbesar:')\n"
    "print(pairs.reindex(pairs.abs().sort_values(ascending=False).index).head(6).round(3).to_string())\n"
    "\n"
    "target_corr = corr[CFG.TARGET].drop(CFG.TARGET)\n"
    "print('\\nKorelasi terhadap target:')\n"
    "print(target_corr.reindex(target_corr.abs().sort_values(ascending=False).index).round(3).to_string())"
))

cells.append(insight("correlation"))

cells.append(md(
    "## Mutual Information dan Cramer's V\n\n"
    "Mutual information dihitung pada train saja, setelah imputasi sementara dengan median "
    "atau label kosong. Imputasi ini hanya untuk pemeringkatan dan tidak dipakai pada model. "
    "Fitur turunan disertakan untuk menilai manfaatnya."
))

cells.append(code(
    "def cramers_v(a: pd.Series, b: pd.Series) -> float:\n"
    "    table = pd.crosstab(a, b)\n"
    "    chi2 = stats.chi2_contingency(table, correction=False)[0]\n"
    "    return float(np.sqrt(chi2 / (table.values.sum() * (min(table.shape) - 1))))\n"
    "\n"
    "mi_frame = train_t[CFG.FEATURES + ['arrival_ym', 'booking_ym']].copy()\n"
    "mi_frame['total_nights'] = mi_frame['weekend_nights'] + mi_frame['weekday_nights']\n"
    "mi_frame['special_requests_missing'] = mi_frame['special_requests'].isna().astype(int)\n"
    "\n"
    "cat_like = CFG.CAT_COLS + ['arrival_ym', 'booking_ym']\n"
    "num_like = [c for c in mi_frame.columns if c not in cat_like]\n"
    "mi_frame[num_like] = mi_frame[num_like].fillna(mi_frame[num_like].median())\n"
    "mi_frame[cat_like] = OrdinalEncoder().fit_transform(mi_frame[cat_like].fillna('kosong').astype(str))\n"
    "\n"
    "discrete = [c != 'days_before_arrival' and c != 'nightly_rate_idr' for c in mi_frame.columns]\n"
    "mi = pd.Series(\n"
    "    mutual_info_classif(mi_frame, train[CFG.TARGET], discrete_features=discrete, random_state=CFG.SEED),\n"
    "    index=mi_frame.columns,\n"
    ").sort_values()\n"
    "\n"
    "fig, ax = plt.subplots(figsize=(7, 5.2))\n"
    "ax.barh(mi.index, mi.values, color=CFG.NEG_COLOR, height=0.7)\n"
    "ax.set_title('Mutual information terhadap is_canceled')\n"
    "ax.set_xlabel('Mutual information (nat)')\n"
    "ax.grid(axis='y', visible=False)\n"
    "save_fig('mutual_information')\n"
    "\n"
    "print(mi.sort_values(ascending=False).round(4).to_string())\n"
    "print('\\nCramer V terhadap target:')\n"
    "for col in cat_like:\n"
    "    print(f'{col:18} {cramers_v(train_t[col].fillna(\"kosong\"), train[CFG.TARGET]):.3f}')"
))

cells.append(insight("relevance"))

# ---------------------------------------------------------------------------
# Section 8: Train vs test
# ---------------------------------------------------------------------------
cells.append(section("Perbandingan Train dan Test", "8"))

cells.append(md(
    "Skor validasi hanya mewakili skor test bila kedua berkas berasal dari distribusi yang "
    "sama. Population Stability Index (PSI) dihitung dengan bin dari train, lalu test "
    "dievaluasi pada bin tersebut, sehingga kedua berkas tetap terpisah. Nilai di bawah 0,1 "
    "umumnya dianggap stabil."
))

cells.append(code(
    "def psi(expected: pd.Series, actual: pd.Series, bins: int = 10, eps: float = 1e-4) -> float:\n"
    "    if expected.nunique() > 20:\n"
    "        edges = np.unique(expected.dropna().quantile(np.linspace(0, 1, bins + 1)))\n"
    "        edges[0], edges[-1] = -np.inf, np.inf\n"
    "        expected, actual = pd.cut(expected, edges), pd.cut(actual, edges)\n"
    "    e = expected.astype(object).where(expected.notna(), 'kosong').value_counts(normalize=True)\n"
    "    a = actual.astype(object).where(actual.notna(), 'kosong').value_counts(normalize=True)\n"
    "    e, a = e.align(a, fill_value=0)\n"
    "    e, a = e.clip(lower=eps), a.clip(lower=eps)\n"
    "    return float(((a - e) * np.log(a / e)).sum())\n"
    "\n"
    "drift = pd.DataFrame({\n"
    "    'psi': {c: psi(train[c], test[c]) for c in CFG.FEATURES},\n"
    "    'ks': {c: stats.ks_2samp(train[c].dropna(), test[c].dropna()).statistic for c in CFG.NUM_COLS},\n"
    "}).sort_values('psi', ascending=False)\n"
    "display(drift.round(4))\n"
    "\n"
    "for col in CFG.CAT_COLS:\n"
    "    unseen = set(test[col].dropna()) - set(train[col].dropna())\n"
    "    print(f'{col:18} kategori baru pada test: {unseen or \"tidak ada\"}')\n"
    "\n"
    "share = monthly['test_share']\n"
    "print(f'Porsi test per bulan: min={share.min():.3f}, max={share.max():.3f}')"
))

cells.append(insight("drift"))

# ---------------------------------------------------------------------------
# Section 9: SVM implications
# ---------------------------------------------------------------------------
cells.append(section("Implikasi untuk SVM", "9"))

cells.append(md(
    "Bagian ini menguji langsung pertanyaan perancangan SVM memakai `sklearn.svm.SVC` sebagai "
    "alat ukur cepat. Hasilnya bersifat arahan, bukan hasil akhir, karena hyperparameter belum "
    "dituning. Seluruh preprocessing berada di dalam pipeline dan duplikat dibuang lebih dulu."
))

cells.append(md(
    "## Skala Fitur\n\n"
    "Kernel Gaussian dan polynomial bergantung pada jarak atau hasil kali titik, sehingga "
    "fitur berskala besar dapat mendominasi. Kemiringan juga diperiksa untuk menilai "
    "perlunya transformasi."
))

cells.append(code(
    "scale = pd.DataFrame({\n"
    "    'std': train[CFG.NUM_COLS].std(),\n"
    "    'range': train[CFG.NUM_COLS].max() - train[CFG.NUM_COLS].min(),\n"
    "    'skew': train[CFG.NUM_COLS].skew(),\n"
    "}).sort_values('range', ascending=False)\n"
    "scale['range_vs_median_range'] = scale['range'] / scale['range'].median()\n"
    "scale.round(2)"
))

cells.append(insight("scale"))

cells.append(md(
    "## Pipeline Uji\n\n"
    "Dua set fitur dibandingkan. Set `raw` memakai kolom apa adanya. Set `engineered` "
    "menerapkan temuan EDA, yaitu bulan kedatangan dan bulan pemesanan sebagai kategori, total "
    "malam, log lead time, penanda tarif nol, dan penanda `special_requests` kosong."
))

cells.append(code(
    "def add_features(df: pd.DataFrame) -> pd.DataFrame:\n"
    "    out = df.copy()\n"
    "    out['total_nights'] = out['weekend_nights'] + out['weekday_nights']\n"
    "    out['lead_log'] = np.log1p(out['days_before_arrival'])\n"
    "    out['rate_is_zero'] = out['nightly_rate_idr'].eq(0).astype(float)\n"
    "    out['special_requests_missing'] = out['special_requests'].isna().astype(float)\n"
    "    arrival = pd.to_datetime(\n"
    "        {'year': out['arrival_year'], 'month': out['arrival_month'], 'day': out['arrival_day']}\n"
    "    )\n"
    "    booking = arrival - pd.to_timedelta(out['days_before_arrival'], unit='D')\n"
    "    out['arrival_ym'] = arrival.dt.strftime('%Y-%m')\n"
    "    out['booking_ym'] = booking.dt.strftime('%Y-%m')\n"
    "    return out\n"
    "\n"
    "FEATURE_SETS = {\n"
    "    'raw': {'num': CFG.NUM_COLS, 'cat': CFG.CAT_COLS},\n"
    "    'engineered': {\n"
    "        'num': CFG.COUNT_COLS + CFG.CONT_COLS + CFG.BIN_COLS + [\n"
    "            'total_nights', 'lead_log', 'rate_is_zero', 'special_requests_missing',\n"
    "        ],\n"
    "        'cat': CFG.CAT_COLS + ['arrival_ym', 'booking_ym'],\n"
    "    },\n"
    "}\n"
    "\n"
    "def make_preprocessor(feature_set: str, scale: bool = True) -> Pipeline:\n"
    "    spec = FEATURE_SETS[feature_set]\n"
    "    num_steps = [('impute', SimpleImputer(strategy='median'))]\n"
    "    if scale:\n"
    "        num_steps.append(('scale', StandardScaler()))\n"
    "    cat_steps = [\n"
    "        ('impute', SimpleImputer(strategy='constant', fill_value='kosong')),\n"
    "        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),\n"
    "    ]\n"
    "    columns = ColumnTransformer([\n"
    "        ('num', Pipeline(num_steps), spec['num']),\n"
    "        ('cat', Pipeline(cat_steps), spec['cat']),\n"
    "    ])\n"
    "    return Pipeline([('features', FunctionTransformer(add_features)), ('columns', columns)])\n"
    "\n"
    "def make_model(feature_set: str, estimator, scale: bool = True) -> Pipeline:\n"
    "    return Pipeline([('prep', make_preprocessor(feature_set, scale)), ('clf', estimator)])\n"
    "\n"
    "clean = train.drop_duplicates(subset=CFG.FEATURES + [CFG.TARGET]).reset_index(drop=True)\n"
    "X_all, y_all = clean[CFG.FEATURES], clean[CFG.TARGET]\n"
    "\n"
    "for name in FEATURE_SETS:\n"
    "    width = make_preprocessor(name).fit_transform(X_all).shape[1]\n"
    "    print(f'Dimensi set {name}: {width}')\n"
    "print(f'Baris setelah buang duplikat: {len(clean)}')"
))

cells.append(md(
    "## Reduksi Dimensi\n\n"
    "PCA dipasang pada set `engineered` untuk melihat jumlah komponen yang menyimpan variansi "
    "serta keterpisahan kelas pada dua komponen pertama. Proyeksi ini juga menjadi kandidat "
    "visualisasi bidang pemisah pada bonus SVM."
))

cells.append(code(
    "Z = make_preprocessor('engineered').fit_transform(X_all)\n"
    "pca = PCA(random_state=CFG.SEED).fit(Z)\n"
    "cumulative = np.cumsum(pca.explained_variance_ratio_)\n"
    "coords = pca.transform(Z)[:, :2]\n"
    "\n"
    "rng = np.random.default_rng(CFG.SEED)\n"
    "idx = rng.choice(len(Z), 4000, replace=False)\n"
    "\n"
    "fig, axes = plt.subplots(1, 2, figsize=(11, 4))\n"
    "axes[0].plot(np.arange(1, len(cumulative) + 1), cumulative, color=CFG.NEG_COLOR, linewidth=2)\n"
    "axes[0].axhline(0.9, color=CFG.MUTED, linewidth=1, linestyle='--')\n"
    "axes[0].set_ylim(0, 1.02)\n"
    "axes[0].set_title('Variansi kumulatif PCA')\n"
    "axes[0].set_xlabel('Jumlah komponen')\n"
    "axes[0].set_ylabel('Proporsi variansi')\n"
    "\n"
    "for value, color, label in [(0, CFG.NEG_COLOR, 'Tidak batal'), (1, CFG.POS_COLOR, 'Batal')]:\n"
    "    mask = y_all.values[idx] == value\n"
    "    axes[1].scatter(*coords[idx][mask].T, s=8, alpha=0.45, color=color, label=label, linewidths=0)\n"
    "axes[1].set_title('Proyeksi dua komponen utama (sampel 4.000 baris)')\n"
    "axes[1].set_xlabel('PC1')\n"
    "axes[1].set_ylabel('PC2')\n"
    "axes[1].legend(markerscale=2)\n"
    "save_fig('pca_projection')\n"
    "\n"
    "for target in (0.8, 0.9, 0.95, 0.99):\n"
    "    print(f'Komponen untuk {target:.0%} variansi: {int(np.searchsorted(cumulative, target) + 1)} dari {Z.shape[1]}')\n"
    "print(f'Variansi dua komponen pertama: {cumulative[1]:.3f}')\n"
    "\n"
    "auc_2d = cross_validate(\n"
    "    LogisticRegression(max_iter=1000), coords, y_all, scoring='roc_auc',\n"
    "    cv=StratifiedKFold(CFG.PROBE_FOLDS, shuffle=True, random_state=CFG.SEED),\n"
    ")['test_score'].mean()\n"
    "print(f'ROC-AUC model linear pada 2 komponen (pembanding kasar): {auc_2d:.3f}')"
))

cells.append(insight("pca"))

cells.append(md(
    "## Efek Kelompok dan Skema Validasi\n\n"
    "Pesanan dengan tanggal kedatangan dan lead time identik kemungkinan berasal dari rombongan "
    "yang sama. Bila rombongan batal bersama, fold acak dapat memecah satu rombongan ke latih dan "
    "validasi. Keseragaman label dibandingkan terhadap dua pembanding, yaitu tingkat batal global "
    "dan probabilitas per baris dari model tanpa informasi kelompok. Skor fold acak lalu "
    "dibandingkan dengan fold yang menjaga satu kelompok tetap utuh."
))

cells.append(code(
    "def booking_key(df: pd.DataFrame) -> pd.Series:\n"
    "    arrival = pd.to_datetime(\n"
    "        {'year': df['arrival_year'], 'month': df['arrival_month'], 'day': df['arrival_day']}\n"
    "    )\n"
    "    lead = df['days_before_arrival']\n"
    "    return (arrival.dt.strftime('%Y%m%d') + '|' + lead.astype('Int64').astype(str)).where(lead.notna())\n"
    "\n"
    "def evaluate_splits(splits):\n"
    "    rows, oof = [], np.zeros(len(X_all))\n"
    "    for fit_idx, val_idx in splits:\n"
    "        X_fit, y_fit = X_all.iloc[fit_idx], y_all.iloc[fit_idx]\n"
    "        X_val, y_val = X_all.iloc[val_idx], y_all.iloc[val_idx]\n"
    "        booster = make_model('engineered', HistGradientBoostingClassifier(random_state=CFG.SEED))\n"
    "        svm = make_model('engineered', SVC(kernel='rbf'))\n"
    "        oof[val_idx] = booster.fit(X_fit, y_fit).predict_proba(X_val)[:, 1]\n"
    "        scores = svm.fit(X_fit, y_fit).decision_function(X_val)\n"
    "        rows.append({\n"
    "            'boosting_auc': roc_auc_score(y_val, oof[val_idx]),\n"
    "            'svm_auc': roc_auc_score(y_val, scores),\n"
    "            'svm_macro_f1': f1_score(y_val, scores > 0, average='macro'),\n"
    "        })\n"
    "    return pd.DataFrame(rows).mean(), oof\n"
    "\n"
    "keys = booking_key(X_all)\n"
    "group_ids = keys.fillna(pd.Series(X_all.index.astype(str), index=X_all.index))\n"
    "\n"
    "random_splits = StratifiedKFold(CFG.PROBE_FOLDS, shuffle=True, random_state=CFG.SEED).split(X_all, y_all)\n"
    "group_splits = GroupKFold(CFG.PROBE_FOLDS).split(X_all, y_all, group_ids)\n"
    "random_score, _ = evaluate_splits(random_splits)\n"
    "group_score, group_oof = evaluate_splits(group_splits)\n"
    "display(pd.DataFrame({'fold_acak': random_score, 'fold_kelompok': group_score}).round(4))\n"
    "\n"
    "frame = pd.DataFrame({'key': keys, 'y': y_all, 'p': np.clip(group_oof, 1e-6, 1 - 1e-6)}).dropna(subset=['key'])\n"
    "frame['log_p'], frame['log_q'] = np.log(frame['p']), np.log1p(-frame['p'])\n"
    "groups = frame.groupby('key').agg(\n"
    "    size=('y', 'size'), rate=('y', 'mean'), log_p=('log_p', 'sum'), log_q=('log_q', 'sum')\n"
    ")\n"
    "base = y_all.mean()\n"
    "groups['pure'] = groups['rate'].isin([0, 1])\n"
    "groups['expected_global'] = base ** groups['size'] + (1 - base) ** groups['size']\n"
    "groups['expected_model'] = np.exp(groups['log_p']) + np.exp(groups['log_q'])\n"
    "groups['bin'] = pd.cut(groups['size'], [1, 4, 9, 1000], labels=['2-4', '5-9', '10+'])\n"
    "\n"
    "purity = groups.dropna(subset=['bin']).groupby('bin', observed=True).agg(\n"
    "    n_grup=('size', 'size'), n_baris=('size', 'sum'), seragam=('pure', 'mean'),\n"
    "    harapan_global=('expected_global', 'mean'), harapan_model=('expected_model', 'mean'),\n"
    ")\n"
    "display(purity.round(3))\n"
    "\n"
    "seen_keys = set(keys.dropna())\n"
    "print(f'Baris test dengan kunci kelompok yang ada di train: {booking_key(test).isin(seen_keys).mean():.3f}')"
))

cells.append(insight("groups"))

cells.append(md(
    "## Uji Kernel dan Scaling\n\n"
    "Kernel, set fitur, scaling, dan bobot kelas dibandingkan dengan hyperparameter bawaan "
    "pada sampel berstrata. Logistic Regression dan gradient boosting hanya menjadi pembanding "
    "untuk memperkirakan batas atas sinyal, dan tidak akan digunakan untuk submission."
))

cells.append(code(
    "X_probe, _, y_probe, _ = train_test_split(\n"
    "    X_all, y_all, train_size=CFG.PROBE_SAMPLE, stratify=y_all, random_state=CFG.SEED\n"
    ")\n"
    "cv = StratifiedKFold(CFG.PROBE_FOLDS, shuffle=True, random_state=CFG.SEED)\n"
    "scoring = ['f1_macro', 'roc_auc', 'accuracy']\n"
    "\n"
    "experiments = {\n"
    "    'SVM linear, raw': make_model('raw', SVC(kernel='linear')),\n"
    "    'SVM poly d=3, raw': make_model('raw', SVC(kernel='poly', degree=3, coef0=1)),\n"
    "    'SVM gaussian, raw': make_model('raw', SVC(kernel='rbf')),\n"
    "    'SVM linear, engineered': make_model('engineered', SVC(kernel='linear')),\n"
    "    'SVM poly d=3, engineered': make_model('engineered', SVC(kernel='poly', degree=3, coef0=1)),\n"
    "    'SVM gaussian, engineered': make_model('engineered', SVC(kernel='rbf')),\n"
    "    'SVM gaussian, engineered, balanced': make_model('engineered', SVC(kernel='rbf', class_weight='balanced')),\n"
    "    'SVM gaussian, engineered, tanpa scaling': make_model('engineered', SVC(kernel='rbf'), scale=False),\n"
    "    'Pembanding: logistic, engineered': make_model('engineered', LogisticRegression(max_iter=2000)),\n"
    "    'Pembanding: gradient boosting, raw': make_model(\n"
    "        'raw', HistGradientBoostingClassifier(random_state=CFG.SEED)\n"
    "    ),\n"
    "}\n"
    "\n"
    "rows = []\n"
    "for name, model in experiments.items():\n"
    "    start = time.perf_counter()\n"
    "    result = cross_validate(model, X_probe, y_probe, cv=cv, scoring=scoring, n_jobs=-1)\n"
    "    rows.append({\n"
    "        'eksperimen': name,\n"
    "        'macro_f1': result['test_f1_macro'].mean(),\n"
    "        'macro_f1_std': result['test_f1_macro'].std(),\n"
    "        'roc_auc': result['test_roc_auc'].mean(),\n"
    "        'accuracy': result['test_accuracy'].mean(),\n"
    "        'detik': time.perf_counter() - start,\n"
    "    })\n"
    "\n"
    "probe = pd.DataFrame(rows).set_index('eksperimen')\n"
    "probe.round(4)"
))

cells.append(insight("kernel"))

cells.append(md(
    "## Ukuran Data Latih\n\n"
    "Biaya SVM kernel tumbuh jauh lebih cepat dari jumlah data, dan matriks kernel penuh tidak "
    "muat di memori untuk seluruh train. Kurva berikut mengukur skor, waktu latih, dan jumlah "
    "support vector terhadap ukuran sampel pada holdout tetap. Hasilnya menentukan kebutuhan "
    "solver pada implementasi from scratch."
))

cells.append(code(
    "X_dev, X_hold, y_dev, y_hold = train_test_split(\n"
    "    X_all, y_all, test_size=CFG.HOLDOUT_SIZE, stratify=y_all, random_state=CFG.SEED\n"
    ")\n"
    "\n"
    "rows = []\n"
    "for size in CFG.SAMPLE_SIZES + [len(X_dev)]:\n"
    "    if size < len(X_dev):\n"
    "        X_sub, _, y_sub, _ = train_test_split(\n"
    "            X_dev, y_dev, train_size=size, stratify=y_dev, random_state=CFG.SEED\n"
    "        )\n"
    "    else:\n"
    "        X_sub, y_sub = X_dev, y_dev\n"
    "\n"
    "    model = make_model('engineered', SVC(kernel='rbf'))\n"
    "    start = time.perf_counter()\n"
    "    model.fit(X_sub, y_sub)\n"
    "    fit_time = time.perf_counter() - start\n"
    "\n"
    "    start = time.perf_counter()\n"
    "    scores = model.decision_function(X_hold)\n"
    "    predict_time = time.perf_counter() - start\n"
    "    predictions = (scores > 0).astype(int)\n"
    "\n"
    "    rows.append({\n"
    "        'n_train': size,\n"
    "        'macro_f1': f1_score(y_hold, predictions, average='macro'),\n"
    "        'roc_auc': roc_auc_score(y_hold, scores),\n"
    "        'accuracy': accuracy_score(y_hold, predictions),\n"
    "        'n_support': int(model.named_steps['clf'].n_support_.sum()),\n"
    "        'fit_detik': fit_time,\n"
    "        'predict_detik': predict_time,\n"
    "        'kernel_matrix_gb': size ** 2 * 8 / 1e9,\n"
    "    })\n"
    "\n"
    "curve = pd.DataFrame(rows).set_index('n_train')\n"
    "display(curve.round(4))\n"
    "\n"
    "fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))\n"
    "for metric, color, label in [('roc_auc', CFG.NEG_COLOR, 'ROC-AUC'), ('macro_f1', CFG.POS_COLOR, 'Macro F1')]:\n"
    "    axes[0].plot(curve.index, curve[metric], color=color, linewidth=2, marker='o', markersize=5, label=label)\n"
    "axes[0].set_xscale('log')\n"
    "axes[0].set_title('Skor holdout terhadap ukuran data')\n"
    "axes[0].set_xlabel('Jumlah data latih')\n"
    "axes[0].legend()\n"
    "\n"
    "axes[1].plot(curve.index, curve['fit_detik'], color=CFG.NEG_COLOR, linewidth=2, marker='o', markersize=5)\n"
    "axes[1].set_xscale('log')\n"
    "axes[1].set_yscale('log')\n"
    "axes[1].set_title('Waktu latih libsvm')\n"
    "axes[1].set_xlabel('Jumlah data latih')\n"
    "axes[1].set_ylabel('Detik')\n"
    "\n"
    "axes[2].plot(curve.index, curve['n_support'], color=CFG.NEG_COLOR, linewidth=2, marker='o', markersize=5)\n"
    "axes[2].set_xscale('log')\n"
    "axes[2].set_yscale('log')\n"
    "axes[2].set_title('Jumlah support vector')\n"
    "axes[2].set_xlabel('Jumlah data latih')\n"
    "save_fig('svm_sample_size')\n"
    "\n"
    "growth = np.polyfit(np.log(curve.index), np.log(curve['fit_detik']), 1)[0]\n"
    "print(f'Eksponen pertumbuhan waktu latih: n^{growth:.2f}')\n"
    "\n"
    "reference = make_model('engineered', HistGradientBoostingClassifier(random_state=CFG.SEED))\n"
    "reference_score = reference.fit(X_dev, y_dev).predict_proba(X_hold)[:, 1]\n"
    "print(f'Pembanding boosting pada data yang sama: ROC-AUC={roc_auc_score(y_hold, reference_score):.4f}, '\n"
    "      f'Macro F1={f1_score(y_hold, reference_score > 0.5, average=\"macro\"):.4f}')\n"
    "\n"
    "n_full = len(X_all)\n"
    "print(f'Matriks kernel penuh n={n_full}: {n_full ** 2 * 8 / 1e9:.2f} GB (float64), '\n"
    "      f'{n_full ** 2 * 4 / 1e9:.2f} GB (float32)')"
))

cells.append(insight("sample_size"))

cells.append(md(
    "## Threshold untuk Macro F1\n\n"
    "Batas keputusan bawaan SVM berada pada skor 0 dan tidak dioptimalkan untuk Macro F1. Pergeseran "
    "threshold dibandingkan dengan pembobotan kelas memakai prediksi out-of-fold pada data "
    "pengembangan, sehingga holdout tidak dipakai untuk memilih threshold."
))

cells.append(code(
    "def oof_scores(class_weight):\n"
    "    scores = np.zeros(len(X_sel))\n"
    "    for fit_idx, val_idx in cv.split(X_sel, y_sel):\n"
    "        model = make_model('engineered', SVC(kernel='rbf', class_weight=class_weight))\n"
    "        model.fit(X_sel.iloc[fit_idx], y_sel.iloc[fit_idx])\n"
    "        scores[val_idx] = model.decision_function(X_sel.iloc[val_idx])\n"
    "    return scores\n"
    "\n"
    "X_sel, _, y_sel, _ = train_test_split(\n"
    "    X_dev, y_dev, train_size=8000, stratify=y_dev, random_state=CFG.SEED\n"
    ")\n"
    "thresholds = np.linspace(-1.5, 1.5, 61)\n"
    "\n"
    "fig, ax = plt.subplots(figsize=(7, 3.4))\n"
    "rows = []\n"
    "for label, weight, color in [('Tanpa bobot', None, CFG.NEG_COLOR), ('Balanced', 'balanced', CFG.POS_COLOR)]:\n"
    "    scores = oof_scores(weight)\n"
    "    curve_f1 = np.array([f1_score(y_sel, scores > t, average='macro') for t in thresholds])\n"
    "    best = int(curve_f1.argmax())\n"
    "    ax.plot(thresholds, curve_f1, color=color, linewidth=2, label=label)\n"
    "    rows.append({\n"
    "        'konfigurasi': label,\n"
    "        'macro_f1_t0': f1_score(y_sel, scores > 0, average='macro'),\n"
    "        'threshold_terbaik': thresholds[best],\n"
    "        'macro_f1_terbaik': curve_f1[best],\n"
    "    })\n"
    "ax.axvline(0, color=CFG.MUTED, linewidth=1, linestyle='--')\n"
    "ax.set_title('Macro F1 out-of-fold terhadap threshold (8.000 baris)')\n"
    "ax.set_xlabel('Threshold pada skor keputusan')\n"
    "ax.set_ylabel('Macro F1')\n"
    "ax.legend()\n"
    "save_fig('threshold_macro_f1')\n"
    "\n"
    "pd.DataFrame(rows).set_index('konfigurasi').round(4)"
))

cells.append(insight("threshold"))

# ---------------------------------------------------------------------------
# Section 10: Conclusion
# ---------------------------------------------------------------------------
cells.append(section("Kesimpulan dan Rekomendasi", "10"))

cells.append(md(INSIGHTS.get("conclusion", "Belum diisi.")))

# ---------------------------------------------------------------------------
# Notebook writer
# ---------------------------------------------------------------------------
nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.14.7"},
    },
    "cells": cells,
}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Wrote {len(cells)} cells -> {OUT}")
